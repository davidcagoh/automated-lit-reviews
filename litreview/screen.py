"""
LLM screening of candidate papers against inclusion criteria.

Supports four backends:
  - "groq"        : GROQ API (OpenAI-compatible, llama-3.3-70b-versatile by default)
  - "gemini"      : Google Gemini via OpenAI-compat endpoint (gemini-2.5-flash by default)
  - "openrouter"  : OpenRouter (any free/paid model, qwen/qwen3-30b-a3b:free by default)
  - "anthropic"   : Anthropic API (claude-haiku-4-5-20251001 by default)

Pass a comma-separated list to enable automatic failover, e.g. --backend gemini,groq.
The screener will rotate to the next backend after _CONSECUTIVE_ERROR_LIMIT consecutive
errors on the current one, and print a live progress summary every _CHUNK_SIZE papers.

Uncertain papers stay 'pending' with reasoning stored in rejection_reason
prefixed by "UNCERTAIN: " for human review.

QA loop (screen_project with qa=True):
  After bulk screening, a stratified sample of recently-screened papers is re-checked
  with Claude Haiku (Anthropic). If agreement < qa_threshold, disagreements are printed
  and the run stops — the caller should refine criteria and re-screen before continuing.
  Recommended: qa_sample=75, qa_threshold=0.90.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import settings

logger = logging.getLogger(__name__)

_GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
_GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Groq model priority (paid tier, --rate-sleep 0 recommended):
#   llama-3.3-70b-versatile                  — default; best accuracy (~100% oracle agreement)
#   meta-llama/llama-4-scout-17b-16e-instruct — good quality/speed balance ($0.10/1k)
#   openai/gpt-oss-20b                       — fastest + cheapest ($0.077/1k), lower accuracy
#   llama-3.1-8b-instant                     — cheapest ($0.037/1k), test JSON reliability first
_DEFAULT_GROQ_MODEL       = "llama-3.3-70b-versatile"
_DEFAULT_GEMINI_MODEL     = "gemini-2.5-flash"
_DEFAULT_OPENROUTER_MODEL = "qwen/qwen3-30b-a3b:free"
_DEFAULT_ANTHROPIC_MODEL  = "claude-haiku-4-5-20251001"

# Seconds to sleep between calls per backend (stay under rate limits)
_RATE_SLEEP = {
    "groq":       2.0,   # paid Developer plan: 30 RPM / 12K TPM for llama-3.3-70b-versatile (~2.5s/paper TPM-limited)
    "gemini":     2.0,   # paid tier, ~30 RPM to avoid burst 429s
    "openrouter": 6.0,   # conservative; free models vary
    "anthropic":  0.5,
}

# Rotate to next backend after this many back-to-back errors on the current one
_CONSECUTIVE_ERROR_LIMIT = 5

# Rough API round-trip latency per backend (seconds, on top of rate sleep)
_EST_LATENCY = {
    "gemini":     0.8,
    "groq":       0.3,
    "openrouter": 3.0,
    "anthropic":  0.6,
}

# Prompt for confirmation when batch exceeds this size
_CONFIRM_THRESHOLD = 200

# Print a progress summary every N papers
_CHUNK_SIZE = 50

# Billing/quota dashboard URLs — shown in the interactive exhaustion prompt
_BILLING_URLS = {
    "gemini":     "https://aistudio.google.com",
    "groq":       "https://console.groq.com",
    "openrouter": "https://openrouter.ai/account",
    "anthropic":  "https://console.anthropic.com",
}


def _get_sleep(backend: str, override: float | None) -> float:
    """Return seconds to sleep after each call, respecting any CLI override."""
    if override is not None:
        return override
    return _RATE_SLEEP.get(backend, 0.0)


def _prompt_backend_action(
    backend: str,
    next_backend: str | None,
    console: Any,
) -> str:
    """Interactively ask the user what to do after repeated failures.

    Returns 'retry', 'switch', or 'quit'.
    """
    url = _BILLING_URLS.get(backend, "your provider's dashboard")
    console.print(
        f"\n[bold red]Backend '{backend}' has failed {_CONSECUTIVE_ERROR_LIMIT} "
        f"times in a row.[/bold red]"
    )
    console.print(
        f"[yellow]  → Check your quota / billing at: {url}[/yellow]"
    )
    if next_backend:
        prompt_text = f"  [r]etry {backend} / [s]witch to {next_backend} / [q]uit > "
    else:
        prompt_text = f"  [r]etry {backend} / [q]uit (no more backends) > "

    while True:
        try:
            choice = input(prompt_text).strip().lower()
        except EOFError:
            return "quit"
        if choice in ("r", "retry"):
            return "retry"
        if choice in ("s", "switch") and next_backend:
            return "switch"
        if choice in ("q", "quit"):
            return "quit"
        console.print("  Enter r, s, or q.")

_SYSTEM_PROMPT = """\
You are a systematic literature review screener. Given a paper's title and abstract, \
decide whether it meets the inclusion criteria below.

INCLUSION CRITERIA:
{criteria}

Respond with a JSON object with exactly two keys:
- "decision": one of "include", "exclude", or "uncertain"
- "reasoning": a concise (1-3 sentence) justification

If the abstract is missing, base your decision on the title alone and note this in reasoning.\
"""


# ── OpenAI-compatible API call (Groq, Gemini, OpenRouter) ─────────────────────

def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=5, max=60),
    retry=retry_if_exception(_is_retryable),
)
def _call_openai_compat(messages: list[dict], model: str, api_key: str, url: str) -> str:
    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Anthropic API call ────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=5, max=60),
    retry=retry_if_exception(_is_retryable),
)
def _call_anthropic(system: str, user: str, model: str, api_key: str) -> str:
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 256,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0.1,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


# ── Backend probe ─────────────────────────────────────────────────────────────

def probe_backend(backend: str) -> float | None:
    """
    Send one minimal request to *backend*. Returns latency in seconds, or None if unavailable.
    Used before batch runs to rank available backends by speed.
    """
    _defaults = {
        "anthropic": _DEFAULT_ANTHROPIC_MODEL,
        "gemini": _DEFAULT_GEMINI_MODEL,
        "openrouter": _DEFAULT_OPENROUTER_MODEL,
        "groq": _DEFAULT_GROQ_MODEL,
    }
    model = _defaults.get(backend, _DEFAULT_GROQ_MODEL)
    system = "You are a classifier. Respond with JSON only."
    user = 'Title: Test\n\nAbstract: Test.\n\nRespond: {"decision":"exclude","reasoning":"test"}'
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    t0 = time.monotonic()
    try:
        if backend == "anthropic":
            key = settings.anthropic_api_key
            if not key:
                return None
            _call_anthropic(system, user, model, key)
        elif backend == "gemini":
            key = settings.gemini_api_key
            if not key:
                return None
            _call_openai_compat(messages, model, key, _GEMINI_URL)
        elif backend == "openrouter":
            key = settings.openrouter_api_key
            if not key:
                return None
            _call_openai_compat(messages, model, key, _OPENROUTER_URL)
        else:  # groq
            key = settings.groq_api_key
            if not key:
                return None
            _call_openai_compat(messages, model, key, _GROQ_URL)
        return time.monotonic() - t0
    except Exception:
        return None


# ── Screening logic ───────────────────────────────────────────────────────────

def screen_paper(
    title: str,
    abstract: str | None,
    criteria: str,
    model: str | None = None,
    api_key: str | None = None,
    backend: str = "groq",
) -> dict[str, str]:
    """
    Screen a single paper. Returns {"decision": "include|exclude|uncertain", "reasoning": "..."}.
    backend: one of "groq", "gemini", "openrouter", "anthropic"
    """
    system = _SYSTEM_PROMPT.format(criteria=criteria)
    user_msg = f"Title: {title}\n\nAbstract: {abstract or '(not available)'}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    if backend == "anthropic":
        resolved_api_key = api_key or settings.anthropic_api_key
        if not resolved_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        resolved_model = model or _DEFAULT_ANTHROPIC_MODEL
        raw = _call_anthropic(system, user_msg, resolved_model, resolved_api_key)
    elif backend == "gemini":
        resolved_api_key = api_key or settings.gemini_api_key
        if not resolved_api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        resolved_model = model or _DEFAULT_GEMINI_MODEL
        raw = _call_openai_compat(messages, resolved_model, resolved_api_key, _GEMINI_URL)
    elif backend == "openrouter":
        resolved_api_key = api_key or settings.openrouter_api_key
        if not resolved_api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        resolved_model = model or _DEFAULT_OPENROUTER_MODEL
        raw = _call_openai_compat(messages, resolved_model, resolved_api_key, _OPENROUTER_URL)
    else:  # groq (default)
        resolved_api_key = api_key or settings.groq_api_key
        if not resolved_api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        resolved_model = model or _DEFAULT_GROQ_MODEL
        raw = _call_openai_compat(messages, resolved_model, resolved_api_key, _GROQ_URL)

    try:
        result = json.loads(raw)
        if result.get("decision") not in ("include", "exclude", "uncertain"):
            raise ValueError(f"Unexpected decision value: {result.get('decision')!r}")
        return {"decision": result["decision"], "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Could not parse LLM response (%s) — raw: %.200s", e, raw)
        return {"decision": "uncertain", "reasoning": f"Parse error: {raw[:200]}"}


# ── Project-level screening ───────────────────────────────────────────────────

def screen_project(
    client: Any,
    project_id: str,
    screening_round: int,
    model: str | None = None,
    dry_run: bool = False,
    backend: str = "gemini",
    limit: int | None = None,
    rate_sleep_override: float | None = None,
    include_uncertain: bool = False,
    interactive: bool = True,
    skip_confirmation: bool = False,
) -> dict[str, int]:
    """
    Screen all 'pending' papers for *project_id* using the latest criteria.

    backend can be a comma-separated list for automatic failover, e.g. "gemini,groq".
    The screener probes each backend at startup (ranked by latency), starts with the
    fastest, and rotates to the next after _CONSECUTIVE_ERROR_LIMIT consecutive errors.

    rate_sleep_override: if set, overrides per-backend _RATE_SLEEP for all backends.
    include_uncertain: if True, clears the "UNCERTAIN: " prefix from pending papers
        so they receive a clean re-screen without carrying forward stale reasoning.
    interactive: if True (default when stdin is a tty), prompts the user on backend
        exhaustion instead of raising immediately.

    Decision mapping to DB inclusion_status:
      include   → 'included'
      exclude   → 'excluded'  (reasoning stored in rejection_reason)
      uncertain → stays 'pending', reasoning stored as "UNCERTAIN: ..." in rejection_reason

    Returns {"total", "included", "excluded", "uncertain", "skipped"}.
    """
    from .db import get_papers, get_current_criteria, update_paper_screening, log_iteration
    from rich.console import Console

    console = Console()

    criteria_version, criteria_content = get_current_criteria(client, project_id)
    if not criteria_content:
        raise RuntimeError("No criteria found for this project. Run 'litreview ingest' first.")

    # Optionally clear UNCERTAIN: prefix so previously-uncertain papers get a clean re-screen
    if include_uncertain and not dry_run:
        resp = (
            client.table("papers")
            .update({"rejection_reason": None})
            .eq("project_id", project_id)
            .eq("inclusion_status", "pending")
            .like("rejection_reason", "UNCERTAIN:%")
            .execute()
        )
        cleared = len(resp.data) if resp.data else 0
        if cleared:
            console.print(f"[cyan]Cleared UNCERTAIN prefix from {cleared} papers for clean re-screen.[/cyan]")

    pending = get_papers(client, project_id, status="pending")
    if limit is not None:
        pending = pending[:limit]

    # Show how many of the pending batch are previously-uncertain
    n_uncertain = sum(
        1 for p in pending
        if (p.get("rejection_reason") or "").startswith("UNCERTAIN:")
    )
    if n_uncertain:
        console.print(f"[dim]  ({n_uncertain} of {len(pending)} pending are previously-uncertain)[/dim]")

    # Parse and probe backends
    backend_list = [b.strip() for b in backend.split(",") if b.strip()]
    if len(backend_list) > 1:
        console.print(f"[cyan]Probing {len(backend_list)} backends…[/cyan]")
        latencies: dict[str, float] = {}
        for b in backend_list:
            t = probe_backend(b)
            status = f"{t*1000:.0f}ms" if t is not None else "unavailable"
            console.print(f"  {b}: {status}")
            if t is not None:
                latencies[b] = t
        if not latencies:
            raise RuntimeError("No backends available — check API keys and connectivity.")
        # Sort by latency; unavailable backends go to end of rotation
        available = sorted(latencies, key=lambda b: latencies[b])
        unavailable = [b for b in backend_list if b not in latencies]
        backend_list = available + unavailable
        console.print(f"[green]Using order: {' → '.join(backend_list)}[/green]")

    active_backend_idx = 0
    active_backend = backend_list[active_backend_idx]

    _defaults = {
        "anthropic": _DEFAULT_ANTHROPIC_MODEL,
        "gemini": _DEFAULT_GEMINI_MODEL,
        "openrouter": _DEFAULT_OPENROUTER_MODEL,
        "groq": _DEFAULT_GROQ_MODEL,
    }
    effective_model = model or _defaults.get(active_backend, _DEFAULT_GROQ_MODEL)

    # Upfront time estimate
    sleep_s_est = _get_sleep(active_backend, rate_sleep_override)
    est_per_paper = sleep_s_est + _EST_LATENCY.get(active_backend, 1.0)
    est_total_min = (len(pending) * est_per_paper) / 60
    console.print(
        f"Screening round {screening_round} | "
        f"backend={active_backend} | "
        f"{len(pending)} pending | "
        f"~{est_per_paper:.1f}s/paper | "
        f"ETA ~{est_total_min:.0f} min"
    )

    # Confirm before committing to a large batch
    if not skip_confirmation and interactive and sys.stdin.isatty() and len(pending) > _CONFIRM_THRESHOLD:
        console.print(
            f"[yellow]  That's a large batch ({len(pending)} papers ≈ {est_total_min:.0f} min).[/yellow]"
        )
        try:
            confirm = input("  Proceed? [y/N] ").strip().lower()
        except EOFError:
            confirm = "y"
        if confirm not in ("y", "yes"):
            raise RuntimeError("Screening cancelled by user.")

    counts: dict[str, int] = {
        "total": len(pending), "include": 0, "exclude": 0, "uncertain": 0, "skipped": 0
    }
    consecutive_errors = 0
    chunk_start_time = time.monotonic()
    chunk_start_i = 0

    for i, paper in enumerate(pending):
        title = paper["title"]
        abstract = paper.get("abstract")

        # Chunk progress summary
        if i > 0 and i % _CHUNK_SIZE == 0:
            elapsed = time.monotonic() - chunk_start_time
            chunk_papers = i - chunk_start_i
            rate = elapsed / chunk_papers if chunk_papers else 0
            remaining = len(pending) - i
            eta_min = (remaining * rate) / 60
            done_so_far = counts["include"] + counts["exclude"] + counts["uncertain"]
            console.print(
                f"[bold cyan][Progress {i}/{len(pending)}][/bold cyan] "
                f"{counts['include']} incl  {counts['exclude']} excl  "
                f"{counts['uncertain']} unc  {counts['skipped']} skip | "
                f"{rate:.1f}s/paper | ETA ~{eta_min:.0f}min | backend={active_backend}"
            )
            chunk_start_time = time.monotonic()
            chunk_start_i = i

        try:
            result = screen_paper(
                title, abstract, criteria_content,
                model=effective_model, backend=active_backend,
            )
            consecutive_errors = 0
        except Exception as e:
            logger.error("[%d/%d] SKIP '%s': %s", i + 1, len(pending), title[:55], e)
            counts["skipped"] += 1
            consecutive_errors += 1

            # Prompt or rotate after too many consecutive errors
            if consecutive_errors >= _CONSECUTIVE_ERROR_LIMIT:
                next_idx = active_backend_idx + 1
                has_next = next_idx < len(backend_list)
                next_backend_name = backend_list[next_idx] if has_next else None

                if interactive and sys.stdin.isatty():
                    action = _prompt_backend_action(active_backend, next_backend_name, console)
                else:
                    # Non-interactive: auto-switch if possible, otherwise hard-stop
                    action = "switch" if has_next else "quit"

                if action == "retry":
                    consecutive_errors = 0
                    console.print(f"[yellow]Retrying with {active_backend}…[/yellow]")
                elif action == "switch":
                    old = active_backend
                    active_backend_idx = next_idx
                    active_backend = backend_list[active_backend_idx]
                    effective_model = model or _defaults.get(active_backend, _DEFAULT_GROQ_MODEL)
                    consecutive_errors = 0
                    # Sleep with new backend's rate before the first call to it
                    switch_sleep = _get_sleep(active_backend, rate_sleep_override)
                    if switch_sleep:
                        time.sleep(switch_sleep)
                    console.print(
                        f"[yellow][Backend switch] {old} → {active_backend} "
                        f"after {_CONSECUTIVE_ERROR_LIMIT} consecutive errors[/yellow]"
                    )
                else:  # quit
                    raise RuntimeError(
                        f"Screening stopped: '{active_backend}' failed "
                        f"{_CONSECUTIVE_ERROR_LIMIT} times with no fallback. "
                        f"Check quota at {_BILLING_URLS.get(active_backend, 'your provider')}."
                    )
            continue

        decision = result["decision"]
        reasoning = result["reasoning"]
        counts[decision] += 1

        if decision == "include":
            db_status = "included"
            db_reason = None
        elif decision == "exclude":
            db_status = "excluded"
            db_reason = reasoning
        else:  # uncertain
            db_status = "pending"
            db_reason = f"UNCERTAIN: {reasoning}"

        logger.info(
            "[%d/%d] %-8s '%s'",
            i + 1, len(pending), decision.upper(), title[:55],
        )

        if not dry_run:
            update_paper_screening(
                client,
                paper_id=paper["paper_id"],
                status=db_status,
                rejection_reason=db_reason,
                screening_round=screening_round,
                criteria_version=criteria_version,
            )

        sleep_s = _get_sleep(active_backend, rate_sleep_override)
        if sleep_s:
            time.sleep(sleep_s)

    # Final summary
    screened = counts["total"] - counts["skipped"]
    console.print(
        f"\n[bold]Screening round {screening_round} complete[/bold] | "
        f"backend(s) used: {', '.join(backend_list[:active_backend_idx+1])} | "
        f"{screened} screened"
    )

    if not dry_run and screened > 0:
        yield_rate = counts["include"] / screened if screened else 0.0
        log_iteration(
            client,
            project_id=project_id,
            round_number=screening_round,
            yield_rate=yield_rate,
            criteria_version=criteria_version,
        )

    return counts


# ── QA sampling ───────────────────────────────────────────────────────────────

def qa_screen(
    client: Any,
    project_id: str,
    screening_round: int,
    sample_size: int = 75,
    threshold: float = 0.90,
    qa_model: str = _DEFAULT_ANTHROPIC_MODEL,
) -> dict:
    """
    Quality-check a stratified sample of papers screened in *screening_round*
    by re-screening them with Claude Haiku (Anthropic) and comparing decisions.

    Sampling strategy: 30% from includes, 70% from excludes — overweights borderline
    includes where calibration errors matter most.

    Returns a dict:
      {
        "agreement": float,          # fraction of matched decisions
        "total_sampled": int,
        "disagreements": [           # list of disagreement dicts
          {"title": ..., "bulk_decision": ..., "qa_decision": ..., "qa_reasoning": ...}
        ],
        "passed": bool,              # agreement >= threshold
      }

    If 'passed' is False, the caller should print disagreements, refine criteria,
    and re-screen before proceeding — re-running QA without criteria changes will
    produce the same result.
    """
    import random
    from .db import get_current_criteria
    from rich.console import Console

    console = Console()

    criteria_version, criteria_content = get_current_criteria(client, project_id)

    # Fetch all papers screened in this round (included + excluded, not seeds)
    included = (
        client.table("papers")
        .select("paper_id, title, abstract, inclusion_status")
        .eq("project_id", project_id)
        .eq("screening_round", screening_round)
        .eq("inclusion_status", "included")
        .neq("source", "seed")
        .execute()
        .data or []
    )
    excluded = (
        client.table("papers")
        .select("paper_id, title, abstract, inclusion_status")
        .eq("project_id", project_id)
        .eq("screening_round", screening_round)
        .eq("inclusion_status", "excluded")
        .neq("source", "seed")
        .execute()
        .data or []
    )

    # Stratified sample: 30% includes, 70% excludes (proportional to their relative importance)
    n_inc = min(round(sample_size * 0.30), len(included))
    n_exc = min(sample_size - n_inc, len(excluded))
    sample = random.sample(included, n_inc) + random.sample(excluded, n_exc)
    random.shuffle(sample)

    if not sample:
        console.print("[yellow]QA: no papers to sample from this screening round.[/yellow]")
        return {"agreement": 1.0, "total_sampled": 0, "disagreements": [], "passed": True}

    console.print(
        f"\n[bold cyan]QA check[/bold cyan] | "
        f"sample={len(sample)} ({n_inc} incl + {n_exc} excl) | "
        f"threshold={threshold:.0%} | model={qa_model}"
    )

    api_key = settings.anthropic_api_key
    if not api_key:
        # No API key: dump sample for in-chat QA instead of calling the API
        import json, tempfile, os
        out_path = os.path.join(tempfile.gettempdir(), f"qa_sample_{project_id[:8]}.json")
        with open(out_path, "w") as f:
            json.dump(
                [
                    {
                        "title": p["title"],
                        "abstract": p.get("abstract"),
                        "bulk_decision": "include" if p["inclusion_status"] == "included" else "exclude",
                    }
                    for p in sample
                ],
                f, indent=2,
            )
        console.print(
            f"\n[yellow]ANTHROPIC_API_KEY not set — cannot run automated QA.[/yellow]\n"
            f"Sample written to: [bold]{out_path}[/bold]\n\n"
            f"To QA in-chat, ask Claude Code:\n"
            f'  "Read {out_path} and screen each paper against the current criteria '
            f"for project '{project_id}', then write your decisions to "
            f"/tmp/qa_decisions_{project_id[:8]}.json and compare against bulk_decision.\""
        )
        return {"agreement": None, "total_sampled": len(sample), "disagreements": [], "passed": None}

    agreed = 0
    disagreements = []

    for i, paper in enumerate(sample):
        title = paper["title"]
        abstract = paper.get("abstract")
        bulk_decision = "include" if paper["inclusion_status"] == "included" else "exclude"

        try:
            result = screen_paper(title, abstract, criteria_content, backend="anthropic", model=qa_model)
            qa_decision = result["decision"]
            qa_reasoning = result["reasoning"]
        except Exception as e:
            console.print(f"  [{i+1}/{len(sample)}] QA ERROR: {e}")
            continue

        match = (
            (bulk_decision == "include" and qa_decision == "include") or
            (bulk_decision == "exclude" and qa_decision in ("exclude", "uncertain"))
        )
        if match:
            agreed += 1
        else:
            disagreements.append({
                "title": title,
                "bulk_decision": bulk_decision,
                "qa_decision": qa_decision,
                "qa_reasoning": qa_reasoning,
            })
        console.print(
            f"  [{i+1:02d}/{len(sample)}] bulk={bulk_decision:<8} qa={qa_decision:<8} "
            f"{'✓' if match else '✗'} {title[:55]}"
        )
        time.sleep(0.5)  # Haiku rate limit

    agreement = agreed / len(sample) if sample else 1.0
    passed = agreement >= threshold

    console.print(
        f"\n[bold]QA result:[/bold] {agreed}/{len(sample)} agreed "
        f"({agreement:.0%}) — {'[green]PASSED ✓[/green]' if passed else '[red]FAILED ✗[/red]'} "
        f"(threshold {threshold:.0%})"
    )

    if not passed and disagreements:
        console.print("\n[bold red]Disagreements (bulk vs QA):[/bold red]")
        for d in disagreements:
            direction = f"bulk={d['bulk_decision']} → qa={d['qa_decision']}"
            console.print(f"  [{direction}] {d['title'][:70]}")
            console.print(f"    QA reasoning: {d['qa_reasoning']}")
        console.print(
            "\n[yellow]Action required:[/yellow] Review disagreements above, refine criteria "
            "in the DB (db.upsert_criteria), reset-screening, and re-screen. "
            "Re-running QA without criteria changes will produce the same result."
        )

    return {
        "agreement": agreement,
        "total_sampled": len(sample),
        "disagreements": disagreements,
        "passed": passed,
    }
