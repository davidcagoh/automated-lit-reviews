"""
LLM screening of candidate papers against inclusion criteria.

Supports four backends:
  - "groq"        : GROQ API (OpenAI-compatible, llama-3.3-70b-versatile by default)
  - "gemini"      : Google Gemini via OpenAI-compat endpoint (gemini-2.0-flash by default)
  - "openrouter"  : OpenRouter (any free/paid model, qwen/qwen3-30b-a3b:free by default)
  - "anthropic"   : Anthropic API (claude-haiku-4-5-20251001 by default)

Uncertain papers stay 'pending' with reasoning stored in rejection_reason
prefixed by "UNCERTAIN: " for human review.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import settings

logger = logging.getLogger(__name__)

_GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
_GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_GROQ_MODEL       = "llama-3.3-70b-versatile"
_DEFAULT_GEMINI_MODEL     = "gemini-2.5-flash"
_DEFAULT_OPENROUTER_MODEL = "qwen/qwen3-30b-a3b:free"
_DEFAULT_ANTHROPIC_MODEL  = "claude-haiku-4-5-20251001"

# Seconds to sleep between calls per backend (stay under free-tier rate limits)
_RATE_SLEEP = {
    "groq":       1.2,   # ~30 RPM free tier
    "gemini":     4.5,   # 15 RPM free tier (Google AI Studio)
    "openrouter": 6.0,   # conservative; free models vary
    "anthropic":  0.0,
}

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

def _is_retryable_anthropic(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=5, max=60),
    retry=retry_if_exception(_is_retryable_anthropic),
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

    backend: "groq" (default) or "anthropic"
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
    backend: str = "groq",
) -> dict[str, int]:
    """
    Screen all 'pending' papers for *project_id* using the latest criteria.

    Decision mapping to DB inclusion_status:
      include   → 'included'
      exclude   → 'excluded'  (reasoning stored in rejection_reason)
      uncertain → stays 'pending', reasoning stored as "UNCERTAIN: ..." in rejection_reason

    Returns {"total", "included", "excluded", "uncertain", "skipped"}.
    """
    from .db import get_papers, get_current_criteria, update_paper_screening, log_iteration

    criteria_version, criteria_content = get_current_criteria(client, project_id)
    if not criteria_content:
        raise RuntimeError("No criteria found for this project. Run 'litreview ingest' first.")

    pending = get_papers(client, project_id, status="pending")
    _defaults = {
        "anthropic": _DEFAULT_ANTHROPIC_MODEL,
        "gemini": _DEFAULT_GEMINI_MODEL,
        "openrouter": _DEFAULT_OPENROUTER_MODEL,
        "groq": _DEFAULT_GROQ_MODEL,
    }
    effective_model = model or _defaults.get(backend, _DEFAULT_GROQ_MODEL)
    logger.info(
        "Screening %d pending papers (criteria v%d, backend=%s, model=%s, round=%d)",
        len(pending), criteria_version, backend, effective_model, screening_round,
    )

    counts: dict[str, int] = {
        "total": len(pending), "include": 0, "exclude": 0, "uncertain": 0, "skipped": 0
    }

    for i, paper in enumerate(pending):
        title = paper["title"]
        abstract = paper.get("abstract")

        try:
            result = screen_paper(title, abstract, criteria_content, model=model, backend=backend)
        except Exception as e:
            logger.error("[%d/%d] SKIP '%s': %s", i + 1, len(pending), title[:55], e)
            counts["skipped"] += 1
            continue

        decision = result["decision"]
        reasoning = result["reasoning"]
        counts[decision] += 1

        # Map to DB enum values
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

        sleep_s = _RATE_SLEEP.get(backend, 0.0)
        if sleep_s:
            time.sleep(sleep_s)

    # Log iteration stats
    screened = counts["total"] - counts["skipped"]
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
