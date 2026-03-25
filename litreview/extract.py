"""
Structured extraction of key information from included papers.

Runs post-stability: for each included paper, calls an LLM to extract a
fixed set of universal fields plus any project-specific fields defined in
project.toml under [[extraction.extra_fields]].

Input: title + abstract (abstract-first; skips gracefully if missing).
Output: written to the `extractions` table as JSONB.
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
_DEFAULT_ANTHROPIC_MODEL  = "claude-sonnet-4-6"

_RATE_SLEEP = {
    "groq":       1.2,
    "gemini":     4.5,
    "openrouter": 6.0,
    "anthropic":  0.0,
}

# ── Universal extraction schema ───────────────────────────────────────────────

UNIVERSAL_FIELDS: list[dict[str, str]] = [
    {
        "name": "contribution_type",
        "description": (
            "Primary type of contribution. One of: theoretical, empirical, survey, framework, other."
        ),
    },
    {
        "name": "problem_statement",
        "description": (
            "The precise problem being solved or question being answered. "
            "Be specific — include mathematical setting or domain if stated."
        ),
    },
    {
        "name": "setting",
        "description": (
            "Assumptions, constraints, and domain of the work "
            "(e.g. 'finite metric spaces', 'supervised classification with i.i.d. data', "
            "'undirected graphs with bounded degree'). Null if not applicable."
        ),
    },
    {
        "name": "methods",
        "description": (
            "Techniques, algorithms, architectures, or proof strategies used. "
            "List the primary ones."
        ),
    },
    {
        "name": "main_results",
        "description": (
            "Key theorems, bounds, empirical findings, or claims. "
            "State them precisely if possible (e.g. 'O(n log n) approximation ratio'). "
            "Return as a list."
        ),
    },
    {
        "name": "limitations",
        "description": (
            "Explicitly stated or apparent scope restrictions, failure modes, "
            "or assumptions that limit generality. Null if none stated."
        ),
    },
    {
        "name": "related_work_positioning",
        "description": (
            "How this paper situates itself relative to prior work: "
            "what it improves on, contrasts with, or extends."
        ),
    },
    {
        "name": "open_questions",
        "description": (
            "Future work, open problems, or unsolved questions the paper raises. "
            "Null if none stated."
        ),
    },
]


def _build_system_prompt(fields: list[dict[str, str]]) -> str:
    field_lines = "\n".join(
        f'- "{f["name"]}": {f["description"]}' for f in fields
    )
    return (
        "You are a research assistant extracting structured information from academic papers "
        "for a systematic literature review. Given a paper's title and abstract, extract the "
        "following fields and return them as a single JSON object.\n\n"
        "FIELDS:\n"
        f"{field_lines}\n\n"
        "Rules:\n"
        "- Return only the JSON object, no other text.\n"
        "- Use null (not empty string) for fields where the abstract provides no information.\n"
        "- For list fields (methods, main_results), return a JSON array.\n"
        "- Be precise and concise. Do not hallucinate details not present in the abstract.\n"
        "- If the abstract is missing, extract what you can from the title alone and note "
        "  '(title only)' in the problem_statement field."
    )


# ── LLM API calls ─────────────────────────────────────────────────────────────

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
            "max_tokens": 1024,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0.1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


# ── Core extraction logic ─────────────────────────────────────────────────────

def extract_paper(
    title: str,
    abstract: str | None,
    extra_fields: list[dict[str, str]] | None = None,
    model: str | None = None,
    backend: str = "groq",
) -> dict[str, Any]:
    """
    Extract structured fields from a single paper's title and abstract.
    Returns a dict ready to be stored as JSONB in extractions.data.
    """
    fields = UNIVERSAL_FIELDS + (extra_fields or [])
    system = _build_system_prompt(fields)
    user_msg = f"Title: {title}\n\nAbstract: {abstract or '(not available)'}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    if backend == "anthropic":
        api_key = settings.anthropic_api_key
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        raw = _call_anthropic(system, user_msg, model or _DEFAULT_ANTHROPIC_MODEL, api_key)
    elif backend == "gemini":
        api_key = settings.gemini_api_key
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        raw = _call_openai_compat(messages, model or _DEFAULT_GEMINI_MODEL, api_key, _GEMINI_URL)
    elif backend == "openrouter":
        api_key = settings.openrouter_api_key
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        raw = _call_openai_compat(messages, model or _DEFAULT_OPENROUTER_MODEL, api_key, _OPENROUTER_URL)
    else:  # groq
        api_key = settings.groq_api_key
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        raw = _call_openai_compat(messages, model or _DEFAULT_GROQ_MODEL, api_key, _GROQ_URL)

    try:
        # Strip markdown code fences if the model wraps its JSON
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Could not parse extraction response (%s) — raw: %.300s", e, raw)
        return {"_parse_error": raw[:500]}


# ── Project-level extraction ──────────────────────────────────────────────────

def extract_project(
    client: Any,
    project_id: str,
    extra_fields: list[dict[str, str]] | None = None,
    model: str | None = None,
    backend: str = "groq",
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, int]:
    """
    Extract structured fields for all included papers in a project.

    Skips papers that already have an extraction unless force=True.
    Returns {"total", "extracted", "skipped_existing", "skipped_no_abstract", "failed"}.
    """
    from .db import get_papers

    included = [p for p in get_papers(client, project_id, status="included")]
    if not included:
        logger.warning("No included papers found for extraction.")
        return {"total": 0, "extracted": 0, "skipped_existing": 0, "skipped_no_abstract": 0, "failed": 0}

    # Fetch existing extraction paper_ids to skip
    existing_ids: set[str] = set()
    if not force:
        res = (
            client.table("extractions")
            .select("paper_id")
            .eq("project_id", project_id)
            .execute()
        )
        existing_ids = {row["paper_id"] for row in res.data}

    counts: dict[str, int] = {
        "total": len(included),
        "extracted": 0,
        "skipped_existing": 0,
        "skipped_no_abstract": 0,
        "failed": 0,
    }

    for i, paper in enumerate(included):
        paper_id = paper["paper_id"]
        title = paper["title"]
        abstract = paper.get("abstract")

        if not force and paper_id in existing_ids:
            logger.debug("[%d/%d] SKIP (already extracted) '%s'", i + 1, len(included), title[:55])
            counts["skipped_existing"] += 1
            continue

        if not abstract:
            logger.info("[%d/%d] SKIP (no abstract) '%s'", i + 1, len(included), title[:55])
            counts["skipped_no_abstract"] += 1
            continue

        try:
            data = extract_paper(title, abstract, extra_fields=extra_fields, model=model, backend=backend)
        except Exception as e:
            logger.error("[%d/%d] FAIL '%s': %s", i + 1, len(included), title[:55], e)
            counts["failed"] += 1
            continue

        logger.info("[%d/%d] EXTRACTED '%s'", i + 1, len(included), title[:55])

        if not dry_run:
            client.table("extractions").upsert(
                {"paper_id": paper_id, "project_id": project_id, "data": data},
                on_conflict="paper_id,project_id",
            ).execute()

        counts["extracted"] += 1

        time.sleep(_RATE_SLEEP.get(backend, 0.0))

    return counts
