"""
Instrumented wrapper around the LLM screening call.

Re-implements the HTTP layer from litreview/screen.py (without tenacity) so we
can capture the full resp.json() and extract token usage. One 429-retry is added
with a 5-second wait — sufficient for a small test set; the tournament is not
trying to saturate rate limits.

Returns per-paper dicts with decision, reasoning, token counts, latency, and
parse_ok so metrics.py can compute cost and accuracy downstream.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from litreview.config import settings
from litreview.screen import (
    _GROQ_URL,
    _GEMINI_URL,
    _OPENROUTER_URL,
)

# ── Cost table (USD per 1M tokens, update when prices change) ─────────────────
# Format: "backend/model" → (input_per_1m, output_per_1m)
# Groq: https://groq.com/pricing  Gemini: https://ai.google.dev/pricing
COST_PER_1M: dict[str, tuple[float, float]] = {
    "groq/llama-3.3-70b-versatile":                    (0.59, 0.79),
    "groq/llama-3.1-8b-instant":                       (0.05, 0.08),
    "groq/meta-llama/llama-4-scout-17b-16e-instruct":  (0.11, 0.34),
    "gemini/gemini-2.5-flash":                         (0.15, 0.60),
}
# Keep old name as alias for backwards compatibility with metrics.py import
COST_PER_1K = COST_PER_1M

# ── Optimal rate sleeps (seconds between calls) ───────────────────────────────
# Based on Developer plan limits at ~600 tokens/call.
# groq/llama-3.3-70b: TPM-bound (12K/600=20 calls/min → 3.0s)
# groq/llama-3.1-8b:  TPM-bound (6K/600=10 calls/min  → 6.0s)
# groq/llama-4-scout: RPM-bound (30 RPM               → 2.0s)
# gemini paid Tier 1: RPM-bound (300 RPM              → 0.2s)
DEFAULT_RATE_SLEEP: dict[str, float] = {
    "groq/llama-3.3-70b-versatile":                    3.0,
    "groq/llama-3.1-8b-instant":                       6.0,
    "groq/meta-llama/llama-4-scout-17b-16e-instruct":  2.0,
    "gemini/gemini-2.5-flash":                         0.2,
}


def _api_key(backend: str) -> str:
    keys = {
        "groq":       settings.groq_api_key,
        "gemini":     settings.gemini_api_key,
        "openrouter": settings.openrouter_api_key,
        "anthropic":  settings.anthropic_api_key,
    }
    key = keys.get(backend)
    if not key:
        raise RuntimeError(f"No API key configured for backend '{backend}'")
    return key


def _raw_call_openai_compat(
    system: str, user: str, model: str, api_key: str, url: str
) -> dict[str, Any]:
    """Make one OpenAI-compat call, retry once on 429. Returns full resp.json()."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(2):
        resp = httpx.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 429 and attempt == 0:
            time.sleep(5)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # unreachable but satisfies type checker


def _raw_call_anthropic(
    system: str, user: str, model: str, api_key: str
) -> dict[str, Any]:
    """Make one Anthropic call, retry once on 429. Returns full resp.json()."""
    payload = {
        "model": model,
        "max_tokens": 256,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": 0.1,
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    for attempt in range(2):
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload, timeout=30,
        )
        if resp.status_code == 429 and attempt == 0:
            time.sleep(5)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()


def _extract_content(backend: str, raw: dict[str, Any]) -> str:
    if backend == "anthropic":
        return raw["content"][0]["text"]
    return raw["choices"][0]["message"]["content"]


def _parse_decision(content: str) -> tuple[str, str, bool]:
    """Parse JSON decision from LLM response. Returns (decision, reasoning, parse_ok)."""
    try:
        parsed = json.loads(content)
        decision = parsed.get("decision", "uncertain")
        reasoning = parsed.get("reasoning", "")
        if decision not in ("include", "exclude", "uncertain"):
            raise ValueError(f"Invalid decision: {decision!r}")
        return decision, reasoning, True
    except (json.JSONDecodeError, ValueError):
        return "uncertain", content[:200], False


def screen_paper_instrumented(
    title: str,
    abstract: str | None,
    criteria: str,
    model: str,
    backend: str,
    prompt_template: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Screen one paper and return rich metrics alongside the decision.

    Returns:
        {
            "decision":      "include" | "exclude" | "uncertain",
            "reasoning":     str,
            "input_tokens":  int,
            "output_tokens": int,
            "latency_s":     float,
            "parse_ok":      bool,
            "est_cost_usd":  float,
        }
    """
    # Load prompt template from file path or use provided string
    if prompt_template is None:
        _default = Path(__file__).parent / "prompts" / "v1_baseline.txt"
        prompt_template = _default.read_text()

    system = prompt_template.format(criteria=criteria)
    user_msg = f"Title: {title}\n\nAbstract: {abstract or '(not available)'}"

    key = api_key or _api_key(backend)

    t0 = time.monotonic()
    url_map = {"groq": _GROQ_URL, "gemini": _GEMINI_URL, "openrouter": _OPENROUTER_URL}
    if backend == "anthropic":
        raw = _raw_call_anthropic(system, user_msg, model, key)
    else:
        raw = _raw_call_openai_compat(system, user_msg, model, key, url_map[backend])
    latency = time.monotonic() - t0

    # Extract token usage (field names differ by provider)
    usage = raw.get("usage", {})
    input_tokens  = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)

    # tiktoken fallback if backend doesn't return usage (e.g. some OpenRouter free models)
    if not input_tokens:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            input_tokens  = len(enc.encode(system + user_msg))
            output_tokens = 80  # rough estimate
        except ImportError:
            input_tokens, output_tokens = 0, 0

    content = _extract_content(backend, raw)
    decision, reasoning, parse_ok = _parse_decision(content)

    # Cost estimate
    cost_key = f"{backend}/{model}"
    prices = COST_PER_1K.get(cost_key, (0.0, 0.0))
    est_cost = input_tokens / 1_000_000 * prices[0] + output_tokens / 1_000_000 * prices[1]

    return {
        "decision":     decision,
        "reasoning":    reasoning,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s":    round(latency, 3),
        "parse_ok":     parse_ok,
        "est_cost_usd": round(est_cost, 6),
    }
