"""
Synthesis: generate a draft literature review section from structured extractions.

Loads all extractions for a project, joins with paper metadata, and calls an LLM
to produce a cohesive markdown narrative grouped by theme.

For large corpora (>150 papers), automatically falls back to map-reduce:
  1. Synthesize thematic mini-summaries per chunk of ~50 papers
  2. Synthesize those summaries into a final narrative

Output is written to projects/<slug>/literature-review.md.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import settings

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_CHUNK_SIZE = 50          # papers per map-reduce chunk
_MAPREDUCE_THRESHOLD = 150  # use map-reduce above this many papers

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_FULL = """\
You are an expert researcher writing a literature review section for an academic paper.
You have structured extractions from {n} papers included in a systematic review.

Project: {project_name}
Inclusion criteria summary: {criteria}

Your task:
1. Group the papers into coherent thematic clusters based on their content.
2. Within each theme, synthesise findings — identify agreements, contradictions,
   progressions, and complementary results.
3. Highlight open questions and gaps that emerge across the literature.
4. Write in scholarly academic prose, not as a list of summaries.
5. Cite every paper using EXACTLY the citation key shown in its --- (Key) --- header.
   You may cite inline as "Author (Year)" or parenthetically as "(Author Year)",
   but the author string and year must match the key exactly (including & vs "and").
6. Structure the output as markdown with ## subsection headings.
7. End with a ## Gaps and Open Questions section.

Be precise about mathematical results and theoretical claims where stated.
Do not introduce claims not supported by the extractions provided.\
"""

_SYSTEM_MAP = """\
You are summarising a subset of papers for a literature review on: {project_name}

For the {n} papers below, produce a structured JSON object with:
- "themes": list of thematic clusters, each with:
    - "label": short theme name
    - "paper_ids": list of paper s2_ids in this theme
    - "summary": 2-4 sentence synthesis of this cluster
- "key_results": list of the most significant results across these papers (with citations)
- "open_questions": list of open questions raised

Be precise. Cite using EXACTLY the key from each paper's --- (Key) --- header.\
"""

_SYSTEM_REDUCE = """\
You are writing a literature review section for an academic paper.
Project: {project_name}
Criteria: {criteria}

You have thematic summaries from {n_chunks} batches covering {n_papers} papers total.
Synthesise these into a single coherent literature review:
1. Merge overlapping themes across batches.
2. Write scholarly academic prose with inline citations using the exact keys from the summaries.
3. Use ## subsection headings per theme.
4. End with ## Gaps and Open Questions.

Do not introduce claims not present in the summaries provided.\
"""


# ── LLM call ─────────────────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=10, max=90),
    retry=retry_if_exception(_is_retryable),
)
def _call_anthropic(system: str, user: str, model: str, api_key: str, max_tokens: int = 4096) -> str:
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0.3,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


# ── Formatting helpers ────────────────────────────────────────────────────────

def _format_authors(authors_json: Any) -> str:
    """Return a short author string like 'Smith et al.' or 'Smith & Jones'."""
    if not authors_json:
        return "Unknown"
    authors = authors_json if isinstance(authors_json, list) else []
    names = [a.get("name", "") for a in authors if a.get("name")]
    if not names:
        return "Unknown"
    last_names = [n.split()[-1] for n in names]
    if len(last_names) == 1:
        return last_names[0]
    if len(last_names) == 2:
        return f"{last_names[0]} & {last_names[1]}"
    return f"{last_names[0]} et al."


def _format_paper_block(paper: dict, extraction: dict) -> str:
    """Format one paper's metadata + extraction as a text block for the prompt."""
    authors = _format_authors(paper.get("authors"))
    year = paper.get("year", "n.d.")
    title = paper.get("title", "Untitled")
    venue = paper.get("venue") or ""
    citation_key = f"({authors} {year})"

    lines = [
        f"--- {citation_key} ---",
        f"Title: {title}",
    ]
    if venue:
        lines.append(f"Venue: {venue}")
    lines.append(f"S2 ID: {paper.get('s2_id', '')}")

    data = extraction.get("data", {})
    for field, value in data.items():
        if value is None or field == "_parse_error":
            continue
        if isinstance(value, list):
            value = "; ".join(str(v) for v in value)
        lines.append(f"{field}: {value}")

    return "\n".join(lines)


# ── Bibliography ──────────────────────────────────────────────────────────────

def _build_citation_lookup(pairs: list[tuple[dict, dict]]) -> dict[str, dict]:
    """
    Return a dict mapping every citation key variant → paper dict.

    Keys are produced by _format_authors + year, e.g. "(Bangachev & Bresler 2024)".
    We also add an "and" variant since LLMs sometimes write "and" instead of "&".
    Longer keys are tested first (handled by the caller sorting by length).
    """
    lookup: dict[str, dict] = {}
    for paper, _ in pairs:
        authors = _format_authors(paper.get("authors"))
        year = paper.get("year", "n.d.")
        key = f"({authors} {year})"
        lookup[key] = paper
        alt = key.replace(" & ", " and ")
        if alt != key:
            lookup[alt] = paper
    return lookup


def _apply_numbered_bibliography(text: str, lookup: dict[str, dict]) -> str:
    """
    Replace (Key) / Key (Year) citation strings with [N] and append ## References.

    Processes citations in order of first appearance in the text.
    """
    paper_to_num: dict[str, int] = {}
    num_to_paper: dict[int, dict] = {}
    counter = 1

    def assign(paper: dict) -> int:
        nonlocal counter
        t = paper["title"]
        if t not in paper_to_num:
            paper_to_num[t] = counter
            num_to_paper[counter] = paper
            counter += 1
        return paper_to_num[t]

    # Find all citation spans; longest keys first to avoid partial shadowing
    sorted_keys = sorted(lookup, key=len, reverse=True)
    matches: list[tuple[int, int, dict]] = []  # (start, end, paper)

    for key in sorted_keys:
        paper = lookup[key]
        inner = key[1:-1]  # strip outer parens: "AuthorStr Year"
        yr_m = re.search(r"\s+(\d{4}[a-z]?)$", inner)
        if not yr_m:
            continue
        author_part = inner[: yr_m.start()]
        year_part = yr_m.group(1)

        # Parenthetical form: (AuthorStr Year)
        pat_paren = re.compile(r"\(" + re.escape(inner) + r"\)")
        # Inline form: AuthorStr (Year)
        pat_inline = re.compile(re.escape(author_part) + r"\s+\(" + re.escape(year_part) + r"\)")

        for pat in (pat_paren, pat_inline):
            for m in pat.finditer(text):
                if not any(s <= m.start() < e for s, e, _ in matches):
                    matches.append((m.start(), m.end(), paper))

    if not matches:
        logger.warning("No citation keys found in synthesised text — bibliography not added.")
        return text

    # Assign numbers in first-appearance order, then replace in reverse
    matches.sort(key=lambda x: x[0])
    span_to_num = {(s, e): assign(p) for s, e, p in matches}

    result = text
    for (start, end), n in sorted(span_to_num.items(), reverse=True):
        result = result[:start] + f"[{n}]" + result[end:]

    # Build references section
    ref_lines = ["\n\n---\n\n## References\n"]
    for n in range(1, counter):
        p = num_to_paper[n]
        all_names = [a.get("name", "") for a in (p.get("authors") or []) if a.get("name")]
        author_str = "; ".join(all_names) if all_names else "Unknown"
        year = p.get("year", "n.d.")
        title = p.get("title") or "Untitled"
        venue = p.get("venue") or ""
        doi = p.get("doi") or ""
        arxiv = p.get("arxiv_id") or ""

        line = f"{n}. {author_str} ({year}). *{title}*."
        if venue:
            line += f" {venue}."
        if doi:
            line += f" https://doi.org/{doi}"
        elif arxiv:
            line += f" arXiv:{arxiv}"
        ref_lines.append(line)

    logger.info("Bibliography: %d references added.", counter - 1)
    return result + "\n".join(ref_lines)


# ── Main synthesis logic ──────────────────────────────────────────────────────

def synthesize_project(
    client: Any,
    project_id: str,
    project_name: str,
    criteria: str,
    output_path: Path,
    model: str = _DEFAULT_MODEL,
) -> str:
    """
    Load extractions, call the LLM, write markdown to output_path.
    Returns the generated markdown string.
    """
    api_key = settings.anthropic_api_key
    if not api_key or api_key == "your-anthropic-key":
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add your key to .env: ANTHROPIC_API_KEY=sk-ant-..."
        )

    # ── Load extractions + join with paper metadata ───────────────────────────
    extractions_res = (
        client.table("extractions")
        .select("paper_id, data")
        .eq("project_id", project_id)
        .execute()
    )
    extractions_by_pid = {row["paper_id"]: row for row in extractions_res.data}

    if not extractions_by_pid:
        raise RuntimeError(
            "No extractions found. Run 'litreview extract' first."
        )

    papers = {
        p["paper_id"]: p
        for p in client.table("papers")
        .select("*")
        .eq("project_id", project_id)
        .eq("inclusion_status", "included")
        .execute()
        .data
    }

    # Only synthesise papers that have extractions
    pairs = [
        (papers[pid], extractions_by_pid[pid])
        for pid in extractions_by_pid
        if pid in papers
    ]
    pairs.sort(key=lambda x: x[0].get("year") or 0)

    n = len(pairs)
    logger.info("Synthesising %d papers (model=%s)", n, model)

    citation_lookup = _build_citation_lookup(pairs)

    if n <= _MAPREDUCE_THRESHOLD:
        raw = _synthesize_full(pairs, project_name, criteria, model, api_key)
    else:
        raw = _synthesize_mapreduce(pairs, project_name, criteria, model, api_key)

    markdown = _apply_numbered_bibliography(raw, citation_lookup)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote literature review to %s", output_path)

    return markdown


def _synthesize_full(
    pairs: list[tuple[dict, dict]],
    project_name: str,
    criteria: str,
    model: str,
    api_key: str,
) -> str:
    """Single-pass synthesis for corpora up to _MAPREDUCE_THRESHOLD papers."""
    paper_blocks = "\n\n".join(_format_paper_block(p, e) for p, e in pairs)
    system = _SYSTEM_FULL.format(
        n=len(pairs),
        project_name=project_name,
        criteria=criteria.strip()[:800],   # truncate very long criteria
    )
    user = f"Here are the {len(pairs)} papers to synthesise:\n\n{paper_blocks}"
    return _call_anthropic(system, user, model, api_key, max_tokens=6000)


def _synthesize_mapreduce(
    pairs: list[tuple[dict, dict]],
    project_name: str,
    criteria: str,
    model: str,
    api_key: str,
) -> str:
    """Map-reduce synthesis for large corpora."""
    chunks = [pairs[i : i + _CHUNK_SIZE] for i in range(0, len(pairs), _CHUNK_SIZE)]
    logger.info("Map phase: %d chunks of ~%d papers", len(chunks), _CHUNK_SIZE)

    chunk_summaries: list[str] = []
    for idx, chunk in enumerate(chunks):
        logger.info("Map chunk %d/%d (%d papers)", idx + 1, len(chunks), len(chunk))
        paper_blocks = "\n\n".join(_format_paper_block(p, e) for p, e in chunk)
        system = _SYSTEM_MAP.format(n=len(chunk), project_name=project_name)
        user = f"Papers:\n\n{paper_blocks}"
        summary = _call_anthropic(system, user, model, api_key, max_tokens=2048)
        chunk_summaries.append(f"## Batch {idx + 1}\n{summary}")

    logger.info("Reduce phase: synthesising %d batch summaries", len(chunk_summaries))
    system = _SYSTEM_REDUCE.format(
        project_name=project_name,
        criteria=criteria.strip()[:800],
        n_chunks=len(chunks),
        n_papers=len(pairs),
    )
    user = "Thematic summaries from all batches:\n\n" + "\n\n".join(chunk_summaries)
    return _call_anthropic(system, user, model, api_key, max_tokens=6000)
