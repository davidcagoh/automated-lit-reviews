"""
Compare pipeline corpus against a list of papers from a Claude.ai chat session.

Usage:
    python scripts/compare_with_chat.py \
        --config projects/geometry-testing-simplicial-complexes/project.toml \
        --chat-papers chat_papers.txt

chat_papers.txt format — one paper per line, any of:
    - "Author et al. (Year). Title."
    - "Title (Year)"
    - arXiv ID (e.g. 2305.04802)
    - DOI (e.g. 10.1145/3519935.3519989)
    - Free-form title string

The script resolves each entry against Semantic Scholar, then cross-references
against the pipeline DB to produce three lists:
    A. In pipeline, not mentioned by chat  → pipeline-only
    B. Mentioned by chat, in pipeline      → overlap
    C. Mentioned by chat, not in pipeline  → chat-only (missed by pipeline)

For C, it also checks whether each paper actually exists in S2 (hallucination check).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from litreview.db import get_client, get_or_create_project, get_papers
from litreview.ingest import fetch_s2_by_title, fetch_s2_by_id, _arxiv_to_s2_id, _doi_to_s2_id
from rapidfuzz import fuzz


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_line(line: str) -> str:
    """Return the most useful search string from a raw bibliography line."""
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    # Strip leading numbers/bullets
    line = re.sub(r"^\d+[\.\)]\s*", "", line)
    # arXiv ID
    m = re.match(r"(\d{4}\.\d{4,5})(v\d+)?$", line.split()[0])
    if m:
        return m.group(1)
    # DOI
    if re.match(r"10\.\d{4,}/", line):
        return line.split()[0]
    if line.lower().startswith("doi:"):
        return line[4:].strip().split()[0]
    if line.lower().startswith("arxiv:"):
        return line[6:].strip().split()[0]
    # Otherwise treat as title/free-form — strip author prefix patterns like
    # "Smith et al. (2023)." or "Smith, J. & Jones, A. (2022)."
    line = re.sub(r"^[A-Z][^.]+(?:et al\.?|&[^(]+)?\s*\(\d{4}\)[.,]?\s*", "", line)
    # Strip trailing DOI/URL
    line = re.sub(r"\s+https?://\S+$", "", line)
    line = re.sub(r"\s+doi:\S+$", "", line, flags=re.IGNORECASE)
    return line.strip(' "\'.*')


def _resolve_to_s2(query: str) -> tuple[str | None, str | None]:
    """
    Resolve a query string to (s2_id, title).
    Returns (None, None) if not found.
    """
    query = query.strip()
    if not query:
        return None, None

    # arXiv
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", query):
        p = fetch_s2_by_id(_arxiv_to_s2_id(query))
        if p:
            return p.s2_id, p.title
        return None, None

    # DOI
    if re.match(r"^10\.\d{4,}/", query):
        p = fetch_s2_by_id(_doi_to_s2_id(query))
        if p:
            return p.s2_id, p.title
        return None, None

    # Title search
    candidates = fetch_s2_by_title(query, top_k=3)
    if candidates:
        best = max(candidates, key=lambda c: fuzz.token_sort_ratio(
            query.lower(), (c.title or "").lower()
        ))
        score = fuzz.token_sort_ratio(query.lower(), (best.title or "").lower())
        if score >= 60:
            return best.s2_id, best.title
        # Return anyway but flag low confidence
        return best.s2_id, f"[LOW CONF {score}%] {best.title}"
    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pipeline corpus vs Claude chat paper list.")
    parser.add_argument("--config", required=True, type=Path, help="Path to project.toml")
    parser.add_argument("--chat-papers", required=True, type=Path, help="Text file with one paper per line")
    parser.add_argument("--out", type=Path, default=None, help="Write report to this file (default: stdout)")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    project_name = cfg["project"]["name"]

    client = get_client()
    project_id = get_or_create_project(client, project_name)

    # Load pipeline corpus
    pipeline_papers = get_papers(client, project_id, status="included")
    pipeline_s2_ids = {p["s2_id"] for p in pipeline_papers if p.get("s2_id")}
    pipeline_dois   = {p["doi"].lower() for p in pipeline_papers if p.get("doi")}
    pipeline_by_s2  = {p["s2_id"]: p for p in pipeline_papers if p.get("s2_id")}

    # Index pipeline by normalised title for fuzzy fallback
    pipeline_titles = [(p["title"], p) for p in pipeline_papers if p.get("title")]

    def in_pipeline(s2_id: str | None, resolved_title: str | None) -> dict | None:
        if s2_id and s2_id in pipeline_s2_ids:
            return pipeline_by_s2[s2_id]
        if resolved_title:
            clean = re.sub(r"\[LOW CONF.*?\]\s*", "", resolved_title)
            for title, p in pipeline_titles:
                if fuzz.token_sort_ratio(clean.lower(), title.lower()) >= 88:
                    return p
        return None

    # Parse chat paper list
    lines = args.chat_papers.read_text().splitlines()
    queries = [_parse_line(l) for l in lines]
    queries = [q for q in queries if q]

    print(f"Resolving {len(queries)} papers from chat list against S2...\n", file=sys.stderr)

    chat_resolved: list[dict] = []
    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q[:70]}", file=sys.stderr)
        s2_id, title = _resolve_to_s2(q)
        chat_resolved.append({
            "query": q,
            "s2_id": s2_id,
            "resolved_title": title,
            "exists_in_s2": s2_id is not None,
            "pipeline_match": in_pipeline(s2_id, title),
        })
        time.sleep(0.5)

    # Categorise
    overlap     = [r for r in chat_resolved if r["pipeline_match"] is not None]
    chat_only   = [r for r in chat_resolved if r["pipeline_match"] is None and r["exists_in_s2"]]
    hallucinated = [r for r in chat_resolved if not r["exists_in_s2"]]

    # Pipeline-only: in pipeline but s2_id never appeared in chat_resolved
    chat_s2_ids = {r["s2_id"] for r in chat_resolved if r["s2_id"]}
    pipeline_only = [
        p for p in pipeline_papers
        if p.get("s2_id") not in chat_s2_ids
        and not any(
            fuzz.token_sort_ratio(p["title"].lower(), (r["resolved_title"] or "").lower()) >= 88
            for r in chat_resolved if r["resolved_title"]
        )
    ]

    # ── Report ────────────────────────────────────────────────────────────────
    out = open(args.out, "w") if args.out else sys.stdout

    def section(title: str) -> None:
        print(f"\n{'='*70}", file=out)
        print(f"  {title}", file=out)
        print(f"{'='*70}", file=out)

    section("SUMMARY")
    print(f"  Pipeline corpus         : {len(pipeline_papers)} papers", file=out)
    print(f"  Chat list (raw)         : {len(queries)} entries", file=out)
    print(f"  Resolved in S2          : {len(chat_resolved) - len(hallucinated)}", file=out)
    print(f"  Hallucinated / not found: {len(hallucinated)}", file=out)
    print(f"", file=out)
    print(f"  Overlap (both)          : {len(overlap)}", file=out)
    print(f"  Pipeline-only (missed by chat)  : {len(pipeline_only)}", file=out)
    print(f"  Chat-only (missed by pipeline)  : {len(chat_only)}", file=out)

    if hallucinated:
        section(f"HALLUCINATED / NOT FOUND IN S2 ({len(hallucinated)})")
        for r in hallucinated:
            print(f"  - {r['query'][:90]}", file=out)

    section(f"OVERLAP — in both ({len(overlap)})")
    for r in overlap:
        p = r["pipeline_match"]
        print(f"  ✓ {r['resolved_title'][:75]}", file=out)

    section(f"PIPELINE-ONLY — missed by chat ({len(pipeline_only)})")
    pipeline_only.sort(key=lambda p: (p.get("depth", 0), p.get("year") or 0))
    for p in pipeline_only:
        authors = p.get("authors") or []
        last = [a["name"].split()[-1] for a in authors if a.get("name")]
        cite = f"{last[0]} et al." if len(last) > 2 else " & ".join(last) if last else "Unknown"
        print(f"  - ({cite} {p.get('year','n.d.')}) {p['title'][:65]}  [depth={p.get('depth',0)}, src={p.get('source','')}]", file=out)

    section(f"CHAT-ONLY — missed by pipeline ({len(chat_only)})")
    for r in chat_only:
        conf = "[LOW CONF] " if "LOW CONF" in (r["resolved_title"] or "") else ""
        print(f"  - {conf}{r['resolved_title'][:75]}", file=out)
        print(f"    query: {r['query'][:70]}", file=out)
        print(f"    s2_id: {r['s2_id']}", file=out)

    if args.out:
        out.close()
        print(f"\nReport written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
