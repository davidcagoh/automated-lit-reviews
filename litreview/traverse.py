"""
Citation traversal: expand the corpus via OpenAlex reference and citation graphs.

For each seed paper (depth=0) with a known OpenAlex ID:
  - backward: fetch the papers it references
  - forward:  fetch the papers that cite it

Results are upserted into `papers` and edges into `citations`.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Literal

from supabase import Client

from .db import get_papers, upsert_paper, upsert_citation
from .ingest import deduplicate
from .models import Author, Paper

logger = logging.getLogger(__name__)

Direction = Literal["forward", "backward", "both"]

_OA_PREFIX = "https://openalex.org/"


# ── OpenAlex helpers ──────────────────────────────────────────────────────────

def _short_id(oa_id: str) -> str:
    """'https://openalex.org/W123' → 'W123'."""
    return oa_id.replace(_OA_PREFIX, "").strip("/")


def _oa_work_to_paper(work: dict) -> Paper | None:
    title = (work.get("title") or "").strip()
    if not title:
        return None

    doi = None
    if work.get("doi"):
        doi = work["doi"].replace("https://doi.org/", "").strip()

    # ArXiv ID from locations
    arxiv_id = None
    for loc in (work.get("locations") or []):
        url = loc.get("landing_page_url") or ""
        if "arxiv.org/abs/" in url:
            arxiv_id = url.split("/abs/")[-1].split("v")[0]
            break
    if not arxiv_id:
        arxiv_raw = (work.get("ids") or {}).get("arxiv", "")
        if arxiv_raw:
            arxiv_id = arxiv_raw.split("/abs/")[-1].split("v")[0]

    # Abstract from inverted index
    abstract = None
    aii = work.get("abstract_inverted_index")
    if aii:
        positions = [(pos, word) for word, pos_list in aii.items() for pos in pos_list]
        abstract = " ".join(w for _, w in sorted(positions))

    venue = None
    primary = work.get("primary_location") or {}
    src = primary.get("source") or {}
    venue = src.get("display_name")

    authors = [
        Author(
            name=(a.get("author") or {}).get("display_name", ""),
            openalex_id=(a.get("author") or {}).get("id"),
        )
        for a in (work.get("authorships") or [])
    ]

    return Paper(
        openalex_id=work.get("id"),
        doi=doi,
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        year=work.get("publication_year"),
        venue=venue,
        authors=authors,
    )


def _resolve_openalex_id(paper_row: dict) -> str | None:
    """Return openalex_id from DB row, or look it up if missing."""
    if paper_row.get("openalex_id"):
        return paper_row["openalex_id"]
    try:
        from pyalex import Works
        doi = paper_row.get("doi")
        title = paper_row.get("title", "")
        results = Works().filter(doi=doi).get() if doi else Works().search_filter(title=title).get(per_page=3)
        if results:
            return results[0].get("id")
    except Exception as e:
        logger.warning("Could not find OpenAlex ID for '%s': %s", paper_row.get("title", "")[:50], e)
    return None


def _fetch_references(openalex_id: str) -> list[dict]:
    """Papers that this work cites (backward). Fetched via referenced_works IDs."""
    try:
        from pyalex import Works
        work = Works()[openalex_id]
        ref_ids = [_short_id(r) for r in (work.get("referenced_works") or [])]
        if not ref_ids:
            return []
        results: list[dict] = []
        # OpenAlex allows pipe-separated OR filters; batch in chunks of 50
        for i in range(0, len(ref_ids), 50):
            chunk = ref_ids[i : i + 50]
            batch = Works().filter(openalex_id="|".join(chunk)).get(per_page=50)
            results.extend(batch)
            time.sleep(0.3)
        return results
    except Exception as e:
        logger.warning("References fetch failed for %s: %s", openalex_id, e)
        return []


def _fetch_citing(openalex_id: str, max_results: int = 300) -> list[dict]:
    """Papers that cite this work (forward)."""
    try:
        from pyalex import Works
        return Works().filter(cites=_short_id(openalex_id)).get(per_page=min(max_results, 200))
    except Exception as e:
        logger.warning("Citing-works fetch failed for %s: %s", openalex_id, e)
        return []


# ── Public entry point ────────────────────────────────────────────────────────

def traverse_citations(
    client: Client,
    project_id: str,
    direction: Direction = "both",
    from_depth: int = 0,
) -> dict[str, int]:
    """
    Expand the corpus via citation edges, starting from papers at *from_depth*.

    from_depth=0  → starts from seeds (depth 0 → produces depth 1 papers)
    from_depth=1  → starts from included depth-1 papers (→ depth 2 papers);
                    falls back to ALL depth-1 papers if none are screened yet

    Returns {"candidates": N, "papers_upserted": N, "edges_stored": N}.
    """
    all_papers = get_papers(client, project_id)
    at_depth = [p for p in all_papers if p.get("depth", 0) == from_depth]

    if from_depth == 0:
        starting_set = at_depth  # all seeds
    else:
        included = [p for p in at_depth if p.get("inclusion_status") == "included"]
        starting_set = included if included else at_depth
        logger.info(
            "Depth-%d frontier: %d included papers (of %d total at depth %d)%s",
            from_depth + 1, len(included), len(at_depth), from_depth,
            " — using all (none screened yet)" if not included else "",
        )

    if not starting_set:
        logger.warning("No papers at depth=%d to traverse from.", from_depth)
        return {"candidates": 0, "papers_upserted": 0, "edges_stored": 0}

    logger.info("Traversing from %d papers at depth=%d (direction=%s)", len(starting_set), from_depth, direction)

    candidate_works: list[dict] = []           # raw OpenAlex works
    edge_map: dict[str, str] = {}              # oa_id → source paper_id (for edge storage)
    target_depth = from_depth + 1

    for source_paper in starting_set:
        source_id = source_paper["paper_id"]
        oa_id = _resolve_openalex_id(source_paper)
        if not oa_id:
            logger.warning("No OpenAlex ID for '%s'; skipping.", source_paper.get("title", "")[:50])
            continue

        if direction in ("backward", "both"):
            refs = _fetch_references(oa_id)
            logger.info("  ← backward: %d refs from '%s'", len(refs), source_paper["title"][:45])
            for w in refs:
                candidate_works.append(w)
                edge_map[w.get("id", "")] = source_id
            time.sleep(0.5)

        if direction in ("forward", "both"):
            citing = _fetch_citing(oa_id)
            logger.info("  → forward:  %d citing from '%s'", len(citing), source_paper["title"][:45])
            for w in citing:
                candidate_works.append(w)
                edge_map[w.get("id", "")] = source_id
            time.sleep(0.5)

    # Parse → Paper objects
    parsed: list[Paper] = []
    for work in candidate_works:
        p = _oa_work_to_paper(work)
        if p:
            p.project_id = project_id
            p.source = "citation"
            p.depth = target_depth
            parsed.append(p)

    # Dedup within this batch
    parsed = deduplicate(parsed)

    # Build existing-ID sets so we can count truly new papers and protect seeds
    existing = all_papers  # already fetched above
    existing_s2  = {p["s2_id"]       for p in existing if p.get("s2_id")}
    existing_doi = {p["doi"]          for p in existing if p.get("doi")}
    existing_oa  = {p["openalex_id"]  for p in existing if p.get("openalex_id")}

    # Seeds must never be overwritten by traversal upserts
    seed_dois  = {p["doi"]          for p in existing if p.get("source") == "seed" and p.get("doi")}
    seed_s2ids = {p["s2_id"]        for p in existing if p.get("source") == "seed" and p.get("s2_id")}
    seed_oa    = {p["openalex_id"]  for p in existing if p.get("source") == "seed" and p.get("openalex_id")}

    papers_upserted = 0
    edges_stored = 0
    # Pre-populate oa_to_db_id from existing rows so seeds get edges without upserting
    oa_to_db_id: dict[str, str] = {
        p["openalex_id"]: p["paper_id"]
        for p in existing if p.get("openalex_id")
    }

    for paper in parsed:
        is_seed = (
            (paper.doi         and paper.doi         in seed_dois)
            or (paper.s2_id    and paper.s2_id       in seed_s2ids)
            or (paper.openalex_id and paper.openalex_id in seed_oa)
        )
        if is_seed:
            # Already a seed — skip upsert so source/depth are not overwritten
            continue

        is_new = not (
            (paper.s2_id       and paper.s2_id       in existing_s2)
            or (paper.doi      and paper.doi          in existing_doi)
            or (paper.openalex_id and paper.openalex_id in existing_oa)
        )
        try:
            row = upsert_paper(client, paper)
            oa_to_db_id[paper.openalex_id or ""] = row["paper_id"]
            if is_new:
                papers_upserted += 1
        except Exception as e:
            logger.warning("Upsert failed for '%s': %s", paper.title[:50], e)

    for oa_id, seed_paper_id in edge_map.items():
        candidate_db_id = oa_to_db_id.get(oa_id)
        if not candidate_db_id or candidate_db_id == seed_paper_id:
            continue
        try:
            upsert_citation(client, project_id, seed_paper_id, candidate_db_id)
            edges_stored += 1
        except Exception:
            pass

    return {
        "candidates": len(candidate_works),
        "papers_upserted": papers_upserted,
        "edges_stored": edges_stored,
    }
