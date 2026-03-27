"""
Fetch paper metadata + SPECTER2 embeddings from Semantic Scholar and OpenAlex,
merge them, and return a Paper model ready for DB upsert.
"""
from __future__ import annotations

import re
import time
import logging
from typing import Sequence

import httpx
from rapidfuzz import fuzz
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, RetryError

from .config import settings
from .models import Author, Paper

logger = logging.getLogger(__name__)

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS_BASE = "paperId,externalIds,title,abstract,year,venue,authors,citationCount,referenceCount"
_S2_FIELDS_WITH_EMBEDDING = _S2_FIELDS_BASE + ",embedding"


def _s2_fields() -> str:
    """Include embedding only when an API key is present (required by S2)."""
    return _S2_FIELDS_WITH_EMBEDDING if settings.s2_api_key else _S2_FIELDS_BASE

# ── Helpers ───────────────────────────────────────────────────────────────────

def _s2_headers() -> dict[str, str]:
    h = {"Accept": "application/json"}
    if settings.s2_api_key:
        h["x-api-key"] = settings.s2_api_key
    return h


def _arxiv_to_s2_id(arxiv_id: str) -> str:
    """Convert bare arXiv ID to S2 lookup key, e.g. '1411.5713' → 'ARXIV:1411.5713'."""
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id.strip())  # strip version suffix
    return f"ARXIV:{arxiv_id}"


def _doi_to_s2_id(doi: str) -> str:
    return f"DOI:{doi}"


def _parse_s2_paper(data: dict) -> Paper:
    authors = [
        Author(name=a.get("name", ""), s2_id=a.get("authorId"))
        for a in data.get("authors", [])
    ]
    ext = data.get("externalIds") or {}
    doi = ext.get("DOI")
    arxiv_id = ext.get("ArXiv")
    embedding = None
    emb_data = data.get("embedding")
    if emb_data and isinstance(emb_data, dict):
        embedding = emb_data.get("vector")

    return Paper(
        s2_id=data.get("paperId"),
        doi=doi,
        arxiv_id=arxiv_id,
        title=data["title"],
        abstract=data.get("abstract"),
        year=data.get("year"),
        venue=data.get("venue"),
        authors=authors,
        embedding=embedding,
        citation_count=data.get("citationCount"),
        reference_count=data.get("referenceCount"),
    )


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def _s2_is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30), retry=retry_if_exception(_s2_is_retryable))
def _s2_get(url: str, params: dict | None = None) -> dict:
    resp = httpx.get(url, headers=_s2_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_s2_by_id(paper_id: str) -> Paper | None:
    """
    paper_id can be:
      - S2 hash        e.g. '649def34f8be52c8b66281af98ae884c09aef38a'
      - 'ARXIV:1411.5713'
      - 'DOI:10.xxx/yyy'
    """
    try:
        data = _s2_get(f"{_S2_BASE}/paper/{paper_id}", params={"fields": _s2_fields()})
        return _parse_s2_paper(data)
    except (httpx.HTTPStatusError, RetryError) as e:
        logger.warning("S2 fetch failed for %s: %s", paper_id, e)
        return None


def fetch_s2_by_title(title: str, top_k: int = 3) -> list[Paper]:
    """Search S2 by title query and return up to top_k results."""
    try:
        data = _s2_get(
            f"{_S2_BASE}/paper/search",
            params={"query": title, "fields": _s2_fields(), "limit": top_k},
        )
        return [_parse_s2_paper(p) for p in data.get("data", [])]
    except Exception as e:
        logger.warning("S2 title search failed for '%s': %s", title, e)
        return []


def fetch_s2_batch(paper_ids: list[str]) -> list[Paper]:
    """Batch fetch up to 500 papers from S2."""
    if not paper_ids:
        return []
    try:
        resp = httpx.post(
            f"{_S2_BASE}/paper/batch",
            headers=_s2_headers(),
            params={"fields": _s2_fields()},
            json={"ids": paper_ids},
            timeout=60,
        )
        resp.raise_for_status()
        return [_parse_s2_paper(p) for p in resp.json() if p]
    except Exception as e:
        logger.warning("S2 batch fetch failed: %s", e)
        return []


# ── OpenAlex ──────────────────────────────────────────────────────────────────

def _fetch_openalex_by_doi(doi: str) -> dict | None:
    try:
        from pyalex import Works
        results = Works().filter(doi=doi).get()
        return results[0] if results else None
    except Exception as e:
        logger.warning("OpenAlex DOI fetch failed for %s: %s", doi, e)
        return None


def _fetch_openalex_by_title(title: str) -> dict | None:
    try:
        from pyalex import Works
        results = Works().search_filter(title=title).get(per_page=3)
        return results[0] if results else None
    except Exception as e:
        logger.warning("OpenAlex title fetch failed for '%s': %s", title, e)
        return None


def _merge_openalex(paper: Paper, oa_data: dict) -> Paper:
    """Fill in any gaps in *paper* from OpenAlex data."""
    if not paper.openalex_id:
        paper.openalex_id = oa_data.get("id")
    if not paper.doi and oa_data.get("doi"):
        paper.doi = oa_data["doi"].replace("https://doi.org/", "")
    if not paper.abstract and oa_data.get("abstract_inverted_index"):
        paper.abstract = _reconstruct_abstract(oa_data["abstract_inverted_index"])
    if not paper.year and oa_data.get("publication_year"):
        paper.year = oa_data["publication_year"]
    if not paper.venue:
        source = (oa_data.get("primary_location") or {}).get("source") or {}
        paper.venue = source.get("display_name")
    if not paper.authors:
        paper.authors = [
            Author(
                name=a.get("author", {}).get("display_name", ""),
                openalex_id=a.get("author", {}).get("id"),
            )
            for a in oa_data.get("authorships", [])
        ]
    if paper.citation_count is None and oa_data.get("cited_by_count") is not None:
        paper.citation_count = oa_data["cited_by_count"]
    if paper.reference_count is None and oa_data.get("referenced_works_count") is not None:
        paper.reference_count = oa_data["referenced_works_count"]
    return paper


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """Convert OpenAlex abstract_inverted_index back to a string."""
    positions: list[tuple[int, str]] = []
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions.append((pos, word))
    return " ".join(w for _, w in sorted(positions))


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(papers: list[Paper], threshold: int = 92) -> list[Paper]:
    """
    Remove duplicates: first by DOI, then by fuzzy title similarity.
    Returns a deduplicated list preserving the first occurrence.
    """
    seen_dois: set[str] = set()
    kept: list[Paper] = []

    for paper in papers:
        # DOI dedup
        if paper.doi:
            norm_doi = paper.doi.lower().strip()
            if norm_doi in seen_dois:
                continue
            seen_dois.add(norm_doi)

        # Title fuzzy dedup
        is_dup = any(
            fuzz.token_sort_ratio(paper.title.lower(), k.title.lower()) >= threshold
            for k in kept
        )
        if not is_dup:
            kept.append(paper)

    return kept


# ── Public entry point ────────────────────────────────────────────────────────

def _classify_identifier(identifier: str) -> tuple[str, str]:
    """Return (kind, value). Kind: 'arxiv' | 'doi' | 'title'."""
    identifier = identifier.strip()
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", identifier):
        return "arxiv", identifier
    if re.match(r"^10\.\d{4,}/", identifier):
        return "doi", identifier
    if identifier.lower().startswith("arxiv:"):
        return "arxiv", identifier[6:]
    if identifier.lower().startswith("doi:"):
        return "doi", identifier[4:]
    return "title", identifier


def ingest_papers(
    identifiers: Sequence[str],
    project_id: str,
    source: str = "seed",
    depth: int = 0,
    enrich_openalex: bool = True,
) -> list[Paper]:
    """
    Resolve each identifier (DOI, arXiv ID, or title string) via Semantic Scholar
    and optionally OpenAlex. Returns deduplicated Paper objects ready for upsert.
    """
    papers: list[Paper] = []

    for ident in identifiers:
        kind, value = _classify_identifier(ident)
        paper: Paper | None = None

        if kind == "arxiv":
            paper = fetch_s2_by_id(_arxiv_to_s2_id(value))
        elif kind == "doi":
            paper = fetch_s2_by_id(_doi_to_s2_id(value))
        else:
            # Title search — take best match above similarity threshold
            candidates = fetch_s2_by_title(value, top_k=3)
            for c in candidates:
                if fuzz.token_sort_ratio(value.lower(), c.title.lower()) >= 75:
                    paper = c
                    break
            if paper is None and candidates:
                logger.warning(
                    "No strong title match for '%s'; best was '%s'",
                    value,
                    candidates[0].title,
                )
                paper = candidates[0]

        if paper is None:
            logger.error("Could not resolve identifier: %s", ident)
            continue

        # Enrich from OpenAlex
        if enrich_openalex:
            oa_data = None
            if paper.doi:
                oa_data = _fetch_openalex_by_doi(paper.doi)
            if oa_data is None:
                oa_data = _fetch_openalex_by_title(paper.title)
            if oa_data:
                paper = _merge_openalex(paper, oa_data)

        paper.project_id = project_id
        paper.source = source
        paper.depth = depth
        papers.append(paper)

        time.sleep(0.5)  # gentle rate limiting

    return deduplicate(papers)
