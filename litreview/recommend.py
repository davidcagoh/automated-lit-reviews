"""
Semantic Scholar paper recommendations: find papers semantically similar
to the included set, using SPECTER2 embeddings via the S2 Recommendations API.

Included + seed papers → positive examples
Excluded papers        → negative examples

Requires SEMANTIC_SCHOLAR_API_KEY for best results (higher rate limits,
larger result sets). The free tier is heavily rate-limited and may 403.
"""
from __future__ import annotations

import logging

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .ingest import _parse_s2_paper, _s2_fields, _s2_headers, deduplicate
from .models import Paper

logger = logging.getLogger(__name__)

_S2_RECO_URL = "https://api.semanticscholar.org/recommendations/v1/papers"
_MAX_POSITIVE = 100   # S2 API hard limit
_MAX_NEGATIVE = 100   # S2 API hard limit


def _s2_is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(_s2_is_retryable),
)
def _post_recommendations(
    positive_ids: list[str],
    negative_ids: list[str],
    limit: int,
) -> list[dict]:
    resp = httpx.post(
        _S2_RECO_URL,
        headers=_s2_headers(),
        params={"fields": _s2_fields(), "limit": limit},
        json={
            "positivePaperIds": positive_ids[:_MAX_POSITIVE],
            "negativePaperIds": negative_ids[:_MAX_NEGATIVE],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("recommendedPapers", [])


def fetch_s2_recommendations(
    project_id: str,
    positive_s2_ids: list[str],
    negative_s2_ids: list[str],
    depth: int,
    limit: int = 500,
) -> list[Paper]:
    """
    Call the S2 Recommendations API and return parsed Paper objects ready for upsert.

    positive_s2_ids: S2 IDs of seed + included papers (what we want more of).
    negative_s2_ids: S2 IDs of excluded papers (what we want to avoid).
    depth: depth to assign returned candidates (use frontier_depth + 1).
    limit: max recommendations to request (capped at 500 by S2).

    If fewer than 1 positive ID is available, returns [] — the API requires at least one.
    When the positive set exceeds 100 IDs, we sample the most recent 100 (highest depth
    included papers are most representative of the evolved criteria).
    """
    if not positive_s2_ids:
        logger.warning("No positive S2 IDs — skipping recommendations.")
        return []

    # If we have more than the API limit, prefer the most recently included papers
    # (passed in at the end of the list by the caller — caller should sort by depth desc)
    pos = positive_s2_ids[-_MAX_POSITIVE:]
    neg = negative_s2_ids[-_MAX_NEGATIVE:]

    try:
        raw = _post_recommendations(pos, neg, limit)
    except (httpx.HTTPStatusError, RetryError) as e:
        logger.warning("S2 recommendations request failed: %s", e)
        return []

    papers: list[Paper] = []
    for item in raw:
        try:
            paper = _parse_s2_paper(item)
            paper.project_id = project_id
            paper.source = "search"
            paper.depth = depth
            papers.append(paper)
        except Exception as e:
            logger.debug("Failed to parse recommended paper: %s", e)

    logger.info("S2 recommendations: %d raw → %d after dedup", len(raw), len(papers))
    return deduplicate(papers)
