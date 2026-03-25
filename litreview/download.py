"""
Download open-access PDFs for included papers into projects/<slug>/included-pdfs/.

Sources tried in order:
  1. arXiv (if paper has arxiv_id)
  2. S2 openAccessPdf field
  3. Skip (log warning)
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)

_S2_BASE = "https://api.semanticscholar.org/graph/v1"


def _s2_headers() -> dict:
    h = {"Accept": "application/json"}
    if settings.s2_api_key:
        h["x-api-key"] = settings.s2_api_key
    return h


def _download_bytes(url: str) -> bytes | None:
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30,
                         headers={"User-Agent": "litreview-download/0.1"})
        if resp.status_code == 200 and b"%PDF" in resp.content[:10]:
            return resp.content
        logger.debug("PDF download returned %d for %s", resp.status_code, url)
    except Exception as e:
        logger.debug("PDF download error for %s: %s", url, e)
    return None


def _get_pdf_url(paper: dict) -> tuple[str | None, str]:
    """
    Return (url, source_label) for the best available PDF.
    source_label is one of: "arxiv", "s2_oa", "none"
    """
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        clean = re.sub(r"v\d+$", "", arxiv_id.strip())
        return f"https://arxiv.org/pdf/{clean}.pdf", "arxiv"

    # Try S2 openAccessPdf
    s2_id = paper.get("s2_id")
    doi = paper.get("doi")
    lookup = s2_id or (f"DOI:{doi}" if doi else None)
    if lookup:
        try:
            resp = httpx.get(
                f"{_S2_BASE}/paper/{lookup}",
                headers=_s2_headers(),
                params={"fields": "openAccessPdf,externalIds"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                ext = data.get("externalIds") or {}
                if ext.get("ArXiv"):
                    clean = re.sub(r"v\d+$", "", ext["ArXiv"])
                    return f"https://arxiv.org/pdf/{clean}.pdf", "arxiv"
                pdf_info = data.get("openAccessPdf") or {}
                if pdf_info.get("url"):
                    return pdf_info["url"], "s2_oa"
        except Exception as e:
            logger.debug("S2 PDF URL lookup failed for %s: %s", lookup, e)

    return None, "none"


def download_included_pdfs(
    papers: list[dict],
    dest_dir: Path,
    skip_existing: bool = True,
) -> dict[str, int]:
    """
    Download PDFs for *papers* into *dest_dir*.

    Returns {"downloaded": N, "skipped": N, "failed": N}.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"downloaded": 0, "skipped": 0, "failed": 0}

    for paper in papers:
        title = (paper.get("title") or "")[:60]
        paper_id = paper.get("paper_id", "unknown")

        # Determine filename — prefer arxiv_id, fall back to paper_id prefix
        arxiv_id = paper.get("arxiv_id")
        if arxiv_id:
            filename = re.sub(r"v\d+$", "", arxiv_id.strip()) + ".pdf"
        else:
            filename = paper_id[:8] + ".pdf"

        dest_path = dest_dir / filename

        if skip_existing and dest_path.exists():
            logger.debug("Skipping (exists): %s", filename)
            counts["skipped"] += 1
            continue

        url, source = _get_pdf_url(paper)
        if url is None:
            logger.warning("No PDF source found for '%s'", title)
            counts["failed"] += 1
            continue

        pdf_bytes = _download_bytes(url)
        if pdf_bytes:
            dest_path.write_bytes(pdf_bytes)
            logger.info("Downloaded [%s] %s → %s", source, title, filename)
            counts["downloaded"] += 1
        else:
            logger.warning("Download failed for '%s' (url: %s)", title, url)
            counts["failed"] += 1

        time.sleep(0.5)  # gentle rate limiting

    return counts
