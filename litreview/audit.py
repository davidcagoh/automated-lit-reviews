"""
Traversal audit: for each seed, download the paper PDF, parse the references
section, and compare against what was actually captured in the DB.

PDF is the ground truth. S2's /references API is used as a fallback when no
open-access PDF is available.

PDF sources tried in order:
  1. arXiv PDF (if paper has arxiv_id)
  2. S2 openAccessPdf field
  3. Fall back to S2 /references API
"""
from __future__ import annotations

import io
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from supabase import Client

from .config import settings
from .db import get_papers
from .traverse import _resolve_openalex_id

logger = logging.getLogger(__name__)

_S2_BASE = "https://api.semanticscholar.org/graph/v1"


# ── PDF download ──────────────────────────────────────────────────────────────

def _s2_headers() -> dict:
    h = {"Accept": "application/json"}
    if settings.s2_api_key:
        h["x-api-key"] = settings.s2_api_key
    return h


def _get_local_pdf(paper_row: dict, project_dir: Path | None) -> bytes | None:
    """
    Look for a local PDF in the project directory.
    Matches on arXiv ID prefix (ignoring version suffix).
    """
    if not project_dir or not project_dir.exists():
        return None
    arxiv_id = paper_row.get("arxiv_id")
    if not arxiv_id:
        return None
    base = re.sub(r"v\d+$", "", arxiv_id.strip())
    for pdf in project_dir.glob("*.pdf"):
        stem = re.sub(r"v\d+$", "", pdf.stem)
        if stem == base:
            logger.info("  Using local PDF: %s", pdf.name)
            return pdf.read_bytes()
    return None


def _download_pdf(url: str) -> bytes | None:
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30,
                         headers={"User-Agent": "litreview-audit/0.1"})
        if resp.status_code == 200 and b"%PDF" in resp.content[:10]:
            return resp.content
        logger.warning("PDF download failed (status %d): %s", resp.status_code, url)
    except Exception as e:
        logger.warning("PDF download error: %s", e)
    return None


def _get_pdf(paper_row: dict, project_dir: Path | None = None) -> bytes | None:
    """Local file first, then arXiv, then S2 openAccessPdf."""
    pdf = _get_local_pdf(paper_row, project_dir)
    if pdf:
        return pdf

    arxiv_id = paper_row.get("arxiv_id")
    if arxiv_id:
        clean = re.sub(r"v\d+$", "", arxiv_id.strip())
        return _download_pdf(f"https://arxiv.org/pdf/{clean}.pdf")

    s2_id = paper_row.get("s2_id")
    doi = paper_row.get("doi")
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
                    return _download_pdf(f"https://arxiv.org/pdf/{clean}.pdf")
                pdf_info = data.get("openAccessPdf") or {}
                if pdf_info.get("url"):
                    return _download_pdf(pdf_info["url"])
        except Exception as e:
            logger.warning("S2 PDF URL lookup failed: %s", e)
    return None


# ── PDF reference parsing ─────────────────────────────────────────────────────

def _extract_references_section(text: str) -> str:
    """
    Find and return everything from the References/Bibliography header onward.
    Tries headings from last to first, returning the first one that yields >3 entries.
    """
    pattern = re.compile(
        r"(?:^|\n)[ \t]*(?:\d+\s+)?(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)[ \t]*\n",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        matches = list(re.finditer(r"(?m)^References\s*$", text, re.IGNORECASE))
    if not matches:
        return ""
    # Try from last to first; use first candidate that has >3 parseable entries
    for m in reversed(matches):
        section = text[m.end():]
        if len(_split_into_entries(section)) > 3:
            return section
    return text[matches[-1].end():]


def _split_into_entries(ref_section: str) -> list[str]:
    """
    Split the reference block into individual entries.
    Handles common formats: [1], [AB12], numbered lines, author-year keys.
    Only splits on keys that appear at the START of a line to avoid
    splitting on inline citations in the body/appendix.
    """
    # Bracket at start of line: [1], [ABC12], [AB+12], etc.
    bracket = re.compile(r"(?m)^[ \t]*(?=\[[A-Za-z0-9+,\s]{1,20}\]\s)")
    parts = bracket.split(ref_section)
    if len(parts) > 2:
        return [p.strip() for p in parts if p.strip()]

    # Numbered at line start: "1." or "1) "
    numbered = re.compile(r"(?m)^(?:\d{1,3}[\.\)])\s+")
    parts = numbered.split(ref_section)
    if len(parts) > 2:
        return [p.strip() for p in parts if p.strip()]

    # Fall back: split on blank lines
    return [p.strip() for p in re.split(r"\n{2,}", ref_section) if p.strip()]


def _parse_entry(raw: str) -> dict:
    """
    Extract title, year, doi from a raw reference string.
    Returns dict with keys: raw, title, year, doi.
    """
    raw = re.sub(r"\s+", " ", raw).strip()

    # DOI
    doi_m = re.search(r"\b(10\.\d{4,}/\S+)", raw)
    doi = doi_m.group(1).rstrip(".,)") if doi_m else None

    # Year: four-digit number 1900–2099
    year_m = re.findall(r"\b(1[89]\d{2}|20[0-2]\d)\b", raw)
    year = int(year_m[0]) if year_m else None

    # Title heuristic: text in quotes, or after author(s) before year
    title = None
    # Quoted title (straight or curly)
    q = re.search(r'["\u201c](.{10,120})["\u201d]', raw)
    if q:
        title = q.group(1).strip()
    else:
        # Try "In: <venue>" pattern — title is likely just before "In:"
        in_m = re.search(r"(.{10,120}?)\.\s+[Ii]n:", raw)
        if in_m:
            # strip leading author block (ends at first ". ")
            candidate = in_m.group(1)
            # Author block usually ends with a period; title follows
            parts = candidate.split(". ", 1)
            title = parts[-1].strip() if len(parts) > 1 else candidate.strip()

    if not title:
        title = raw[:80]

    return {"raw": raw, "title": title, "year": year, "doi": doi}


def _parse_pdf_references(pdf_bytes: bytes) -> list[dict]:
    """Download and parse a PDF, returning a list of parsed reference dicts."""
    try:
        import pypdf
    except ImportError:
        logger.error("pypdf not installed; run: pip install pypdf")
        return []

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages)
    except Exception as e:
        logger.warning("PDF text extraction failed: %s", e)
        return []

    ref_section = _extract_references_section(full_text)
    if not ref_section:
        logger.warning("Could not find References section in PDF.")
        return []

    entries = _split_into_entries(ref_section)
    return [_parse_entry(e) for e in entries if len(e) > 20]


# ── S2 fallback ───────────────────────────────────────────────────────────────

def _fetch_s2_references(s2_id: str) -> list[dict]:
    refs = []
    offset, limit = 0, 1000
    while True:
        try:
            resp = httpx.get(
                f"{_S2_BASE}/paper/{s2_id}/references",
                headers=_s2_headers(),
                params={"fields": "paperId,title,year,externalIds", "limit": limit, "offset": offset},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("S2 references fallback failed: %s", e)
            break
        items = data.get("data") or []
        for item in items:
            cp = item.get("citedPaper") or {}
            ext = cp.get("externalIds") or {}
            refs.append({
                "title": cp.get("title") or "",
                "year": cp.get("year"),
                "doi": ext.get("DOI"),
                "s2_id": cp.get("paperId"),
                "raw": cp.get("title") or "",
            })
        if len(items) < limit:
            break
        offset += limit
        time.sleep(0.3)
    return refs


# ── Matching helpers ──────────────────────────────────────────────────────────

def _norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.lower().strip().lstrip("https://doi.org/")


def _build_db_lookup(papers: list[dict]) -> tuple[dict, dict]:
    by_doi = {_norm_doi(p["doi"]): p for p in papers if p.get("doi")}
    by_s2  = {p["s2_id"]: p for p in papers if p.get("s2_id")}
    return by_doi, by_s2


def _find_in_db(ref: dict, by_doi: dict, by_s2: dict, db_titles: list[tuple[str, str]]) -> bool:
    if ref.get("s2_id") and ref["s2_id"] in by_s2:
        return True
    nd = _norm_doi(ref.get("doi"))
    if nd and nd in by_doi:
        return True
    # Search each DB paper's title inside the raw entry text.
    # This is robust to garbled title extraction — we don't need a clean title,
    # just need to know if a DB paper we already have is mentioned in this entry.
    from rapidfuzz import fuzz
    raw = re.sub(r"\s+", " ", (ref.get("raw") or "")).lower()
    if len(raw) >= 20:
        for db_title, _ in db_titles:
            if len(db_title) < 10:
                continue
            if fuzz.partial_ratio(db_title, raw) >= 88:
                return True
    return False


def _looks_like_math_garbage(entry: str) -> bool:
    """Return True if the entry looks like body math text, not a bibliography entry."""
    stripped = entry.lstrip("[]0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+,. ")
    math_chars = set("→≤≥≠∈∉⊂⊆∪∩∀∃∑∏∫∂∇±×÷αβγδεζηθλμνξπρστφψω")
    if any(c in math_chars for c in stripped[:30]):
        return True
    # Math interval/tuple brackets like [0,1] or [1,2,3] — NOT bibliography keys [1] or [42]
    if re.match(r"^\[\s*\d[\d\s]*[,\.]\s*\d", entry.strip()):
        return True
    return False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RefEntry:
    title: str
    year: int | None
    doi: str | None
    captured: bool = False


@dataclass
class AuditResult:
    seed_title: str
    seed_paper_id: str
    source: str   # "pdf" | "s2_api"
    refs: list[RefEntry] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.refs)

    @property
    def n_captured(self) -> int:
        return sum(1 for r in self.refs if r.captured)

    @property
    def n_missing(self) -> int:
        return self.total - self.n_captured

    @property
    def capture_rate(self) -> float:
        return self.n_captured / self.total if self.total else 1.0

    @property
    def missing(self) -> list[RefEntry]:
        return [r for r in self.refs if not r.captured]


# ── Core audit logic ──────────────────────────────────────────────────────────

def _audit_paper(
    paper: dict,
    by_doi: dict,
    by_s2: dict,
    db_titles: list[tuple[str, str]],
    pdf_dir: Path | None = None,
) -> AuditResult:
    """Audit a single paper's references against the DB. Returns an AuditResult."""
    title = paper.get("title", "")
    ref_dicts: list[dict] = []
    ref_source = "s2_api"

    pdf_bytes = _get_pdf(paper, pdf_dir)
    if pdf_bytes:
        ref_dicts = _parse_pdf_references(pdf_bytes)
        if ref_dicts:
            ref_source = "pdf"
            logger.info("  Parsed %d references from PDF.", len(ref_dicts))

    if not ref_dicts:
        s2_id = paper.get("s2_id")
        if not s2_id and paper.get("doi"):
            try:
                r = httpx.get(f"{_S2_BASE}/paper/DOI:{paper['doi']}",
                              headers=_s2_headers(), params={"fields": "paperId"}, timeout=15)
                if r.status_code == 200:
                    s2_id = r.json().get("paperId")
            except Exception:
                pass
        if s2_id:
            ref_dicts = _fetch_s2_references(s2_id)
            logger.info("  S2 API returned %d references.", len(ref_dicts))
        else:
            logger.warning("  No PDF and no S2 ID — cannot audit '%s'.", title[:50])

    ref_entries = []
    seen: set[str] = set()
    for ref in ref_dicts:
        if _looks_like_math_garbage(ref.get("raw") or ref.get("title") or ""):
            continue
        key = _norm_doi(ref.get("doi")) or (ref.get("title") or "").lower()[:60]
        if key in seen or not key:
            continue
        seen.add(key)
        ref_entries.append(RefEntry(
            title=ref.get("title") or "—",
            year=ref.get("year"),
            doi=ref.get("doi"),
            captured=_find_in_db(ref, by_doi, by_s2, db_titles),
        ))

    result = AuditResult(
        seed_title=title,
        seed_paper_id=paper["paper_id"],
        source=ref_source,
        refs=ref_entries,
    )
    logger.info(
        "  %d / %d refs captured (%.0f%%) — %d missing  [source: %s]",
        result.n_captured, result.total, result.capture_rate * 100,
        result.n_missing, ref_source,
    )
    return result


def _build_shared_lookup(client: Client, project_id: str) -> tuple[list[dict], dict, dict, list[tuple[str, str]]]:
    all_papers = get_papers(client, project_id)
    by_doi, by_s2 = _build_db_lookup(all_papers)
    db_titles = [(p["title"].strip().lower(), p["paper_id"]) for p in all_papers if p.get("title")]
    return all_papers, by_doi, by_s2, db_titles


# ── Public entry points ────────────────────────────────────────────────────────

def audit_traversal(
    client: Client,
    project_id: str,
    project_dir: Path | None = None,
) -> list[AuditResult]:
    """Audit seed papers' references against the DB corpus."""
    all_papers, by_doi, by_s2, db_titles = _build_shared_lookup(client, project_id)
    seeds = [p for p in all_papers if p.get("source") == "seed"]
    results = []
    for seed in seeds:
        logger.info("Auditing seed '%s' …", seed.get("title", "")[:65])
        results.append(_audit_paper(seed, by_doi, by_s2, db_titles, pdf_dir=project_dir))
    return results


def audit_included(
    client: Client,
    project_id: str,
    pdf_dir: Path,
    depth: int | None = None,
) -> list[AuditResult]:
    """
    Audit included papers' references against the DB corpus.
    PDFs must be pre-downloaded into *pdf_dir* (typically projects/<slug>/included-pdfs/).
    Optionally filter to a specific depth.
    """
    all_papers, by_doi, by_s2, db_titles = _build_shared_lookup(client, project_id)
    candidates = [p for p in all_papers if p.get("inclusion_status") == "included"
                  and p.get("source") != "seed"]
    if depth is not None:
        candidates = [p for p in candidates if p.get("depth") == depth]

    results = []
    for paper in candidates:
        logger.info("Auditing included '%s' …", paper.get("title", "")[:65])
        results.append(_audit_paper(paper, by_doi, by_s2, db_titles, pdf_dir=pdf_dir))
    return results


# ── Missing ref recovery ──────────────────────────────────────────────────────

def recover_missing_refs(
    client: Client,
    project_id: str,
    audit_results: list[AuditResult],
    parent_depth: int = 0,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    For each missing ref in *audit_results*, attempt to fetch it from S2 by title
    and upsert it into the DB as a citation candidate.

    Returns {"recovered": N, "not_found": N, "skipped": N}.
    """
    from rapidfuzz import fuzz
    from .ingest import fetch_s2_by_title, fetch_s2_by_id, _doi_to_s2_id
    from .db import upsert_paper
    from .models import Paper

    counts: dict[str, int] = {"recovered": 0, "not_found": 0, "skipped": 0}
    seen_titles: set[str] = set()

    for result in audit_results:
        for ref in result.missing:
            title = (ref.title or "").strip()
            if not title or title == "—" or len(title) < 10:
                counts["skipped"] += 1
                continue

            norm = title.lower()[:80]
            if norm in seen_titles:
                counts["skipped"] += 1
                continue
            seen_titles.add(norm)

            # Try DOI first
            paper: Paper | None = None
            if ref.doi:
                paper = fetch_s2_by_id(_doi_to_s2_id(ref.doi))

            # Fall back to title search
            if paper is None:
                candidates = fetch_s2_by_title(title, top_k=3)
                for c in candidates:
                    if fuzz.token_sort_ratio(title.lower(), c.title.lower()) >= 80:
                        paper = c
                        break

            if paper is None:
                logger.debug("Could not recover: '%s'", title[:60])
                counts["not_found"] += 1
                time.sleep(0.3)
                continue

            paper.project_id = project_id
            paper.source = "citation"
            paper.depth = parent_depth + 1

            if not dry_run:
                upsert_paper(client, paper)
            logger.info("Recovered: '%s'", paper.title[:60])
            counts["recovered"] += 1
            time.sleep(0.5)

    return counts
