from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class InclusionStatus(str, Enum):
    pending = "pending"
    included = "included"
    excluded = "excluded"


class Author(BaseModel):
    name: str
    s2_id: str | None = None
    openalex_id: str | None = None


class Paper(BaseModel):
    paper_id: str | None = None       # UUID set by DB
    project_id: str | None = None

    # External IDs
    doi: str | None = None
    arxiv_id: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None

    # Metadata
    title: str
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[Author] = Field(default_factory=list)

    # Citation counts
    citation_count: int | None = None   # incoming: times cited (hub signal)
    reference_count: int | None = None  # outgoing: references made (survey signal)

    # Embedding (SPECTER2, dim 768)
    embedding: list[float] | None = None

    # Screening
    inclusion_status: InclusionStatus = InclusionStatus.pending
    screening_round: int | None = None
    criteria_version: int | None = None
    rejection_reason: str | None = None

    # Provenance
    source: str = "seed"   # 'seed' | 'citation' | 'search'
    depth: int = 0

    # Fields managed exclusively by the screener — never overwrite via ingest/traverse upsert
    _SCREENER_FIELDS = {"inclusion_status", "rejection_reason", "screening_round", "criteria_version"}

    def db_dict(self) -> dict[str, Any]:
        """Return a dict suitable for Supabase upsert (no None paper_id).

        Screening fields are excluded so that traversal/ingest upserts never
        overwrite decisions made by the screener.  New rows get the DB-level
        default (inclusion_status = 'pending'); existing rows keep their state.
        """
        d = self.model_dump(exclude={"paper_id"} | self._SCREENER_FIELDS)
        d["authors"] = [a.model_dump() for a in self.authors]
        return {k: v for k, v in d.items() if v is not None}
