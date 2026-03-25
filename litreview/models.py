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

    def db_dict(self) -> dict[str, Any]:
        """Return a dict suitable for Supabase upsert (no None paper_id)."""
        d = self.model_dump(exclude={"paper_id"})
        d["authors"] = [a.model_dump() for a in self.authors]
        if d["embedding"] is not None:
            # Supabase expects a list; pgvector handles conversion
            pass
        return {k: v for k, v in d.items() if v is not None}
