from __future__ import annotations

from typing import Any
from supabase import create_client, Client
from .config import settings
from .models import Paper


def get_client() -> Client:
    return create_client(settings.supabase_url, settings.supabase_key)


# ── Projects ──────────────────────────────────────────────────────────────────

def get_or_create_project(client: Client, name: str) -> str:
    """Return the project_id for *name*, creating the row if absent."""
    res = client.table("projects").select("project_id").eq("name", name).execute()
    if res.data:
        return res.data[0]["project_id"]
    res = client.table("projects").insert({"name": name}).execute()
    return res.data[0]["project_id"]


# ── Papers ────────────────────────────────────────────────────────────────────

def upsert_paper(client: Client, paper: Paper) -> dict[str, Any]:
    """
    Upsert by (project_id, s2_id) if available, else (project_id, doi).
    Returns the DB row.
    """
    row = paper.db_dict()

    # Supabase upsert needs the conflict target expressed via on_conflict
    if paper.s2_id:
        conflict = "project_id,s2_id"
    elif paper.doi:
        conflict = "project_id,doi"
    else:
        # No stable ID — just insert; duplicates handled by title dedup upstream
        conflict = None

    if conflict:
        res = (
            client.table("papers")
            .upsert(row, on_conflict=conflict)
            .execute()
        )
    else:
        res = client.table("papers").insert(row).execute()

    return res.data[0]


def get_papers(
    client: Client,
    project_id: str,
    status: str | None = None,
) -> list[dict[str, Any]]:
    q = client.table("papers").select("*").eq("project_id", project_id)
    if status:
        q = q.eq("inclusion_status", status)
    return q.execute().data


# ── Citations ─────────────────────────────────────────────────────────────────

def upsert_citation(
    client: Client,
    project_id: str,
    citing_paper_id: str,
    cited_paper_id: str,
) -> None:
    client.table("citations").upsert(
        {
            "project_id": project_id,
            "citing_paper_id": citing_paper_id,
            "cited_paper_id": cited_paper_id,
        },
        on_conflict="citing_paper_id,cited_paper_id",
    ).execute()


# ── Screening ─────────────────────────────────────────────────────────────────

def reset_screening(
    client: Client,
    project_id: str,
    depth: int | None = None,
) -> int:
    """
    Reset screened papers back to 'pending', clearing rejection_reason/screening_round.
    If depth is given, only resets papers at that depth. Returns the count reset.
    """
    q = (
        client.table("papers")
        .update({"inclusion_status": "pending", "rejection_reason": None, "screening_round": None})
        .eq("project_id", project_id)
        .neq("inclusion_status", "pending")
        .neq("source", "seed")  # never reset seed papers
    )
    if depth is not None:
        q = q.eq("depth", depth)
    result = q.execute()
    return len(result.data)


def update_paper_screening(
    client: Client,
    paper_id: str,
    status: str,
    rejection_reason: str | None = None,
    screening_round: int | None = None,
    criteria_version: int | None = None,
) -> None:
    """Update a paper's inclusion_status and screening metadata."""
    update: dict[str, Any] = {"inclusion_status": status}
    if rejection_reason is not None:
        update["rejection_reason"] = rejection_reason
    if screening_round is not None:
        update["screening_round"] = screening_round
    if criteria_version is not None:
        update["criteria_version"] = criteria_version
    client.table("papers").update(update).eq("paper_id", paper_id).execute()


# ── Criteria ──────────────────────────────────────────────────────────────────

def get_current_criteria(client: Client, project_id: str) -> tuple[int, str]:
    """Return (version, content) for the latest criteria version, or (0, '') if none."""
    res = (
        client.table("criteria")
        .select("version,content")
        .eq("project_id", project_id)
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    if not res.data:
        return 0, ""
    row = res.data[0]
    return row["version"], row["content"]


def upsert_criteria(
    client: Client,
    project_id: str,
    version: int,
    content: str,
    trigger: str | None = None,
) -> None:
    client.table("criteria").upsert(
        {
            "project_id": project_id,
            "version": version,
            "content": content,
            "trigger": trigger,
        },
        on_conflict="project_id,version",
    ).execute()


# ── Iterations ─────────────────────────────────────────────────────────────────

def log_iteration(
    client: Client,
    project_id: str,
    round_number: int,
    yield_rate: float | None = None,
    overlap_rate: float | None = None,
    mean_embedding_distance: float | None = None,
    query: str | None = None,
    criteria_version: int | None = None,
) -> None:
    """Insert a row into iterations to record stability metrics for a round."""
    row: dict[str, Any] = {"project_id": project_id, "round": round_number}
    if yield_rate is not None:
        row["yield_rate"] = yield_rate
    if overlap_rate is not None:
        row["overlap_rate"] = overlap_rate
    if mean_embedding_distance is not None:
        row["mean_embedding_distance"] = mean_embedding_distance
    if query is not None:
        row["query"] = query
    if criteria_version is not None:
        row["criteria_version"] = criteria_version
    client.table("iterations").insert(row).execute()


def get_frontier_depth(client: Client, project_id: str) -> int:
    """Return the max depth of included papers (the current traversal frontier)."""
    papers = get_papers(client, project_id, status="included")
    if not papers:
        return 0
    return max(p.get("depth", 0) for p in papers)


def get_iterations(client: Client, project_id: str) -> list[dict[str, Any]]:
    """Return all iteration rows for a project, ordered by round ascending."""
    return (
        client.table("iterations")
        .select("*")
        .eq("project_id", project_id)
        .order("round", desc=False)
        .execute()
        .data
    )


def is_stable(
    client: Client,
    project_id: str,
    yield_threshold: float = 0.05,
    consecutive_rounds: int = 2,
    new_papers_threshold: int = 10,
    new_papers_count: int | None = None,
) -> tuple[bool, str]:
    """
    Check whether the pipeline has converged.

    Stable if EITHER:
      (a) The last `consecutive_rounds` rounds all have yield_rate < yield_threshold, OR
      (b) new_papers_count is provided and < new_papers_threshold (traversal exhausted)

    Returns (stable: bool, reason: str).
    """
    if new_papers_count is not None and new_papers_count < new_papers_threshold:
        return True, f"traversal exhausted ({new_papers_count} new papers < threshold {new_papers_threshold})"

    iters = get_iterations(client, project_id)
    if len(iters) < consecutive_rounds:
        return False, f"only {len(iters)} rounds logged, need {consecutive_rounds}"

    recent = iters[-consecutive_rounds:]
    low_yield = [r for r in recent if r.get("yield_rate") is not None and r["yield_rate"] < yield_threshold]
    if len(low_yield) == consecutive_rounds:
        rates = ", ".join(f"{r['yield_rate']:.1%}" for r in recent)
        return True, f"yield rate below {yield_threshold:.0%} for last {consecutive_rounds} rounds ({rates})"

    last = iters[-1]
    return False, f"round {last['round']} yield={last.get('yield_rate', 'n/a')}"


def next_round_number(client: Client, project_id: str) -> int:
    """Return the next screening round number (max existing + 1, or 1)."""
    res = (
        client.table("iterations")
        .select("round")
        .eq("project_id", project_id)
        .order("round", desc=True)
        .limit(1)
        .execute()
    )
    return (res.data[0]["round"] + 1) if res.data else 1
