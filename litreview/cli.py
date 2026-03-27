"""
litreview CLI — entry point for the automated literature review pipeline.

Commands:
  ingest   Ingest seed papers for a project.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="litreview",
    help="Automated literature review pipeline.",
    no_args_is_help=True,
)


@app.callback()
def global_options(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")


# ── ingest ────────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to project TOML file."
    ),
    project_name: Optional[str] = typer.Argument(
        None, help="Project name (if not using --config)."
    ),
    identifiers: Optional[list[str]] = typer.Argument(
        None, help="DOIs, arXiv IDs, or title strings (if not using --config)."
    ),
    no_openalex: bool = typer.Option(
        False, "--no-openalex", help="Skip OpenAlex enrichment."
    ),
) -> None:
    """Ingest seed papers into Supabase for a project."""
    import tomllib

    from litreview.db import get_client, get_or_create_project, upsert_paper, upsert_criteria
    from litreview.ingest import ingest_papers

    criteria_content: str | None = None

    if config:
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        project_name = cfg["project"]["name"]
        ids: list[str] = cfg["seeds"]["identifiers"]
        criteria_content = cfg.get("criteria", {}).get("v1")
    elif project_name and identifiers:
        ids = list(identifiers)
    else:
        console.print("[red]Provide --config or both project_name and identifiers.[/red]")
        raise typer.Exit(1)

    client = get_client()
    project_id = get_or_create_project(client, project_name)
    console.print(f"[bold]Project:[/bold] {project_name}  ({project_id})")

    if criteria_content:
        upsert_criteria(client, project_id, version=1, content=criteria_content, trigger="initial")
        console.print("[green]Criteria v1 stored.[/green]")

    console.print(f"\nResolving {len(ids)} seed identifiers …")
    papers = ingest_papers(
        ids,
        project_id=project_id,
        source="seed",
        depth=0,
        enrich_openalex=not no_openalex,
    )

    if not papers:
        console.print("[red]No papers resolved. Check identifiers and API keys.[/red]")
        raise typer.Exit(1)

    table = Table("Title", "Year", "S2 ID", "DOI", "Embedding?")
    for paper in papers:
        upsert_paper(client, paper)
        table.add_row(
            (paper.title or "")[:60],
            str(paper.year or ""),
            paper.s2_id or "",
            paper.doi or "",
            "yes" if paper.embedding else "no",
        )

    console.print(table)
    console.print(f"\n[green]Upserted {len(papers)} papers into '{project_name}'.[/green]")


# ── shared helpers ────────────────────────────────────────────────────────────

def _run_audit(client, project_id: str, con: Console, project_dir: Optional[Path] = None) -> None:
    from litreview.audit import audit_traversal
    con.print("\n[bold]Traversal audit[/bold]")
    results = audit_traversal(client, project_id, project_dir=project_dir)
    for r in results:
        rate_color = "green" if r.capture_rate >= 0.9 else "yellow" if r.capture_rate >= 0.7 else "red"
        src_label = "[dim](pdf)[/dim]" if r.source == "pdf" else "[dim](s2 api)[/dim]"
        con.print(
            f"  [bold]{r.seed_title[:65]}[/bold]  "
            f"[{rate_color}]{r.n_captured}/{r.total} ({r.capture_rate:.0%})[/{rate_color}]"
            f"  —  {r.n_missing} missing  {src_label}"
        )
        if r.missing:
            t = Table("Title", "Year", "DOI", show_header=True)
            for m in sorted(r.missing, key=lambda x: x.year or 0):
                t.add_row(
                    (m.title or "—")[:65],
                    str(m.year or "?"),
                    m.doi or "—",
                )
            con.print(t)


# ── traverse ──────────────────────────────────────────────────────────────────

@app.command()
def traverse(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project TOML."),
    project_name: Optional[str] = typer.Argument(None, help="Project name."),
    direction: str = typer.Option("both", "--direction", "-d", help="forward | backward | both"),
    from_depth: int = typer.Option(0, "--from-depth", help="Traverse from papers at this depth (0=seeds, 1=depth-1 included, …)."),
) -> None:
    """Expand the corpus by traversing citation edges from seed papers via OpenAlex."""
    import tomllib
    from litreview.db import get_client, get_or_create_project
    from litreview.traverse import traverse_citations

    if config:
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        name = cfg["project"]["name"]
    elif project_name:
        name = project_name
    else:
        console.print("[red]Provide --config or project_name.[/red]")
        raise typer.Exit(1)

    if direction not in ("forward", "backward", "both"):
        console.print("[red]--direction must be forward, backward, or both.[/red]")
        raise typer.Exit(1)

    project_dir = config.parent if config else None

    client = get_client()
    project_id = get_or_create_project(client, name)
    console.print(f"[bold]Project:[/bold] {name}  ({project_id})")
    console.print(f"Traversing citations ({direction}) …\n")

    stats = traverse_citations(client, project_id, direction=direction, from_depth=from_depth)  # type: ignore[arg-type]

    console.print(f"[green]Done.[/green]")
    console.print(f"  Raw candidates fetched : {stats['candidates']}")
    console.print(f"  New papers upserted    : {stats['papers_upserted']}")
    console.print(f"  Citation edges stored  : {stats['edges_stored']}")

    _run_audit(client, project_id, console, project_dir=project_dir)


# ── screen ─────────────────────────────────────────────────────────────────────

@app.command()
def screen(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project TOML."),
    project_name: Optional[str] = typer.Argument(None, help="Project name."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model ID (defaults per backend)."),
    backend: str = typer.Option("gemini", "--backend", "-b", help="LLM backend: groq, gemini, openrouter, anthropic, or comma-separated list for failover."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print decisions without writing to DB."),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Stop after screening this many papers."),
    rate_sleep: Optional[float] = typer.Option(None, "--rate-sleep", help="Override per-backend rate sleep (seconds). Useful on paid tiers with higher RPM limits."),
    include_uncertain: bool = typer.Option(False, "--include-uncertain", help="Clear UNCERTAIN prefix from pending papers so they get a clean re-screen."),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Never prompt on backend failure — auto-switch or raise immediately."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip large-batch confirmation prompt (>200 papers)."),
) -> None:
    """Screen pending papers against inclusion criteria using an LLM backend."""
    import sys as _sys
    import tomllib
    from litreview.db import get_client, get_or_create_project, next_round_number
    from litreview.screen import screen_project

    _valid_backends = {"groq", "gemini", "openrouter", "anthropic"}
    if not all(b.strip() in _valid_backends for b in backend.split(",")):
        console.print("[red]--backend must be one of: groq, gemini, openrouter, anthropic (or comma-separated).[/red]")
        raise typer.Exit(1)

    if config:
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        name = cfg["project"]["name"]
    elif project_name:
        name = project_name
    else:
        console.print("[red]Provide --config or project_name.[/red]")
        raise typer.Exit(1)

    client = get_client()
    project_id = get_or_create_project(client, name)
    round_number = next_round_number(client, project_id)

    console.print(f"[bold]Project:[/bold] {name}  ({project_id})")
    console.print(f"Screening round {round_number} | backend={backend}" + (f" model={model}" if model else "") + (f" rate-sleep={rate_sleep}s" if rate_sleep is not None else "") + (" [DRY RUN]" if dry_run else "") + "\n")

    is_interactive = not non_interactive and _sys.stdin.isatty()
    try:
        counts = screen_project(
            client, project_id, round_number,
            model=model, dry_run=dry_run, backend=backend, limit=limit,
            rate_sleep_override=rate_sleep,
            include_uncertain=include_uncertain,
            interactive=is_interactive,
            skip_confirmation=yes or not is_interactive,
        )
    except RuntimeError as e:
        console.print(f"\n[bold red]Screening stopped:[/bold red] {e}")
        raise typer.Exit(1)

    table = Table("Metric", "Count")
    table.add_row("Total pending",  str(counts["total"]))
    table.add_row("[green]Included[/green]",  str(counts["include"]))
    table.add_row("[red]Excluded[/red]",  str(counts["exclude"]))
    table.add_row("Uncertain (→ pending)", str(counts["uncertain"]))
    table.add_row("Skipped (error)",  str(counts["skipped"]))
    console.print(table)

    screened = counts["total"] - counts["skipped"]
    if screened:
        console.print(f"\nYield rate: {counts['include'] / screened:.1%}")


# ── recommend ─────────────────────────────────────────────────────────────────

@app.command()
def recommend(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project TOML."),
    project_name: Optional[str] = typer.Argument(None, help="Project name."),
    limit: int = typer.Option(500, "--limit", help="Max recommendations to request from S2 (hard cap: 500)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print candidate count without writing to DB."),
) -> None:
    """
    Fetch semantically similar papers from S2 Recommendations API.

    Uses included + seed papers as positive examples and excluded papers as
    negatives. Upserts candidates with source='search' for the next screening round.
    Requires SEMANTIC_SCHOLAR_API_KEY for reliable results (free tier is rate-limited).
    """
    import tomllib
    from litreview.db import get_client, get_or_create_project, get_papers, get_frontier_depth
    from litreview.recommend import fetch_s2_recommendations

    if config:
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        name = cfg["project"]["name"]
    elif project_name:
        name = project_name
    else:
        console.print("[red]Provide --config or project_name.[/red]")
        raise typer.Exit(1)

    client = get_client()
    project_id = get_or_create_project(client, name)
    console.print(f"[bold]Project:[/bold] {name}  ({project_id})")

    all_papers = get_papers(client, project_id)

    # Positives: seeds (always included) + explicitly included papers, sorted by depth desc
    # so that when we truncate to 100 we keep the most frontier-representative papers
    positives = sorted(
        [p for p in all_papers if p.get("source") == "seed" or p.get("inclusion_status") == "included"],
        key=lambda p: p.get("depth", 0),
        reverse=True,
    )
    negatives = [p for p in all_papers if p.get("inclusion_status") == "excluded"]

    pos_ids = [p["s2_id"] for p in positives if p.get("s2_id")]
    neg_ids = [p["s2_id"] for p in negatives if p.get("s2_id")]

    console.print(f"  Positives (seed + included): {len(positives)}  ({len(pos_ids)} with S2 ID)")
    console.print(f"  Negatives (excluded):        {len(negatives)}  ({len(neg_ids)} with S2 ID)")

    if not pos_ids:
        console.print("[red]No positive papers with S2 IDs found. Run ingest first.[/red]")
        raise typer.Exit(1)

    depth = get_frontier_depth(client, project_id) + 1
    papers = fetch_s2_recommendations(project_id, pos_ids, neg_ids, depth=depth, limit=limit)

    # Filter out papers already in the corpus
    existing_s2  = {p["s2_id"] for p in all_papers if p.get("s2_id")}
    existing_doi = {p["doi"]   for p in all_papers if p.get("doi")}
    new_papers = [
        p for p in papers
        if not (
            (p.s2_id and p.s2_id in existing_s2)
            or (p.doi and p.doi in existing_doi)
        )
    ]

    console.print(f"\n  S2 returned {len(papers)} recommendations → {len(new_papers)} genuinely new")

    if dry_run:
        console.print("[yellow]Dry run — nothing written.[/yellow]")
        return

    from litreview.db import upsert_paper
    upserted = 0
    for paper in new_papers:
        try:
            upsert_paper(client, paper)
            upserted += 1
        except Exception as e:
            logger.warning("Upsert failed for '%s': %s", paper.title[:50], e)

    console.print(f"[green]Upserted {upserted} new candidate papers (source='search').[/green]")


# ── reset-screening ────────────────────────────────────────────────────────────

@app.command(name="reset-screening")
def reset_screening(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project TOML."),
    project_name: Optional[str] = typer.Argument(None, help="Project name."),
    depth: Optional[int] = typer.Option(None, "--depth", help="Only reset papers at this depth (omit for all non-seed papers)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Reset screened papers back to pending (seeds are never touched)."""
    import tomllib
    from litreview.db import get_client, get_or_create_project, reset_screening as db_reset

    if config:
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        name = cfg["project"]["name"]
    elif project_name:
        name = project_name
    else:
        console.print("[red]Provide --config or project_name.[/red]")
        raise typer.Exit(1)

    depth_label = f"depth={depth}" if depth is not None else "all non-seed papers"
    if not yes:
        typer.confirm(f"Reset screening for {depth_label} in '{name}'?", abort=True)

    client = get_client()
    project_id = get_or_create_project(client, name)
    count = db_reset(client, project_id, depth=depth)
    console.print(f"[green]Reset {count} papers to pending.[/green]")


# ── iterate ────────────────────────────────────────────────────────────────────

@app.command()
def iterate(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project TOML."),
    project_name: Optional[str] = typer.Argument(None, help="Project name."),
    direction: str = typer.Option("both", "--direction", "-d"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model ID (defaults per backend)."),
    backend: str = typer.Option("gemini", "--backend", "-b", help="LLM backend: groq, gemini, openrouter, or anthropic."),
    dry_run: bool = typer.Option(False, "--dry-run"),
    loop: bool = typer.Option(False, "--loop", help="Keep iterating until stable."),
    max_rounds: int = typer.Option(10, "--max-rounds", help="Maximum rounds when --loop is set."),
    yield_threshold: float = typer.Option(0.05, "--yield-threshold", help="Yield rate below which pipeline is considered converging."),
    stable_rounds: int = typer.Option(2, "--stable-rounds", help="Consecutive low-yield rounds required to declare stability."),
    use_recommend: bool = typer.Option(False, "--recommend/--no-recommend", help="Run S2 recommendations as a parallel sourcing track alongside citation traversal."),
) -> None:
    """Run one full iteration (or loop until stable): traverse citations → screen pending papers."""
    import tomllib
    from litreview.db import get_client, get_or_create_project, next_round_number, is_stable, get_frontier_depth
    from litreview.traverse import traverse_citations
    from litreview.screen import screen_project

    if config:
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        name = cfg["project"]["name"]
    elif project_name:
        name = project_name
    else:
        console.print("[red]Provide --config or project_name.[/red]")
        raise typer.Exit(1)

    project_dir = config.parent if config else None
    client = get_client()
    project_id = get_or_create_project(client, name)

    console.print(f"[bold]Project:[/bold] {name}  ({project_id})")

    rounds_run = 0
    while True:
        round_number = next_round_number(client, project_id)
        rounds_run += 1
        console.print(f"\n[bold]Round {round_number}[/bold]" + (f"  [{rounds_run}/{max_rounds}]" if loop else "") + "\n")

        # Step 1: traverse from the current frontier (max depth of included papers)
        from_depth = get_frontier_depth(client, project_id)
        console.print(f"[bold]Step 1 — Citation traversal[/bold]  [dim](from depth {from_depth})[/dim]")
        stats = traverse_citations(client, project_id, direction=direction, from_depth=from_depth)  # type: ignore[arg-type]
        new_papers = stats["papers_upserted"]
        console.print(
            f"  {stats['candidates']} candidates → {new_papers} new, "
            f"{stats['edges_stored']} edges\n"
        )
        _run_audit(client, project_id, console, project_dir=project_dir)

        # Step 2 (optional): S2 recommendations
        if use_recommend:
            from litreview.recommend import fetch_s2_recommendations
            from litreview.db import upsert_paper as _upsert_paper
            _all = get_papers(client, project_id)
            _pos = sorted(
                [p for p in _all if p.get("source") == "seed" or p.get("inclusion_status") == "included"],
                key=lambda p: p.get("depth", 0), reverse=True,
            )
            _neg = [p for p in _all if p.get("inclusion_status") == "excluded"]
            _pos_ids = [p["s2_id"] for p in _pos if p.get("s2_id")]
            _neg_ids = [p["s2_id"] for p in _neg if p.get("s2_id")]
            _existing_s2  = {p["s2_id"] for p in _all if p.get("s2_id")}
            _existing_doi = {p["doi"]   for p in _all if p.get("doi")}
            _reco_depth = get_frontier_depth(client, project_id) + 1
            _reco = fetch_s2_recommendations(project_id, _pos_ids, _neg_ids, depth=_reco_depth)
            _reco_new = [
                p for p in _reco
                if not ((p.s2_id and p.s2_id in _existing_s2) or (p.doi and p.doi in _existing_doi))
            ]
            _upserted = 0
            for _p in _reco_new:
                try:
                    _upsert_paper(client, _p)
                    _upserted += 1
                except Exception:
                    pass
            console.print(f"[bold]Step 2 — S2 Recommendations[/bold]  {len(_reco)} returned → [green]{_upserted} new candidates[/green]\n")

        # Step 3: screen
        console.print("[bold]Step 3 — LLM screening[/bold]" if use_recommend else "[bold]Step 2 — LLM screening[/bold]")
        counts = screen_project(client, project_id, round_number, model=model, dry_run=dry_run, backend=backend)

        table = Table("Decision", "Count")
        table.add_row("[green]Included[/green]",   str(counts["include"]))
        table.add_row("[red]Excluded[/red]",        str(counts["exclude"]))
        table.add_row("Uncertain (→ pending)",      str(counts["uncertain"]))
        table.add_row("Skipped (error)",            str(counts["skipped"]))
        console.print(table)

        screened = counts["total"] - counts["skipped"]
        if screened:
            console.print(f"\nYield rate: {counts['include'] / screened:.1%}")

        if not loop:
            console.print("\n[green]Iteration complete.[/green]")
            break

        # Stability check
        stable, reason = is_stable(
            client, project_id,
            yield_threshold=yield_threshold,
            consecutive_rounds=stable_rounds,
            new_papers_count=new_papers,
        )
        if stable:
            console.print(f"\n[bold green]Pipeline stable — stopping.[/bold green]  ({reason})")
            break
        else:
            console.print(f"\n[dim]Stability: {reason} — continuing.[/dim]")

        if rounds_run >= max_rounds:
            console.print(f"\n[yellow]Reached max rounds ({max_rounds}) — stopping.[/yellow]")
            break


# ── audit ─────────────────────────────────────────────────────────────────────

@app.command()
def audit(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project TOML."),
    project_name: Optional[str] = typer.Argument(None, help="Project name."),
    included: bool = typer.Option(False, "--included", help="Audit included papers (from included-pdfs/) instead of seeds."),
    depth: Optional[int] = typer.Option(None, "--depth", help="Filter to papers at this depth (--included only)."),
    recover: bool = typer.Option(False, "--recover", help="Attempt to fetch missing refs via S2 title search and upsert them."),
    dry_run: bool = typer.Option(False, "--dry-run", help="With --recover: show what would be fetched without writing."),
) -> None:
    """
    Audit citation coverage: compare each paper's reference list against the DB corpus.

    Default: audits seed papers (using PDFs from project dir or S2 API fallback).
    --included: audits included non-seed papers (PDFs from included-pdfs/ subfolder).
    --recover: after auditing, fetches missing refs from S2 and upserts them.
    """
    import tomllib
    from litreview.db import get_client, get_or_create_project
    from litreview.audit import audit_included, recover_missing_refs

    if config:
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        name = cfg["project"]["name"]
    elif project_name:
        name = project_name
    else:
        console.print("[red]Provide --config or project_name.[/red]")
        raise typer.Exit(1)

    project_dir = config.parent if config else Path(f"projects/{name}")
    client = get_client()
    project_id = get_or_create_project(client, name)
    console.print(f"[bold]Project:[/bold] {name}  ({project_id})\n")

    if included:
        pdf_dir = project_dir / "included-pdfs"
        if not pdf_dir.exists():
            console.print(f"[yellow]included-pdfs/ not found at {pdf_dir}. Run 'litreview download-pdfs' first.[/yellow]")
            raise typer.Exit(1)
        from litreview.audit import audit_included as _audit_included
        results = _audit_included(client, project_id, pdf_dir=pdf_dir, depth=depth)
        for r in results:
            rate_color = "green" if r.capture_rate >= 0.9 else "yellow" if r.capture_rate >= 0.7 else "red"
            console.print(
                f"  [bold]{r.seed_title[:65]}[/bold]  "
                f"[{rate_color}]{r.n_captured}/{r.total} ({r.capture_rate:.0%})[/{rate_color}]"
                f"  —  {r.n_missing} missing  [dim]({r.source})[/dim]"
            )
            if r.missing:
                t = Table("Title", "Year", "DOI", show_header=True)
                for m in sorted(r.missing, key=lambda x: x.year or 0):
                    t.add_row((m.title or "—")[:65], str(m.year or "?"), m.doi or "—")
                console.print(t)
    else:
        _run_audit(client, project_id, console, project_dir=project_dir)
        results = None  # _run_audit prints inline; recover needs results

    if recover:
        if results is None:
            # Re-run to get results object for recovery
            from litreview.audit import audit_traversal
            results = audit_traversal(client, project_id, project_dir=project_dir)

        all_missing = sum(r.n_missing for r in results)
        console.print(f"\n[bold]Recovering {all_missing} missing refs via S2 …[/bold]" + (" [DRY RUN]" if dry_run else ""))
        parent_depth = max((r.refs[0].year or 0) for r in results if r.refs) if results else 0
        # Derive parent depth from the audited papers' depth
        from litreview.db import get_papers as _get_papers
        all_db = _get_papers(client, project_id)
        pid_to_depth = {p["paper_id"]: p.get("depth", 0) for p in all_db}
        counts_by_parent: dict[str, int] = {"recovered": 0, "not_found": 0, "skipped": 0}
        for r in results:
            pdepth = pid_to_depth.get(r.seed_paper_id, 0)
            c = recover_missing_refs(client, project_id, [r], parent_depth=pdepth, dry_run=dry_run)
            for k in counts_by_parent:
                counts_by_parent[k] += c[k]
        console.print(f"[green]Recovered: {counts_by_parent['recovered']}[/green]  "
                      f"Not found: {counts_by_parent['not_found']}  "
                      f"Skipped: {counts_by_parent['skipped']}")


# ── download-pdfs ──────────────────────────────────────────────────────────────

@app.command(name="download-pdfs")
def download_pdfs(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project TOML."),
    project_name: Optional[str] = typer.Argument(None, help="Project name."),
    depth: Optional[int] = typer.Option(None, "--depth", help="Only download PDFs for included papers at this depth."),
) -> None:
    """Download open-access PDFs for included papers into projects/<slug>/included-pdfs/."""
    import tomllib
    from litreview.db import get_client, get_or_create_project, get_papers
    from litreview.download import download_included_pdfs

    if config:
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        name = cfg["project"]["name"]
    elif project_name:
        name = project_name
    else:
        console.print("[red]Provide --config or project_name.[/red]")
        raise typer.Exit(1)

    project_dir = config.parent if config else Path(f"projects/{name}")
    dest_dir = project_dir / "included-pdfs"

    client = get_client()
    project_id = get_or_create_project(client, name)
    papers = get_papers(client, project_id, status="included")
    # Exclude seeds (already in project root)
    papers = [p for p in papers if p.get("source") != "seed"]
    if depth is not None:
        papers = [p for p in papers if p.get("depth") == depth]

    console.print(f"[bold]Project:[/bold] {name}  ({project_id})")
    console.print(f"Downloading PDFs for {len(papers)} included papers → {dest_dir}\n")

    counts = download_included_pdfs(papers, dest_dir)
    console.print(f"[green]Downloaded: {counts['downloaded']}[/green]  "
                  f"Skipped (exists): {counts['skipped']}  "
                  f"[red]Failed: {counts['failed']}[/red]")


# ── extract ───────────────────────────────────────────────────────────────────

@app.command()
def extract(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project TOML."),
    project_name: Optional[str] = typer.Argument(None, help="Project name."),
    backend: str = typer.Option("groq", "--backend", "-b", help="LLM backend: groq, gemini, openrouter, or anthropic."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model ID (defaults per backend)."),
    force: bool = typer.Option(False, "--force", help="Re-extract papers that already have an extraction."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print counts without writing to DB."),
) -> None:
    """
    Extract structured fields from included papers (run post-stability).

    Extracts: contribution_type, problem_statement, setting, methods, main_results,
    limitations, related_work_positioning, open_questions — plus any extra fields
    defined in [[extraction.extra_fields]] in project.toml.

    Skips papers without abstracts and papers already extracted (unless --force).
    Uses abstract only; does not parse PDFs.
    """
    import tomllib
    from litreview.db import get_client, get_or_create_project
    from litreview.extract import extract_project

    extra_fields: list[dict[str, str]] = []

    if config:
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        name = cfg["project"]["name"]
        for ef in cfg.get("extraction", {}).get("extra_fields", []):
            extra_fields.append({"name": ef["name"], "description": ef["description"]})
    elif project_name:
        name = project_name
    else:
        console.print("[red]Provide --config or project_name.[/red]")
        raise typer.Exit(1)

    client = get_client()
    project_id = get_or_create_project(client, name)
    console.print(f"[bold]Project:[/bold] {name}  ({project_id})")
    console.print(f"Backend: {backend}" + (f"  model={model}" if model else "") + (" [DRY RUN]" if dry_run else ""))
    if extra_fields:
        console.print(f"Extra fields: {', '.join(ef['name'] for ef in extra_fields)}")

    counts = extract_project(
        client, project_id,
        extra_fields=extra_fields or None,
        model=model,
        backend=backend,
        dry_run=dry_run,
        force=force,
    )

    table = Table("Metric", "Count")
    table.add_row("[green]Extracted[/green]",         str(counts["extracted"]))
    table.add_row("Skipped (already done)",           str(counts["skipped_existing"]))
    table.add_row("[yellow]Skipped (no abstract)[/yellow]", str(counts["skipped_no_abstract"]))
    table.add_row("[red]Failed[/red]",                str(counts["failed"]))
    table.add_row("Total included",                   str(counts["total"]))
    console.print(table)


# ── synthesize ────────────────────────────────────────────────────────────────

@app.command()
def synthesize(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project TOML."),
    project_name: Optional[str] = typer.Argument(None, help="Project name."),
    model: str = typer.Option("claude-sonnet-4-6", "--model", "-m", help="Anthropic model ID."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path (default: projects/<slug>/literature-review.md)."),
) -> None:
    """
    Generate a draft literature review from structured extractions (run after 'litreview extract').

    Always uses the Anthropic backend (requires ANTHROPIC_API_KEY).
    For corpora >150 papers, automatically uses map-reduce synthesis.
    Output is written as markdown.
    """
    import tomllib
    from litreview.db import get_client, get_or_create_project, get_current_criteria
    from litreview.synthesize import synthesize_project

    if config:
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        name = cfg["project"]["name"]
        project_dir = config.parent
    elif project_name:
        name = project_name
        project_dir = Path(f"projects/{project_name}")
    else:
        console.print("[red]Provide --config or project_name.[/red]")
        raise typer.Exit(1)

    out_path = output or (project_dir / "literature-review.md")

    client = get_client()
    project_id = get_or_create_project(client, name)
    _, criteria = get_current_criteria(client, project_id)

    console.print(f"[bold]Project:[/bold] {name}  ({project_id})")
    console.print(f"Model: {model}")
    console.print(f"Output: {out_path}\n")

    try:
        markdown = synthesize_project(
            client, project_id,
            project_name=name,
            criteria=criteria,
            output_path=out_path,
            model=model,
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    n_words = len(markdown.split())
    console.print(f"[green]Done.[/green] {n_words} words → {out_path}")


# ── init ───────────────────────────────────────────────────────────────────────

@app.command()
def init(
    folder: Path = typer.Argument(..., help="Folder containing seed PDFs."),
    scope: str = typer.Option(..., "--scope", "-s", help="Research scope / inclusion criteria."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Project slug (defaults to folder name)."),
    run_ingest: bool = typer.Option(True, "--ingest/--no-ingest", help="Run ingest immediately after creating project.toml."),
) -> None:
    """
    Bootstrap a new project from a folder of seed PDFs.

    Scans the folder for PDFs, extracts arXiv IDs from filenames, generates
    project.toml, then optionally runs ingest + traversal.
    """
    import re as _re
    import tomllib

    folder = folder.expanduser().resolve()
    if not folder.exists():
        console.print(f"[red]Folder not found: {folder}[/red]")
        raise typer.Exit(1)

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        console.print(f"[red]No PDF files found in {folder}[/red]")
        raise typer.Exit(1)

    # Extract arXiv IDs from filenames like 2305.04802v2.pdf or 2305.04802.pdf
    arxiv_pattern = _re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")
    identifiers = []
    unrecognised = []
    for pdf in pdfs:
        m = arxiv_pattern.search(pdf.stem)
        if m:
            # Strip version suffix for the identifier
            arxiv_id = _re.sub(r"v\d+$", "", m.group(1))
            identifiers.append(arxiv_id)
        else:
            unrecognised.append(pdf.name)
            console.print(f"[yellow]  Could not extract arXiv ID from: {pdf.name}[/yellow]")

    if not identifiers:
        console.print("[red]No arXiv IDs found in PDF filenames. Name files like '2305.04802v2.pdf'.[/red]")
        raise typer.Exit(1)

    project_slug = name or folder.name
    # Sanitise slug: lowercase, replace spaces/special chars with hyphens
    project_slug = _re.sub(r"[^a-z0-9]+", "-", project_slug.lower()).strip("-")

    toml_path = folder / "project.toml"
    ids_toml = "\n  ".join(f'"{i}",' for i in identifiers)
    toml_content = f"""[project]
name = "{project_slug}"

[seeds]
identifiers = [
  {ids_toml}
]

[criteria]
v1 = \"\"\"
{scope.strip()}
\"\"\"
"""

    if toml_path.exists():
        console.print(f"[yellow]project.toml already exists at {toml_path} — overwriting.[/yellow]")

    toml_path.write_text(toml_content)
    console.print(f"[green]Created {toml_path}[/green]")
    console.print(f"  Project : {project_slug}")
    console.print(f"  Seeds   : {identifiers}")
    if unrecognised:
        console.print(f"  [yellow]Skipped (no arXiv ID): {unrecognised}[/yellow]")

    if not run_ingest:
        console.print("\nDone. Run [bold]litreview ingest --config {toml_path}[/bold] when ready.")
        return

    # Run ingest
    console.print("\n[bold]Step 1 — Ingesting seeds[/bold]")
    from litreview.db import get_client, get_or_create_project, upsert_paper, upsert_criteria
    from litreview.ingest import ingest_papers

    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    criteria_content = cfg.get("criteria", {}).get("v1")
    client = get_client()
    project_id = get_or_create_project(client, project_slug)
    console.print(f"[bold]Project:[/bold] {project_slug}  ({project_id})")

    if criteria_content:
        upsert_criteria(client, project_id, version=1, content=criteria_content, trigger="initial")
        console.print("[green]Criteria v1 stored.[/green]")

    papers = ingest_papers(identifiers, project_id=project_id, source="seed", depth=0, enrich_openalex=True)
    if not papers:
        console.print("[red]No papers resolved. Check identifiers and API keys.[/red]")
        raise typer.Exit(1)

    table = Table("Title", "Year", "S2 ID", "DOI")
    for paper in papers:
        upsert_paper(client, paper)
        table.add_row(
            (paper.title or "")[:60],
            str(paper.year or ""),
            paper.s2_id or "",
            paper.doi or "",
        )
    console.print(table)
    console.print(f"[green]Upserted {len(papers)} seed papers.[/green]")

    # Run traversal + audit
    console.print("\n[bold]Step 2 — Citation traversal[/bold]")
    from litreview.traverse import traverse_citations
    stats = traverse_citations(client, project_id, direction="both")
    console.print(f"  {stats['candidates']} candidates → {stats['papers_upserted']} new, {stats['edges_stored']} edges")

    _run_audit(client, project_id, console, project_dir=folder)

    console.print(f"\n[green]Project '{project_slug}' is ready.[/green]")


# ── run ────────────────────────────────────────────────────────────────────────

@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to project TOML."),
    backend: str = typer.Option("gemini", "--backend", "-b", help="LLM backend: groq, gemini, openrouter, or anthropic."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    direction: str = typer.Option("both", "--direction", "-d"),
    max_rounds: int = typer.Option(10, "--max-rounds", help="Safety cap on iterations."),
    yield_threshold: float = typer.Option(0.05, "--yield-threshold", help="Yield rate below which pipeline is converging."),
    stable_rounds: int = typer.Option(2, "--stable-rounds", help="Consecutive low-yield rounds to declare stability."),
    recover: bool = typer.Option(True, "--recover/--no-recover", help="Recover missing refs via S2 after each audit."),
    use_recommend: bool = typer.Option(True, "--recommend/--no-recommend", help="Run S2 recommendations as a parallel sourcing track alongside citation traversal."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """
    Run the full automated pipeline until stable:

      screen → download PDFs → audit included + recover → traverse → recommend → repeat

    Starts from whatever state the project is in (pending papers, current frontier depth).
    """
    import tomllib
    import time as _time
    from litreview.db import (
        get_client, get_or_create_project, get_papers,
        next_round_number, is_stable, get_frontier_depth,
    )
    from litreview.traverse import traverse_citations
    from litreview.screen import screen_project
    from litreview.download import download_included_pdfs
    from litreview.audit import audit_included, recover_missing_refs

    with open(config, "rb") as f:
        cfg = tomllib.load(f)
    name = cfg["project"]["name"]
    project_dir = config.parent
    included_pdf_dir = project_dir / "included-pdfs"

    client = get_client()
    project_id = get_or_create_project(client, name)

    console.print(f"[bold]Project:[/bold] {name}  ({project_id})")
    console.print(f"[bold]Backend:[/bold] {backend}  |  max_rounds={max_rounds}  yield_threshold={yield_threshold:.0%}\n")

    rounds_run = 0
    last_new_papers = None

    while rounds_run < max_rounds:
        rounds_run += 1
        round_start = _time.time()
        console.rule(f"[bold]Round {rounds_run}[/bold]")

        # ── Step 1: Screen any pending papers ─────────────────────────────────
        pending = [p for p in get_papers(client, project_id) if p["inclusion_status"] == "pending"]
        if pending:
            round_number = next_round_number(client, project_id)
            console.print(f"\n[bold]1 — Screen[/bold]  ({len(pending)} pending, round {round_number})")
            counts = screen_project(
                client, project_id, round_number,
                model=model, dry_run=dry_run, backend=backend,
            )
            screened = counts["total"] - counts["skipped"]
            yield_rate = counts["include"] / screened if screened else 0.0
            console.print(
                f"   [green]+{counts['include']} included[/green]  "
                f"[red]-{counts['exclude']} excluded[/red]  "
                f"~{counts['uncertain']} uncertain  "
                f"yield={yield_rate:.1%}"
            )
        else:
            console.print("\n[bold]1 — Screen[/bold]  [dim]no pending papers[/dim]")

        # ── Step 2: Download PDFs for included papers ──────────────────────────
        included = [p for p in get_papers(client, project_id, status="included")
                    if p.get("source") != "seed"]
        if included:
            console.print(f"\n[bold]2 — Download PDFs[/bold]  ({len(included)} included non-seed papers)")
            dl_counts = download_included_pdfs(included, included_pdf_dir)
            console.print(
                f"   [green]+{dl_counts['downloaded']} downloaded[/green]  "
                f"{dl_counts['skipped']} skipped  "
                f"[red]{dl_counts['failed']} failed[/red]"
            )

        # ── Step 3: Audit included + recover missing refs ──────────────────────
        if recover and included_pdf_dir.exists():
            console.print(f"\n[bold]3 — Audit + Recover[/bold]")
            audit_results = audit_included(client, project_id, pdf_dir=included_pdf_dir)
            total_missing = sum(r.n_missing for r in audit_results)
            avg_capture = (
                sum(r.capture_rate for r in audit_results) / len(audit_results)
                if audit_results else 1.0
            )
            console.print(f"   avg capture rate={avg_capture:.0%}  missing={total_missing}")
            if total_missing > 0:
                pid_to_depth = {p["paper_id"]: p.get("depth", 0)
                                for p in get_papers(client, project_id)}
                rec_total: dict[str, int] = {"recovered": 0, "not_found": 0, "skipped": 0}
                for r in audit_results:
                    pdepth = pid_to_depth.get(r.seed_paper_id, 0)
                    c = recover_missing_refs(client, project_id, [r],
                                            parent_depth=pdepth, dry_run=dry_run)
                    for k in rec_total:
                        rec_total[k] += c[k]
                console.print(
                    f"   [green]+{rec_total['recovered']} recovered[/green]  "
                    f"{rec_total['not_found']} not found  "
                    f"{rec_total['skipped']} skipped"
                )
        else:
            console.print(f"\n[bold]3 — Audit + Recover[/bold]  [dim]skipped (no PDFs yet)[/dim]")

        # ── Step 4: Traverse from current frontier ─────────────────────────────
        from_depth = get_frontier_depth(client, project_id)
        console.print(f"\n[bold]4 — Traverse[/bold]  (from depth {from_depth})")
        stats = traverse_citations(client, project_id, direction=direction, from_depth=from_depth)  # type: ignore[arg-type]
        last_new_papers = stats["papers_upserted"]
        console.print(
            f"   {stats['candidates']} candidates → "
            f"[green]+{last_new_papers} new[/green]  "
            f"{stats['edges_stored']} edges"
        )

        # ── Step 5: S2 recommendations ─────────────────────────────────────────
        if use_recommend:
            from litreview.recommend import fetch_s2_recommendations
            from litreview.db import upsert_paper as _upsert_paper
            _all = get_papers(client, project_id)
            _pos = sorted(
                [p for p in _all if p.get("source") == "seed" or p.get("inclusion_status") == "included"],
                key=lambda p: p.get("depth", 0), reverse=True,
            )
            _neg = [p for p in _all if p.get("inclusion_status") == "excluded"]
            _pos_ids = [p["s2_id"] for p in _pos if p.get("s2_id")]
            _neg_ids = [p["s2_id"] for p in _neg if p.get("s2_id")]
            _existing_s2  = {p["s2_id"] for p in _all if p.get("s2_id")}
            _existing_doi = {p["doi"]   for p in _all if p.get("doi")}
            _reco_depth = get_frontier_depth(client, project_id) + 1
            console.print(f"\n[bold]5 — Recommend[/bold]  ({len(_pos_ids)} positives, {len(_neg_ids)} negatives)")
            _reco = fetch_s2_recommendations(project_id, _pos_ids, _neg_ids, depth=_reco_depth)
            _reco_new = [
                p for p in _reco
                if not ((p.s2_id and p.s2_id in _existing_s2) or (p.doi and p.doi in _existing_doi))
            ]
            _upserted = 0
            for _p in _reco_new:
                try:
                    _upsert_paper(client, _p)
                    _upserted += 1
                except Exception:
                    pass
            console.print(
                f"   {len(_reco)} returned → [green]+{_upserted} new candidates[/green]"
            )

        # ── Stability check ────────────────────────────────────────────────────
        stable, reason = is_stable(
            client, project_id,
            yield_threshold=yield_threshold,
            consecutive_rounds=stable_rounds,
            new_papers_count=last_new_papers,
        )
        elapsed = _time.time() - round_start
        console.print(f"\n   [dim]Round time: {elapsed/60:.1f} min[/dim]")

        if stable:
            console.print(f"\n[bold green]Pipeline stable — stopping.[/bold green]  ({reason})")
            break
        else:
            console.print(f"   [dim]Stability: {reason}[/dim]")

    else:
        console.print(f"\n[yellow]Reached max rounds ({max_rounds}) — stopping.[/yellow]")

    # ── Final summary ──────────────────────────────────────────────────────────
    all_papers = get_papers(client, project_id)
    n_included = sum(1 for p in all_papers if p["inclusion_status"] == "included")
    n_total = len(all_papers)
    console.print(f"\n[bold]Final corpus:[/bold] {n_included} included / {n_total} total")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
