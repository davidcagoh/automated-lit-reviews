"""
Ingest seed papers for a project.

Usage:
    python scripts/ingest_seeds.py <project-name> [identifier ...]

    # Use a project TOML file instead:
    python scripts/ingest_seeds.py --config projects/geometry-testing-simplicial-complexes/project.toml

Identifiers can be DOIs, arXiv IDs (with or without 'arXiv:' prefix), or
quoted title strings.
"""
from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tomllib
from rich.console import Console
from rich.table import Table

from litreview.db import get_client, get_or_create_project, upsert_paper, upsert_criteria
from litreview.ingest import ingest_papers

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
console = Console()


def load_project_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest seed papers into Supabase.")
    parser.add_argument("--config", type=Path, help="Path to project TOML file.")
    parser.add_argument("project_name", nargs="?", help="Project name.")
    parser.add_argument("identifiers", nargs="*", help="DOIs, arXiv IDs, or titles.")
    args = parser.parse_args()

    if args.config:
        cfg = load_project_config(args.config)
        project_name: str = cfg["project"]["name"]
        identifiers: list[str] = cfg["seeds"]["identifiers"]
        criteria_content: str | None = cfg.get("criteria", {}).get("v1")
    else:
        if not args.project_name or not args.identifiers:
            parser.error("Provide --config or both project_name and identifiers.")
        project_name = args.project_name
        identifiers = args.identifiers
        criteria_content = None

    client = get_client()

    # Ensure project exists
    project_id = get_or_create_project(client, project_name)
    console.print(f"[bold]Project:[/bold] {project_name}  ({project_id})")

    # Store criteria v1 if provided
    if criteria_content:
        upsert_criteria(client, project_id, version=1, content=criteria_content, trigger="initial")
        console.print("[green]Criteria v1 stored.[/green]")

    # Fetch papers
    console.print(f"\nResolving {len(identifiers)} seed identifiers …")
    papers = ingest_papers(identifiers, project_id=project_id, source="seed", depth=0)

    if not papers:
        console.print("[red]No papers resolved. Check identifiers and API keys.[/red]")
        sys.exit(1)

    # Upsert into DB
    table = Table("Title", "Year", "S2 ID", "DOI", "Embedding?")
    for paper in papers:
        row = upsert_paper(client, paper)
        table.add_row(
            (paper.title or "")[:60],
            str(paper.year or ""),
            paper.s2_id or "",
            paper.doi or "",
            "yes" if paper.embedding else "no",
        )

    console.print(table)
    console.print(f"\n[green]Upserted {len(papers)} papers into '{project_name}'.[/green]")


if __name__ == "__main__":
    main()
