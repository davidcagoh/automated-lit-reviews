"""
Compare screening decisions across models.

Ground truth: Claude Sonnet decisions saved to /tmp/claude_decisions.json
  {"title": {"decision": "include|exclude|uncertain", "reasoning": "..."}, ...}

DB: Haiku (or GROQ) decisions stored in Supabase after a screen run.

Usage:
  python scripts/compare_screeners.py --config projects/.../project.toml
  python scripts/compare_screeners.py --config ... --ground-truth /tmp/claude_decisions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Allow running without installing the package ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from litreview.db import get_client, get_or_create_project, get_papers


def _load_ground_truth(path: str) -> dict[str, str]:
    """Load {title: decision} from a JSON file produced during manual screening."""
    with open(path) as f:
        raw = json.load(f)
    # Support both {title: decision_str} and {title: {decision: ..., reasoning: ...}}
    result = {}
    for title, val in raw.items():
        if isinstance(val, dict):
            result[title] = val["decision"]
        else:
            result[title] = val
    return result


def _normalise(title: str) -> str:
    """Lower-case + strip for fuzzy matching."""
    return title.lower().strip()


def _match(title: str, ground_truth: dict[str, str]) -> str | None:
    """Find a ground truth decision for a DB paper title (exact then normalised)."""
    if title in ground_truth:
        return ground_truth[title]
    norm = _normalise(title)
    for gt_title, decision in ground_truth.items():
        if _normalise(gt_title) == norm:
            return decision
    return None


def compare(
    project_id: str,
    client,
    ground_truth: dict[str, str],
    label_gt: str = "Claude Sonnet",
    label_db: str = "Haiku",
) -> None:
    papers = get_papers(client, project_id)
    # Only non-seed papers that have been screened (not pending)
    screened = [
        p for p in papers
        if p.get("source") != "seed" and p.get("inclusion_status") != "pending"
    ]

    # Map DB status back to decision vocabulary
    def db_decision(p: dict) -> str:
        status = p["inclusion_status"]
        reason = p.get("rejection_reason") or ""
        if status == "included":
            return "include"
        elif status == "excluded":
            return "exclude"
        elif reason.startswith("UNCERTAIN"):
            return "uncertain"
        return "pending"

    agree = 0
    disagree = 0
    no_gt = 0

    errors: dict[str, list[str]] = {}  # mismatch type → list of titles

    for p in screened:
        title = p["title"] or ""
        db_dec = db_decision(p)
        gt_dec = _match(title, ground_truth)

        if gt_dec is None:
            no_gt += 1
            continue

        if db_dec == gt_dec:
            agree += 1
        else:
            disagree += 1
            key = f"{label_db}={db_dec.upper()[:3]} {label_gt}={gt_dec.upper()[:3]}"
            errors.setdefault(key, []).append(title)

    total = agree + disagree
    pct = agree / total * 100 if total else 0

    print(f"\n{'='*65}")
    print(f"  {label_db}  vs  {label_gt}")
    print(f"  Agreement: {agree}/{total} ({pct:.0f}%)")
    if no_gt:
        print(f"  (skipped {no_gt} papers with no ground-truth entry)")
    print(f"{'='*65}")

    # Summary by mismatch type
    if errors:
        print("\nDisagreement breakdown:")
        for mtype, titles in sorted(errors.items(), key=lambda x: -len(x[1])):
            print(f"  [{mtype}]  ×{len(titles)}")

        print("\nDisagreement detail:")
        for mtype, titles in sorted(errors.items(), key=lambda x: -len(x[1])):
            for t in titles:
                print(f"  [{mtype}]  {t[:75]}")
    else:
        print("\nPerfect agreement!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare screener decisions vs ground truth.")
    parser.add_argument("--config", required=True, help="Path to project.toml")
    parser.add_argument(
        "--ground-truth",
        default="/tmp/claude_decisions.json",
        help="Path to Claude Sonnet ground-truth JSON (default: /tmp/claude_decisions.json)",
    )
    parser.add_argument("--label-gt", default="ClaudeSonnet", help="Label for ground truth model")
    parser.add_argument("--label-db", default="Haiku", help="Label for DB model being compared")
    args = parser.parse_args()

    import tomllib
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    project_name = cfg["project"]["name"]

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"[error] Ground truth file not found: {gt_path}")
        sys.exit(1)

    ground_truth = _load_ground_truth(str(gt_path))
    print(f"Loaded {len(ground_truth)} ground-truth decisions from {gt_path}")

    client = get_client()
    project_id = get_or_create_project(client, project_name)

    compare(project_id, client, ground_truth, label_gt=args.label_gt, label_db=args.label_db)


if __name__ == "__main__":
    main()
