"""
Pull a stratified test set from Supabase for manual ground-truth labeling.

Sampling strategy (120 papers, ~33% positive rate):
  40  included papers   — random sample
  40  excluded papers   — random sample
  40  borderline papers — excluded papers whose rejection_reason contains
                          "uncertain", "borderline", "possibly", "unclear", or
                          had an "UNCERTAIN:" prefix (the hardest cases)

Why 33% positives: at the natural ~13% rate, a model that always-excludes gets
87% accuracy — not meaningful. At 33%, it only gets 67%, so accuracy is
discriminating. Borderline oversampling concentrates the test set on the cases
where model calibration matters most.

Output JSON: {title: {decision, abstract, paper_id}}
After manual review / Claude in-chat screening, strip to {title: "include"|"exclude"}
and save as ground_truth.json.

Usage:
  python optimizations/screening/tournament/sampler.py \
      --config projects/geometry-testing-simplicial-complexes/project.toml \
      --out optimizations/screening/tournament/candidate_test_set.json
  python ... --dry-run   # prints stats only, no file written
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from litreview.db import get_client, get_or_create_project, get_papers


_BORDERLINE_KEYWORDS = ("uncertain", "borderline", "possibly", "unclear", "marginal")


def _is_borderline(paper: dict) -> bool:
    reason = (paper.get("rejection_reason") or "").lower()
    return reason.startswith("uncertain:") or any(kw in reason for kw in _BORDERLINE_KEYWORDS)


def sample_test_set(
    project_id: str,
    client,
    n_include: int = 40,
    n_exclude: int = 40,
    n_borderline: int = 40,
    seed: int = 42,
) -> list[dict]:
    """
    Returns a list of paper dicts with keys: title, abstract, paper_id, gt_decision.
    gt_decision is the DB decision ("include"/"exclude") — used as a starting point
    for Claude's in-chat review, not as final ground truth.
    """
    rng = random.Random(seed)
    # Query included and excluded separately to avoid the 1000-row Supabase default
    # cap (projects with thousands of pending papers would otherwise drown out screened ones)
    included_raw = (
        client.table("papers")
        .select("paper_id,title,abstract,source,inclusion_status,rejection_reason")
        .eq("project_id", project_id)
        .eq("inclusion_status", "included")
        .neq("source", "seed")
        .limit(5000)
        .execute()
        .data or []
    )
    excluded_raw = (
        client.table("papers")
        .select("paper_id,title,abstract,source,inclusion_status,rejection_reason")
        .eq("project_id", project_id)
        .eq("inclusion_status", "excluded")
        .neq("source", "seed")
        .limit(5000)
        .execute()
        .data or []
    )
    included = included_raw
    excluded = excluded_raw
    borderline = [p for p in excluded if _is_borderline(p)]
    regular_excluded = [p for p in excluded if not _is_borderline(p)]

    # Sample each stratum; redistribute borderline shortfall to regular excluded
    sampled_include    = rng.sample(included, min(n_include, len(included)))
    sampled_borderline = rng.sample(borderline, min(n_borderline, len(borderline)))
    borderline_deficit = n_borderline - len(sampled_borderline)
    sampled_exclude    = rng.sample(
        regular_excluded, min(n_exclude + borderline_deficit, len(regular_excluded))
    )

    result = []
    for p in sampled_include:
        result.append({
            "paper_id":   p["paper_id"],
            "title":      p["title"],
            "abstract":   p.get("abstract"),
            "gt_decision": "include",
            "stratum":    "included",
        })
    for p in sampled_borderline:
        result.append({
            "paper_id":   p["paper_id"],
            "title":      p["title"],
            "abstract":   p.get("abstract"),
            "gt_decision": "exclude",
            "stratum":    "borderline",
            "rejection_reason": p.get("rejection_reason", ""),
        })
    for p in sampled_exclude:
        result.append({
            "paper_id":   p["paper_id"],
            "title":      p["title"],
            "abstract":   p.get("abstract"),
            "gt_decision": "exclude",
            "stratum":    "excluded",
        })

    rng.shuffle(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a stratified test set from the DB.")
    parser.add_argument("--config", required=True, help="Path to project.toml")
    parser.add_argument("--out", default="optimizations/screening/tournament/candidate_test_set.json",
                        help="Output path for the candidate test set JSON")
    parser.add_argument("--n-include",    type=int, default=40)
    parser.add_argument("--n-exclude",    type=int, default=40)
    parser.add_argument("--n-borderline", type=int, default=40)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats only, don't write output file")
    parser.add_argument("--as-ground-truth", action="store_true",
                        help="Write {title: decision} directly as ground_truth.json "
                             "using DB decisions, skipping the in-chat labeling step. "
                             "The candidate_test_set.json (with abstracts) is still "
                             "written alongside so the runner has abstracts for LLM calls.")
    args = parser.parse_args()

    import tomllib
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    project_name = cfg["project"]["name"]

    client = get_client()
    project_id = get_or_create_project(client, project_name)

    papers = sample_test_set(
        project_id, client,
        n_include=args.n_include,
        n_exclude=args.n_exclude,
        n_borderline=args.n_borderline,
        seed=args.seed,
    )

    strata = {"included": 0, "borderline": 0, "excluded": 0}
    for p in papers:
        strata[p["stratum"]] += 1

    n_include = strata["included"]
    n_total   = len(papers)
    print(f"\nTest set: {n_total} papers")
    print(f"  included:   {strata['included']}")
    print(f"  borderline: {strata['borderline']}")
    print(f"  excluded:   {strata['excluded']}")
    print(f"  positive rate: {n_include/n_total:.0%}")
    no_abstract = sum(1 for p in papers if not p.get("abstract"))
    print(f"  missing abstracts: {no_abstract}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"\nWritten to: {out_path}")

    if args.as_ground_truth:
        # Write ground_truth.json directly from DB decisions alongside the candidate set
        gt = {p["title"]: p["gt_decision"] for p in papers if p.get("title")}
        gt_path = Path(args.out).parent / "ground_truth.json"
        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2)
        print(f"Ground truth written to: {gt_path}  ({len(gt)} entries, using DB decisions)")
        print("\nNext: python run_tournament.py --round 1 --config rounds/round_1.toml")
    else:
        print("\nNext step: share this file with Claude in chat to generate ground_truth.json.")
        print('  Ask Claude to screen each paper against the current criteria and write')
        print('  decisions to optimizations/screening/tournament/ground_truth.json')
        print('  as {"title": "include"|"exclude", ...}')
        print("  (Or re-run with --as-ground-truth to use DB decisions directly.)")


if __name__ == "__main__":
    main()
