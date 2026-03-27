"""
Run a tournament round: screen a labeled test set with each contestant config,
compute agreement vs ground truth, and update the leaderboard.

Workflow:
  1. python sampler.py --config ... --out candidate_test_set.json
  2. Share candidate_test_set.json with Claude in chat → Claude writes ground_truth.json
  3. python run_tournament.py --round 1 --config rounds/round_1.toml
  4. Read results/round_1_results.json → Claude analyzes disagreements, writes
     prompts/v2_<description>.txt and rounds/round_2.toml
  5. Repeat from step 3

Usage:
  python optimizations/screening/tournament/run_tournament.py \
      --round 1 --config optimizations/screening/tournament/rounds/round_1.toml
      [--contestant groq-llama70b]   # run only this contestant
      [--resume]                     # skip contestants with complete result files
      [--dry-run]                    # random decisions, no API calls
      [--gemini-tier paid|free]      # paid=0.2s sleep, free=4.0s (default: paid)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import tomllib
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from litreview.db import get_client, get_or_create_project, get_current_criteria
from optimizations.screening.tournament.wrapper import screen_paper_instrumented
from optimizations.screening.tournament.metrics import compute_metrics

TOURNAMENT_DIR = Path(__file__).parent
console = Console()

_GEMINI_RATE_SLEEP = {"paid": 0.2, "free": 4.0}


def _load_ground_truth(path: Path) -> dict[str, str]:
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for title, val in raw.items():
        result[title] = val["decision"] if isinstance(val, dict) else val
    return result


def _load_prompt(name: str) -> str:
    path = TOURNAMENT_DIR / "prompts" / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text()


def _results_path(results_dir: Path, round_n: int, contestant_id: str) -> Path:
    return results_dir / f"round_{round_n}_{contestant_id}.json"


def _is_complete(path: Path, n_papers: int) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return len(data) >= n_papers
    except Exception:
        return False


def run_contestant(
    contestant: dict,
    test_papers: list[dict],
    ground_truth: dict[str, str],
    criteria: str,
    results_path: Path,
    dry_run: bool = False,
    gemini_tier: str = "paid",
) -> list[dict]:
    """
    Screen all test papers for one contestant. Writes results incrementally to
    results_path after each paper so a crash can be resumed.

    Returns list of per-paper result dicts.
    """
    cid = contestant["id"]
    backend = contestant["backend"]
    model = contestant["model"]
    prompt_name = contestant.get("prompt", "v1_baseline")
    rate_sleep = contestant.get("rate_sleep", 2.0)

    # Override Gemini rate sleep based on tier
    if backend == "gemini":
        rate_sleep = _GEMINI_RATE_SLEEP.get(gemini_tier, 0.2)

    prompt_template = _load_prompt(prompt_name)

    # Load partial results if resuming mid-run
    existing: list[dict] = []
    if results_path.exists():
        try:
            existing = json.loads(results_path.read_text())
        except Exception:
            existing = []
    done_titles = {r["title"] for r in existing}
    remaining = [p for p in test_papers if p["title"] not in done_titles]

    if existing:
        console.print(f"  [dim]Resuming {cid}: {len(existing)} done, {len(remaining)} remaining[/dim]")

    results = list(existing)
    n_total = len(test_papers)

    for i, paper in enumerate(remaining):
        title    = paper["title"]
        abstract = paper.get("abstract")
        global_i = len(results) + 1

        if dry_run:
            # Random decisions for testing metrics math
            decision = random.choice(["include", "exclude", "exclude"])
            result = {
                "title":         title,
                "decision":      decision,
                "reasoning":     "[dry-run]",
                "input_tokens":  450,
                "output_tokens": 60,
                "latency_s":     0.01,
                "parse_ok":      True,
                "est_cost_usd":  0.0,
            }
        else:
            try:
                result = screen_paper_instrumented(
                    title, abstract, criteria, model, backend,
                    prompt_template=prompt_template,
                )
            except Exception as e:
                console.print(f"  [red][{global_i}/{n_total}] ERROR on '{title[:50]}': {e}[/red]")
                result = {
                    "decision": "uncertain", "reasoning": f"ERROR: {e}",
                    "input_tokens": 0, "output_tokens": 0,
                    "latency_s": 0.0, "parse_ok": False, "est_cost_usd": 0.0,
                }

        result["title"] = title
        gt = ground_truth.get(title) or next(
            (v for k, v in ground_truth.items() if k.lower().strip() == title.lower().strip()), None
        )
        result["gt"] = gt

        match_symbol = ""
        if gt:
            model_include = result["decision"] == "include"
            gt_include    = gt == "include"
            match_symbol  = "[green]✓[/green]" if model_include == gt_include else "[red]✗[/red]"

        console.print(
            f"  [{global_i:03d}/{n_total}] "
            f"[cyan]{result['decision']:<8}[/cyan] gt={gt or '?':<8} "
            f"{match_symbol} {title[:55]}"
        )

        results.append(result)
        results_path.write_text(json.dumps(results, indent=2))

        if not dry_run and i < len(remaining) - 1:
            time.sleep(rate_sleep)

    return results


def run_round(
    round_n: int,
    config_path: Path,
    contestant_filter: str | None,
    resume: bool,
    dry_run: bool,
    gemini_tier: str,
) -> None:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    tournament = cfg["tournament"]
    project_name = tournament["project"]
    contestants = cfg["contestant"]

    if contestant_filter:
        contestants = [c for c in contestants if c["id"] == contestant_filter]
        if not contestants:
            console.print(f"[red]Contestant '{contestant_filter}' not found in config.[/red]")
            sys.exit(1)

    # Ground truth: {title: "include"|"exclude"} — labeled by Claude in-chat
    gt_path = TOURNAMENT_DIR / "ground_truth.json"
    if not gt_path.exists() and not dry_run:
        console.print(f"[red]ground_truth.json not found at {gt_path}[/red]")
        console.print("Run sampler.py, then have Claude screen candidate_test_set.json in chat.")
        sys.exit(1)
    ground_truth = _load_ground_truth(gt_path) if gt_path.exists() else {}

    # Test papers: list of {title, abstract, ...} from sampler.py
    # Separate from ground_truth so we have abstracts for the LLM calls.
    candidate_path = TOURNAMENT_DIR / "candidate_test_set.json"
    if candidate_path.exists():
        with open(candidate_path) as f:
            test_papers = json.load(f)
        # Only screen papers that have a ground-truth entry
        test_papers = [p for p in test_papers if p.get("title") in ground_truth
                       or p.get("title", "").lower().strip() in
                       {k.lower().strip() for k in ground_truth}]
    elif ground_truth:
        # Fallback: derive paper list from ground truth keys (no abstracts)
        test_papers = [{"title": t, "abstract": None} for t in ground_truth]
    elif dry_run:
        test_papers = [{"title": f"Test paper {i}", "abstract": "Abstract."} for i in range(20)]
        ground_truth = {p["title"]: random.choice(["include", "exclude", "exclude"])
                        for p in test_papers}
    else:
        console.print("[red]No candidate_test_set.json or ground_truth.json found.[/red]")
        sys.exit(1)

    # Load criteria from DB
    client = get_client()
    project_id = get_or_create_project(client, project_name)
    _, criteria = get_current_criteria(client, project_id)

    results_dir = TOURNAMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    console.print(f"\n[bold]Tournament Round {round_n}[/bold]  "
                  f"project={project_name}  papers={len(test_papers)}  "
                  f"contestants={len(contestants)}  "
                  f"{'[yellow]DRY-RUN[/yellow]' if dry_run else ''}")

    all_contestant_results: dict[str, list[dict]] = {}

    for contestant in contestants:
        cid = contestant["id"]
        rpath = _results_path(results_dir, round_n, cid)

        if resume and _is_complete(rpath, len(test_papers)):
            console.print(f"\n[dim]Skipping {cid} (complete)[/dim]")
            with open(rpath) as f:
                all_contestant_results[cid] = json.load(f)
            continue

        console.print(f"\n[bold cyan]── {cid}[/bold cyan]  "
                      f"backend={contestant['backend']}  model={contestant['model']}  "
                      f"prompt={contestant.get('prompt','v1_baseline')}")

        decisions = run_contestant(
            contestant, test_papers, ground_truth, criteria, rpath,
            dry_run=dry_run, gemini_tier=gemini_tier,
        )
        all_contestant_results[cid] = decisions

    # ── Aggregate results ─────────────────────────────────────────────────────
    aggregated: dict[str, dict] = {}
    for cid, decisions in all_contestant_results.items():
        metrics = compute_metrics(decisions, ground_truth)
        contestant_cfg = next(c for c in cfg["contestant"] if c["id"] == cid)
        aggregated[cid] = {
            "backend":  contestant_cfg["backend"],
            "model":    contestant_cfg["model"],
            "prompt":   contestant_cfg.get("prompt", "v1_baseline"),
            "metrics":  metrics,
            "decisions": decisions,
        }

    round_results = {
        "round":         round_n,
        "test_set_size": len(test_papers),
        "run_at":        datetime.now(timezone.utc).isoformat(),
        "dry_run":       dry_run,
        "contestants":   aggregated,
    }
    round_path = results_dir / f"round_{round_n}_results.json"
    round_path.write_text(json.dumps(round_results, indent=2))
    console.print(f"\n[green]Results written to {round_path}[/green]")

    # ── Update leaderboard ────────────────────────────────────────────────────
    lb_path = results_dir / "leaderboard.json"
    leaderboard: list[dict] = []
    if lb_path.exists():
        try:
            leaderboard = json.loads(lb_path.read_text())
        except Exception:
            leaderboard = []

    # Remove any previous entries for this round (allow re-runs)
    leaderboard = [e for e in leaderboard if e["round"] != round_n]
    for cid, data in aggregated.items():
        m = data["metrics"]
        leaderboard.append({
            "round":               round_n,
            "contestant_id":       cid,
            "backend":             data["backend"],
            "model":               data["model"],
            "prompt":              data["prompt"],
            "agreement":           m["agreement"],
            "false_negative_rate": m["false_negative_rate"],
            "mean_latency_s":      m["mean_latency_s"],
            "est_cost_usd":        m["est_cost_usd"],
            "cost_per_1k_papers":  m["cost_per_1k_papers"],
            "n_matched":           m["n_matched"],
            "dry_run":             dry_run,
        })
    leaderboard.sort(key=lambda x: (-x["agreement"], x["false_negative_rate"]))
    lb_path.write_text(json.dumps(leaderboard, indent=2))

    # ── Print leaderboard table ───────────────────────────────────────────────
    _print_leaderboard(leaderboard, highlight_round=round_n)
    console.print(
        f"\n[dim]Next: review round_{round_n}_results.json, then share with Claude to "
        f"write prompts/v{round_n+1}_<description>.txt and rounds/round_{round_n+1}.toml[/dim]"
    )


def _print_leaderboard(leaderboard: list[dict], highlight_round: int) -> None:
    table = Table(title="Leaderboard", show_header=True, header_style="bold")
    table.add_column("Rank", justify="right", width=5)
    table.add_column("Contestant", width=28)
    table.add_column("Prompt", width=18)
    table.add_column("Agree %", justify="right", width=8)
    table.add_column("FNR %", justify="right", width=7)
    table.add_column("Latency", justify="right", width=8)
    table.add_column("Cost/1k", justify="right", width=9)
    table.add_column("Rnd", justify="right", width=4)

    for rank, entry in enumerate(leaderboard, 1):
        style = "bold" if entry["round"] == highlight_round else "dim"
        table.add_row(
            str(rank),
            entry["contestant_id"],
            entry["prompt"],
            f"{entry['agreement']:.0%}",
            f"{entry['false_negative_rate']:.0%}",
            f"{entry['mean_latency_s']:.2f}s",
            f"${entry['cost_per_1k_papers']:.2f}",
            str(entry["round"]),
            style=style,
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tournament round.")
    parser.add_argument("--round",  type=int, required=True, help="Round number")
    parser.add_argument("--config", required=True, help="Path to rounds/round_N.toml")
    parser.add_argument("--contestant", default=None, help="Run only this contestant ID")
    parser.add_argument("--resume",  action="store_true", help="Skip complete contestants")
    parser.add_argument("--dry-run", action="store_true",
                        help="No API calls; random decisions. Validates metrics math.")
    parser.add_argument("--gemini-tier", choices=["paid", "free"], default="paid",
                        help="Gemini rate tier: paid=0.2s sleep, free=4.0s (default: paid)")
    args = parser.parse_args()

    run_round(
        round_n=args.round,
        config_path=Path(args.config),
        contestant_filter=args.contestant,
        resume=args.resume,
        dry_run=args.dry_run,
        gemini_tier=args.gemini_tier,
    )


if __name__ == "__main__":
    main()
