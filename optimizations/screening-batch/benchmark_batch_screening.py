"""
Benchmark: does batching papers per LLM request degrade screening accuracy?

Takes a stratified sample of already-screened papers from the DB, re-screens them
with Gemini (or any backend) at multiple batch sizes, and measures agreement against
the existing DB decisions (used as ground truth).

Motivation
----------
Sending N papers per request shares the system prompt (criteria) across the batch,
reducing total tokens and API calls ~N-fold. The risk is cross-paper contamination —
borderline decisions may be nudged by their neighbors. This script quantifies that.

Usage
-----
  python optimizations/screening-batch/benchmark_batch_screening.py \\
      --config projects/self-supervised-pretraining/project.toml \\
      --batch-sizes 1,5,10,20 \\
      --sample-size 100 \\
      --backend gemini

Output
------
  batch_size | agreement | inc_agree | exc_agree | uncertain_rate | n_calls | s/paper
  -----------+-----------+-----------+-----------+----------------+---------+--------
           1 |     0.91  |     0.85  |     0.94  |          0.03  |     100 |    2.8
           5 |     0.89  |     0.83  |     0.92  |          0.05  |      20 |    0.7
          10 |     0.88  |     0.82  |     0.90  |          0.06  |      10 |    0.4
          20 |     0.84  |     0.78  |     0.87  |          0.09  |       5 |    0.3

Decision rule: pick the largest batch_size with agreement within 3pp of the
batch_size=1 baseline. If uncertain_rate rises sharply, the model is struggling.

Note: this script never writes to the DB — it is read-only.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import tomllib
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from litreview.db import get_client, get_or_create_project, get_papers, get_current_criteria
from litreview.screen import screen_paper, screen_papers_batch


def _sample_papers(
    client,
    project_id: str,
    sample_size: int,
    seed: int,
) -> list[dict]:
    """Stratified sample: 30% included / 70% excluded, excluding seeds and pending."""
    all_papers = get_papers(client, project_id)
    included = [
        p for p in all_papers
        if p.get("inclusion_status") == "included" and p.get("source") != "seed"
    ]
    excluded = [
        p for p in all_papers
        if p.get("inclusion_status") == "excluded" and p.get("source") != "seed"
    ]

    rng = random.Random(seed)
    n_inc = min(round(sample_size * 0.30), len(included))
    n_exc = min(sample_size - n_inc, len(excluded))

    sampled = rng.sample(included, n_inc) + rng.sample(excluded, n_exc)
    rng.shuffle(sampled)
    return sampled


def _db_decision(paper: dict) -> str:
    status = paper["inclusion_status"]
    if status == "included":
        return "include"
    elif status == "excluded":
        return "exclude"
    return "uncertain"


def _run_batch_size(
    papers: list[dict],
    criteria: str,
    batch_size: int,
    backend: str,
    model: str | None,
    rate_sleep: float,
) -> dict:
    """Re-screen all papers at the given batch_size and return metrics."""
    agreed = 0
    inc_agreed = 0
    inc_total = 0
    exc_agreed = 0
    exc_total = 0
    uncertain_count = 0
    n_calls = 0

    t0 = time.monotonic()

    i = 0
    while i < len(papers):
        chunk = papers[i : i + batch_size]

        if batch_size == 1:
            p = chunk[0]
            results = [screen_paper(p["title"], p.get("abstract"), criteria, backend=backend, model=model)]
        else:
            results = screen_papers_batch(chunk, criteria, backend=backend, model=model)

        n_calls += 1

        for paper, result in zip(chunk, results):
            gt = _db_decision(paper)
            pred = result["decision"]

            if pred == "uncertain":
                uncertain_count += 1

            if gt == "include":
                inc_total += 1
                if pred == "include":
                    inc_agreed += 1
                    agreed += 1
            elif gt == "exclude":
                exc_total += 1
                if pred in ("exclude", "uncertain"):  # conservative: uncertain counts as agree on excludes
                    exc_agreed += 1
                    agreed += 1

        i += len(chunk)

        if rate_sleep > 0:
            time.sleep(rate_sleep)

    elapsed = time.monotonic() - t0
    n = len(papers)

    return {
        "batch_size": batch_size,
        "n_papers": n,
        "n_calls": n_calls,
        "agreement": agreed / n if n else 0.0,
        "inc_agree": inc_agreed / inc_total if inc_total else float("nan"),
        "exc_agree": exc_agreed / exc_total if exc_total else float("nan"),
        "uncertain_rate": uncertain_count / n if n else 0.0,
        "s_per_paper": elapsed / n if n else 0.0,
    }


def _fmt(v: float) -> str:
    if v != v:  # nan
        return "  n/a "
    return f"{v:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark batch screening accuracy vs single-paper baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to project.toml")
    parser.add_argument(
        "--batch-sizes", default="1,5,10,20",
        help="Comma-separated batch sizes to test (default: 1,5,10,20)",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Papers to sample (default 100)")
    parser.add_argument("--backend", default="gemini", help="LLM backend (default: gemini)")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument(
        "--rate-sleep", type=float, default=None,
        help="Seconds to sleep between API calls (default: per-backend default)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    project_name = cfg["project"]["name"]

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Per-backend rate sleep defaults (mirror screen.py)
    _RATE_SLEEP = {"groq": 2.0, "gemini": 2.0, "openrouter": 6.0, "anthropic": 0.5}
    rate_sleep = args.rate_sleep if args.rate_sleep is not None else _RATE_SLEEP.get(args.backend, 1.0)

    client = get_client()
    project_id = get_or_create_project(client, project_name)
    _, criteria = get_current_criteria(client, project_id)

    if not criteria:
        print(f"[error] No criteria found for project '{project_name}'")
        sys.exit(1)

    papers = _sample_papers(client, project_id, args.sample_size, args.seed)
    n_inc = sum(1 for p in papers if p["inclusion_status"] == "included")
    n_exc = len(papers) - n_inc
    print(f"\nProject:     {project_name}")
    print(f"Backend:     {args.backend}" + (f"  model={args.model}" if args.model else ""))
    print(f"Sample:      {len(papers)} papers ({n_inc} included / {n_exc} excluded)")
    print(f"Rate sleep:  {rate_sleep}s  (override with --rate-sleep)")
    print(f"Batch sizes: {batch_sizes}")
    print()

    results = []
    for bs in batch_sizes:
        n_calls_est = (len(papers) + bs - 1) // bs
        est_min = n_calls_est * (rate_sleep + 1.0) / 60
        print(f"[batch_size={bs}]  ~{n_calls_est} calls, ETA ~{est_min:.1f}min …", flush=True)
        r = _run_batch_size(papers, criteria, bs, args.backend, args.model, rate_sleep)
        results.append(r)
        print(
            f"  agreement={r['agreement']:.3f}  inc={r['inc_agree']:.3f}  "
            f"exc={r['exc_agree']:.3f}  uncertain={r['uncertain_rate']:.3f}  "
            f"{r['s_per_paper']:.2f}s/paper"
        )

    # Summary table
    baseline = results[0]["agreement"] if results else 0.0
    print("\n" + "="*75)
    print(f"{'batch_size':>10} | {'agreement':>9} | {'inc_agree':>9} | {'exc_agree':>9} | {'uncertain%':>10} | {'n_calls':>7} | {'s/paper':>7} | {'vs baseline':>11}")
    print("-"*75)
    for r in results:
        delta = r["agreement"] - baseline
        delta_str = f"{delta:+.3f}" if r["batch_size"] != results[0]["batch_size"] else "baseline"
        print(
            f"{r['batch_size']:>10} | "
            f"{_fmt(r['agreement']):>9} | "
            f"{_fmt(r['inc_agree']):>9} | "
            f"{_fmt(r['exc_agree']):>9} | "
            f"{r['uncertain_rate']:>10.3f} | "
            f"{r['n_calls']:>7} | "
            f"{r['s_per_paper']:>7.2f} | "
            f"{delta_str:>11}"
        )
    print("="*75)
    print(
        "\nDecision rule: pick the largest batch_size with agreement within 0.03 of baseline.\n"
        "Watch for rising uncertain_rate — indicates the model is struggling with batch context.\n"
    )


if __name__ == "__main__":
    main()
