"""
Metrics for the screening tournament.

Two primary metrics per contestant:
  agreement          — % of papers where model decision matches ground truth
  false_negative_rate — % of ground-truth includes that the model missed
                        (exclude|uncertain when gt=include)

We keep it to these two because:
- The test set is constructed at ~33% positives (not the natural ~13%), so
  accuracy is already discriminating — a model that always-excludes only gets
  67%, not 87%.
- In a lit review, missing a relevant paper (false negative) is the costly error.
  FNR makes that explicit without requiring full precision/recall.

Title matching reuses _normalise/_match from scripts/compare_screeners.py for
consistency with the existing QA tooling.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.compare_screeners import _normalise, _match  # noqa: F401 — re-exported


def compute_metrics(
    decisions: list[dict[str, Any]],
    ground_truth: dict[str, str],
) -> dict[str, Any]:
    """
    Compute agreement and false-negative rate for one contestant's decisions.

    decisions: list of {title, decision, input_tokens, output_tokens, latency_s,
                        parse_ok, est_cost_usd, ...}
    ground_truth: {title: "include"|"exclude"}

    Returns:
        {
            "agreement":          float,  # 0–1
            "false_negative_rate": float,  # 0–1
            "n_matched":          int,    # papers with ground truth entry
            "n_agree":            int,
            "n_gt_include":       int,    # ground-truth includes in matched set
            "n_fn":               int,    # missed includes
            "mean_latency_s":     float,
            "parse_fail_rate":    float,
            "est_cost_usd":       float,  # total for this contestant run
            "cost_per_1k_papers": float,  # extrapolated
        }
    """
    matched: list[tuple[dict, str]] = []
    for d in decisions:
        title = d.get("title", "")
        gt = _match(title, ground_truth)
        if gt is not None:
            matched.append((d, gt))

    n = len(matched)
    if n == 0:
        return {"agreement": 0.0, "false_negative_rate": 0.0, "n_matched": 0,
                "n_agree": 0, "n_gt_include": 0, "n_fn": 0,
                "mean_latency_s": 0.0, "parse_fail_rate": 0.0,
                "est_cost_usd": 0.0, "cost_per_1k_papers": 0.0}

    agree = 0
    fn = 0
    gt_includes = 0
    for d, gt in matched:
        model_dec = d["decision"]
        # "uncertain" treated as non-include (matches pipeline behaviour — stays pending)
        model_include = model_dec == "include"
        gt_include = gt == "include"

        if gt_include:
            gt_includes += 1
            if not model_include:
                fn += 1

        if model_include == gt_include:
            agree += 1

    total_cost = sum(d.get("est_cost_usd", 0.0) for d in decisions)
    mean_latency = sum(d.get("latency_s", 0.0) for d in decisions) / len(decisions)
    parse_fails = sum(1 for d in decisions if not d.get("parse_ok", True))

    cost_per_1k = (total_cost / len(decisions) * 1000) if decisions else 0.0

    return {
        "agreement":           round(agree / n, 4),
        "false_negative_rate": round(fn / gt_includes, 4) if gt_includes else 0.0,
        "n_matched":           n,
        "n_agree":             agree,
        "n_gt_include":        gt_includes,
        "n_fn":                fn,
        "mean_latency_s":      round(mean_latency, 3),
        "parse_fail_rate":     round(parse_fails / len(decisions), 4),
        "est_cost_usd":        round(total_cost, 4),
        "cost_per_1k_papers":  round(cost_per_1k, 4),
    }
