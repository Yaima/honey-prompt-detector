#!/usr/bin/env python3
"""
Revision Experiment: c_min Sensitivity Sweep

Addresses reviewer item (iv): sensitivity analysis of early-exit threshold.

Uses cached per-sample CSVs (no API calls needed) to compute:
  - Early-exit % as a function of c_min
  - F1, Precision, Recall at each c_min
  - Number of Stage 4 calls saved

Usage:
  cd honey-prompt-detector
  python scripts/experiment_revision_cmin_sweep.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
DATA_DIR = REPO / "data"


def load_per_sample(filename):
    """Load per-sample CSV with all stage predictions."""
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_metrics(y_true, y_pred):
    """Compute precision, recall, F1."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def bootstrap_ci(y_true, y_pred, n_boot=500, seed=42):
    """95% bootstrap CI for F1."""
    rng = np.random.RandomState(seed)
    f1s = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        m = compute_metrics(yt, yp)
        f1s.append(m["f1"])
    return [round(np.percentile(f1s, 2.5), 4), round(np.percentile(f1s, 97.5), 4)]


def run_cmin_sweep():
    print("=" * 70)
    print("REVISION EXPERIMENT: c_min Sensitivity Sweep (GPT-4o-mini)")
    print("=" * 70)

    # Load GPT-4o-mini per-sample results
    rows = load_per_sample("full_pipeline_per_sample_gpt-4o-mini.csv")
    print(f"Loaded {len(rows)} samples")

    # Parse data
    y_true = []
    local_pred = []
    local_conf = []
    s4_pred = []
    s4_conf = []

    for r in rows:
        y_true.append(int(r["label"]))
        local_pred.append(int(r["full_pred"]))
        local_conf.append(float(r["full_conf"]))
        s4_pred.append(int(r.get("stage4_pred", 0)))
        s4_conf.append(float(r.get("stage4_conf", 0.0)))

    n = len(y_true)

    # Sweep c_min values
    c_min_values = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    results = []
    print(f"\n{'c_min':>6s}  {'Early%':>7s}  {'S4 Saved':>8s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}  {'F1 CI':>18s}  {'Overrides':>9s}  {'Ov.Prec':>8s}")
    print(f"{'-'*6}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*18}  {'-'*9}  {'-'*8}")

    for c_min in c_min_values:
        # Apply gated aggregation at this c_min
        final_pred = []
        early_exits = 0
        overrides = 0
        override_correct = 0

        for i in range(n):
            lp = local_pred[i]
            lc = local_conf[i]
            s4p = s4_pred[i]

            if lp == 1 and lc >= c_min:
                # High-confidence local → early exit (skip Stage 4)
                final_pred.append(1)
                early_exits += 1
                # This is an "override" if Stage 4 would have disagreed
                if s4p == 0:
                    overrides += 1
                    if y_true[i] == 1:
                        override_correct += 1
            elif s4p == 1:
                # Stage 4 detects
                final_pred.append(1)
            elif lp == 1 and s4p == 0:
                # Local flagged but below c_min AND Stage 4 says benign → suppress
                final_pred.append(0)
            else:
                # Both agree benign
                final_pred.append(s4p)

        m = compute_metrics(y_true, final_pred)
        ci = bootstrap_ci(y_true, final_pred)
        early_pct = round(100 * early_exits / n, 1)
        s4_saved = early_exits
        override_prec = round(override_correct / overrides, 3) if overrides > 0 else 0.0

        row = {
            "c_min": c_min,
            "early_exit_pct": early_pct,
            "stage4_calls_saved": s4_saved,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "f1_ci": ci,
            "overrides": overrides,
            "override_precision": override_prec,
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        }
        results.append(row)

        print(f"{c_min:6.2f}  {early_pct:6.1f}%  {s4_saved:8d}  {m['precision']:7.4f}  {m['recall']:7.4f}  {m['f1']:7.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]  {overrides:9d}  {override_prec:8.3f}")

    # Save
    output_path = RESULTS_DIR / "experiment_revision_cmin_sweep.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_cmin_sweep()
