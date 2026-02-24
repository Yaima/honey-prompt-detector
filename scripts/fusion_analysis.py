#!/usr/bin/env python3
"""
Fusion Analysis: HIVE + Sentinel Ensemble
==========================================
Analyzes per-sample predictions from HIVE and Sentinel to find ensemble
strategies that outperform both individual systems.

Strategies tested:
  1. OR  (flag if either detects)       — maximizes recall
  2. AND (flag if both detect)          — maximizes precision
  3. Sentinel-as-Stage4 (replace GPT-4o-mini with Sentinel)
  4. Majority vote (HIVE local + Sentinel + HIVE-Stage4)
  5. Confidence-weighted (average confidences, sweep thresholds)

Requirements:
    pip install pandas scikit-learn numpy

Usage:
    python scripts/fusion_analysis.py

Expects these files in results/:
    - full_pipeline_per_sample.csv  (HIVE)
    - sentinel_per_sample.csv       (Sentinel)
    - baseline_protectai_per_sample.csv (ProtectAI, optional)
"""

import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_hive_results():
    """Load HIVE full pipeline per-sample results."""
    path = RESULTS_DIR / "full_pipeline_per_sample.csv"
    df = pd.read_csv(path)
    print(f"HIVE: {len(df)} samples loaded")
    print(f"  Columns: {list(df.columns)}")
    return df


def load_sentinel_results():
    """Load Sentinel per-sample results."""
    path = RESULTS_DIR / "sentinel_per_sample.csv"
    df = pd.read_csv(path)
    print(f"Sentinel: {len(df)} samples loaded")
    return df


def load_protectai_results():
    """Load ProtectAI per-sample results (optional)."""
    path = RESULTS_DIR / "baseline_protectai_per_sample.csv"
    if not path.exists():
        print("ProtectAI results not found, skipping")
        return None
    df = pd.read_csv(path)
    print(f"ProtectAI: {len(df)} samples loaded")
    return df


def compute_metrics(labels, preds, confs=None):
    """Compute standard metrics."""
    m = {
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
    }
    if confs is not None and len(np.unique(labels)) > 1:
        m["auc_roc"] = round(roc_auc_score(labels, confs), 4)
    cm = confusion_matrix(labels, preds)
    m["tp"] = int(cm[1][1])
    m["tn"] = int(cm[0][0])
    m["fp"] = int(cm[0][1])
    m["fn"] = int(cm[1][0])
    m["fpr"] = round(m["fp"] / (m["fp"] + m["tn"]), 4) if (m["fp"] + m["tn"]) > 0 else 0
    return m


def print_metrics(name, m):
    print(f"\n  {name}:")
    print(f"    Precision: {m['precision']}")
    print(f"    Recall:    {m['recall']}")
    print(f"    F1:        {m['f1']}")
    print(f"    Accuracy:  {m['accuracy']}")
    if "auc_roc" in m:
        print(f"    AUC-ROC:   {m['auc_roc']}")
    print(f"    FPR:       {m['fpr']}")
    print(f"    TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}")


def main():
    print("=" * 60)
    print("FUSION ANALYSIS: HIVE + Sentinel Ensemble")
    print("=" * 60)

    # Load data
    hive_df = load_hive_results()
    sentinel_df = load_sentinel_results()
    protectai_df = load_protectai_results()

    # Align by index
    n = min(len(hive_df), len(sentinel_df))
    print(f"\nAligning {n} samples...")

    # Extract arrays
    labels = hive_df["label"].values[:n].astype(int)

    # HIVE predictions
    hive_full_pred = hive_df["full_pred"].values[:n].astype(int)
    hive_full_conf = hive_df["full_conf"].values[:n].astype(float)

    # HIVE local-only (stages 1-3)
    hive_local_pred = hive_df["stage1_pred"].values[:n].astype(int)
    hive_local_conf = hive_df["stage1_conf"].values[:n].astype(float)

    # HIVE Stage 4 alone
    hive_s4_pred = hive_df["stage4_pred"].values[:n].astype(int)
    hive_s4_conf = hive_df["stage4_conf"].values[:n].astype(float)

    # Sentinel predictions
    sentinel_pred = sentinel_df["sentinel_pred"].values[:n].astype(int)
    sentinel_conf = sentinel_df["sentinel_confidence"].values[:n].astype(float)

    # ProtectAI (if available)
    if protectai_df is not None:
        protectai_pred = protectai_df["model_pred"].values[:n].astype(int)
        protectai_conf = protectai_df["model_confidence"].values[:n].astype(float)

    print(f"Labels: {labels.sum()} positive, {(1-labels).sum()} negative")

    # ---------- Individual baselines ----------
    print("\n" + "=" * 60)
    print("INDIVIDUAL SYSTEMS")
    print("=" * 60)

    baselines = {
        "HIVE Full Pipeline (1+2+3+4)": (hive_full_pred, hive_full_conf),
        "HIVE Local Only (1+2+3)": (hive_local_pred, hive_local_conf),
        "HIVE Stage 4 Alone": (hive_s4_pred, hive_s4_conf),
        "Sentinel": (sentinel_pred, sentinel_conf),
    }
    if protectai_df is not None:
        baselines["ProtectAI"] = (protectai_pred, protectai_conf)

    baseline_metrics = {}
    for name, (pred, conf) in baselines.items():
        m = compute_metrics(labels, pred, conf)
        baseline_metrics[name] = m
        print_metrics(name, m)

    # ---------- Disagreement analysis ----------
    print("\n" + "=" * 60)
    print("DISAGREEMENT ANALYSIS (HIVE Full vs Sentinel)")
    print("=" * 60)

    both_correct = ((hive_full_pred == labels) & (sentinel_pred == labels)).sum()
    hive_only_correct = ((hive_full_pred == labels) & (sentinel_pred != labels)).sum()
    sentinel_only_correct = ((hive_full_pred != labels) & (sentinel_pred == labels)).sum()
    both_wrong = ((hive_full_pred != labels) & (sentinel_pred != labels)).sum()

    print(f"  Both correct:         {both_correct} ({both_correct/n*100:.1f}%)")
    print(f"  HIVE only correct:    {hive_only_correct} ({hive_only_correct/n*100:.1f}%)")
    print(f"  Sentinel only correct:{sentinel_only_correct} ({sentinel_only_correct/n*100:.1f}%)")
    print(f"  Both wrong:           {both_wrong} ({both_wrong/n*100:.1f}%)")

    # Break down by class
    for cls, cls_name in [(1, "Malicious"), (0, "Benign")]:
        mask = labels == cls
        nc = mask.sum()
        h_right = ((hive_full_pred[mask] == labels[mask])).sum()
        s_right = ((sentinel_pred[mask] == labels[mask])).sum()
        both_r = ((hive_full_pred[mask] == labels[mask]) & (sentinel_pred[mask] == labels[mask])).sum()
        either_r = ((hive_full_pred[mask] == labels[mask]) | (sentinel_pred[mask] == labels[mask])).sum()
        print(f"\n  {cls_name} ({nc} samples):")
        print(f"    HIVE correct:     {h_right} ({h_right/nc*100:.1f}%)")
        print(f"    Sentinel correct: {s_right} ({s_right/nc*100:.1f}%)")
        print(f"    Both correct:     {both_r} ({both_r/nc*100:.1f}%)")
        print(f"    Either correct:   {either_r} ({either_r/nc*100:.1f}%)")

    # ---------- Fusion strategies ----------
    print("\n" + "=" * 60)
    print("FUSION STRATEGIES")
    print("=" * 60)

    strategies = {}

    # Strategy 1: OR (flag if either HIVE-full OR Sentinel detects)
    or_pred = ((hive_full_pred == 1) | (sentinel_pred == 1)).astype(int)
    or_conf = np.maximum(hive_full_conf, sentinel_conf)
    strategies["1. OR (HIVE-full | Sentinel)"] = compute_metrics(labels, or_pred, or_conf)

    # Strategy 2: AND (flag if both detect)
    and_pred = ((hive_full_pred == 1) & (sentinel_pred == 1)).astype(int)
    and_conf = np.minimum(hive_full_conf, sentinel_conf)
    strategies["2. AND (HIVE-full & Sentinel)"] = compute_metrics(labels, and_pred, and_conf)

    # Strategy 3: Sentinel-as-Stage4 (HIVE local stages 1-3 OR Sentinel)
    # This replaces GPT-4o-mini with Sentinel
    s_as_s4_pred = ((hive_local_pred == 1) | (sentinel_pred == 1)).astype(int)
    s_as_s4_conf = np.maximum(hive_local_conf, sentinel_conf)
    strategies["3. HIVE-local | Sentinel (replace Stage4)"] = compute_metrics(labels, s_as_s4_pred, s_as_s4_conf)

    # Strategy 4: HIVE-local AND Sentinel (high-precision combo)
    local_and_s = ((hive_local_pred == 1) & (sentinel_pred == 1)).astype(int)
    strategies["4. HIVE-local & Sentinel (high-precision)"] = compute_metrics(labels, local_and_s)

    # Strategy 5: Majority vote (HIVE-local, Sentinel, HIVE-Stage4)
    votes = hive_local_pred + sentinel_pred + hive_s4_pred
    maj_pred = (votes >= 2).astype(int)
    avg_conf = (hive_local_conf + sentinel_conf + hive_s4_conf) / 3
    strategies["5. Majority vote (local + Sentinel + S4)"] = compute_metrics(labels, maj_pred, avg_conf)

    # Strategy 6: Sentinel primary + HIVE-local boost
    # Use Sentinel as primary, but also flag anything HIVE local catches
    s_plus_local = ((sentinel_pred == 1) | (hive_local_pred == 1)).astype(int)
    strategies["6. Sentinel | HIVE-local"] = compute_metrics(labels, s_plus_local)

    # Strategy 7: Confidence-weighted average, sweep thresholds
    avg_conf_hs = (hive_full_conf + sentinel_conf) / 2.0
    best_thresh_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.30, 0.70, 0.01):
        t_pred = (avg_conf_hs >= t).astype(int)
        t_f1 = f1_score(labels, t_pred, zero_division=0)
        if t_f1 > best_thresh_f1:
            best_thresh_f1 = t_f1
            best_thresh = t
    opt_pred = (avg_conf_hs >= best_thresh).astype(int)
    strategies[f"7. Avg confidence, τ={best_thresh:.2f}"] = compute_metrics(labels, opt_pred, avg_conf_hs)

    # Strategy 8: Weighted confidence (Sentinel 0.6, HIVE 0.4)
    weighted_conf = 0.6 * sentinel_conf + 0.4 * hive_full_conf
    best_wt_f1 = 0
    best_wt = 0.5
    for t in np.arange(0.30, 0.70, 0.01):
        t_pred = (weighted_conf >= t).astype(int)
        t_f1 = f1_score(labels, t_pred, zero_division=0)
        if t_f1 > best_wt_f1:
            best_wt_f1 = t_f1
            best_wt = t
    wt_pred = (weighted_conf >= best_wt).astype(int)
    strategies[f"8. Weighted (0.6*Sentinel + 0.4*HIVE), τ={best_wt:.2f}"] = compute_metrics(labels, wt_pred, weighted_conf)

    if protectai_df is not None:
        # Strategy 9: Triple ensemble (Sentinel + ProtectAI + HIVE-full majority)
        triple_votes = sentinel_pred + protectai_pred + hive_full_pred
        triple_pred = (triple_votes >= 2).astype(int)
        triple_conf = (sentinel_conf + protectai_conf + hive_full_conf) / 3
        strategies["9. Triple majority (Sentinel+ProtectAI+HIVE)"] = compute_metrics(labels, triple_pred, triple_conf)

    for name, m in strategies.items():
        print_metrics(name, m)

    # ---------- Summary table ----------
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<52} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6}")
    print("-" * 76)

    # Add baselines
    for name, m in baseline_metrics.items():
        print(f"{name:<52} {m['precision']:>6} {m['recall']:>6} {m['f1']:>6} {m['fpr']:>6}")
    print("-" * 76)
    for name, m in strategies.items():
        marker = " ***" if m["f1"] > max(baseline_metrics["Sentinel"]["f1"], baseline_metrics["HIVE Full Pipeline (1+2+3+4)"]["f1"]) else ""
        print(f"{name:<52} {m['precision']:>6} {m['recall']:>6} {m['f1']:>6} {m['fpr']:>6}{marker}")

    print("\n*** = outperforms both Sentinel and HIVE individually")

    # ---------- Save results ----------
    output = {
        "baselines": baseline_metrics,
        "strategies": strategies,
        "disagreement": {
            "both_correct": int(both_correct),
            "hive_only_correct": int(hive_only_correct),
            "sentinel_only_correct": int(sentinel_only_correct),
            "both_wrong": int(both_wrong),
        }
    }
    out_path = RESULTS_DIR / "fusion_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
