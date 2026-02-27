#!/usr/bin/env python3
"""
Compute gated aggregation metrics at c_local ≥ 0.95 and c_local = 1.00 with Sentinel as Stage 4,
using EXISTING cached per-sample results.

Metrics computed:
  - Precision, Recall, F1 (with 95% bootstrap CIs, 500 iterations)
  - Override count (samples with local override active)
  - Override precision (fraction of overrides that are TP)
  - FP count
  - Suppression count and suppression precision
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

RESULTS_DIR = Path(__file__).parent.parent / "results"
BOOTSTRAP_ITERATIONS = 500
BOOTSTRAP_CI = 0.95


def compute_metrics_with_ci(labels, preds):
    """Compute precision, recall, F1 with 95% bootstrap confidence intervals."""
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    f = f1_score(labels, preds, zero_division=0)
    
    # Bootstrap for CI
    n = len(labels)
    p_boots = []
    r_boots = []
    f_boots = []
    
    np.random.seed(42)
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = np.random.choice(n, size=n, replace=True)
        labels_boot = labels[indices]
        preds_boot = preds[indices]
        
        p_boot = precision_score(labels_boot, preds_boot, zero_division=0)
        r_boot = recall_score(labels_boot, preds_boot, zero_division=0)
        f_boot = f1_score(labels_boot, preds_boot, zero_division=0)
        
        p_boots.append(p_boot)
        r_boots.append(r_boot)
        f_boots.append(f_boot)
    
    alpha = (1 - BOOTSTRAP_CI) / 2
    p_ci = [np.percentile(p_boots, alpha * 100), np.percentile(p_boots, (1 - alpha) * 100)]
    r_ci = [np.percentile(r_boots, alpha * 100), np.percentile(r_boots, (1 - alpha) * 100)]
    f_ci = [np.percentile(f_boots, alpha * 100), np.percentile(f_boots, (1 - alpha) * 100)]
    
    return {
        "precision": float(p),
        "precision_ci": [float(p_ci[0]), float(p_ci[1])],
        "recall": float(r),
        "recall_ci": [float(r_ci[0]), float(r_ci[1])],
        "f1": float(f),
        "f1_ci": [float(f_ci[0]), float(f_ci[1])]
    }


def compute_gated_aggregation(labels, local_pred, local_conf, stage4_pred, gate_threshold):
    """
    Apply gated aggregation logic:
    - If stage4_says_attack: final = attack
    - Elif local_says_attack AND local_conf >= gate_threshold: final = attack (override)
    - Else: final = stage4_decision
    
    Returns (predictions, override_count, override_precision, fp_count, suppression_count, suppression_precision)
    """
    n = len(labels)
    gated_pred = np.zeros(n, dtype=int)
    
    override_indices = []
    suppression_indices = []
    fp_indices = []
    
    for i in range(n):
        local_flag = local_pred[i] == 1
        stage4_flag = stage4_pred[i] == 1
        
        if stage4_flag:
            # Stage 4 says attack -> flag it
            gated_pred[i] = 1
        elif local_flag:
            # Local says attack, Stage 4 disagrees
            if local_conf[i] >= gate_threshold:
                # High local confidence -> override Stage 4
                gated_pred[i] = 1
                override_indices.append(i)
            else:
                # Low local confidence -> suppress (follow Stage 4)
                gated_pred[i] = 0
                suppression_indices.append(i)
        else:
            # Both benign
            gated_pred[i] = 0
    
    # Compute override metrics
    override_count = len(override_indices)
    override_tp = sum(labels[i] == 1 for i in override_indices)
    override_precision = override_tp / override_count if override_count > 0 else 0
    
    # Compute FP count: predicted 1 but label is 0
    cm = confusion_matrix(labels, gated_pred)
    fp_count = cm[0, 1]  # False positives
    
    # Compute suppression metrics
    suppression_count = len(suppression_indices)
    suppression_tp = sum(labels[i] == 1 for i in suppression_indices)
    suppression_precision = suppression_tp / suppression_count if suppression_count > 0 else 0
    
    return {
        "predictions": gated_pred,
        "override_count": override_count,
        "override_precision": float(override_precision),
        "fp_count": int(fp_count),
        "suppression_count": suppression_count,
        "suppression_precision": float(suppression_precision)
    }


def main():
    print("=" * 70)
    print("GATED AGGREGATION EXPERIMENT")
    print("=" * 70)
    
    # Load data
    full_df = pd.read_csv(RESULTS_DIR / "full_pipeline_per_sample.csv")
    sentinel_df = pd.read_csv(RESULTS_DIR / "sentinel_per_sample.csv")
    
    n = len(full_df)
    print(f"\nLoaded {n} samples from full_pipeline_per_sample.csv")
    print(f"Loaded {len(sentinel_df)} samples from sentinel_per_sample.csv")
    
    # Extract relevant columns
    labels = full_df["label"].values.astype(int)
    local_pred = full_df["full_pred"].values.astype(int)
    local_conf = full_df["full_conf"].values.astype(float)
    stage4_pred = sentinel_df["sentinel_pred"].values.astype(int)
    
    # Verify alignment
    if len(labels) != len(stage4_pred):
        raise ValueError(f"Misaligned data: {len(labels)} vs {len(stage4_pred)}")
    
    print(f"Data verified: {n} samples aligned")
    print(f"\nLabel distribution: {np.sum(labels)} attacks, {n - np.sum(labels)} benign")
    print(f"Local (Stages 1-3) - Attacks predicted: {np.sum(local_pred)}")
    print(f"Sentinel (Stage 4) - Attacks predicted: {np.sum(stage4_pred)}")
    
    # Compute Sentinel standalone metrics (reference)
    print("\n" + "=" * 70)
    print("SENTINEL STANDALONE (Reference)")
    print("=" * 70)
    sentinel_metrics = compute_metrics_with_ci(labels, stage4_pred)
    print(f"Precision: {sentinel_metrics['precision']:.4f} [{sentinel_metrics['precision_ci'][0]:.4f}, {sentinel_metrics['precision_ci'][1]:.4f}]")
    print(f"Recall:    {sentinel_metrics['recall']:.4f} [{sentinel_metrics['recall_ci'][0]:.4f}, {sentinel_metrics['recall_ci'][1]:.4f}]")
    print(f"F1:        {sentinel_metrics['f1']:.4f} [{sentinel_metrics['f1_ci'][0]:.4f}, {sentinel_metrics['f1_ci'][1]:.4f}]")
    
    # Compute gated aggregation for each threshold
    results = {
        "sentinel_standalone": sentinel_metrics,
        "thresholds": {}
    }
    
    gate_thresholds = [0.90, 0.95, 1.00]
    
    for gate_thresh in gate_thresholds:
        print("\n" + "=" * 70)
        print(f"GATED AGGREGATION (c_local >= {gate_thresh})")
        print("=" * 70)
        
        gated_data = compute_gated_aggregation(
            labels, local_pred, local_conf, stage4_pred, gate_thresh
        )
        
        gated_pred = gated_data["predictions"]
        metrics = compute_metrics_with_ci(labels, gated_pred)
        
        print(f"Precision: {metrics['precision']:.4f} [{metrics['precision_ci'][0]:.4f}, {metrics['precision_ci'][1]:.4f}]")
        print(f"Recall:    {metrics['recall']:.4f} [{metrics['recall_ci'][0]:.4f}, {metrics['recall_ci'][1]:.4f}]")
        print(f"F1:        {metrics['f1']:.4f} [{metrics['f1_ci'][0]:.4f}, {metrics['f1_ci'][1]:.4f}]")
        
        print(f"\nOverride count (local override active): {gated_data['override_count']}")
        print(f"Override precision (TP / overrides): {gated_data['override_precision']:.4f}")
        print(f"FP count: {gated_data['fp_count']}")
        print(f"Suppression count (Stage 4 suppressed): {gated_data['suppression_count']}")
        print(f"Suppression precision (TP / suppressions): {gated_data['suppression_precision']:.4f}")
        
        threshold_key = f"{gate_thresh:.2f}"
        results["thresholds"][threshold_key] = {
            "precision": metrics["precision"],
            "precision_ci": metrics["precision_ci"],
            "recall": metrics["recall"],
            "recall_ci": metrics["recall_ci"],
            "f1": metrics["f1"],
            "f1_ci": metrics["f1_ci"],
            "overrides": gated_data["override_count"],
            "override_precision": gated_data["override_precision"],
            "fp_count": gated_data["fp_count"],
            "suppressions": gated_data["suppression_count"],
            "suppression_precision": gated_data["suppression_precision"]
        }
    
    # Save results
    output_file = RESULTS_DIR / "experiment_gated_thresholds.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Results saved to {output_file}")
    print("=" * 70)
    
    # Print summary
    print("\nSUMMARY TABLE:")
    print("-" * 100)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Overrides':<12} {'Suppressions':<12}")
    print("-" * 100)
    print(f"{'Sentinel':<12} {sentinel_metrics['precision']:.4f}{'':>6} {sentinel_metrics['recall']:.4f}{'':>6} {sentinel_metrics['f1']:.4f}{'':>6} {'N/A':<12} {'N/A':<12}")
    for gate_thresh in gate_thresholds:
        threshold_key = f"{gate_thresh:.2f}"
        res = results["thresholds"][threshold_key]
        print(f"{gate_thresh:<12.2f} {res['precision']:.4f}{'':>6} {res['recall']:.4f}{'':>6} {res['f1']:.4f}{'':>6} {res['overrides']:<12} {res['suppressions']:<12}")
    print("-" * 100)


if __name__ == "__main__":
    main()
