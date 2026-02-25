#!/usr/bin/env python3
"""
Generate publication-quality ROC and PR curves from evaluation results.
"""

import json
import sys
from pathlib import Path

# Check if matplotlib is available, install if needed
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Installing matplotlib...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "matplotlib"])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np


def main():
    # Load evaluation report
    report_path = Path("results/full_evaluation_report.json")
    if not report_path.exists():
        print(f"Error: {report_path} not found. Run comprehensive_evaluation.py first.")
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    detectors = report["detectors"]

    # Extract colors and markers for each detector
    colors = {
        "heuristic_rules_engine": "#1f77b4",  # Blue
        "keyword_baseline": "#ff7f0e",  # Orange
        "heuristic_baseline": "#2ca02c",  # Green
    }

    names = {
        "heuristic_rules_engine": "HIVE (Heuristic Rules)",
        "keyword_baseline": "Keyword Baseline",
        "heuristic_baseline": "Heuristic Baseline",
    }

    # =========================================================================
    # ROC Curve
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    for detector_key, detector_data in detectors.items():
        roc_points = detector_data["roc_analysis"]["curve_points"]
        auc = detector_data["roc_analysis"]["auc"]

        fprs = [p["fpr"] for p in roc_points]
        tprs = [p["tpr"] for p in roc_points]

        ax.plot(
            fprs,
            tprs,
            label=f"{names[detector_key]} (AUC={auc:.3f})",
            linewidth=2.5,
            color=colors[detector_key],
        )

    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier (AUC=0.5)", alpha=0.7)

    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curves - HIVE Prompt Injection Detector Evaluation", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    roc_path = Path("results/roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {roc_path}")
    plt.close()

    # =========================================================================
    # PR Curve (Precision-Recall)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    for detector_key, detector_data in detectors.items():
        pr_points = detector_data["pr_analysis"]["curve_points"]
        auc_pr = detector_data["pr_analysis"]["auc"]

        recalls = [p["recall"] for p in pr_points]
        precisions = [p["precision"] for p in pr_points]

        ax.plot(
            recalls,
            precisions,
            label=f"{names[detector_key]} (AUC={auc_pr:.3f})",
            linewidth=2.5,
            color=colors[detector_key],
        )

    # Baseline line: precision = # attacks / total samples
    n_attacks = report["metadata"]["attack_samples"]
    n_total = report["metadata"]["total_samples"]
    baseline_precision = n_attacks / n_total
    ax.axhline(y=baseline_precision, color="k", linestyle="--", linewidth=1.5,
               label=f"Random Baseline (Precision={baseline_precision:.3f})", alpha=0.7)

    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title("PR Curves - HIVE Prompt Injection Detector Evaluation", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    pr_path = Path("results/pr_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {pr_path}")
    plt.close()

    # =========================================================================
    # Performance Comparison Bar Chart
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    detector_keys = list(detectors.keys())
    detector_names = [names[k] for k in detector_keys]

    metrics_to_plot = [
        ("metrics_at_0.5", "precision", "Precision at Threshold 0.5"),
        ("metrics_at_0.5", "recall", "Recall at Threshold 0.5"),
        ("metrics_at_0.5", "f1", "F1 Score at Threshold 0.5"),
        ("roc_analysis", "auc", "AUC-ROC"),
    ]

    for idx, (ax, (report_key, metric_key, title)) in enumerate(zip(axes.flat, metrics_to_plot)):
        values = []
        for det_key in detector_keys:
            if report_key == "metrics_at_0.5":
                val = detectors[det_key][report_key][metric_key]
            elif report_key == "roc_analysis":
                val = detectors[det_key][report_key][metric_key]
            else:
                val = detectors[det_key][report_key][metric_key]
            values.append(val)

        bars = ax.bar(detector_names, values, color=[colors[k] for k in detector_keys], alpha=0.8, edgecolor="black")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight="bold")

        ax.set_ylabel(metric_key.capitalize(), fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    comparison_path = Path("results/performance_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {comparison_path}")
    plt.close()

    print("\nAll curves generated successfully!")


if __name__ == "__main__":
    main()
