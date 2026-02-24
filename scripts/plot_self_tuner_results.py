#!/usr/bin/env python3
"""
Generate plots from self-tuner analysis results.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    """Generate and save plots."""
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results"

    # Load history
    with open(results_dir / "tuner_history.json") as f:
        history = json.load(f)

    # Load analysis results
    with open(results_dir / "self_tuner_analysis.json") as f:
        analysis = json.load(f)

    threshold_history = history["threshold_history"]
    confidence_history = history["confidence_history"]
    shift_round = history["shift_round"]

    # Figure 1: Threshold Convergence
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Self-Tuner Convergence Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Threshold over all rounds
    ax = axes[0, 0]
    ax.plot(threshold_history, linewidth=1.5, label="Threshold", color="steelblue")
    ax.axhline(y=0.72, color="red", linestyle="--", label="Optimal (0.72)", alpha=0.7)
    ax.fill_between(range(len(threshold_history)), 0.67, 0.77, alpha=0.1, color="green")
    ax.set_xlabel("Round")
    ax.set_ylabel("Threshold Value")
    ax.set_title("Threshold Evolution (Full 1500 Rounds)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Early convergence phase
    ax = axes[0, 1]
    early_rounds = min(400, len(threshold_history))
    ax.plot(range(early_rounds), threshold_history[:early_rounds], linewidth=2, label="Threshold", color="steelblue")
    ax.axhline(y=0.72, color="red", linestyle="--", label="Optimal (0.72)", alpha=0.7)
    ax.fill_between(range(early_rounds), 0.67, 0.77, alpha=0.1, color="green")
    convergence_round = analysis["convergence_metrics"]["rounds_to_convergence"]
    if convergence_round > 0:
        ax.axvline(x=convergence_round, color="orange", linestyle=":", label=f"Convergence @ {convergence_round}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Threshold Value")
    ax.set_title("Early Convergence Phase (First 400 Rounds)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Distribution shift response
    ax = axes[1, 0]
    start = shift_round - 100
    end = min(shift_round + 300, len(threshold_history))
    rounds_around_shift = range(start, end)
    ax.plot(rounds_around_shift, threshold_history[start:end], linewidth=2, label="Threshold", color="steelblue")
    ax.axvline(x=shift_round, color="red", linestyle="--", linewidth=2, label="Distribution Shift")
    ax.axhspan(shift_round, shift_round + 50, alpha=0.1, color="orange", label="Recovery Window")
    ax.set_xlabel("Round")
    ax.set_ylabel("Threshold Value")
    ax.set_title(f"Response to Distribution Shift (Round {shift_round})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Stability metrics
    ax = axes[1, 1]
    window_size = 100
    rolling_std = [
        np.std(threshold_history[i : i + window_size])
        for i in range(0, len(threshold_history) - window_size, window_size // 2)
    ]
    window_centers = [i + window_size // 2 for i in range(0, len(threshold_history) - window_size, window_size // 2)]

    ax.bar(window_centers, rolling_std, width=50, color="teal", alpha=0.7, label="Threshold Std Dev (100-round windows)")
    ax.axhline(y=np.mean(rolling_std), color="red", linestyle="--", label=f"Mean Std Dev ({np.mean(rolling_std):.4f})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Threshold Stability Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = results_dir / "self_tuner_convergence.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved convergence plot to {plot_path}")
    plt.close()

    # Figure 2: Detailed shift analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribution Shift Stability Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Threshold behavior around shift
    ax = axes[0, 0]
    start = shift_round - 200
    end = min(shift_round + 400, len(threshold_history))
    rounds = range(start, end)
    ax.plot(rounds, threshold_history[start:end], linewidth=2, color="steelblue", label="Threshold")
    ax.axvline(x=shift_round, color="red", linestyle="--", linewidth=2, label="Distribution Shift")
    ax.fill_between(rounds, min(threshold_history[start:end]) - 0.05, max(threshold_history[start:end]) + 0.05,
                     alpha=0.1, color="blue")
    ax.set_xlabel("Round")
    ax.set_ylabel("Threshold")
    ax.set_title("Threshold Behavior During Shift (±200/+400 rounds)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Recovery rate
    ax = axes[0, 1]
    pre_shift_threshold = analysis["distribution_shift_metrics"]["pre_shift_threshold"]
    post_shift_mean = np.mean(threshold_history[shift_round + 200 : shift_round + 350])
    recovery_time = analysis["distribution_shift_metrics"]["recovery_time"]

    recovery_window = range(shift_round, min(shift_round + 300, len(threshold_history)))
    recovery_thresholds = threshold_history[shift_round : shift_round + 300]

    ax.plot(recovery_window, recovery_thresholds, linewidth=2, color="purple", marker="o", markersize=3, label="Threshold")
    ax.axhline(y=pre_shift_threshold, color="green", linestyle="--", label=f"Pre-shift baseline ({pre_shift_threshold:.4f})")
    if recovery_time > 0:
        ax.axvline(x=shift_round + recovery_time, color="orange", linestyle=":", linewidth=2,
                   label=f"Recovered @ +{recovery_time}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Threshold")
    ax.set_title(f"Recovery Trajectory (Recovery time: {recovery_time} rounds)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Confidence distribution before/after shift
    ax = axes[1, 0]
    pre_shift_conf = confidence_history[: shift_round - 100]
    post_shift_conf = confidence_history[shift_round + 100 : shift_round + 300]

    bins = np.linspace(0, 1, 20)
    ax.hist(pre_shift_conf, bins=bins, alpha=0.6, label="Pre-shift confidences", color="blue", edgecolor="black")
    ax.hist(post_shift_conf, bins=bins, alpha=0.6, label="Post-shift confidences", color="red", edgecolor="black")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Confidence Distribution Before vs After Shift")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Volatility analysis
    ax = axes[1, 1]
    volatility_window = 50
    volatilities = [
        np.std(threshold_history[i : i + volatility_window])
        for i in range(0, len(threshold_history) - volatility_window)
    ]

    ax.plot(volatilities, linewidth=1.5, color="darkred", label="Threshold volatility (50-round window)")
    ax.axvline(x=shift_round, color="red", linestyle="--", linewidth=2, label="Distribution Shift")
    ax.axhspan(shift_round - 100, shift_round + 200, alpha=0.1, color="orange", label="Impact zone")
    ax.set_xlabel("Round")
    ax.set_ylabel("Volatility (Std Dev)")
    ax.set_title("Threshold Volatility Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = results_dir / "self_tuner_shift.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved shift analysis plot to {plot_path}")
    plt.close()

    print("\nPlots generated successfully!")


if __name__ == "__main__":
    plot_results()
