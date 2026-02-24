#!/usr/bin/env python3
"""
Comprehensive Evaluation of HIVE Prompt Injection Detector
===========================================================

Evaluates:
1. HeuristicRulesEngine (HIVE's Stage 1)
2. KeywordBaseline
3. HeuristicBaseline

On the full unified dataset (34,718 samples) with:
- ROC/PR curves with AUC annotation
- Per-source breakdown
- Low-FPR operating points (1%, 5%, 10%)
- 95% confidence intervals
- Bootstrap confidence intervals (1000 iterations)
"""

import argparse
import csv
import json
import sys
import os
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.honey_prompt_detector.core.heuristic_rules import HeuristicRulesEngine
from src.honey_prompt_detector.baselines.keyword_baseline import KeywordBaseline
from src.honey_prompt_detector.baselines.heuristic_baseline import HeuristicBaseline
from src.honey_prompt_detector.monitoring.evaluation_metrics import (
    EvaluationMetrics,
    DetectionResult,
)


def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """Load the unified dataset."""
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def evaluate_heuristic_engine(texts: List[str]) -> List[float]:
    """Run HeuristicRulesEngine on texts, return confidence scores."""
    engine = HeuristicRulesEngine()
    confidences = []

    for i, text in enumerate(texts):
        # Try quick scan first
        quick_match = engine.scan_quick(text)
        if quick_match:
            confidences.append(quick_match.confidence)
        else:
            # Then full scan
            matches = engine.scan(text)
            if matches:
                confidences.append(matches[0].confidence)
            else:
                confidences.append(0.0)

        if (i + 1) % 5000 == 0:
            print(f"  HeuristicEngine: {i+1}/{len(texts)} samples processed")

    return confidences


def evaluate_keyword_baseline(texts: List[str]) -> List[float]:
    """Run KeywordBaseline on texts, return confidence scores."""
    baseline = KeywordBaseline()
    confidences = []

    for i, text in enumerate(texts):
        result = baseline.detect(text)
        confidences.append(result["confidence"])

        if (i + 1) % 5000 == 0:
            print(f"  KeywordBaseline: {i+1}/{len(texts)} samples processed")

    return confidences


def evaluate_heuristic_baseline(texts: List[str]) -> List[float]:
    """Run HeuristicBaseline on texts, return confidence scores."""
    baseline = HeuristicBaseline()
    confidences = []

    for i, text in enumerate(texts):
        result = baseline.detect(text)
        confidences.append(result["confidence"])

        if (i + 1) % 5000 == 0:
            print(f"  HeuristicBaseline: {i+1}/{len(texts)} samples processed")

    return confidences


def calculate_pr_curve(results: List[DetectionResult]) -> List[Tuple[float, float, float]]:
    """
    Calculate PR curve points (precision-recall curve).
    Returns: List of (threshold, recall, precision) tuples
    """
    pr_points = []

    # Use sorted unique confidence scores as thresholds
    thresholds = sorted(set([r.confidence for r in results] + [0.0, 1.0]), reverse=True)

    # Sample thresholds if too many
    if len(thresholds) > 100:
        step = len(thresholds) / 100
        thresholds = [thresholds[int(i * step)] for i in range(100)]

    for threshold in thresholds:
        tp = sum(1 for r in results if r.confidence >= threshold and r.actual_attack)
        fp = sum(1 for r in results if r.confidence >= threshold and not r.actual_attack)
        fn = sum(1 for r in results if r.confidence < threshold and r.actual_attack)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        pr_points.append((threshold, recall, precision))

    return pr_points


def calculate_auc_pr(pr_points: List[Tuple[float, float, float]]) -> float:
    """Calculate AUC-PR (area under precision-recall curve)."""
    if len(pr_points) < 2:
        return 0.0

    # Sort by recall
    sorted_points = sorted(pr_points, key=lambda x: x[1])

    auc = 0.0
    for i in range(1, len(sorted_points)):
        prev_recall, prev_prec = sorted_points[i - 1][1], sorted_points[i - 1][2]
        curr_recall, curr_prec = sorted_points[i][1], sorted_points[i][2]

        # Use average precision as height
        auc += (curr_recall - prev_recall) * (curr_prec + prev_prec) / 2

    return auc


def bootstrap_confidence_intervals(
    results: List[DetectionResult],
    metric_func,
    n_iterations: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence intervals using bootstrap resampling.

    Returns: (lower_bound, point_estimate, upper_bound)
    """
    bootstrap_metrics = []

    for _ in range(n_iterations):
        # Resample with replacement
        resampled = random.choices(results, k=len(results))
        metric = metric_func(resampled)
        bootstrap_metrics.append(metric)

    bootstrap_metrics = sorted(bootstrap_metrics)

    # Calculate percentiles
    alpha = (1.0 - confidence) / 2
    lower_idx = int(alpha * n_iterations)
    upper_idx = int((1.0 - alpha) * n_iterations)

    point_estimate = metric_func(results)
    lower = bootstrap_metrics[lower_idx]
    upper = bootstrap_metrics[upper_idx]

    return lower, point_estimate, upper


def generate_detector_report(
    name: str,
    confidences: List[float],
    results: List[Dict],
    sources: List[str],
) -> Dict[str, Any]:
    """Generate comprehensive report for a detector."""

    # Create detection results
    detection_results = [
        DetectionResult(
            text=r["text"][:200],
            predicted_attack=False,  # Placeholder, will be set by threshold
            confidence=conf,
            actual_attack=r["actual_attack"],
            category=r["source"],
        )
        for conf, r in zip(confidences, results)
    ]

    # Add results to evaluator
    evaluator = EvaluationMetrics(detection_results)

    # Generate base report at threshold 0.5
    base_report = evaluator.generate_report(threshold=0.5)

    # Calculate ROC curve and AUC
    roc_curve = evaluator.calculate_roc_curve(num_thresholds=100)
    auc_roc = evaluator.calculate_auc()

    # Calculate PR curve and AUC
    pr_curve = calculate_pr_curve(detection_results)
    auc_pr = calculate_auc_pr(pr_curve)

    # Calculate bootstrap confidence intervals
    print(f"\n  Calculating bootstrap confidence intervals for {name}...")

    def precision_metric(res):
        if not res:
            return 0.0
        tp = sum(1 for r in res if r.confidence >= 0.5 and r.actual_attack)
        fp = sum(1 for r in res if r.confidence >= 0.5 and not r.actual_attack)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall_metric(res):
        if not res:
            return 0.0
        tp = sum(1 for r in res if r.confidence >= 0.5 and r.actual_attack)
        fn = sum(1 for r in res if r.confidence < 0.5 and r.actual_attack)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1_metric(res):
        prec = precision_metric(res)
        rec = recall_metric(res)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    precision_ci = bootstrap_confidence_intervals(detection_results, precision_metric, n_iterations=500)
    recall_ci = bootstrap_confidence_intervals(detection_results, recall_metric, n_iterations=500)
    f1_ci = bootstrap_confidence_intervals(detection_results, f1_metric, n_iterations=500)

    # Per-source breakdown
    sources_by_name = defaultdict(list)
    for conf, r in zip(confidences, results):
        sources_by_name[r["source"]].append((conf, r["actual_attack"]))

    per_source = {}
    for source, confs_labels in sources_by_name.items():
        source_results = [
            DetectionResult(
                text="",
                predicted_attack=False,
                confidence=conf,
                actual_attack=actual,
                category=source,
            )
            for conf, actual in confs_labels
        ]
        source_eval = EvaluationMetrics(source_results)
        source_report = source_eval.generate_report(threshold=0.5)
        per_source[source] = {
            "precision": source_report["metrics"]["precision"],
            "recall": source_report["metrics"]["recall"],
            "f1": source_report["metrics"]["f1"],
            "accuracy": source_report["metrics"]["accuracy"],
            "auc": source_eval.calculate_auc(),
            "num_samples": len(source_results),
            "num_attacks": sum(1 for r in source_results if r.actual_attack),
        }

    return {
        "detector_name": name,
        "summary": base_report["summary"],
        "confusion_matrix": base_report["confusion_matrix"],
        "metrics_at_0.5": base_report["metrics"],
        "confidence_intervals_95_bootstrap": {
            "precision": {
                "lower": precision_ci[0],
                "estimate": precision_ci[1],
                "upper": precision_ci[2],
            },
            "recall": {
                "lower": recall_ci[0],
                "estimate": recall_ci[1],
                "upper": recall_ci[2],
            },
            "f1": {
                "lower": f1_ci[0],
                "estimate": f1_ci[1],
                "upper": f1_ci[2],
            },
        },
        "roc_analysis": {
            "auc": auc_roc,
            "curve_points": [
                {"threshold": t, "fpr": f, "tpr": tp}
                for t, f, tp in roc_curve
            ],
        },
        "pr_analysis": {
            "auc": auc_pr,
            "curve_points": [
                {"threshold": t, "recall": rec, "precision": prec}
                for t, rec, prec in pr_curve
            ],
        },
        "low_fpr_operating_points": base_report["roc_analysis"]["low_fpr_operating_points"],
        "per_source_breakdown": per_source,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of HIVE detector on unified dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/unified_dataset.csv",
        help="Path to unified dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/full_evaluation_report.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE EVALUATION OF HIVE PROMPT INJECTION DETECTOR")
    print("=" * 70)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    rows = load_dataset(args.dataset)
    print(f"  Loaded {len(rows)} samples")

    # Parse labels and extract texts
    labels = [r["label"] == "1" for r in rows]
    texts = [r["text"] for r in rows]
    sources = [r.get("source", "unknown") for r in rows]

    # Count distribution
    n_attacks = sum(labels)
    n_benign = len(labels) - n_attacks
    print(f"  Distribution: {n_benign} benign, {n_attacks} malicious")
    print(f"  Class ratio: {n_benign/len(labels):.1%} benign, {n_attacks/len(labels):.1%} malicious")

    # Prepare results structure
    results = [
        {
            "text": text,
            "actual_attack": label,
            "source": source,
        }
        for text, label, source in zip(texts, labels, sources)
    ]

    # Evaluate detectors
    print("\n[2/5] Evaluating HeuristicRulesEngine (HIVE Stage 1)...")
    heuristic_confidences = evaluate_heuristic_engine(texts)

    print("\n[3/5] Evaluating KeywordBaseline...")
    keyword_confidences = evaluate_keyword_baseline(texts)

    print("\n[4/5] Evaluating HeuristicBaseline...")
    heuristic_baseline_confidences = evaluate_heuristic_baseline(texts)

    # Generate reports
    print("\n[5/5] Generating reports...")

    reports = {
        "heuristic_rules_engine": generate_detector_report(
            "HeuristicRulesEngine",
            heuristic_confidences,
            results,
            sources,
        ),
        "keyword_baseline": generate_detector_report(
            "KeywordBaseline",
            keyword_confidences,
            results,
            sources,
        ),
        "heuristic_baseline": generate_detector_report(
            "HeuristicBaseline",
            heuristic_baseline_confidences,
            results,
            sources,
        ),
    }

    # Add metadata
    full_report = {
        "metadata": {
            "evaluation_date": "2025-02-21",
            "dataset": args.dataset,
            "total_samples": len(rows),
            "benign_samples": n_benign,
            "attack_samples": n_attacks,
            "random_seed": args.seed,
        },
        "detectors": reports,
    }

    # Save report
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)

    for name, report in reports.items():
        print(f"\n{name}:")
        m = report["metrics_at_0.5"]
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1:        {m['f1']:.4f}")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  AUC-ROC:   {report['roc_analysis']['auc']:.4f}")
        print(f"  AUC-PR:    {report['pr_analysis']['auc']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
