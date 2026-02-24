#!/usr/bin/env python3
"""
End-to-End Evaluation of HIVE Prompt Injection Detector
========================================================

Evaluates the FULL HIVE pipeline:
1. Stage 1: Heuristic rules (YARA-style pattern matching)
2. Stage 3: Honey-prompt token detection (exact + variations + obfuscation)

Compares:
- Stage 1 only (baseline heuristics)
- Full pipeline (Stage 1 + Stage 3 token matching)
- Existing baselines (KeywordBaseline, HeuristicBaseline)

On the unified dataset (34,718 samples) with:
- Per-detector evaluation with metrics
- ROC/PR curves with AUC annotation
- Per-source breakdown
- Timing analysis
- Confusion matrices
- 95% bootstrap confidence intervals

Key Features:
- Tests honey token matching WITHOUT requiring embeddings (Stage 2)
- Creates synthetic honey tokens to simulate real canary detection
- Tests canonicalization/obfuscation resistance
- No LLM API calls required (fully local evaluation)
"""

import argparse
import csv
import json
import sys
import os
import random
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.honey_prompt_detector.core.detector import Detector
from src.honey_prompt_detector.core.heuristic_rules import HeuristicRulesEngine
from src.honey_prompt_detector.core.honey_prompt import HoneyPrompt
from src.honey_prompt_detector.baselines.keyword_baseline import KeywordBaseline
from src.honey_prompt_detector.baselines.heuristic_baseline import HeuristicBaseline
from src.honey_prompt_detector.monitoring.evaluation_metrics import (
    EvaluationMetrics,
    DetectionResult,
)


# Simple context evaluator that doesn't require embeddings
class SimpleContextEvaluator:
    """Lightweight context evaluator for end-to-end testing."""

    def adjust_confidence(self, confidence: float, context: str, expected_context: str) -> float:
        """Adjust confidence based on context similarity (without embeddings)."""
        # Simple heuristic: check for keyword overlap
        if not context or not expected_context:
            return confidence

        context_words = set(context.lower().split())
        expected_words = set(expected_context.lower().split())

        if not expected_words:
            return confidence

        # Jaccard similarity
        intersection = context_words & expected_words
        union = context_words | expected_words
        similarity = len(intersection) / len(union) if union else 0.0

        # Adjust confidence based on similarity
        adjustment = (similarity - 0.5) * 0.2  # Small adjustment
        return min(1.0, max(0.0, confidence + adjustment))


def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """Load the unified dataset."""
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def create_synthetic_honey_tokens() -> List[HoneyPrompt]:
    """
    Create synthetic honey tokens for testing Stage 3 detection.

    These are "canary tokens" that we place in the dataset and expect
    the detector to find when they appear in attack texts.
    """
    return [
        HoneyPrompt(
            base_token="HONEY_TOKEN_INJECTION_DETECTOR_CANARY_001",
            category="direct_injection",
            sensitivity=0.95,
            context="system prompt injection attack detection",
            variations=[
                "HONEY_TOKEN_INJECTION_DETECTOR_CANARY_001",
                "HONEY TOKEN INJECTION DETECTOR CANARY 001",
                "honey_token_injection_detector_canary_001",
            ],
        ),
        HoneyPrompt(
            base_token="__HIVE_SECURITY_MONITOR_TRIGGER__",
            category="context_manipulation",
            sensitivity=0.90,
            context="prompt manipulation detection monitoring",
            variations=[
                "__HIVE_SECURITY_MONITOR_TRIGGER__",
                "HIVE_SECURITY_MONITOR_TRIGGER",
                "__hive_security_monitor_trigger__",
            ],
        ),
        HoneyPrompt(
            base_token="[ATTACK_PATTERN_CANARY_DETECTION]",
            category="jailbreak",
            sensitivity=0.85,
            context="adversarial attack pattern recognition",
            variations=[
                "[ATTACK_PATTERN_CANARY_DETECTION]",
                "ATTACK_PATTERN_CANARY_DETECTION",
                "[attack_pattern_canary_detection]",
            ],
        ),
    ]


def evaluate_stage1_only(texts: List[str]) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Evaluate Stage 1 ONLY: Heuristic rules.

    Returns: (confidences, detection_details)
    """
    engine = HeuristicRulesEngine()
    confidences = []
    details = []

    for i, text in enumerate(texts):
        # Try quick scan first (HIGH/CRITICAL)
        quick_match = engine.scan_quick(text)
        if quick_match:
            confidences.append(quick_match.confidence)
            details.append({
                "stage": "Stage 1 (Heuristic - Quick)",
                "matched": True,
                "rule_name": quick_match.rule_name,
                "severity": quick_match.severity.value,
            })
        else:
            # Then full scan (MEDIUM/LOW)
            matches = engine.scan(text)
            if matches:
                best_match = matches[0]
                confidences.append(best_match.confidence)
                details.append({
                    "stage": "Stage 1 (Heuristic - Full)",
                    "matched": True,
                    "rule_name": best_match.rule_name,
                    "severity": best_match.severity.value,
                })
            else:
                confidences.append(0.0)
                details.append({
                    "stage": "Stage 1 (Heuristic)",
                    "matched": False,
                    "rule_name": None,
                    "severity": None,
                })

        if (i + 1) % 5000 == 0:
            print(f"  Stage 1 Evaluation: {i+1}/{len(texts)} samples")

    return confidences, details


def evaluate_full_pipeline(
    texts: List[str],
    honey_tokens: List[HoneyPrompt],
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Evaluate FULL pipeline: Stage 1 + Stage 3 (honey token matching).

    Stage 2 (attack memory) is skipped because it requires embeddings.

    Returns: (confidences, detection_details)
    """
    heuristic_engine = HeuristicRulesEngine()
    context_evaluator = SimpleContextEvaluator()

    # Create detector instance
    detector = Detector(
        context_evaluator=context_evaluator,
        heuristic_engine=heuristic_engine,
        attack_memory=None,  # No embeddings available
    )

    confidences = []
    details = []

    for i, text in enumerate(texts):
        max_confidence = 0.0
        detection_info = {
            "stage": "Full Pipeline",
            "matched": False,
            "stages_checked": [],
            "best_match_stage": None,
            "best_match_rule": None,
        }

        # Run through Stage 1 first
        heuristic_result = detector._check_heuristics(text)
        detection_info["stages_checked"].append("Stage 1")

        if heuristic_result["matched"]:
            max_confidence = heuristic_result["confidence"]
            detection_info["matched"] = True
            detection_info["best_match_stage"] = "Stage 1 (Heuristic)"
            detection_info["best_match_rule"] = heuristic_result.get("rule_name")
        else:
            # Stage 1 didn't match, try Stage 3 with honey tokens
            detection_info["stages_checked"].append("Stage 3 (Honey Tokens)")

            for honey_token in honey_tokens:
                # Test exact match
                if honey_token.base_token in text:
                    max_confidence = 1.0
                    detection_info["matched"] = True
                    detection_info["best_match_stage"] = "Stage 3 (Exact Token Match)"
                    detection_info["best_match_rule"] = f"token:{honey_token.base_token[:20]}"
                    break

                # Test variations
                for variation in honey_token.variations:
                    if variation in text:
                        max_confidence = 0.9
                        detection_info["matched"] = True
                        detection_info["best_match_stage"] = "Stage 3 (Token Variation)"
                        detection_info["best_match_rule"] = f"variation:{variation[:20]}"
                        break

                if detection_info["matched"]:
                    break

                # Test obfuscation resistance
                canonical_text = detector.canonicalizer.canonicalize(text, aggressive=True)
                canonical_token = detector.canonicalizer.canonicalize(
                    honey_token.base_token, aggressive=True
                )

                if canonical_token in canonical_text:
                    max_confidence = 0.85
                    detection_info["matched"] = True
                    detection_info["best_match_stage"] = "Stage 3 (Obfuscated Token)"
                    detection_info["best_match_rule"] = f"obfuscated:{honey_token.base_token[:20]}"
                    break

        confidences.append(max_confidence)
        details.append(detection_info)

        if (i + 1) % 5000 == 0:
            print(f"  Full Pipeline Evaluation: {i+1}/{len(texts)} samples")

    return confidences, details


def evaluate_keyword_baseline(texts: List[str]) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Evaluate KeywordBaseline."""
    baseline = KeywordBaseline()
    confidences = []
    details = []

    for i, text in enumerate(texts):
        result = baseline.detect(text)
        confidences.append(result["confidence"])
        details.append({
            "stage": "Baseline (Keyword)",
            "matched": result.get("matched", result["confidence"] > 0.5),
        })

        if (i + 1) % 5000 == 0:
            print(f"  Keyword Baseline: {i+1}/{len(texts)} samples")

    return confidences, details


def evaluate_heuristic_baseline(texts: List[str]) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Evaluate HeuristicBaseline."""
    baseline = HeuristicBaseline()
    confidences = []
    details = []

    for i, text in enumerate(texts):
        result = baseline.detect(text)
        confidences.append(result["confidence"])
        details.append({
            "stage": "Baseline (Heuristic)",
            "matched": result.get("matched", result["confidence"] > 0.5),
        })

        if (i + 1) % 5000 == 0:
            print(f"  Heuristic Baseline: {i+1}/{len(texts)} samples")

    return confidences, details


def calculate_pr_curve(
    results: List[DetectionResult],
) -> List[Tuple[float, float, float]]:
    """Calculate PR curve points (threshold, recall, precision)."""
    pr_points = []

    thresholds = sorted(set([r.confidence for r in results] + [0.0, 1.0]), reverse=True)

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
    """Calculate AUC-PR."""
    if len(pr_points) < 2:
        return 0.0

    sorted_points = sorted(pr_points, key=lambda x: x[1])
    auc = 0.0
    for i in range(1, len(sorted_points)):
        prev_recall, prev_prec = sorted_points[i - 1][1], sorted_points[i - 1][2]
        curr_recall, curr_prec = sorted_points[i][1], sorted_points[i][2]
        auc += (curr_recall - prev_recall) * (curr_prec + prev_prec) / 2

    return auc


def bootstrap_confidence_intervals(
    results: List[DetectionResult],
    metric_func,
    n_iterations: int = 500,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence intervals."""
    bootstrap_metrics = []

    for _ in range(n_iterations):
        resampled = random.choices(results, k=len(results))
        metric = metric_func(resampled)
        bootstrap_metrics.append(metric)

    bootstrap_metrics = sorted(bootstrap_metrics)

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

    detection_results = [
        DetectionResult(
            text=r["text"][:200],
            predicted_attack=False,
            confidence=conf,
            actual_attack=r["actual_attack"],
            category=r["source"],
        )
        for conf, r in zip(confidences, results)
    ]

    evaluator = EvaluationMetrics(detection_results)
    base_report = evaluator.generate_report(threshold=0.5)
    roc_curve = evaluator.calculate_roc_curve(num_thresholds=100)
    auc_roc = evaluator.calculate_auc()
    pr_curve = calculate_pr_curve(detection_results)
    auc_pr = calculate_auc_pr(pr_curve)

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

    precision_ci = bootstrap_confidence_intervals(detection_results, precision_metric)
    recall_ci = bootstrap_confidence_intervals(detection_results, recall_metric)
    f1_ci = bootstrap_confidence_intervals(detection_results, f1_metric)

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
        "low_fpr_operating_points": base_report["roc_analysis"].get("low_fpr_operating_points", []),
        "per_source_breakdown": per_source,
    }


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation of HIVE detector (Stage 1 + Stage 3)"
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
        default="results/end_to_end_evaluation_report.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (useful for testing)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("END-TO-END EVALUATION OF HIVE PROMPT INJECTION DETECTOR")
    print("=" * 80)

    # Load dataset
    print("\n[1/7] Loading dataset...")
    rows = load_dataset(args.dataset)

    if args.max_samples:
        rows = rows[:args.max_samples]

    print(f"  Loaded {len(rows)} samples")

    labels = [r["label"] == "1" for r in rows]
    texts = [r["text"] for r in rows]
    sources = [r.get("source", "unknown") for r in rows]

    n_attacks = sum(labels)
    n_benign = len(labels) - n_attacks
    print(f"  Distribution: {n_benign} benign, {n_attacks} malicious")
    print(f"  Class ratio: {n_benign/len(labels):.1%} benign, {n_attacks/len(labels):.1%} malicious")

    results = [
        {
            "text": text,
            "actual_attack": label,
            "source": source,
        }
        for text, label, source in zip(texts, labels, sources)
    ]

    # Create synthetic honey tokens
    print("\n[2/7] Creating synthetic honey tokens...")
    honey_tokens = create_synthetic_honey_tokens()
    print(f"  Created {len(honey_tokens)} synthetic canary tokens")

    # Evaluate Stage 1 only
    print("\n[3/7] Evaluating Stage 1 (Heuristics only)...")
    t_start = time.time()
    stage1_confidences, stage1_details = evaluate_stage1_only(texts)
    stage1_time = time.time() - t_start
    print(f"  Completed in {stage1_time:.2f}s")

    # Evaluate full pipeline
    print("\n[4/7] Evaluating full pipeline (Stage 1 + Stage 3 tokens)...")
    t_start = time.time()
    full_confidences, full_details = evaluate_full_pipeline(texts, honey_tokens)
    full_time = time.time() - t_start
    print(f"  Completed in {full_time:.2f}s")

    # Evaluate baselines
    print("\n[5/7] Evaluating KeywordBaseline...")
    keyword_confidences, _ = evaluate_keyword_baseline(texts)

    print("\n[6/7] Evaluating HeuristicBaseline...")
    heuristic_baseline_confidences, _ = evaluate_heuristic_baseline(texts)

    # Generate reports
    print("\n[7/7] Generating reports...")

    reports = {
        "stage1_only": generate_detector_report(
            "Stage 1 Only (Heuristics)",
            stage1_confidences,
            results,
            sources,
        ),
        "full_pipeline": generate_detector_report(
            "Full Pipeline (Stage 1 + Stage 3 Tokens)",
            full_confidences,
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

    # Calculate improvement metrics
    print("\nCalculating Stage 1 vs Full Pipeline improvements...")

    def calc_improvements(confidences_before, confidences_after):
        """Calculate improvement when adding Stage 3."""
        improvements = {
            "samples_with_increased_confidence": 0,
            "samples_with_decreased_confidence": 0,
            "avg_confidence_increase": 0.0,
            "max_confidence_increase": 0.0,
        }

        increases = []
        for before, after in zip(confidences_before, confidences_after):
            diff = after - before
            if diff > 0:
                improvements["samples_with_increased_confidence"] += 1
                increases.append(diff)
            elif diff < 0:
                improvements["samples_with_decreased_confidence"] += 1

        if increases:
            improvements["avg_confidence_increase"] = np.mean(increases)
            improvements["max_confidence_increase"] = np.max(increases)

        return improvements

    pipeline_improvements = calc_improvements(stage1_confidences, full_confidences)

    # Build full report
    full_report = {
        "metadata": {
            "evaluation_date": "2026-02-21",
            "dataset": args.dataset,
            "total_samples": len(rows),
            "benign_samples": n_benign,
            "attack_samples": n_attacks,
            "random_seed": args.seed,
            "max_samples_limit": args.max_samples,
        },
        "pipeline_analysis": {
            "stage1_only_time_seconds": stage1_time,
            "full_pipeline_time_seconds": full_time,
            "pipeline_overhead_seconds": full_time - stage1_time,
            "honey_tokens_count": len(honey_tokens),
            "stage1_vs_full_improvement": pipeline_improvements,
        },
        "detectors": reports,
    }

    # Save report
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)

    for name, report in reports.items():
        print(f"\n{name}:")
        m = report["metrics_at_0.5"]
        print(f"  Precision: {m['precision']:.4f} (95% CI: [{report['confidence_intervals_95_bootstrap']['precision']['lower']:.4f}, {report['confidence_intervals_95_bootstrap']['precision']['upper']:.4f}])")
        print(f"  Recall:    {m['recall']:.4f} (95% CI: [{report['confidence_intervals_95_bootstrap']['recall']['lower']:.4f}, {report['confidence_intervals_95_bootstrap']['recall']['upper']:.4f}])")
        print(f"  F1:        {m['f1']:.4f} (95% CI: [{report['confidence_intervals_95_bootstrap']['f1']['lower']:.4f}, {report['confidence_intervals_95_bootstrap']['f1']['upper']:.4f}])")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  AUC-ROC:   {report['roc_analysis']['auc']:.4f}")
        print(f"  AUC-PR:    {report['pr_analysis']['auc']:.4f}")

    print("\n" + "=" * 80)
    print("STAGE 1 vs FULL PIPELINE COMPARISON")
    print("=" * 80)
    print(f"Stage 1 Evaluation Time:     {full_report['pipeline_analysis']['stage1_only_time_seconds']:.2f}s")
    print(f"Full Pipeline Time:          {full_report['pipeline_analysis']['full_pipeline_time_seconds']:.2f}s")
    print(f"Pipeline Overhead:           {full_report['pipeline_analysis']['pipeline_overhead_seconds']:.2f}s")

    improvements = full_report['pipeline_analysis']['stage1_vs_full_improvement']
    print(f"\nConfidence Score Improvements (Stage 1 -> Full):")
    print(f"  Samples with increased confidence: {improvements['samples_with_increased_confidence']}")
    print(f"  Samples with decreased confidence: {improvements['samples_with_decreased_confidence']}")
    print(f"  Average increase (when improved):  {improvements['avg_confidence_increase']:.4f}")
    print(f"  Maximum increase:                  {improvements['max_confidence_increase']:.4f}")

    # Show Stage 1 vs Full Pipeline metrics
    s1_metrics = reports["stage1_only"]["metrics_at_0.5"]
    fp_metrics = reports["full_pipeline"]["metrics_at_0.5"]

    print(f"\nMetric Changes (Stage 1 -> Full):")
    print(f"  Precision: {s1_metrics['precision']:.4f} -> {fp_metrics['precision']:.4f} ({(fp_metrics['precision']-s1_metrics['precision'])*100:+.2f}%)")
    print(f"  Recall:    {s1_metrics['recall']:.4f} -> {fp_metrics['recall']:.4f} ({(fp_metrics['recall']-s1_metrics['recall'])*100:+.2f}%)")
    print(f"  F1:        {s1_metrics['f1']:.4f} -> {fp_metrics['f1']:.4f} ({(fp_metrics['f1']-s1_metrics['f1'])*100:+.2f}%)")
    print(f"  AUC-ROC:   {reports['stage1_only']['roc_analysis']['auc']:.4f} -> {reports['full_pipeline']['roc_analysis']['auc']:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
