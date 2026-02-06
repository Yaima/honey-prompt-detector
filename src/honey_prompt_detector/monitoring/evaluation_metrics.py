"""
Evaluation Metrics Module

Addresses reviewer feedback:
"No comparison to recent detectors with low-FPR calibration"
"Report precision/recall/F1 and ROC/AUC"

Provides:
1. Standard classification metrics (precision, recall, F1, accuracy)
2. ROC/AUC analysis
3. Low-FPR operating point calibration (PromptShield-style)
4. Per-category breakdown
5. Threshold calibration for FPR budgets
6. Confidence interval calculation
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DetectionResult:
    """Single detection result for evaluation."""

    text: str
    predicted_attack: bool
    confidence: float
    actual_attack: bool
    category: str = "unknown"


@dataclass
class ClassificationMetrics:
    """Classification metrics for a given threshold."""

    threshold: float
    tp: int = 0  # True positives
    fp: int = 0  # False positives
    tn: int = 0  # True negatives
    fn: int = 0  # False negatives

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / total"""
        if self.total == 0:
            return 0.0
        return (self.tp + self.tn) / self.total

    @property
    def fpr(self) -> float:
        """False Positive Rate = FP / (FP + TN)"""
        if self.fp + self.tn == 0:
            return 0.0
        return self.fp / (self.fp + self.tn)

    @property
    def tpr(self) -> float:
        """True Positive Rate (same as recall) = TP / (TP + FN)"""
        return self.recall

    @property
    def fnr(self) -> float:
        """False Negative Rate = FN / (FN + TP)"""
        if self.fn + self.tp == 0:
            return 0.0
        return self.fn / (self.fn + self.tp)

    @property
    def specificity(self) -> float:
        """Specificity = TN / (TN + FP)"""
        if self.tn + self.fp == 0:
            return 0.0
        return self.tn / (self.tn + self.fp)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "fpr": self.fpr,
            "tpr": self.tpr,
            "fnr": self.fnr,
            "specificity": self.specificity,
        }


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics calculator.

    Features:
    - Multi-threshold analysis
    - ROC curve and AUC calculation
    - Low-FPR operating point calibration
    - Per-category breakdown
    - Confidence intervals
    """

    def __init__(self, results: Optional[List[DetectionResult]] = None):
        self.results: List[DetectionResult] = results or []
        self._roc_cache: Optional[List[Tuple[float, float, float]]] = None

    def add_result(self, result: DetectionResult):
        """Add a detection result."""
        self.results.append(result)
        self._roc_cache = None  # Invalidate cache

    def add_results(self, results: List[DetectionResult]):
        """Add multiple detection results."""
        self.results.extend(results)
        self._roc_cache = None

    def calculate_metrics(self, threshold: float = 0.5) -> ClassificationMetrics:
        """
        Calculate classification metrics at a given threshold.

        Predictions with confidence >= threshold are classified as attacks.
        """
        metrics = ClassificationMetrics(threshold=threshold)

        for result in self.results:
            # Predict attack if confidence >= threshold
            predicted = result.confidence >= threshold

            if predicted and result.actual_attack:
                metrics.tp += 1
            elif predicted and not result.actual_attack:
                metrics.fp += 1
            elif not predicted and not result.actual_attack:
                metrics.tn += 1
            else:  # not predicted and actual_attack
                metrics.fn += 1

        return metrics

    def calculate_roc_curve(self, num_thresholds: int = 100) -> List[Tuple[float, float, float]]:
        """
        Calculate ROC curve points.

        Returns: List of (threshold, FPR, TPR) tuples
        """
        if self._roc_cache is not None:
            return self._roc_cache

        roc_points = []

        # Use sorted unique confidence scores as thresholds
        thresholds = sorted(set([r.confidence for r in self.results] + [0.0, 1.0]), reverse=True)

        # If too many thresholds, sample
        if len(thresholds) > num_thresholds:
            step = len(thresholds) / num_thresholds
            thresholds = [thresholds[int(i * step)] for i in range(num_thresholds)]

        for threshold in thresholds:
            metrics = self.calculate_metrics(threshold)
            roc_points.append((threshold, metrics.fpr, metrics.tpr))

        self._roc_cache = roc_points
        return roc_points

    def calculate_auc(self) -> float:
        """
        Calculate Area Under the ROC Curve (AUC).

        Uses trapezoidal rule for numerical integration.
        """
        roc_curve = self.calculate_roc_curve()

        if len(roc_curve) < 2:
            return 0.0

        # Sort by FPR for proper integration
        roc_curve_sorted = sorted(roc_curve, key=lambda x: x[1])

        auc = 0.0
        for i in range(1, len(roc_curve_sorted)):
            prev_fpr, prev_tpr = roc_curve_sorted[i - 1][1], roc_curve_sorted[i - 1][2]
            curr_fpr, curr_tpr = roc_curve_sorted[i][1], roc_curve_sorted[i][2]

            # Trapezoidal area
            auc += (curr_fpr - prev_fpr) * (curr_tpr + prev_tpr) / 2

        return auc

    def find_threshold_for_fpr(self, target_fpr: float) -> Tuple[float, ClassificationMetrics]:
        """
        Find threshold that achieves target FPR.

        This is critical for low-FPR operating point calibration
        as requested by reviewers (PromptShield-style).

        Args:
            target_fpr: Target false positive rate (e.g., 0.01 for 1%)

        Returns:
            (threshold, metrics) tuple
        """
        roc_curve = self.calculate_roc_curve()

        # Find threshold that gives FPR closest to (but not exceeding) target
        best_threshold = 1.0
        best_diff = float("inf")

        for threshold, fpr, tpr in roc_curve:
            if fpr <= target_fpr:
                diff = target_fpr - fpr
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = threshold

        return best_threshold, self.calculate_metrics(best_threshold)

    def calculate_tpr_at_fpr(self, target_fpr: float) -> float:
        """
        Calculate TPR at a specific FPR operating point.

        This is the standard metric for comparing detectors at
        low-FPR regimes (e.g., TPR@1%FPR).
        """
        _, metrics = self.find_threshold_for_fpr(target_fpr)
        return metrics.tpr

    def calibrate_for_fpr_budget(self, fpr_budget: float) -> Dict[str, Any]:
        """
        Calibrate threshold to meet an FPR budget.

        Args:
            fpr_budget: Maximum acceptable FPR (e.g., 0.01 for 1%)

        Returns:
            Calibration result with threshold and expected metrics
        """
        threshold, metrics = self.find_threshold_for_fpr(fpr_budget)

        return {
            "fpr_budget": fpr_budget,
            "calibrated_threshold": threshold,
            "achieved_fpr": metrics.fpr,
            "achieved_tpr": metrics.tpr,
            "achieved_precision": metrics.precision,
            "achieved_recall": metrics.recall,
            "achieved_f1": metrics.f1,
            "total_samples": metrics.total,
        }

    def calculate_per_category_metrics(self, threshold: float = 0.5) -> Dict[str, ClassificationMetrics]:
        """
        Calculate metrics per attack category.

        Useful for understanding performance across different attack types.
        """
        category_results: Dict[str, List[DetectionResult]] = defaultdict(list)

        for result in self.results:
            category_results[result.category].append(result)

        category_metrics = {}
        for category, results in category_results.items():
            evaluator = EvaluationMetrics(results)
            category_metrics[category] = evaluator.calculate_metrics(threshold)

        return category_metrics

    def calculate_confidence_interval(
        self, metric: str, threshold: float = 0.5, confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for a metric using Wilson score.

        Args:
            metric: 'precision', 'recall', 'f1', or 'accuracy'
            threshold: Classification threshold
            confidence: Confidence level (default 95%)

        Returns:
            (lower_bound, point_estimate, upper_bound)
        """
        metrics = self.calculate_metrics(threshold)

        if metric == "precision":
            p = metrics.precision
            n = metrics.tp + metrics.fp
        elif metric == "recall":
            p = metrics.recall
            n = metrics.tp + metrics.fn
        elif metric == "accuracy":
            p = metrics.accuracy
            n = metrics.total
        else:  # f1 or other
            return (metrics.f1, metrics.f1, metrics.f1)  # No CI for F1

        if n == 0:
            return (0.0, 0.0, 0.0)

        # Wilson score interval
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return (lower, p, upper)

    def generate_report(self, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Includes all metrics requested by reviewers.
        """
        metrics = self.calculate_metrics(threshold)
        auc = self.calculate_auc()

        # Low-FPR operating points
        low_fpr_points = {}
        for fpr_target in [0.001, 0.005, 0.01, 0.05, 0.1]:
            calibration = self.calibrate_for_fpr_budget(fpr_target)
            low_fpr_points[f"{fpr_target*100:.1f}%_fpr"] = {
                "threshold": calibration["calibrated_threshold"],
                "tpr": calibration["achieved_tpr"],
                "precision": calibration["achieved_precision"],
            }

        # Confidence intervals
        precision_ci = self.calculate_confidence_interval("precision", threshold)
        recall_ci = self.calculate_confidence_interval("recall", threshold)

        # Per-category breakdown
        category_metrics = self.calculate_per_category_metrics(threshold)
        category_summary = {
            cat: {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "support": m.tp + m.fn,  # Number of actual positives
            }
            for cat, m in category_metrics.items()
        }

        return {
            "summary": {
                "total_samples": metrics.total,
                "attack_samples": metrics.tp + metrics.fn,
                "benign_samples": metrics.tn + metrics.fp,
                "threshold": threshold,
            },
            "confusion_matrix": {
                "tp": metrics.tp,
                "fp": metrics.fp,
                "tn": metrics.tn,
                "fn": metrics.fn,
            },
            "metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "accuracy": metrics.accuracy,
                "fpr": metrics.fpr,
                "fnr": metrics.fnr,
                "specificity": metrics.specificity,
            },
            "confidence_intervals_95": {
                "precision": {"lower": precision_ci[0], "estimate": precision_ci[1], "upper": precision_ci[2]},
                "recall": {"lower": recall_ci[0], "estimate": recall_ci[1], "upper": recall_ci[2]},
            },
            "roc_analysis": {
                "auc": auc,
                "low_fpr_operating_points": low_fpr_points,
            },
            "per_category": category_summary,
        }

    def print_report(self, threshold: float = 0.5):
        """Print formatted evaluation report."""
        report = self.generate_report(threshold)

        print("\n" + "=" * 70)
        print("EVALUATION METRICS REPORT")
        print("=" * 70)

        print("\n[Summary]")
        print(f"  Total samples: {report['summary']['total_samples']}")
        print(f"  Attack samples: {report['summary']['attack_samples']}")
        print(f"  Benign samples: {report['summary']['benign_samples']}")
        print(f"  Threshold: {report['summary']['threshold']}")

        print("\n[Confusion Matrix]")
        cm = report["confusion_matrix"]
        print(f"  TP: {cm['tp']:5d}  FP: {cm['fp']:5d}")
        print(f"  FN: {cm['fn']:5d}  TN: {cm['tn']:5d}")

        print("\n[Classification Metrics]")
        m = report["metrics"]
        print(f"  Precision:   {m['precision']:.4f}")
        print(f"  Recall:      {m['recall']:.4f}")
        print(f"  F1 Score:    {m['f1']:.4f}")
        print(f"  Accuracy:    {m['accuracy']:.4f}")
        print(f"  FPR:         {m['fpr']:.4f}")
        print(f"  Specificity: {m['specificity']:.4f}")

        print("\n[95% Confidence Intervals]")
        ci = report["confidence_intervals_95"]
        print(f"  Precision: [{ci['precision']['lower']:.4f}, {ci['precision']['upper']:.4f}]")
        print(f"  Recall:    [{ci['recall']['lower']:.4f}, {ci['recall']['upper']:.4f}]")

        print("\n[ROC Analysis]")
        print(f"  AUC: {report['roc_analysis']['auc']:.4f}")
        print("\n  Low-FPR Operating Points:")
        for name, data in report["roc_analysis"]["low_fpr_operating_points"].items():
            print(f"    {name}: TPR={data['tpr']:.4f}, Threshold={data['threshold']:.4f}")

        if report["per_category"]:
            print("\n[Per-Category Breakdown]")
            for cat, data in report["per_category"].items():
                print(f"  {cat}:")
                print(
                    f"    Precision: {data['precision']:.4f}, Recall: {data['recall']:.4f}, "
                    f"F1: {data['f1']:.4f}, Support: {data['support']}"
                )

        print("\n" + "=" * 70)


# Convenience function for quick evaluation
def evaluate_detector(
    predictions: List[Tuple[bool, float]],
    labels: List[bool],
    categories: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Quick evaluation of detector predictions.

    Args:
        predictions: List of (predicted_attack, confidence) tuples
        labels: List of actual attack labels
        categories: Optional list of attack categories
        threshold: Classification threshold

    Returns:
        Evaluation report
    """
    if categories is None:
        categories = ["unknown"] * len(predictions)

    results = [
        DetectionResult(text="", predicted_attack=pred[0], confidence=pred[1], actual_attack=label, category=cat)
        for pred, label, cat in zip(predictions, labels, categories)
    ]

    evaluator = EvaluationMetrics(results)
    return evaluator.generate_report(threshold)
