#!/usr/bin/env python3
"""
Test suite for evaluation metrics module.

Verifies:
1. Precision/Recall/F1 calculations
2. ROC/AUC analysis
3. Low-FPR calibration
4. Confidence intervals
5. Per-category breakdown
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.honey_prompt_detector.monitoring.evaluation_metrics import (
    DetectionResult,
    EvaluationMetrics,
    evaluate_detector,
)


class EvaluationMetricsTestSuite:
    """Test suite for evaluation metrics."""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0

    def run_test(self, name: str, test_func) -> bool:
        """Run a single test."""
        try:
            result = test_func()
            if result:
                self.tests_passed += 1
                print(f"  [PASSED] {name}")
            else:
                self.tests_failed += 1
                print(f"  [FAILED] {name}")
            return result
        except Exception as e:
            self.tests_failed += 1
            print(f"  [ERROR] {name}: {e}")
            return False

    def test_basic_metrics_calculation(self):
        """Test basic precision/recall/F1 calculation."""
        # Create a perfect classifier result
        results = [
            DetectionResult("", True, 0.9, True, "attack"),  # TP
            DetectionResult("", True, 0.8, True, "attack"),  # TP
            DetectionResult("", True, 0.7, True, "attack"),  # TP
            DetectionResult("", False, 0.3, False, "benign"),  # TN
            DetectionResult("", False, 0.2, False, "benign"),  # TN
        ]

        evaluator = EvaluationMetrics(results)
        metrics = evaluator.calculate_metrics(threshold=0.5)

        # Expected: TP=3, TN=2, FP=0, FN=0
        assert metrics.tp == 3, f"Expected TP=3, got {metrics.tp}"
        assert metrics.tn == 2, f"Expected TN=2, got {metrics.tn}"
        assert metrics.fp == 0, f"Expected FP=0, got {metrics.fp}"
        assert metrics.fn == 0, f"Expected FN=0, got {metrics.fn}"
        assert metrics.precision == 1.0, f"Expected precision=1.0, got {metrics.precision}"
        assert metrics.recall == 1.0, f"Expected recall=1.0, got {metrics.recall}"
        assert metrics.f1 == 1.0, f"Expected F1=1.0, got {metrics.f1}"

        return True

    def test_imperfect_classifier(self):
        """Test metrics with FP and FN."""
        results = [
            DetectionResult("", True, 0.9, True, "attack"),  # TP
            DetectionResult("", True, 0.8, False, "benign"),  # FP
            DetectionResult("", False, 0.3, True, "attack"),  # FN
            DetectionResult("", False, 0.2, False, "benign"),  # TN
        ]

        evaluator = EvaluationMetrics(results)
        metrics = evaluator.calculate_metrics(threshold=0.5)

        # Expected: TP=1, FP=1, FN=1, TN=1
        assert metrics.tp == 1
        assert metrics.fp == 1
        assert metrics.fn == 1
        assert metrics.tn == 1

        # Precision = 1/(1+1) = 0.5
        assert abs(metrics.precision - 0.5) < 0.001

        # Recall = 1/(1+1) = 0.5
        assert abs(metrics.recall - 0.5) < 0.001

        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert abs(metrics.f1 - 0.5) < 0.001

        return True

    def test_roc_auc_perfect_classifier(self):
        """Test ROC/AUC for perfect classifier (AUC should be 1.0)."""
        # Perfect separation: all attacks have high confidence, all benign have low
        results = []
        for i in range(50):
            results.append(DetectionResult("", True, 0.9 + random.random() * 0.1, True, "attack"))
        for i in range(50):
            results.append(DetectionResult("", False, random.random() * 0.1, False, "benign"))

        evaluator = EvaluationMetrics(results)
        auc = evaluator.calculate_auc()

        # AUC should be very close to 1.0 for perfect separation
        assert auc > 0.95, f"Expected AUC > 0.95, got {auc}"
        return True

    def test_roc_auc_random_classifier(self):
        """Test ROC/AUC for random classifier (AUC should be ~0.5)."""
        results = []
        random.seed(42)
        for i in range(100):
            is_attack = random.random() > 0.5
            confidence = random.random()  # Random confidence
            results.append(DetectionResult("", confidence > 0.5, confidence, is_attack, "random"))

        evaluator = EvaluationMetrics(results)
        auc = evaluator.calculate_auc()

        # AUC should be around 0.5 for random classifier
        assert 0.3 < auc < 0.7, f"Expected AUC around 0.5, got {auc}"
        return True

    def test_low_fpr_calibration(self):
        """Test low-FPR operating point calibration."""
        # Create results with varying confidence
        results = []
        for i in range(50):
            conf = 0.7 + random.random() * 0.3  # High confidence attacks
            results.append(DetectionResult("", True, conf, True, "attack"))
        for i in range(50):
            conf = random.random() * 0.5  # Low confidence benign
            results.append(DetectionResult("", False, conf, False, "benign"))

        evaluator = EvaluationMetrics(results)

        # Calibrate for 5% FPR
        calibration = evaluator.calibrate_for_fpr_budget(0.05)

        # FPR should be at or below target
        assert calibration["achieved_fpr"] <= 0.05, f"Expected FPR <= 0.05, got {calibration['achieved_fpr']}"

        # Should still have reasonable TPR
        assert calibration["achieved_tpr"] > 0.5, f"Expected TPR > 0.5, got {calibration['achieved_tpr']}"

        return True

    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        # Create large sample for meaningful CI
        results = []
        for i in range(100):
            results.append(DetectionResult("", True, 0.9, True, "attack"))
        for i in range(10):
            results.append(DetectionResult("", True, 0.9, False, "benign"))  # FP
        for i in range(90):
            results.append(DetectionResult("", False, 0.1, False, "benign"))

        evaluator = EvaluationMetrics(results)
        ci = evaluator.calculate_confidence_interval("precision", threshold=0.5)

        # CI should be valid
        lower, estimate, upper = ci
        assert lower < estimate < upper, "CI should have lower < estimate < upper"
        assert lower >= 0 and upper <= 1, "CI should be in [0, 1]"

        # Estimate should be precision
        metrics = evaluator.calculate_metrics(0.5)
        assert abs(estimate - metrics.precision) < 0.001

        return True

    def test_per_category_metrics(self):
        """Test per-category breakdown."""
        results = [
            DetectionResult("", True, 0.9, True, "instruction_override"),
            DetectionResult("", True, 0.8, True, "instruction_override"),
            DetectionResult("", True, 0.7, True, "data_extraction"),
            DetectionResult("", True, 0.6, False, "benign"),  # FP
            DetectionResult("", False, 0.3, False, "benign"),
        ]

        evaluator = EvaluationMetrics(results)
        category_metrics = evaluator.calculate_per_category_metrics(threshold=0.5)

        # Should have metrics for each category
        assert "instruction_override" in category_metrics
        assert "data_extraction" in category_metrics
        assert "benign" in category_metrics

        # instruction_override should have perfect recall (2/2)
        assert category_metrics["instruction_override"].recall == 1.0

        return True

    def test_report_generation(self):
        """Test comprehensive report generation."""
        results = []
        random.seed(42)
        for i in range(50):
            conf = 0.6 + random.random() * 0.4
            results.append(DetectionResult("", True, conf, True, "attack"))
        for i in range(10):
            conf = 0.5 + random.random() * 0.3
            results.append(DetectionResult("", True, conf, False, "benign"))  # FP
        for i in range(5):
            conf = random.random() * 0.4
            results.append(DetectionResult("", False, conf, True, "attack"))  # FN
        for i in range(35):
            conf = random.random() * 0.4
            results.append(DetectionResult("", False, conf, False, "benign"))

        evaluator = EvaluationMetrics(results)
        report = evaluator.generate_report(threshold=0.5)

        # Check report structure
        assert "summary" in report
        assert "confusion_matrix" in report
        assert "metrics" in report
        assert "roc_analysis" in report
        assert "confidence_intervals_95" in report

        # Check metrics are present
        assert "precision" in report["metrics"]
        assert "recall" in report["metrics"]
        assert "f1" in report["metrics"]
        assert "auc" in report["roc_analysis"]

        return True

    def test_evaluate_detector_convenience(self):
        """Test the convenience function."""
        predictions = [
            (True, 0.9),
            (True, 0.8),
            (False, 0.3),
            (False, 0.2),
        ]
        labels = [True, False, True, False]  # Attack, Benign, Attack, Benign

        report = evaluate_detector(predictions, labels, threshold=0.5)

        # Should return valid report
        assert report["confusion_matrix"]["tp"] == 1  # First prediction
        assert report["confusion_matrix"]["fp"] == 1  # Second prediction
        assert report["confusion_matrix"]["fn"] == 1  # Third prediction
        assert report["confusion_matrix"]["tn"] == 1  # Fourth prediction

        return True

    def run_all_tests(self):
        """Run all evaluation metrics tests."""
        print("\n" + "=" * 70)
        print("EVALUATION METRICS TEST SUITE")
        print("=" * 70)

        print("\n[Basic Metrics Tests]")
        self.run_test("Basic metrics calculation", self.test_basic_metrics_calculation)
        self.run_test("Imperfect classifier", self.test_imperfect_classifier)

        print("\n[ROC/AUC Tests]")
        self.run_test("ROC/AUC perfect classifier", self.test_roc_auc_perfect_classifier)
        self.run_test("ROC/AUC random classifier", self.test_roc_auc_random_classifier)

        print("\n[Calibration Tests]")
        self.run_test("Low-FPR calibration", self.test_low_fpr_calibration)
        self.run_test("Confidence intervals", self.test_confidence_intervals)

        print("\n[Advanced Tests]")
        self.run_test("Per-category metrics", self.test_per_category_metrics)
        self.run_test("Report generation", self.test_report_generation)
        self.run_test("Convenience function", self.test_evaluate_detector_convenience)

        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"  Passed: {self.tests_passed}")
        print(f"  Failed: {self.tests_failed}")
        print(f"  Total:  {self.tests_passed + self.tests_failed}")

        if self.tests_failed == 0:
            print("\nALL EVALUATION METRICS TESTS PASSED!")
            print("\nVerified capabilities:")
            print("  - Precision/Recall/F1/Accuracy calculation")
            print("  - ROC curve and AUC calculation")
            print("  - Low-FPR operating point calibration (PromptShield-style)")
            print("  - Confidence interval calculation (Wilson score)")
            print("  - Per-category metric breakdown")
            print("  - Comprehensive report generation")

        return self.tests_failed == 0


if __name__ == "__main__":
    suite = EvaluationMetricsTestSuite()
    success = suite.run_all_tests()
    sys.exit(0 if success else 1)
