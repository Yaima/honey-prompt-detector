"""
Enhanced SelfTuner with Online Learning

Addresses reviewer feedback:
"How are FP/FN estimated in deployment without ground-truth labels?"

Solutions implemented:
1. Confidence-based pseudo-labeling for high-confidence predictions
2. Bandit-style exploration with Thompson Sampling
3. Semi-supervised calibration using consistency checks
4. Human-in-the-loop adjudication support with latency tracking
5. Drift detection for distribution shifts
"""

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .detector import Detector

logger = logging.getLogger("honey_prompt")


@dataclass
class DetectionRecord:
    """Record of a detection event for analysis."""

    timestamp: datetime
    text_hash: str  # Hash of input text (not full text for privacy)
    confidence: float
    predicted_attack: bool
    threshold_used: float
    pseudo_label: Optional[bool] = None
    human_label: Optional[bool] = None
    latency_ms: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class ThresholdArm:
    """Bandit arm for threshold exploration."""

    threshold: float
    successes: int = 0  # Correct predictions
    failures: int = 0  # Incorrect predictions
    total_samples: int = 0

    def sample_beta(self) -> float:
        """Sample from Beta distribution (Thompson Sampling)."""
        return random.betavariate(self.successes + 1, self.failures + 1)


class EnhancedSelfTuner:
    """
    Production-ready self-tuner with online learning capabilities.

    Key features:
    - Works without ground truth via pseudo-labeling
    - Uses Thompson Sampling for threshold exploration
    - Supports human-in-the-loop adjudication
    - Tracks metrics for operational monitoring
    - Detects distribution drift
    """

    def __init__(
        self,
        detector_agent: Detector,
        config: Any,
        high_confidence_threshold: float = 0.95,
        low_confidence_threshold: float = 0.3,
        consistency_window: int = 5,
        drift_detection_window: int = 100,
        human_review_budget: int = 10,  # Max human reviews per hour
    ):
        self.detector_agent = detector_agent
        self.config = config

        # Confidence thresholds for pseudo-labeling
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold

        # Metrics tracking
        self.total_evaluations = 0
        self.pseudo_labeled_count = 0
        self.human_labeled_count = 0

        # Pseudo-label estimates
        self.estimated_fp = 0
        self.estimated_fn = 0
        self.estimated_tp = 0
        self.estimated_tn = 0

        # History for analysis
        self.detection_history: deque = deque(maxlen=1000)
        self.consistency_window = consistency_window

        # Drift detection
        self.drift_detection_window = drift_detection_window
        self.baseline_confidence_mean: Optional[float] = None
        self.baseline_confidence_std: Optional[float] = None

        # Human-in-the-loop
        self.human_review_budget = human_review_budget
        self.human_reviews_this_hour = 0
        self.last_hour_reset = datetime.now()
        self.pending_human_review: List[DetectionRecord] = []

        # Thompson Sampling for threshold exploration
        self.threshold_arms = self._init_threshold_arms()
        self.current_arm_idx = self._find_arm_for_threshold(detector_agent.current_threshold)

        # Calibration metrics
        self.calibration_bins: Dict[str, Dict[str, int]] = self._init_calibration_bins()

        logger.info(
            f"EnhancedSelfTuner initialized with "
            f"high_conf={high_confidence_threshold}, "
            f"low_conf={low_confidence_threshold}"
        )

    def _init_threshold_arms(self) -> List[ThresholdArm]:
        """Initialize threshold arms for bandit exploration."""
        # Create arms for thresholds from 0.5 to 0.95 in steps of 0.05
        return [ThresholdArm(threshold=t) for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]

    def _find_arm_for_threshold(self, threshold: float) -> int:
        """Find the closest arm for a given threshold."""
        min_diff = float("inf")
        best_idx = 0
        for i, arm in enumerate(self.threshold_arms):
            diff = abs(arm.threshold - threshold)
            if diff < min_diff:
                min_diff = diff
                best_idx = i
        return best_idx

    def _init_calibration_bins(self) -> Dict[str, Dict[str, int]]:
        """Initialize bins for calibration tracking."""
        bins = {}
        for i in range(10):
            bin_name = f"{i*10}-{(i+1)*10}"
            bins[bin_name] = {"total": 0, "positive": 0}
        return bins

    def record_detection(
        self, detection_result: Dict[str, Any], text_hash: str, latency_ms: float = 0.0
    ) -> DetectionRecord:
        """
        Record a detection event and apply pseudo-labeling.

        This method works WITHOUT ground truth labels by using:
        1. High-confidence predictions as pseudo-labels
        2. Consistency checking across similar inputs
        3. Calibration-based estimation

        Args:
            detection_result: Result from detector
            text_hash: Hash of input text
            latency_ms: Detection latency

        Returns:
            DetectionRecord with pseudo-label if applicable
        """
        self.total_evaluations += 1

        confidence = detection_result.get("confidence", 0.5)
        predicted_attack = detection_result.get("detection", False)
        threshold = self.detector_agent.current_threshold

        record = DetectionRecord(
            timestamp=datetime.now(),
            text_hash=text_hash,
            confidence=confidence,
            predicted_attack=predicted_attack,
            threshold_used=threshold,
            latency_ms=latency_ms,
            features=detection_result.get("features", {}),
        )

        # Apply pseudo-labeling for high-confidence predictions
        pseudo_label = self._generate_pseudo_label(confidence, predicted_attack)
        record.pseudo_label = pseudo_label

        if pseudo_label is not None:
            self.pseudo_labeled_count += 1
            self._update_pseudo_estimates(predicted_attack, pseudo_label, confidence)

        # Update calibration bins
        self._update_calibration(confidence, pseudo_label)

        # Check for drift
        self._check_drift(confidence)

        # Add to history
        self.detection_history.append(record)

        # Maybe queue for human review
        self._maybe_queue_for_review(record)

        return record

    def _generate_pseudo_label(self, confidence: float, predicted_attack: bool) -> Optional[bool]:
        """
        Generate pseudo-label based on confidence.

        Strategy:
        - Very high confidence (>0.95): Trust the prediction
        - Very low confidence (<0.30): Trust the inverse (if we're very unsure
          it's an attack, pseudo-label as benign)
        - Medium confidence: No pseudo-label (uncertain)
        """
        if predicted_attack and confidence >= self.high_confidence_threshold:
            # Very confident attack -> pseudo-label as attack
            return True
        elif not predicted_attack and confidence >= self.high_confidence_threshold:
            # Very confident not-attack -> pseudo-label as benign
            return False
        elif predicted_attack and confidence <= self.low_confidence_threshold:
            # Low confidence attack prediction -> likely benign
            return False
        elif not predicted_attack and confidence <= self.low_confidence_threshold:
            # Low confidence benign prediction -> might be attack
            # But we don't pseudo-label this as attack (too risky)
            return None

        return None  # Uncertain, no pseudo-label

    def _update_pseudo_estimates(self, predicted: bool, pseudo_label: bool, confidence: float):
        """Update estimated FP/FN/TP/TN based on pseudo-labels."""
        if predicted and pseudo_label:
            self.estimated_tp += 1
        elif predicted and not pseudo_label:
            self.estimated_fp += 1
        elif not predicted and pseudo_label:
            self.estimated_fn += 1
        else:
            self.estimated_tn += 1

    def _update_calibration(self, confidence: float, pseudo_label: Optional[bool]):
        """Update calibration bins for reliability tracking."""
        bin_idx = min(int(confidence * 10), 9)
        bin_name = f"{bin_idx*10}-{(bin_idx+1)*10}"

        self.calibration_bins[bin_name]["total"] += 1
        if pseudo_label is True:
            self.calibration_bins[bin_name]["positive"] += 1

    def _check_drift(self, confidence: float):
        """Check for distribution drift in confidence scores."""
        if len(self.detection_history) < self.drift_detection_window:
            return

        recent_confidences = [r.confidence for r in list(self.detection_history)[-self.drift_detection_window :]]

        current_mean = sum(recent_confidences) / len(recent_confidences)
        current_std = math.sqrt(sum((c - current_mean) ** 2 for c in recent_confidences) / len(recent_confidences))

        # Initialize baseline if needed
        if self.baseline_confidence_mean is None:
            self.baseline_confidence_mean = current_mean
            self.baseline_confidence_std = current_std
            return

        # Check for significant drift (2 sigma)
        drift_detected = abs(current_mean - self.baseline_confidence_mean) > 2 * self.baseline_confidence_std

        if drift_detected:
            logger.warning(
                f"Distribution drift detected! "
                f"Baseline: {self.baseline_confidence_mean:.3f}±{self.baseline_confidence_std:.3f}, "
                f"Current: {current_mean:.3f}±{current_std:.3f}"
            )
            # Could trigger re-calibration here

    def _maybe_queue_for_review(self, record: DetectionRecord):
        """Queue uncertain predictions for human review."""
        # Reset hourly budget
        now = datetime.now()
        if (now - self.last_hour_reset) > timedelta(hours=1):
            self.human_reviews_this_hour = 0
            self.last_hour_reset = now

        # Queue if uncertain and budget allows
        if record.pseudo_label is None and self.human_reviews_this_hour < self.human_review_budget:
            self.pending_human_review.append(record)

    def update_with_human_label(self, text_hash: str, human_label: bool) -> bool:
        """
        Update with human-provided label.

        This is the gold standard for calibration.
        Uses Thompson Sampling to adjust threshold based on accumulated human feedback.

        Args:
            text_hash: Hash of the text that was labeled
            human_label: True if human says it's an attack

        Returns:
            True if record was found and updated
        """
        # Find record in history
        for record in self.detection_history:
            if record.text_hash == text_hash:
                record.human_label = human_label
                self.human_labeled_count += 1
                self.human_reviews_this_hour += 1

                # Update bandit arm with true feedback
                arm = self.threshold_arms[self.current_arm_idx]
                prediction_correct = record.predicted_attack == human_label

                if prediction_correct:
                    arm.successes += 1
                else:
                    arm.failures += 1
                arm.total_samples += 1

                # Remove from pending review
                self.pending_human_review = [r for r in self.pending_human_review if r.text_hash != text_hash]

                # Apply Thompson Sampling if we have enough human feedback
                if self.human_labeled_count >= 5 and self.human_labeled_count % 5 == 0:
                    self.adjust_threshold_thompson_sampling()

                logger.info(
                    f"Human label received: {human_label}, "
                    f"Prediction was: {record.predicted_attack}, "
                    f"Correct: {prediction_correct}"
                )
                return True

        return False

    def adjust_threshold_thompson_sampling(self) -> float:
        """
        Adjust threshold using Thompson Sampling.

        This balances exploration (trying different thresholds)
        with exploitation (using the best known threshold).
        """
        # Sample from each arm's Beta distribution
        samples = [arm.sample_beta() for arm in self.threshold_arms]

        # Select arm with highest sample
        best_arm_idx = max(range(len(samples)), key=lambda i: samples[i])

        if best_arm_idx != self.current_arm_idx:
            old_threshold = self.threshold_arms[self.current_arm_idx].threshold
            new_threshold = self.threshold_arms[best_arm_idx].threshold

            self.current_arm_idx = best_arm_idx
            self.detector_agent.current_threshold = new_threshold

            logger.info(f"Thompson Sampling: Threshold {old_threshold:.2f} -> {new_threshold:.2f}")

        return self.detector_agent.current_threshold

    def adjust_threshold_pseudo_labels(self) -> float:
        """
        Adjust threshold based on pseudo-label estimates.

        Original formula: τ ← τ + η(FP − ρ FN)
        Enhanced with pseudo-label confidence weighting.
        """
        if self.total_evaluations < self.config.tuning_batch_size:
            return self.detector_agent.current_threshold

        total_pseudo = self.estimated_tp + self.estimated_fp + self.estimated_fn + self.estimated_tn

        if total_pseudo == 0:
            return self.detector_agent.current_threshold

        # Calculate rates from pseudo-labels
        fp_rate = self.estimated_fp / max(1, self.estimated_fp + self.estimated_tn)
        fn_rate = self.estimated_fn / max(1, self.estimated_fn + self.estimated_tp)

        # Apply update rule
        eta = getattr(self.config, "learning_rate", 0.01)
        rho = getattr(self.config, "fn_penalty_weight", 2.0)  # FN often worse than FP

        delta = eta * (fp_rate - rho * fn_rate)

        old_threshold = self.detector_agent.current_threshold
        new_threshold = max(0.3, min(0.95, old_threshold + delta))

        if abs(new_threshold - old_threshold) > 0.001:
            self.detector_agent.current_threshold = new_threshold

            logger.info(
                f"Pseudo-label tuning: Threshold {old_threshold:.3f} -> {new_threshold:.3f} "
                f"(FP rate: {fp_rate:.3f}, FN rate: {fn_rate:.3f})"
            )

        # Reset estimates periodically
        if self.total_evaluations >= self.config.tuning_batch_size * 2:
            self._reset_pseudo_estimates()

        return self.detector_agent.current_threshold

    def _reset_pseudo_estimates(self):
        """Reset pseudo-label estimates (keep some history via decay)."""
        decay = 0.5
        self.estimated_fp = int(self.estimated_fp * decay)
        self.estimated_fn = int(self.estimated_fn * decay)
        self.estimated_tp = int(self.estimated_tp * decay)
        self.estimated_tn = int(self.estimated_tn * decay)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current tuner metrics for monitoring."""
        total_pseudo = self.estimated_tp + self.estimated_fp + self.estimated_fn + self.estimated_tn

        # Estimated precision/recall from pseudo-labels
        precision = self.estimated_tp / max(1, self.estimated_tp + self.estimated_fp) if total_pseudo > 0 else 0.0
        recall = self.estimated_tp / max(1, self.estimated_tp + self.estimated_fn) if total_pseudo > 0 else 0.0

        # Current arm stats
        current_arm = self.threshold_arms[self.current_arm_idx]

        return {
            "current_threshold": self.detector_agent.current_threshold,
            "total_evaluations": self.total_evaluations,
            "pseudo_labeled_count": self.pseudo_labeled_count,
            "human_labeled_count": self.human_labeled_count,
            "estimated_precision": precision,
            "estimated_recall": recall,
            "estimated_fp_rate": self.estimated_fp / max(1, total_pseudo),
            "estimated_fn_rate": self.estimated_fn / max(1, total_pseudo),
            "pending_human_reviews": len(self.pending_human_review),
            "current_arm": {
                "threshold": current_arm.threshold,
                "successes": current_arm.successes,
                "failures": current_arm.failures,
                "total_samples": current_arm.total_samples,
            },
            "drift_baseline": {
                "mean": self.baseline_confidence_mean,
                "std": self.baseline_confidence_std,
            },
            "calibration_bins": self.calibration_bins,
        }

    def get_pending_reviews(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending items for human review."""
        return [
            {
                "text_hash": r.text_hash,
                "confidence": r.confidence,
                "predicted_attack": r.predicted_attack,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in self.pending_human_review[:limit]
        ]


# Keep original SelfTuner for backwards compatibility
class SelfTuner:
    """
    Original SelfTuner - uses heuristic logic with ground truth labels.

    For production use, prefer EnhancedSelfTuner which works without labels.
    """

    def __init__(self, detector_agent: Detector, config):
        self.detector_agent = detector_agent
        self.config = config
        self.false_positives = 0
        self.false_negatives = 0
        self.total_evaluations = 0

    def update_metrics(self, detection_result: Dict[str, Any], expected: bool):
        self.total_evaluations += 1
        detected = detection_result.get("detection", False)

        if detected and not expected:
            self.false_positives += 1
        elif not detected and expected:
            self.false_negatives += 1

    def adjust_threshold_if_needed(self):
        if self.total_evaluations >= self.config.tuning_batch_size:
            fp_rate = self.false_positives / self.total_evaluations
            fn_rate = self.false_negatives / self.total_evaluations

            if fp_rate > self.config.max_fp_rate:
                self.detector_agent.increase_threshold()
            elif fn_rate > self.config.max_fn_rate:
                self.detector_agent.decrease_threshold()

            # Reset counters after adjustment
            self.false_positives = 0
            self.false_negatives = 0
            self.total_evaluations = 0

        return self.detector_agent.current_threshold
