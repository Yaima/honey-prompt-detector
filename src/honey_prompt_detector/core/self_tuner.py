from typing import Dict, Any

from src.honey_prompt_detector.core.detector import Detector


class SelfTuner:
    """
    Uses heuristic logic to adjust thresholds based on false positives/negatives metrics, not direct AI usage.
    """
    def __init__(self, detector_agent: Detector, config):
        self.detector_agent = detector_agent
        self.config = config
        self.false_positives = 0
        self.false_negatives = 0
        self.total_evaluations = 0

    def update_metrics(self, detection_result: Dict[str, Any], expected: bool):
        self.total_evaluations += 1
        detected = detection_result.get('detection', False)

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
