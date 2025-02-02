# src/honey_prompt_detector/dynamic_adaptation.py

import asyncio
import logging
from typing import Any
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)


class DynamicAdaptation:
    """
    Periodically reviews metrics to adjust detection thresholds dynamically.

    This layer monitors the false positive rate and updates the detector's
    confidence threshold to maintain optimal detection performance.
    """

    def __init__(self, metrics: MetricsCollector, detector: Any, adaptation_interval: int = 60):
        """
        Args:
            metrics: MetricsCollector instance.
            detector: Detector instance (or any object with a 'confidence_threshold' attribute).
            adaptation_interval: Time in seconds between adaptations.
        """
        self.metrics = metrics
        self.detector = detector
        self.adaptation_interval = adaptation_interval
        self.running = False

    async def run(self):
        self.running = True
        while self.running:
            await asyncio.sleep(self.adaptation_interval)
            self.adapt_threshold()

    def adapt_threshold(self):
        """
        Adjust the detector's threshold based on the false positive rate.

        For example, if the false positive rate is high, increase the threshold.
        """
        summary = self.metrics.get_summary()
        false_positive_rate = summary.get('false_positive_rate', 0.0)
        logger.debug(f"Dynamic adaptation: false positive rate {false_positive_rate:.2f}")

        if false_positive_rate > 0.2:
            self.detector.confidence_threshold = min(1.0, self.detector.confidence_threshold + 0.05)
            logger.info(f"Increased detector threshold to {self.detector.confidence_threshold:.2f}")
        elif false_positive_rate < 0.05:
            self.detector.confidence_threshold = max(0.0, self.detector.confidence_threshold - 0.05)
            logger.info(f"Decreased detector threshold to {self.detector.confidence_threshold:.2f}")

    def adjust_threshold_based_on_performance(current_threshold: float, detection_rate: float,
                                              target_rate: float = 0.9, step_size: float = 0.05) -> float:
        """
        Adjust the threshold based on detection performance.
        If detection_rate is below target_rate, lower the threshold by step_size.
        If above, optionally raise it slightly.
        """
        if detection_rate < target_rate:
            new_threshold = max(0.5, current_threshold - step_size)
        else:
            new_threshold = min(1.0, current_threshold + (step_size * 0.4))
        return new_threshold


def main():
    import asyncio
    # For testing, we create a dummy MetricsCollector and a dummy detector.
    class DummyDetector:
        def __init__(self):
            self.confidence_threshold = 0.8

    dummy_detector = DummyDetector()

    # Create a dummy metrics collector with a method get_summary.
    class DummyMetricsCollector:
        def get_summary(self):
            return {'false_positive_rate': 0.25}

    dummy_metrics = DummyMetricsCollector()

    adaptation = DynamicAdaptation(dummy_metrics, dummy_detector, adaptation_interval=2)

    async def run_adaptation():
        # Run adaptation for a few cycles.
        task = asyncio.create_task(adaptation.run())
        await asyncio.sleep(7)
        adaptation.running = False
        await task
        print(f"Final threshold: {dummy_detector.confidence_threshold:.2f}")

    asyncio.run(run_adaptation())


if __name__ == "__main__":
    main()
