# src/honey_prompt_detector/monitoring/metrics.py
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json
from pathlib import Path


class MetricsCollector:
    def __init__(self, metrics_file: Path = None):
        """
        Initialize the metrics collector.

        Args:
            metrics_file: Optional path to store metrics data
        """
        self.metrics_file = metrics_file
        self.reset_metrics()

    def reset_metrics(self):
        """Initialize or reset all metrics counters and storage."""
        self.metrics = {
            'detections': {
                'total': 0,
                'by_type': defaultdict(int),
                'by_confidence': defaultdict(int),
                'false_positives': 0
            },
            'performance': {
                'avg_response_time': 0.0,
                'total_requests': 0,
                'errors': 0
            },
            'patterns': {
                'common_contexts': defaultdict(int),
                'time_distribution': defaultdict(int),
                'token_effectiveness': defaultdict(float)
            },
            'system_health': {
                'last_error': None,
                'error_count': 0,
                'last_checkpoint': datetime.now().isoformat()
            }
        }

    def record_detection(self, detection_info: Dict[str, Any]):
        """
        Record information about a detection event.

        This method updates various metrics based on the detection details,
        helping build a picture of system performance and attack patterns.
        """
        self.metrics['detections']['total'] += 1

        # Record detection type
        if 'match_type' in detection_info:
            self.metrics['detections']['by_type'][detection_info['match_type']] += 1

        # Record confidence level
        if 'confidence' in detection_info:
            confidence_bucket = round(detection_info['confidence'] * 10) / 10
            self.metrics['detections']['by_confidence'][str(confidence_bucket)] += 1

        # Record context patterns if available
        if 'context' in detection_info:
            context_summary = self._summarize_context(detection_info['context'])
            self.metrics['patterns']['common_contexts'][context_summary] += 1

        # Record timing information
        hour = datetime.now().hour
        self.metrics['patterns']['time_distribution'][hour] += 1

    def record_performance(self, response_time: float, is_error: bool = False):
        """
        Record performance metrics for a detection operation.

        Tracks response times and error rates to help monitor system health
        and identify performance issues.
        """
        total = self.metrics['performance']['total_requests']
        avg_time = self.metrics['performance']['avg_response_time']

        # Update running average of response time
        self.metrics['performance']['avg_response_time'] = (
                (avg_time * total + response_time) / (total + 1)
        )
        self.metrics['performance']['total_requests'] += 1

        if is_error:
            self.metrics['performance']['errors'] += 1
            self.metrics['system_health']['error_count'] += 1
            self.metrics['system_health']['last_error'] = datetime.now().isoformat()

    def record_false_positive(self, detection_info: Dict[str, Any]):
        """Record and analyze false positive detections."""
        self.metrics['detections']['false_positives'] += 1

        # Update token effectiveness
        if 'token' in detection_info:
            token = detection_info['token']
            current_effectiveness = self.metrics['patterns']['token_effectiveness'][token]
            # Decrease effectiveness score for tokens that generate false positives
            self.metrics['patterns']['token_effectiveness'][token] = max(
                0.0,
                current_effectiveness - 0.1
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of current metrics.

        Returns a comprehensive overview of system performance,
        detection patterns, and health indicators.
        """
        total_detections = self.metrics['detections']['total']
        false_positive_rate = (
            self.metrics['detections']['false_positives'] / total_detections
            if total_detections > 0 else 0
        )

        return {
            'detection_rate': self.calculate_detection_rate(),
            'false_positive_rate': false_positive_rate,
            'avg_response_time': self.metrics['performance']['avg_response_time'],
            'error_rate': self.calculate_error_rate(),
            'most_common_patterns': self.get_common_patterns(limit=5),
            'system_health': self.get_health_status()
        }

    def calculate_detection_rate(self) -> float:
        """Calculate the overall detection rate."""
        total_requests = self.metrics['performance']['total_requests']
        if total_requests == 0:
            return 0.0
        return self.metrics['detections']['total'] / total_requests

    def calculate_error_rate(self) -> float:
        """Calculate the system error rate."""
        total_requests = self.metrics['performance']['total_requests']
        if total_requests == 0:
            return 0.0
        return self.metrics['performance']['errors'] / total_requests

    def get_common_patterns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most common attack patterns observed."""
        patterns = sorted(
            self.metrics['patterns']['common_contexts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {'pattern': p[0], 'count': p[1]}
            for p in patterns[:limit]
        ]

    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        return {
            'status': 'healthy' if self.calculate_error_rate() < 0.1 else 'degraded',
            'last_error': self.metrics['system_health']['last_error'],
            'error_count': self.metrics['system_health']['error_count']
        }

    def _summarize_context(self, context: str) -> str:
        """Create a summary of detection context for pattern analysis."""
        # Simplified context summary - could be enhanced with NLP
        return context[:50].strip()

    def save_metrics(self):
        """Save current metrics to file if configured."""
        if self.metrics_file:
            # Ensure the parent directory exists
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)

    def load_metrics(self):
        """Load metrics from file if available."""
        if self.metrics_file and self.metrics_file.exists():
            with open(self.metrics_file) as f:
                saved_metrics = json.load(f)
                # Convert defaultdict entries
                saved_metrics['detections']['by_type'] = defaultdict(
                    int, saved_metrics['detections']['by_type']
                )
                saved_metrics['patterns']['common_contexts'] = defaultdict(
                    int, saved_metrics['patterns']['common_contexts']
                )
                self.metrics = saved_metrics

    def record_system_start(self):
        """Record that the system has started."""
        # Add any logic you want—e.g., increment a 'starts' count in the metrics dict
        if 'starts' not in self.metrics['system_health']:
            self.metrics['system_health']['starts'] = 0
        self.metrics['system_health']['starts'] += 1
