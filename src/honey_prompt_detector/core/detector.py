from typing import Dict, Any
import logging
from datetime import datetime
from .honey_prompt import HoneyPrompt
from ..utils.logging import setup_logger

logger = logging.getLogger("honey_prompt")


class Detector:
    """
    Performs advanced text analysis to detect patterns, including exact,
    variational, and obfuscation honey-prompt matches.
    """

    def __init__(self,
                 context_evaluator,
                 initial_threshold: float = 0.80,
                 step: float = 0.02,
                 min_threshold: float = 0.70,
                 max_threshold: float = 0.95):
        self.current_threshold = initial_threshold
        self.step = step
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.context_evaluator = context_evaluator
        self.detection_history = []  # Proper initialization of history

    def increase_threshold(self):
        """Increase threshold to reduce false positives."""
        self.current_threshold = min(self.current_threshold + self.step, self.max_threshold)
        logger.info(f"Increased threshold to {self.current_threshold:.2f}")

    def decrease_threshold(self):
        """Decrease threshold to reduce false negatives."""
        self.current_threshold = max(self.current_threshold - self.step, self.min_threshold)
        logger.info(f"Decreased threshold to {self.current_threshold:.2f}")

    def detect(self, confidence_score: float) -> bool:
        """Make detection decision based on current threshold."""
        return confidence_score >= self.current_threshold

    def analyze_text(
            self,
            text: str,
            honey_prompt: HoneyPrompt,
            context_window_size: int = 100
    ) -> Dict[str, Any]:
        # Determine a local threshold based on the category
        if honey_prompt.category == "direct_injection":
            local_threshold = 0.70
        elif honey_prompt.category == "context_manipulation":
            local_threshold = 0.75
        else:
            local_threshold = self.current_threshold

        logger.debug(f"Using local threshold: {local_threshold} for category: {honey_prompt.category}")

        # Check for exact token match first
        if honey_prompt.base_token in text:
            match_info = self._analyze_exact_match(text, honey_prompt, context_window_size)

            if match_info['confidence'] < local_threshold and honey_prompt.context:
                adjusted_confidence = self.context_evaluator.adjust_confidence(
                    match_info['confidence'],
                    match_info.get('context', ''),
                    honey_prompt.context
                )
                match_info['confidence'] = adjusted_confidence
                logger.debug(f"Adjusted confidence using semantic similarity: {adjusted_confidence}")

            if match_info['confidence'] >= local_threshold:
                self._record_detection(match_info)
                return match_info

        # Check for variations
        variation_match = self._check_variations(text, honey_prompt, context_window_size)
        if variation_match['matched'] and variation_match['confidence'] >= local_threshold:
            self._record_detection(variation_match)
            return variation_match

        # Check for obfuscation attempts
        obfuscation_match = self._check_obfuscation(text, honey_prompt, context_window_size)
        if obfuscation_match['matched'] and obfuscation_match['confidence'] >= local_threshold:
            self._record_detection(obfuscation_match)
            return obfuscation_match

        return {
            'matched': False,
            'confidence': 0.0,
            'match_type': None
        }

    def _analyze_exact_match(self, text: str, honey_prompt: HoneyPrompt, context_window_size: int) -> Dict[str, Any]:
        start_index = text.find(honey_prompt.base_token)
        context_start = max(0, start_index - context_window_size)
        context_end = min(len(text), start_index + len(honey_prompt.base_token) + context_window_size)
        surrounding_context = text[context_start:context_end]
        confidence = 1.0
        logger.debug(
            f"Exact match for '{honey_prompt.base_token}' at index {start_index}. "
            f"Confidence: {confidence}, Context: {surrounding_context}"
        )
        return {
            'matched': True,
            'confidence': confidence,
            'match_type': 'exact',
            'token': honey_prompt.base_token,
            'context': surrounding_context,
            'position': start_index,
            'timestamp': datetime.now()
        }

    def _check_variations(
            self,
            text: str,
            honey_prompt: HoneyPrompt,
            context_window_size: int
    ) -> Dict[str, Any]:
        for variation in honey_prompt.variations:
            if variation in text:
                start_index = text.find(variation)
                context_start = max(0, start_index - context_window_size)
                context_end = min(len(text), start_index + len(variation) + context_window_size)
                surrounding_context = text[context_start:context_end]
                logger.debug(f"Variation match '{variation}' found at index {start_index}")
                return {
                    'matched': True,
                    'confidence': 0.9,
                    'match_type': 'variation',
                    'token': variation,
                    'original_token': honey_prompt.base_token,
                    'context': surrounding_context,
                    'position': start_index,
                    'timestamp': datetime.now()
                }
        return {'matched': False, 'confidence': 0.0}

    def _check_obfuscation(
            self,
            text: str,
            honey_prompt: HoneyPrompt,
            context_window_size: int
    ) -> Dict[str, Any]:
        normalized_text = ''.join(c.lower() for c in text if c.isalnum() or c.isspace())
        normalized_token = ''.join(c.lower() for c in honey_prompt.base_token if c.isalnum() or c.isspace())

        if normalized_token in normalized_text:
            start_index = normalized_text.find(normalized_token)
            context_start = max(0, start_index - context_window_size)
            context_end = min(len(text), start_index + len(normalized_token) + context_window_size)
            surrounding_context = text[context_start:context_end]
            logger.debug(f"Obfuscation match for token '{honey_prompt.base_token}' at index {start_index}")
            return {
                'matched': True,
                'confidence': 0.8,
                'match_type': 'obfuscated',
                'token': honey_prompt.base_token,
                'context': surrounding_context,
                'position': start_index,
                'timestamp': datetime.now()
            }
        return {'matched': False, 'confidence': 0.0}

    def _record_detection(self, detection_info: Dict[str, Any]) -> None:
        self.detection_history.append({
            'timestamp': detection_info['timestamp'],
            'match_type': detection_info['match_type'],
            'confidence': detection_info['confidence'],
            'token': detection_info.get('token'),
            'context': detection_info.get('context')
        })
        logger.warning(
            f"Detection recorded - Type: {detection_info['match_type']}, "
            f"Confidence: {detection_info['confidence']:.2f}"
        )
