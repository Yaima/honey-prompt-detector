# src/honey_prompt_detector/core/detector.py

from typing import Dict, Any, List
import logging
from datetime import datetime
from .honey_prompt import HoneyPrompt
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class Detector:
    """
    Performs advanced text analysis to detect patterns, including exact,
    variational, and obfuscation honey-prompt matches.

    This class implements complex text detection mechanisms and utilities
    to identify patterns within the given text, focusing on honey-prompt
    patterns and variations. The analysis includes handling obfuscation
    cases and recording detailed detection history for later assessment.
    """

    def __init__(self, confidence_threshold: float = 0.8):
        """
        Initialize the detector with configuration parameters.

        Args:
            confidence_threshold: Minimum confidence score to consider a match.
        """
        self.confidence_threshold = confidence_threshold
        self.detection_history: List[Dict[str, Any]] = []

    def analyze_text(
            self,
            text: str,
            honey_prompt: HoneyPrompt,
            context_window_size: int = 100
    ) -> Dict[str, Any]:
        # Determine a local threshold based on the category
        if honey_prompt.category == "direct_injection":
            local_threshold = 0.70  # Lower threshold for direct injections
        elif honey_prompt.category == "context_manipulation":
            local_threshold = 0.75
        else:
            local_threshold = self.confidence_threshold

        logger.debug(f"Using local threshold: {local_threshold} for category: {honey_prompt.category}")

        # Check for exact token match first
        if honey_prompt.base_token in text:
            match_info = self._analyze_exact_match(text, honey_prompt, context_window_size)
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
            f"Exact match for '{honey_prompt.base_token}' at index {start_index}. Confidence: {confidence}, Context: {surrounding_context}")
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
        """Check for known variations of the honey-prompt."""
        for variation in honey_prompt.variations:
            if variation in text:
                start_index = text.find(variation)
                context_start = max(0, start_index - context_window_size)
                context_end = min(len(text), start_index + len(variation) + context_window_size)
                surrounding_context = text[context_start:context_end]
                logger.debug(f"Variation match '{variation}' found at index {start_index}")
                return {
                    'matched': True,
                    'confidence': 0.9,  # Slightly lower confidence for variations
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
        """
        Check for potential obfuscation attempts.
        This method looks for common techniques used to hide or modify tokens.
        """
        normalized_text = text.lower()
        normalized_token = honey_prompt.base_token.lower()

        # Remove common obfuscation characters
        cleaned_text = ''.join(c for c in normalized_text if c.isalnum() or c.isspace())
        cleaned_token = ''.join(c for c in normalized_token if c.isalnum() or c.isspace())

        if cleaned_token in cleaned_text:
            start_index = cleaned_text.find(cleaned_token)
            context_start = max(0, start_index - context_window_size)
            context_end = min(len(text), start_index + len(cleaned_token) + context_window_size)
            surrounding_context = text[context_start:context_end]
            logger.debug(f"Obfuscation match for token '{honey_prompt.base_token}' found at index {start_index}")
            return {
                'matched': True,
                'confidence': 0.8,  # Lower confidence for obfuscated matches
                'match_type': 'obfuscated',
                'token': honey_prompt.base_token,
                'context': surrounding_context,
                'position': start_index,
                'timestamp': datetime.now()
            }
        return {'matched': False, 'confidence': 0.0}

    def _record_detection(self, detection_info: Dict[str, Any]) -> None:
        """Record detection details for later analysis."""
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
