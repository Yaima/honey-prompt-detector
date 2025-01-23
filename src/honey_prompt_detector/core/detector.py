# src/honey_prompt_detector/core/detector.py
from typing import Dict, Any, List, Optional
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

    :ivar confidence_threshold: The minimum confidence score to consider a match.
    :type confidence_threshold: float
    :ivar detection_history: A list storing detection details for analysis.
    :type detection_history: List[Dict[str, Any]]
    """

    def __init__(self, confidence_threshold: float = 0.8):
        """
        Initialize the detector with configuration parameters.

        Args:
            confidence_threshold: Minimum confidence score to consider a match
        """
        self.confidence_threshold = confidence_threshold
        self.detection_history: List[Dict[str, Any]] = []

    def analyze_text(
            self,
            text: str,
            honey_prompt: HoneyPrompt,
            context_window_size: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze text for potential honey-prompt matches.

        This method implements the core detection logic, looking for both exact
        and approximate matches while considering context and patterns.

        Args:
            text: The text to analyze
            honey_prompt: The honey-prompt to look for
            context_window_size: Size of context window around matches

        Returns:
            Dict containing analysis results including match details and confidence
        """
        # Check for exact token match first
        if honey_prompt.base_token in text:
            match_info = self._analyze_exact_match(
                text,
                honey_prompt,
                context_window_size
            )
            if match_info['confidence'] >= self.confidence_threshold:
                self._record_detection(match_info)
                return match_info

        # Check for variations if no exact match found
        variation_match = self._check_variations(
            text,
            honey_prompt,
            context_window_size
        )
        if variation_match['matched']:
            self._record_detection(variation_match)
            return variation_match

        # Check for potential obfuscation attempts
        obfuscation_match = self._check_obfuscation(
            text,
            honey_prompt,
            context_window_size
        )
        if obfuscation_match['matched']:
            self._record_detection(obfuscation_match)
            return obfuscation_match

        return {
            'matched': False,
            'confidence': 0.0,
            'match_type': None
        }

    def _analyze_exact_match(
            self,
            text: str,
            honey_prompt: HoneyPrompt,
            context_window_size: int
    ) -> Dict[str, Any]:
        """Analyze an exact token match and its context."""
        # Find all occurrences of the token
        start_index = text.find(honey_prompt.base_token)

        # Extract surrounding context
        context_start = max(0, start_index - context_window_size)
        context_end = min(
            len(text),
            start_index + len(honey_prompt.base_token) + context_window_size
        )
        surrounding_context = text[context_start:context_end]

        return {
            'matched': True,
            'confidence': 1.0,
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

                # Extract surrounding context
                context_start = max(0, start_index - context_window_size)
                context_end = min(
                    len(text),
                    start_index + len(variation) + context_window_size
                )
                surrounding_context = text[context_start:context_end]

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

        This method looks for common techniques used to hide or modify tokens:
        - Character substitution
        - Added spaces or special characters
        - Case manipulation
        - Unicode tricks
        """
        # Convert to lowercase for case-insensitive matching
        normalized_text = text.lower()
        normalized_token = honey_prompt.base_token.lower()

        # Remove common obfuscation characters
        cleaned_text = ''.join(
            c for c in normalized_text
            if c.isalnum() or c.isspace()
        )
        cleaned_token = ''.join(
            c for c in normalized_token
            if c.isalnum() or c.isspace()
        )

        if cleaned_token in cleaned_text:
            # Find approximate position in original text
            start_index = cleaned_text.find(cleaned_token)

            # Extract surrounding context from original text
            context_start = max(0, start_index - context_window_size)
            context_end = min(
                len(text),
                start_index + len(cleaned_token) + context_window_size
            )
            surrounding_context = text[context_start:context_end]

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
        """Record detection details for analysis and pattern recognition."""
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