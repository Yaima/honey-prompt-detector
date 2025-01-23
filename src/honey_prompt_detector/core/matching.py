# src/honey_prompt_detector/core/matching.py
from typing import Dict, Any, Optional
import re
from difflib import SequenceMatcher
import logging
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class PatternMatcher:
    """
    Provides methods to identify patterns in text using multiple matching strategies.

    This class facilitates pattern matching through exact, fuzzy, and pattern-based
    techniques, enabling detection of obfuscated patterns, as well as case variations.
    The sensitivity of matching operations can also be controlled to prioritize either
    precision or generality based on specific use cases.

    :ivar sensitivity: Sensitivity level for fuzzy and pattern-based matching.
        Values closer to 1.0 enforce stricter matching, while lower values allow
        more lenient matches.
    :type sensitivity: float
    """

    def __init__(self, sensitivity: float = 0.8):
        """
        Initialize the pattern matcher with configurable sensitivity.

        The sensitivity parameter determines how strict the matching should be,
        with higher values requiring closer matches.
        """
        self.sensitivity = sensitivity

    def find_matches(self, text: str, pattern: str) -> Dict[str, Any]:
        """
        Find all instances of a pattern in text using multiple matching strategies.

        This method applies several matching techniques in sequence, starting with
        the most precise and moving to more fuzzy matching if needed. This helps
        catch both exact matches and potential obfuscation attempts.
        """
        # Try exact matching first
        exact_match = self._exact_match(text, pattern)
        if exact_match['found']:
            return exact_match

        # Try fuzzy matching if exact match fails
        fuzzy_match = self._fuzzy_match(text, pattern)
        if fuzzy_match['found'] and fuzzy_match['confidence'] >= self.sensitivity:
            return fuzzy_match

        # Try pattern-based matching for potential obfuscation
        pattern_match = self._pattern_match(text, pattern)
        if pattern_match['found']:
            return pattern_match

        return {'found': False, 'confidence': 0.0}

    def _exact_match(self, text: str, pattern: str) -> Dict[str, Any]:
        """
        Perform exact string matching with case sensitivity handling.

        Returns both case-sensitive and case-insensitive match information
        to help evaluate the confidence of the match.
        """
        # Check for exact, case-sensitive match
        if pattern in text:
            return {
                'found': True,
                'confidence': 1.0,
                'match_type': 'exact',
                'positions': [m.start() for m in re.finditer(re.escape(pattern), text)]
            }

        # Check for case-insensitive match
        pattern_lower = pattern.lower()
        text_lower = text.lower()
        if pattern_lower in text_lower:
            return {
                'found': True,
                'confidence': 0.9,  # Slightly lower confidence for case-insensitive
                'match_type': 'case_insensitive',
                'positions': [m.start() for m in re.finditer(re.escape(pattern_lower), text_lower)]
            }

        return {'found': False, 'confidence': 0.0}

    def _fuzzy_match(self, text: str, pattern: str) -> Dict[str, Any]:
        """
        Perform fuzzy string matching to catch near-matches and variations.

        Uses sequence matching to find similarities even when characters are
        changed, added, or removed. This helps catch attempts to obfuscate
        the pattern.
        """
        best_ratio = 0.0
        best_match = None

        # Slide a window of similar size to the pattern through the text
        window_size = len(pattern) * 2
        for i in range(len(text) - len(pattern) + 1):
            window = text[i:i + window_size]
            ratio = SequenceMatcher(None, pattern.lower(), window.lower()).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = window

        if best_ratio >= self.sensitivity:
            return {
                'found': True,
                'confidence': best_ratio,
                'match_type': 'fuzzy',
                'matched_text': best_match
            }

        return {'found': False, 'confidence': best_ratio}

    def _pattern_match(self, text: str, pattern: str) -> Dict[str, Any]:
        """
        Perform pattern-based matching to catch obfuscation attempts.

        This method looks for common obfuscation techniques like:
        - Added spaces or special characters
        - Unicode character substitution
        - Mixed case patterns
        - Character repetition
        """
        # Create a pattern that allows for common obfuscation techniques
        flexible_pattern = ''
        for char in pattern:
            # Allow optional spaces or special characters between each character
            flexible_pattern += re.escape(char) + r'[\s\W_]*'

        matches = list(re.finditer(flexible_pattern, text, re.IGNORECASE))

        if matches:
            # Calculate confidence based on how much obfuscation was detected
            original_len = len(pattern)
            matched_text = matches[0].group(0)
            obfuscation_ratio = original_len / len(matched_text)

            return {
                'found': True,
                'confidence': 0.7 * obfuscation_ratio,  # Lower confidence for obfuscated matches
                'match_type': 'pattern',
                'matched_text': matched_text,
                'positions': [m.start() for m in matches]
            }

        return {'found': False, 'confidence': 0.0}