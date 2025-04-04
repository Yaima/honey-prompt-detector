# src/honey_prompt_detector/core/honey_prompt.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
import hashlib


@dataclass
class HoneyPrompt:
    base_token: str
    category: str
    sensitivity: float
    context: str
    variations: List[str] = field(default_factory=list)
    detection_rules: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate honey-prompt attributes and setup derived properties."""
        if not 0.0 <= self.sensitivity <= 1.0:
            raise ValueError("Sensitivity must be between 0.0 and 1.0")
        if not self.base_token:
            raise ValueError("Base token cannot be empty")

        # Create a secure hash of the token for internal reference
        self.token_hash = self._create_token_hash()

        # Initialize default detection rules if none provided
        if not self.detection_rules:
            self.detection_rules = {
                'exact_match_weight': 1.0,
                'variation_match_weight': 0.8,
                'context_importance': 0.7,
                'minimum_confidence': 0.6
            }

    def _create_token_hash(self) -> str:
        """Create a secure hash of the token for internal reference."""
        return hashlib.sha256(self.base_token.encode()).hexdigest()[:16]

    def matches_text(self, text: str, context: str = None) -> Dict[str, Any]:
        """
        Check if this honey-prompt appears in the given text.

        Args:
            text: The text to check for token appearances
            context: Optional surrounding context for context-aware matching

        Returns:
            Dict containing match details including confidence score and match type
        """
        # Check for exact match first
        if self.base_token in text:
            confidence = self.detection_rules['exact_match_weight']
            match_type = 'exact'
        else:
            # Check for variations
            for variation in self.variations:
                if variation in text:
                    confidence = self.detection_rules['variation_match_weight']
                    match_type = 'variation'
                    break
            else:
                return {'matched': False, 'confidence': 0.0}

        # Adjust confidence based on context if provided
        if context and self.context:
            context_similarity = self._evaluate_context(context)
            confidence *= (1.0 + (context_similarity - 0.5) *
                           self.detection_rules['context_importance'])

        return {
            'matched': confidence >= self.detection_rules['minimum_confidence'],
            'confidence': min(1.0, confidence),
            'match_type': match_type,
            'token': self.base_token,
            'timestamp': datetime.now()
        }

    def _evaluate_context(self, context: str) -> float:
        """
        Evaluate how well the surrounding context matches expected usage.

        Returns:
            float: Similarity score between 0 and 1
        """
        # This is a simplified context evaluation
        # In a full implementation, you might use more sophisticated
        # text similarity metrics or embeddings
        return 0.5  # Default neutral context similarity