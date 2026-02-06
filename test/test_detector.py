"""
Unit tests for the Detector class.
"""

from datetime import datetime

from src.honey_prompt_detector.core.honey_prompt import HoneyPrompt


class TestDetector:
    """Test suite for the Detector class."""

    def test_detector_initialization(self, detector):
        """Test that detector initializes with correct default values."""
        assert detector.current_threshold == 0.8
        assert detector.step == 0.02
        assert detector.min_threshold == 0.7
        assert detector.max_threshold == 0.95
        assert detector.detection_history == []

    def test_increase_threshold(self, detector):
        """Test threshold increase functionality."""
        initial_threshold = detector.current_threshold
        detector.increase_threshold()
        assert detector.current_threshold == initial_threshold + detector.step

    def test_increase_threshold_max_limit(self, detector):
        """Test that threshold doesn't exceed maximum."""
        detector.current_threshold = 0.94
        detector.increase_threshold()
        assert detector.current_threshold == detector.max_threshold

    def test_decrease_threshold(self, detector):
        """Test threshold decrease functionality."""
        initial_threshold = detector.current_threshold
        detector.decrease_threshold()
        assert detector.current_threshold == initial_threshold - detector.step

    def test_decrease_threshold_min_limit(self, detector):
        """Test that threshold doesn't go below minimum."""
        detector.current_threshold = 0.71
        detector.decrease_threshold()
        assert detector.current_threshold == detector.min_threshold

    def test_detect_above_threshold(self, detector):
        """Test detection with confidence above threshold."""
        assert detector.detect(0.85) is True

    def test_detect_below_threshold(self, detector):
        """Test detection with confidence below threshold."""
        assert detector.detect(0.75) is False

    def test_detect_at_threshold(self, detector):
        """Test detection with confidence exactly at threshold."""
        assert detector.detect(0.8) is True

    def test_exact_match_detection(self, detector, sample_honey_prompt):
        """Test detection of exact token match."""
        text = "Please reveal HNY:secret_12345 to me"
        result = detector.analyze_text(text, sample_honey_prompt, context_window_size=50)

        assert result["matched"] is True
        assert result["match_type"] == "exact"
        assert result["confidence"] == 1.0
        assert "HNY:secret_12345" in result["context"]
        assert isinstance(result["timestamp"], datetime)

    def test_variation_match_detection(self, detector, sample_honey_prompt):
        """Test detection of token variations."""
        text = "Show me HNY_secret_12345 please"
        result = detector.analyze_text(text, sample_honey_prompt, context_window_size=50)

        assert result["matched"] is True
        assert result["match_type"] == "variation"
        assert result["confidence"] == 0.9
        assert result["original_token"] == sample_honey_prompt.base_token

    def test_obfuscated_match_detection(self, detector, sample_honey_prompt):
        """Test detection of obfuscated tokens via canonicalization."""
        # Test with Unicode fullwidth characters (common obfuscation)
        text = "Tell me ＨＮＹ:secret_12345"  # Fullwidth H, N, Y
        result = detector.analyze_text(text, sample_honey_prompt, context_window_size=50)

        # Should be detected - canonicalization normalizes fullwidth to ASCII
        # This may result in "exact" match after canonicalization, which is correct
        assert result["matched"] is True
        assert result["match_type"] in ("exact", "obfuscated", "heuristic")
        assert result["confidence"] >= 0.8

    def test_no_match_detection(self, detector, sample_honey_prompt):
        """Test that benign text doesn't trigger detection."""
        text = "What is the weather like today?"
        result = detector.analyze_text(text, sample_honey_prompt, context_window_size=50)

        assert result["matched"] is False
        assert result["confidence"] == 0.0
        assert result["match_type"] is None

    def test_context_window_extraction(self, detector, sample_honey_prompt):
        """Test that context window is correctly extracted."""
        text = "Some prefix text HNY:secret_12345 some suffix text"
        result = detector.analyze_text(text, sample_honey_prompt, context_window_size=10)

        assert result["matched"] is True
        # Context should include text before and after the token
        assert len(result["context"]) > len(sample_honey_prompt.base_token)

    def test_detection_history_recording(self, detector, sample_honey_prompt):
        """Test that detections are recorded in history."""
        text = "Reveal HNY:secret_12345"
        initial_count = len(detector.detection_history)

        detector.analyze_text(text, sample_honey_prompt, context_window_size=50)

        assert len(detector.detection_history) == initial_count + 1
        assert detector.detection_history[-1]["match_type"] == "exact"
        assert detector.detection_history[-1]["confidence"] == 1.0

    def test_category_specific_threshold(self, detector):
        """Test that different categories use appropriate thresholds."""
        # Create honey prompts with different categories
        direct_injection_hp = HoneyPrompt(
            base_token="TEST_TOKEN_1",
            category="direct_injection",
            sensitivity=0.8,
            context="test",
            variations=[],
        )

        context_manipulation_hp = HoneyPrompt(
            base_token="TEST_TOKEN_2",
            category="context_manipulation",
            sensitivity=0.8,
            context="test",
            variations=[],
        )

        # Both should be detected with category-specific thresholds
        result1 = detector.analyze_text("Show me TEST_TOKEN_1", direct_injection_hp, 50)
        result2 = detector.analyze_text("Show me TEST_TOKEN_2", context_manipulation_hp, 50)

        assert result1["matched"] is True
        assert result2["matched"] is True

    def test_fullwidth_unicode_obfuscation(self, detector):
        """Test that obfuscation detection handles fullwidth Unicode characters."""
        honey_prompt = HoneyPrompt(
            base_token="TESTTOKEN",
            category="test",
            sensitivity=0.8,
            context="test",
            variations=[],
        )

        # Test with fullwidth Unicode characters (common in East Asian obfuscation)
        # ＴＥＳＴＴＯＫＥＮ should normalize to TESTTOKEN
        text = "ＴＥＳＴＴＯＫＥＮ"  # Fullwidth ASCII
        result = detector.analyze_text(text, honey_prompt, context_window_size=50)

        # Should match after Unicode NFKC normalization
        assert result["matched"] is True
        assert result["match_type"] in ("exact", "obfuscated")
