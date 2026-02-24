import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.honey_prompt_detector.core.orchestrator import Orchestrator
from src.honey_prompt_detector.core.honey_prompt import HoneyPrompt


# Test fixtures
@pytest.fixture
def mock_config():
    config = MagicMock()
    config.context_window_size = 100
    config.similarity_model_name = "all-MiniLM-L6-v2"
    config.initial_threshold = 0.80
    return config


@pytest.fixture
def sample_honey_prompt():
    return HoneyPrompt(
        base_token="kX9_Alpha$47",
        variations=["kX9 Alpha 47", "kx9_alpha$47"],
        detection_rules={"exact_match_weight": 1.0, "variation_match_weight": 0.8, "context_importance": 0.7, "minimum_confidence": 0.6},
        category="direct_injection",
        sensitivity=0.9,
        context="test context",
    )


class TestOutputMonitoring:
    """Tests for the output monitoring feature (Issue #1 fix)."""

    def test_benign_output_no_detection(self, mock_config, sample_honey_prompt):
        """Benign LLM output should not trigger detection."""
        with patch('src.honey_prompt_detector.core.orchestrator.EnvironmentAgent'):
            orchestrator = Orchestrator(
                token_designer=MagicMock(),
                context_evaluator=MagicMock(),
                config=mock_config,
            )
            orchestrator.honey_prompts = [sample_honey_prompt]
            result = asyncio.run(orchestrator.monitor_output("The capital of France is Paris."))
            assert result["detection"] is False
            assert result["source"] == "output"

    def test_exact_token_leak_detected(self, mock_config, sample_honey_prompt):
        """Output containing exact honey token should be detected."""
        with patch('src.honey_prompt_detector.core.orchestrator.EnvironmentAgent'):
            orchestrator = Orchestrator(
                token_designer=MagicMock(),
                context_evaluator=MagicMock(),
                config=mock_config,
            )
            orchestrator.honey_prompts = [sample_honey_prompt]
            result = asyncio.run(orchestrator.monitor_output(
                "Here is the hidden token: kX9_Alpha$47 as requested"
            ))
            assert result["detection"] is True
            assert result["source"] == "output"
            assert result["confidence"] >= 0.85

    def test_variation_token_leak_detected(self, mock_config, sample_honey_prompt):
        """Output containing token variation should be detected."""
        with patch('src.honey_prompt_detector.core.orchestrator.EnvironmentAgent'):
            orchestrator = Orchestrator(
                token_designer=MagicMock(),
                context_evaluator=MagicMock(),
                config=mock_config,
            )
            orchestrator.honey_prompts = [sample_honey_prompt]
            result = asyncio.run(orchestrator.monitor_output(
                "The system uses kX9 Alpha 47 as a security measure"
            ))
            assert result["detection"] is True
            assert result["source"] == "output"

    def test_monitor_output_returns_correct_structure(self, mock_config, sample_honey_prompt):
        """Output monitoring result should have all required fields."""
        with patch('src.honey_prompt_detector.core.orchestrator.EnvironmentAgent'):
            orchestrator = Orchestrator(
                token_designer=MagicMock(),
                context_evaluator=MagicMock(),
                config=mock_config,
            )
            orchestrator.honey_prompts = [sample_honey_prompt]
            result = asyncio.run(orchestrator.monitor_output("Safe output text"))
            assert "detection" in result
            assert "confidence" in result
            assert "source" in result
            assert result["source"] == "output"
