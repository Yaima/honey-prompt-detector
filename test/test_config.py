"""
Unit tests for the Config class.
"""

import os
from unittest.mock import patch

import pytest

from src.honey_prompt_detector.utils.config import Config, ConfigurationError


class TestConfig:
    """Test suite for the Config class."""

    @pytest.fixture
    def valid_env_vars(self):
        """Provide valid environment variables for testing."""
        return {
            "OPENAI_API_KEY": "test-api-key-12345",
            "LOG_LEVEL": "INFO",
            "CONFIDENCE_THRESHOLD": "0.8",
            "CONTEXT_WINDOW_SIZE": "100",
            "SYSTEM_CONTEXT": "Test system context",
            "MODEL_NAME": "gpt-4o-mini",
            "TEMPERATURE": "0.2",
            "MAX_TOKENS": "150",
        }

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_required_variables(self):
        """Test that missing required variables raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            Config()
        assert "Missing required variables" in str(exc_info.value)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=True)
    def test_partial_missing_variables(self):
        """Test that partially missing variables raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            Config()
        assert "Missing required variables" in str(exc_info.value)

    @patch.dict(os.environ, clear=True)
    def test_invalid_variable_type(self, valid_env_vars):
        """Test that invalid variable types raise ConfigurationError."""
        valid_env_vars["CONFIDENCE_THRESHOLD"] = "not_a_number"
        with patch.dict(os.environ, valid_env_vars):
            with pytest.raises(ConfigurationError) as exc_info:
                Config()
            assert "Invalid variable types" in str(exc_info.value)

    @patch.dict(os.environ, clear=True)
    def test_valid_configuration(self, valid_env_vars):
        """Test successful configuration loading."""
        with patch.dict(os.environ, valid_env_vars):
            config = Config()
            assert config.openai_api_key == "test-api-key-12345"
            assert config.confidence_threshold == 0.8
            assert config.context_window_size == 100
            assert config.system_context == "Test system context"

    @patch.dict(os.environ, clear=True)
    def test_optional_variables_defaults(self, valid_env_vars):
        """Test that optional variables use default values."""
        with patch.dict(os.environ, valid_env_vars):
            config = Config()
            assert config.model_name == "gpt-4o-mini"
            assert config.temperature == 0.2
            assert config.max_tokens == 150
            assert config.initial_threshold == 0.8

    @patch.dict(os.environ, clear=True)
    def test_invalid_log_level(self, valid_env_vars):
        """Test that invalid log level raises ConfigurationError."""
        valid_env_vars["LOG_LEVEL"] = "INVALID"
        with patch.dict(os.environ, valid_env_vars):
            with pytest.raises(ConfigurationError) as exc_info:
                Config()
            assert "Invalid log level" in str(exc_info.value)

    @patch.dict(os.environ, clear=True)
    def test_valid_log_levels(self, valid_env_vars):
        """Test that all valid log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        for level in valid_levels:
            valid_env_vars["LOG_LEVEL"] = level
            with patch.dict(os.environ, valid_env_vars):
                config = Config()
                assert config.log_level is not None

    @patch.dict(os.environ, clear=True)
    def test_alert_settings_loaded(self, valid_env_vars):
        """Test that alert settings are properly loaded."""
        with patch.dict(os.environ, valid_env_vars):
            config = Config()
            assert "alert_thresholds" in config.alert_settings
            assert config.alert_settings["alert_thresholds"]["CRITICAL"] == 0.9
            assert config.alert_settings["alert_thresholds"]["HIGH"] == 0.8

    @patch.dict(os.environ, clear=True)
    def test_custom_alert_thresholds(self, valid_env_vars):
        """Test that custom alert thresholds can be set."""
        valid_env_vars["ALERT_CRITICAL_THRESHOLD"] = "0.95"
        valid_env_vars["ALERT_HIGH_THRESHOLD"] = "0.85"
        with patch.dict(os.environ, valid_env_vars):
            config = Config()
            assert config.alert_settings["alert_thresholds"]["CRITICAL"] == 0.95
            assert config.alert_settings["alert_thresholds"]["HIGH"] == 0.85

    @patch.dict(os.environ, clear=True)
    def test_as_dict_method(self, valid_env_vars):
        """Test that as_dict returns all configuration values."""
        with patch.dict(os.environ, valid_env_vars):
            config = Config()
            config_dict = config.as_dict()

            assert "openai_api_key" in config_dict
            assert "confidence_threshold" in config_dict
            assert "alert_settings" in config_dict
            assert config_dict["model_name"] == "gpt-4o-mini"

    @patch.dict(os.environ, clear=True)
    def test_similarity_model_name(self, valid_env_vars):
        """Test that similarity model name is set."""
        with patch.dict(os.environ, valid_env_vars):
            config = Config()
            assert config.similarity_model_name == "all-MiniLM-L6-v2"

    @patch.dict(os.environ, clear=True)
    def test_tuning_batch_size(self, valid_env_vars):
        """Test that tuning batch size is set."""
        with patch.dict(os.environ, valid_env_vars):
            config = Config()
            assert config.tuning_batch_size == 10
