# src/honey_prompt_detector/utils/config.py
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class ConfigurationError(Exception):
    """
    Represents an error encountered during configuration processing.

    Attributes:
        message (str): Description of the error encountered.
        config_key (str): Configuration key that caused the error (optional).
    """

    def __init__(self, message: str, config_key: str = None):
        self.message = message
        self.config_key = config_key
        super().__init__(self.__str__())

    def __str__(self):
        base_msg = self.message
        if self.config_key:
            return f"{base_msg} (Config Key: {self.config_key})"
        return base_msg


class Config:
    # Required environment variables with their descriptions and validation functions
    REQUIRED_VARS = {
        'OPENAI_API_KEY': ('OpenAI API key for LLM access', str),
        'LOG_LEVEL': ('Logging level (DEBUG, INFO, WARNING, ERROR)', str),
        'CONFIDENCE_THRESHOLD': ('Minimum confidence for attack detection', float),
        'CONTEXT_WINDOW_SIZE': ('Number of characters around token to analyze', int),
        'SYSTEM_CONTEXT': ('System context for token generation', str)
    }

    # Optional variables with their default values and types
    OPTIONAL_VARS = {
        'MODEL_NAME': ('gpt-4o-mini', str),
        'TEMPERATURE': ('0.2', float),
        'MAX_TOKENS': ('150', int),
        'LOG_FILE': (f'logs/honey_prompt_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', str),
        'METRICS_FILE': (f'metrics/detection_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', str),
        'RETENTION_DAYS': ('30', int),
        'INITIAL_THRESHOLD': ('0.8', float),
    }

    def __init__(self, env_path: Optional[Path] = None):
        """
        Initialize configuration from environment variables.

        Args:
            env_path: Optional path to .env file
        """
        # Load environment variables
        if env_path:
            if not env_path.exists():
                raise ConfigurationError(f"Environment file not found: {env_path}")
            load_dotenv(env_path)
        else:
            load_dotenv()  # Look for .env in current directory

        self._validate_environment()
        self._load_configuration()
        self.similarity_model_name = 'all-MiniLM-L6-v2'

        self.tuning_batch_size = 10

        logger.info("Configuration loaded successfully")

    def _validate_environment(self) -> None:
        """
        Ensure all required environment variables are present and valid.

        Raises:
            ConfigurationError: If any required variable is missing or invalid
        """
        missing_vars = []
        invalid_vars = []

        for var_name, (description, var_type) in self.REQUIRED_VARS.items():
            value = os.getenv(var_name)
            if value is None:
                missing_vars.append(f"{var_name} ({description})")
            else:
                try:
                    var_type(value)
                except ValueError:
                    invalid_vars.append(f"{var_name} (expected {var_type.__name__})")

        if missing_vars or invalid_vars:
            error_msg = []
            if missing_vars:
                error_msg.append("Missing required variables:\n" +
                                 "\n".join(f"- {var}" for var in missing_vars))
            if invalid_vars:
                error_msg.append("Invalid variable types:\n" +
                                 "\n".join(f"- {var}" for var in invalid_vars))
            raise ConfigurationError("\n".join(error_msg))

    def _load_configuration(self) -> None:
        """Load and validate all configuration values."""
        # Load required variables
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.log_level = self._parse_log_level(os.getenv('LOG_LEVEL'))
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD'))
        self.context_window_size = int(os.getenv('CONTEXT_WINDOW_SIZE'))
        self.system_context = os.getenv('SYSTEM_CONTEXT')

        # Load optional variables with defaults
        self.model_name = os.getenv('MODEL_NAME', self.OPTIONAL_VARS['MODEL_NAME'][0])
        self.temperature = float(os.getenv('TEMPERATURE', self.OPTIONAL_VARS['TEMPERATURE'][0]))
        self.max_tokens = int(os.getenv('MAX_TOKENS', self.OPTIONAL_VARS['MAX_TOKENS'][0]))
        log_file_env = os.getenv('LOG_FILE', self.OPTIONAL_VARS['LOG_FILE'][0])
        self.log_file = (PROJECT_ROOT / log_file_env).resolve()
        self.retention_days = int(os.getenv('RETENTION_DAYS', self.OPTIONAL_VARS['RETENTION_DAYS'][0]))
        self.initial_threshold = float(os.getenv('INITIAL_THRESHOLD', self.OPTIONAL_VARS['INITIAL_THRESHOLD'][0]))

        # Explicitly load alert settings
        alert_history_file_env = os.getenv('ALERT_HISTORY_FILE', 'alerts/alert_history.json')
        self.alert_history_file = (PROJECT_ROOT / alert_history_file_env).resolve()

        metrics_file_env = os.getenv('METRICS_FILE', self.OPTIONAL_VARS['METRICS_FILE'][0])
        self.metrics_file = (PROJECT_ROOT / metrics_file_env).resolve()

        self.alert_settings = {
            'alert_history_file': self.alert_history_file,
            'alert_thresholds': {
                'CRITICAL': float(os.getenv('ALERT_CRITICAL_THRESHOLD', '0.9')),
                'HIGH': float(os.getenv('ALERT_HIGH_THRESHOLD', '0.8')),
                'MEDIUM': float(os.getenv('ALERT_MEDIUM_THRESHOLD', '0.7')),
                'LOW': float(os.getenv('ALERT_LOW_THRESHOLD', '0.6')),
            }
        }

        # With email and slack

        # self.alert_settings = {
        #     'email_settings': {
        #         'smtp_server': os.getenv('SMTP_SERVER'),
        #         'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        #         'from_address': os.getenv('EMAIL_FROM'),
        #         'to_address': os.getenv('EMAIL_TO'),
        #         'username': os.getenv('EMAIL_USERNAME'),
        #         'password': os.getenv('EMAIL_PASSWORD'),
        #         'use_tls': os.getenv('EMAIL_USE_TLS', 'true').lower() == 'true'
        #     },
        #     'slack_webhook': os.getenv('SLACK_WEBHOOK'),
        #     'alert_history_file': self.alert_history_file,
        #     'alert_thresholds': {
        #         'CRITICAL': float(os.getenv('ALERT_CRITICAL_THRESHOLD', '0.9')),
        #         'HIGH': float(os.getenv('ALERT_HIGH_THRESHOLD', '0.8')),
        #         'MEDIUM': float(os.getenv('ALERT_MEDIUM_THRESHOLD', '0.7')),
        #         'LOW': float(os.getenv('ALERT_LOW_THRESHOLD', '0.6')),
        #     }
        # }

    def _parse_log_level(self, level_str: str) -> int:
        """Parse and validate logging level."""
        level_mapping = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }

        level_str = level_str.upper()
        if level_str not in level_mapping:
            raise ConfigurationError(
                f"Invalid log level: {level_str}. "
                f"Must be one of: {', '.join(level_mapping.keys())}"
            )

        return level_mapping[level_str]

    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return {
            'openai_api_key': self.openai_api_key,
            'log_level': self.log_level,
            'confidence_threshold': self.confidence_threshold,
            'context_window_size': self.context_window_size,
            'system_context': self.system_context,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'log_file': self.log_file,
            'retention_days': self.retention_days,
            'alert_settings': self.alert_settings,  # Explicitly added
        }