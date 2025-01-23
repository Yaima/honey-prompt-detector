# src/honey_prompt_detector/utils/config.py
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """
    Represents an error encountered during configuration processing.

    This exception is raised when invalid configurations are detected
    or when there are issues related to misconfiguration during program execution.
    """
    pass


class Config:
    """
    The Config class manages application configuration, primarily sourced from
    environment variables. It validates the presence and type of critical settings,
    assigning default values where necessary. Config ensures seamless access
    to configurations, enhancing maintainability and reliability of the application.

    This class is pivotal for centralizing configuration logic, handling
    missing or invalid variables, and converting them into usable formats. It
    supports customization via optional `.env` file loading.

    :ivar REQUIRED_VARS: Dictionary of required environment variables with their
        descriptions and validation functions.
    :type REQUIRED_VARS: Dict[str, Tuple[str, Type[Any]]]
    :ivar OPTIONAL_VARS: Dictionary of optional environment variables with their
        descriptions, default values, and validation functions.
    :type OPTIONAL_VARS: Dict[str, Tuple[str, Type[Any]]]
    :ivar openai_api_key: OpenAI API key used for accessing LLM functionalities.
    :type openai_api_key: str
    :ivar log_level: Logging level for application logs.
    :type log_level: int
    :ivar confidence_threshold: Minimum confidence value for attack detection.
    :type confidence_threshold: float
    :ivar context_window_size: Number of characters around a token to analyze in
        application interactions.
    :type context_window_size: int
    :ivar system_context: Context for system token generation processes.
    :type system_context: str
    :ivar model_name: Name of the model to be used, defaults to 'gpt-4o'.
    :type model_name: str
    :ivar temperature: Sampling temperature for token generation.
    :type temperature: float
    :ivar max_tokens: Maximum number of tokens that can be generated in a response.
    :type max_tokens: int
    :ivar log_file: File path for saving application log entries.
    :type log_file: Path
    :ivar retention_days: Number of days for retaining logs.
    :type retention_days: int
    """

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
        'MODEL_NAME': ('gpt-4o', str),
        'TEMPERATURE': ('0.2', float),
        'MAX_TOKENS': ('1000', int),
        'LOG_FILE': ('logs/honey_prompt_detector.log', str),
        'RETENTION_DAYS': ('30', int)
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
        self.log_file = Path(os.getenv('LOG_FILE', self.OPTIONAL_VARS['LOG_FILE'][0]))
        self.retention_days = int(os.getenv('RETENTION_DAYS', self.OPTIONAL_VARS['RETENTION_DAYS'][0]))

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
            'retention_days': self.retention_days
        }