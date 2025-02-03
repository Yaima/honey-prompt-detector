# src/honey_prompt_detector/utils/validation.py
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


@dataclass
class ValidationResult:
    """
    Represents the result of a validation process.

    This class is used to store the outcome of a validation operation, indicating
    whether it was successful and capturing any errors or warnings that occurred.

    :ivar is_valid: Indicates if the validation process was successful.
    :type is_valid: bool
    :ivar errors: A list of error messages encountered during validation.
    :type errors: List[str]
    :ivar warnings: A list of warning messages encountered during validation.
    :type warnings: List[str]
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class InputValidator:
    """
    Validates inputs for the honey-prompt detector.

    This class provides comprehensive validation for all inputs to ensure
    they meet system requirements and constraints. It helps prevent errors
    and maintain system reliability.
    """

    # Constants for validation rules
    MAX_TEXT_LENGTH = 1000000  # Maximum length of input text
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0
    VALID_ALERT_LEVELS = {'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'}

    @classmethod
    def validate_text_input(cls, text: str) -> ValidationResult:
        """
        Validate text input for analysis.

        This method checks if the input text meets basic requirements:
        - Not empty
        - Within length limits
        - Contains valid characters
        """
        errors = []
        warnings = []

        if not text:
            errors.append("Input text cannot be empty")
            return ValidationResult(False, errors, warnings)

        if len(text) > cls.MAX_TEXT_LENGTH:
            errors.append(
                f"Input text exceeds maximum length of {cls.MAX_TEXT_LENGTH} characters"
            )

        # Check for potentially problematic characters
        if '\x00' in text:
            errors.append("Input text contains null characters")

        if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', text):
            warnings.append("Input text contains control characters")

        return ValidationResult(len(errors) == 0, errors, warnings)

    @classmethod
    def validate_honey_prompt(cls, prompt_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate honey-prompt configuration.

        Checks if the honey-prompt configuration contains all required fields
        and valid values.
        """
        errors = []
        warnings = []

        required_fields = {
            'token': str,
            'category': str,
            'sensitivity': float,
            'context': str
        }

        for field, expected_type in required_fields.items():
            if field not in prompt_data:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(prompt_data[field], expected_type):
                errors.append(
                    f"Invalid type for {field}: expected {expected_type.__name__}, "
                    f"got {type(prompt_data[field]).__name__}"
                )

        if 'sensitivity' in prompt_data:
            sensitivity = prompt_data['sensitivity']
            if not cls.MIN_CONFIDENCE <= sensitivity <= cls.MAX_CONFIDENCE:
                errors.append(
                    f"Sensitivity must be between {cls.MIN_CONFIDENCE} "
                    f"and {cls.MAX_CONFIDENCE}"
                )

        return ValidationResult(len(errors) == 0, errors, warnings)

    @classmethod
    def validate_detection_result(
            cls,
            result: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate detection result structure.

        Ensures the detection result contains all required information
        and valid values.
        """
        errors = []
        warnings = []

        # Check required fields
        required_fields = {
            'detection': bool,
            'confidence': float
        }

        for field, expected_type in required_fields.items():
            if field not in result:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(result[field], expected_type):
                errors.append(
                    f"Invalid type for {field}: expected {expected_type.__name__}, "
                    f"got {type(result[field]).__name__}"
                )

        # Validate confidence score
        if 'confidence' in result:
            confidence = result['confidence']
            if not cls.MIN_CONFIDENCE <= confidence <= cls.MAX_CONFIDENCE:
                errors.append(
                    f"Confidence must be between {cls.MIN_CONFIDENCE} "
                    f"and {cls.MAX_CONFIDENCE}"
                )

        # Check optional fields if present
        if 'match_type' in result and not isinstance(result['match_type'], str):
            errors.append("match_type must be a string")

        if 'explanation' in result and not isinstance(result['explanation'], str):
            errors.append("explanation must be a string")

        return ValidationResult(len(errors) == 0, errors, warnings)

    @classmethod
    def validate_alert_config(cls, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate alert system configuration.

        Checks if the alert configuration contains valid settings for
        all notification channels.
        """
        errors = []
        warnings = []

        # Validate email settings if present
        if email_settings := config.get('email_settings'):
            if not isinstance(email_settings, dict):
                errors.append("email_settings must be a dictionary")
            else:
                required_email_fields = {
                    'smtp_server': str,
                    'smtp_port': int,
                    'from_address': str,
                    'to_address': str
                }

                for field, expected_type in required_email_fields.items():
                    if field not in email_settings:
                        errors.append(f"Missing email setting: {field}")
                    elif not isinstance(email_settings[field], expected_type):
                        errors.append(
                            f"Invalid type for email setting {field}: "
                            f"expected {expected_type.__name__}"
                        )

        # Validate Slack settings if present
        if slack_webhook := config.get('slack_webhook'):
            if not isinstance(slack_webhook, str):
                errors.append("slack_webhook must be a string")
            elif not slack_webhook.startswith(('http://', 'https://')):
                errors.append("slack_webhook must be a valid URL")

        # Validate alert thresholds
        if thresholds := config.get('alert_thresholds'):
            if not isinstance(thresholds, dict):
                errors.append("alert_thresholds must be a dictionary")
            else:
                for level, threshold in thresholds.items():
                    if level not in cls.VALID_ALERT_LEVELS:
                        errors.append(f"Invalid alert level: {level}")
                    if not isinstance(threshold, (int, float)):
                        errors.append(
                            f"Threshold for level {level} must be a number"
                        )
                    elif not cls.MIN_CONFIDENCE <= threshold <= cls.MAX_CONFIDENCE:
                        errors.append(
                            f"Threshold for level {level} must be between "
                            f"{cls.MIN_CONFIDENCE} and {cls.MAX_CONFIDENCE}"
                        )

        return ValidationResult(len(errors) == 0, errors, warnings)

    @classmethod
    def validate_metrics_data(
            cls,
            metrics: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate metrics data structure.

        Ensures metrics data contains valid values and appropriate types
        for all fields.
        """
        errors = []
        warnings = []

        required_sections = {
            'detections': dict,
            'performance': dict,
            'patterns': dict,
            'system_health': dict
        }

        for section, expected_type in required_sections.items():
            if section not in metrics:
                errors.append(f"Missing metrics section: {section}")
            elif not isinstance(metrics[section], expected_type):
                errors.append(
                    f"Invalid type for metrics section {section}: "
                    f"expected {expected_type.__name__}"
                )

        # Validate specific metrics if sections exist
        if 'performance' in metrics:
            perf = metrics['performance']
            if 'avg_response_time' in perf and perf['avg_response_time'] < 0:
                errors.append("Average response time cannot be negative")

        if 'system_health' in metrics:
            health = metrics['system_health']
            if 'error_count' in health and not isinstance(health['error_count'], int):
                errors.append("error_count must be an integer")

        return ValidationResult(len(errors) == 0, errors, warnings)


def validate_file_path(path: Union[str, Path]) -> Path:
    """
    Validate and normalize a file path.

    Args:
        path: File path as string or Path object

    Returns:
        Path object if valid

    Raises:
        ValidationError: If path is invalid or inaccessible
    """
    try:
        path = Path(path).resolve()

        # Check if parent directory exists and is writable
        if not path.parent.exists():
            raise ValidationError(f"Directory does not exist: {path.parent}")
        if not os.access(path.parent, os.W_OK):
            raise ValidationError(f"Directory is not writable: {path.parent}")

        return path

    except Exception as e:
        raise ValidationError(f"Invalid file path: {str(e)}")