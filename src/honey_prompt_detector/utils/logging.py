# src/honey_prompt_detector/utils/logging.py
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class SecurityFormatter(logging.Formatter):
    """
    Custom logging formatter for enhanced log detail and tracking precision.

    This class extends the functionality of the base logging formatter by adding
    additional fields to log records, such as timestamps with microsecond precision,
    process and thread IDs, and source code location details. These enhancements
    are particularly useful for debugging and analyzing logs in applications
    with concurrent operations.

    :ivar default_msec_format: Default format for representing milliseconds in a log.
    :type default_msec_format: str
    """

    def format(self, record):
        # Add timestamp with microsecond precision
        record.created_fmt = datetime.fromtimestamp(record.created).isoformat()

        # Add process and thread IDs for tracking concurrent operations
        record.process_thread = f"[{record.process}:{record.thread}]"

        # Add source code location
        record.source_location = f"{record.filename}:{record.lineno}"

        return super().format(record)


def setup_logger(
        name: str,
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        retention_days: int = 30
) -> logging.Logger:
    """
    Configure a logger with security-focused formatting and optional file output.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to log file
        level: Logging level
        retention_days: Number of days to retain log files

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create security-focused formatter
    formatter = SecurityFormatter(
        '%(created_fmt)s %(process_thread)s %(levelname)s '
        '[%(source_location)s] %(name)s: %(message)s'
    )

    # Console handler with color coding for different levels
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Set up log rotation
        from logging.handlers import TimedRotatingFileHandler
        rotating_handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=retention_days
        )
        rotating_handler.setFormatter(formatter)
        logger.addHandler(rotating_handler)

    return logger


# Example security-relevant log messages
SECURITY_LOG_MESSAGES = {
    'token_generation': 'Generated new honey-prompt token: {token_hash}',
    'token_detection': 'Detected honey-prompt token: {token_hash} (confidence: {confidence})',
    'attack_detected': 'ALERT: Potential prompt injection attack detected! Confidence: {confidence}',
    'false_positive': 'False positive detection recorded for token: {token_hash}',
    'system_startup': 'Honey-prompt detection system initialized with {num_tokens} tokens',
    'api_error': 'API error during {operation}: {error}',
    'config_error': 'Configuration error: {error}',
}