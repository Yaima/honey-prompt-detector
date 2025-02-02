# src/honey_prompt_detector/monitoring/alerts.py

from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
import aiohttp
import asyncio

# Import aiosmtplib for asynchronous email sending.
try:
    from aiosmtplib import send as async_send
except ImportError:
    async_send = None  # Fallback: if not installed, email alerts may not work asynchronously

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alerting mechanisms for detecting and responding to potential
    prompt injection attacks.

    Handles the processes of generating, sending, and recording alerts
    based on provided detection data and configurable channels such as
    email and Slack.

    Attributes:
      config: Configuration dictionary containing alert settings (email, Slack, etc.)
      alert_history: List of previously recorded alert messages.
      alert_history_file: File path to save and load alert history.
    """

    ALERT_LEVELS = {
        'CRITICAL': {'threshold': 0.9, 'color': '#FF0000'},
        'HIGH': {'threshold': 0.8, 'color': '#FFA500'},
        'MEDIUM': {'threshold': 0.7, 'color': '#FFFF00'},
        'LOW': {'threshold': 0.6, 'color': '#00FF00'}
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alert manager with configuration.

        Args:
            config: Dictionary containing alert configuration:
                - email_settings: SMTP server settings
                - slack_webhook: Slack webhook URL
                - alert_thresholds: Confidence thresholds for different alert levels
                - alert_history_file: Path to store alert history
        """
        self.config = config
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_history_file = Path(config.get('alert_history_file', 'alert_history.json'))
        self._load_alert_history()

    async def send_alert(self, detection_info: Dict[str, Any],
                         additional_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send an alert about a potential prompt injection attack.

        Args:
            detection_info: Information about the detected attack.
            additional_context: Optional additional context about the detection.

        Returns:
            bool: True if the alert was sent successfully, False otherwise.
        """
        try:
            # Determine alert level based on detection confidence.
            alert_level = self._determine_alert_level(detection_info.get('confidence', 0.0))
            alert_message = self._create_alert_message(detection_info, alert_level, additional_context)
            self._record_alert(alert_message, alert_level)

            alert_tasks = []

            # Asynchronous email alert using aiosmtplib (if available and configured)
            if email_settings := self.config.get('email_settings'):
                alert_tasks.append(self._send_email_alert(alert_message, email_settings))

            # Slack alerts via aiohttp.
            if slack_webhook := self.config.get('slack_webhook'):
                alert_tasks.append(self._send_slack_alert(alert_message, slack_webhook, alert_level))

            if alert_tasks:
                await asyncio.gather(*alert_tasks)

            logger.info(
                f"Alert sent successfully - Level: {alert_level}, Confidence: {detection_info.get('confidence', 0.0):.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
            return False

    def _determine_alert_level(self, confidence: float) -> str:
        """Determine appropriate alert level based on detection confidence."""
        for level, settings in self.ALERT_LEVELS.items():
            if confidence >= settings['threshold']:
                return level
        return 'LOW'

    def _create_alert_message(self, detection_info: Dict[str, Any], alert_level: str,
                              additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a structured alert message with all relevant information."""
        timestamp = datetime.now().isoformat()
        message = {
            'alert_id': f"ALERT_{timestamp}_{alert_level}",
            'timestamp': timestamp,
            'alert_level': alert_level,
            'detection_info': detection_info,
            'confidence': detection_info.get('confidence', 0.0),
            'match_type': detection_info.get('match_type', 'unknown'),
            'context': detection_info.get('context', '')
        }
        if additional_context:
            message['additional_context'] = additional_context
        return message

    async def _send_email_alert(self, alert_message: Dict[str, Any], email_settings: Dict[str, Any]) -> None:
        """Send alert via email asynchronously using aiosmtplib."""
        # Use aiosmtplib if available; otherwise, raise an error.
        if async_send is None:
            raise RuntimeError("aiosmtplib is not installed; cannot send email asynchronously.")

        try:
            msg = MIMEText(self._format_alert_for_email(alert_message), 'plain')
            msg['Subject'] = f"[{alert_message['alert_level']}] Prompt Injection Alert - {alert_message['timestamp']}"
            msg['From'] = email_settings['from_address']
            msg['To'] = email_settings['to_address']

            await async_send(
                msg,
                hostname=email_settings['smtp_server'],
                port=email_settings['smtp_port'],
                username=email_settings.get('username'),
                password=email_settings.get('password'),
                start_tls=email_settings.get('use_tls', True)
            )
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            raise

    async def _send_slack_alert(self, alert_message: Dict[str, Any], webhook_url: str, alert_level: str) -> None:
        """Send alert to Slack channel via aiohttp."""
        try:
            slack_message = {
                'attachments': [{
                    'color': self.ALERT_LEVELS[alert_level]['color'],
                    'title': f"Prompt Injection Alert - {alert_level}",
                    'text': self._format_alert_for_slack(alert_message),
                    'fields': [
                        {'title': 'Confidence', 'value': f"{alert_message['confidence']:.2f}", 'short': True},
                        {'title': 'Match Type', 'value': alert_message['match_type'], 'short': True}
                    ]
                }]
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=slack_message) as response:
                    if response.status not in (200, 201):
                        raise RuntimeError(f"Slack API returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            raise

    def _format_alert_for_email(self, alert_message: Dict[str, Any]) -> str:
        """Format alert message for email delivery."""
        return f"""
Prompt Injection Alert

Alert Level: {alert_message['alert_level']}
Timestamp: {alert_message['timestamp']}
Confidence: {alert_message['confidence']:.2f}

Detection Details:
- Match Type: {alert_message['match_type']}
- Context: {alert_message['context']}

{alert_message.get('additional_context', '')}

This is an automated alert from the Honey-Prompt Detection System.
"""

    def _format_alert_for_slack(self, alert_message: Dict[str, Any]) -> str:
        """Format alert message for Slack delivery."""
        return (
            f"*Prompt Injection Detected*\n\n"
            f"*Level:* {alert_message['alert_level']}\n"
            f"*Confidence:* {alert_message['confidence']:.2f}\n"
            f"*Match Type:* {alert_message['match_type']}\n\n"
            f"*Context:*\n```{alert_message['context']}```"
        )

    def _record_alert(self, alert_message: Dict[str, Any], alert_level: str) -> None:
        """Record alert details in history and save to file."""
        self.alert_history.append({
            'timestamp': alert_message['timestamp'],
            'level': alert_level,
            'message': alert_message
        })
        # Keep only the last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        self._save_alert_history()

    def _save_alert_history(self) -> None:
        """Save the alert history to file."""
        try:
            with open(self.alert_history_file, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert history: {str(e)}")

    def _load_alert_history(self) -> None:
        """Load alert history from file if it exists."""
        try:
            if self.alert_history_file.exists():
                with open(self.alert_history_file) as f:
                    self.alert_history = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load alert history: {str(e)}")
            self.alert_history = []

    async def get_recent_alerts(self, limit: int = 10, min_level: str = 'LOW') -> List[Dict[str, Any]]:
        """
        Retrieve recent alerts filtered by minimum alert level.

        Args:
            limit: Maximum number of alerts to return.
            min_level: Minimum alert level (e.g., 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL').

        Returns:
            A list of recent alert dictionaries.
        """
        min_threshold = self.ALERT_LEVELS[min_level]['threshold']
        filtered_alerts = [
            alert for alert in self.alert_history
            if self.ALERT_LEVELS[alert['level']]['threshold'] >= min_threshold
        ]
        return sorted(filtered_alerts, key=lambda x: x['timestamp'], reverse=True)[:limit]
