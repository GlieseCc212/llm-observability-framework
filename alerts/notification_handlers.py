"""
Notification Handlers for LLM Observability Framework

Different notification channels for sending alerts including:
- Email notifications
- Slack notifications
- Webhook notifications
- SMS notifications (optional)
"""

import smtplib
import json
import logging
import requests
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .alert_manager import AlertEvent


class NotificationHandler(ABC):
    """Abstract base class for notification handlers"""
    
    @abstractmethod
    def send_notification(self, alert_event: AlertEvent) -> None:
        """Send notification for alert event"""
        pass


@dataclass
class EmailConfig:
    """Email configuration"""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_address: str
    use_tls: bool = True


class EmailNotifier(NotificationHandler):
    """Email notification handler"""
    
    def __init__(self, config: EmailConfig, recipients: list):
        """
        Initialize EmailNotifier
        
        Args:
            config: EmailConfig object with SMTP settings
            recipients: List of email addresses to send alerts to
        """
        self.config = config
        self.recipients = recipients
        self.logger = logging.getLogger(__name__)
    
    def send_notification(self, alert_event: AlertEvent) -> None:
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.from_address
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[{alert_event.severity.upper()}] LLM Alert: {alert_event.alert_type}"
            
            # Create email body
            body = self._create_email_body(alert_event)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                server.login(self.config.username, self.config.password)
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent for event {alert_event.event_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            raise
    
    def _create_email_body(self, alert_event: AlertEvent) -> str:
        """Create HTML email body"""
        severity_color = {
            'critical': '#FF0000',
            'warning': '#FFA500',
            'medium': '#FFFF00',
            'low': '#00FF00'
        }.get(alert_event.severity, '#808080')
        
        return f"""
        <html>
        <head>
            <style>
                .alert-container {{
                    font-family: Arial, sans-serif;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    border: 2px solid {severity_color};
                    border-radius: 10px;
                }}
                .alert-header {{
                    background-color: {severity_color};
                    color: white;
                    padding: 15px;
                    margin: -20px -20px 20px -20px;
                    border-radius: 8px 8px 0 0;
                    font-size: 18px;
                    font-weight: bold;
                }}
                .alert-details {{
                    margin: 10px 0;
                }}
                .metric-value {{
                    font-weight: bold;
                    color: {severity_color};
                }}
                .footer {{
                    margin-top: 20px;
                    padding-top: 15px;
                    border-top: 1px solid #ccc;
                    font-size: 12px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="alert-container">
                <div class="alert-header">
                    🚨 {alert_event.severity.upper()} ALERT: {alert_event.alert_type}
                </div>
                
                <div class="alert-details">
                    <strong>Message:</strong> {alert_event.message}<br><br>
                    <strong>Model:</strong> {alert_event.model_name}<br>
                    <strong>Metric:</strong> {alert_event.metric_name}<br>
                    <strong>Current Value:</strong> <span class="metric-value">{alert_event.current_value}</span><br>
                    <strong>Threshold:</strong> {alert_event.threshold}<br>
                    <strong>Timestamp:</strong> {alert_event.timestamp}<br>
                    <strong>Event ID:</strong> {alert_event.event_id}
                </div>
                
                <div class="footer">
                    This is an automated alert from the LLM Observability Framework.<br>
                    Please investigate and take appropriate action.
                </div>
            </div>
        </body>
        </html>
        """


@dataclass
class SlackConfig:
    """Slack configuration"""
    webhook_url: str
    channel: Optional[str] = None
    username: Optional[str] = "LLM Observability"
    icon_emoji: Optional[str] = ":warning:"


class SlackNotifier(NotificationHandler):
    """Slack notification handler"""
    
    def __init__(self, config: SlackConfig):
        """
        Initialize SlackNotifier
        
        Args:
            config: SlackConfig object with webhook settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def send_notification(self, alert_event: AlertEvent) -> None:
        """Send Slack notification"""
        try:
            # Create Slack message
            message = self._create_slack_message(alert_event)
            
            # Send to Slack
            response = requests.post(
                self.config.webhook_url,
                json=message,
                timeout=30
            )
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent for event {alert_event.event_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            raise
    
    def _create_slack_message(self, alert_event: AlertEvent) -> Dict[str, Any]:
        """Create Slack message payload"""
        severity_color = {
            'critical': '#FF0000',
            'warning': '#FFA500',
            'medium': '#FFFF00',
            'low': '#00FF00'
        }.get(alert_event.severity, '#808080')
        
        severity_emoji = {
            'critical': ':rotating_light:',
            'warning': ':warning:',
            'medium': ':yellow_circle:',
            'low': ':green_circle:'
        }.get(alert_event.severity, ':grey_circle:')
        
        message = {
            "username": self.config.username,
            "icon_emoji": self.config.icon_emoji,
            "attachments": [{
                "color": severity_color,
                "title": f"{severity_emoji} {alert_event.severity.upper()} Alert: {alert_event.alert_type}",
                "text": alert_event.message,
                "fields": [
                    {
                        "title": "Model",
                        "value": alert_event.model_name,
                        "short": True
                    },
                    {
                        "title": "Metric",
                        "value": alert_event.metric_name,
                        "short": True
                    },
                    {
                        "title": "Current Value",
                        "value": str(alert_event.current_value),
                        "short": True
                    },
                    {
                        "title": "Threshold",
                        "value": str(alert_event.threshold),
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": alert_event.timestamp,
                        "short": False
                    }
                ],
                "footer": "LLM Observability Framework",
                "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png",
                "ts": int(alert_event.timestamp.replace(':', '').replace('-', '').replace('T', '').replace('.', '')[:10]) if alert_event.timestamp else None
            }]
        }
        
        if self.config.channel:
            message["channel"] = self.config.channel
        
        return message


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    url: str
    method: str = "POST"
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30
    retry_count: int = 3


class WebhookNotifier(NotificationHandler):
    """Generic webhook notification handler"""
    
    def __init__(self, config: WebhookConfig):
        """
        Initialize WebhookNotifier
        
        Args:
            config: WebhookConfig object with webhook settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def send_notification(self, alert_event: AlertEvent) -> None:
        """Send webhook notification"""
        payload = self._create_webhook_payload(alert_event)
        headers = self.config.headers or {'Content-Type': 'application/json'}
        
        for attempt in range(self.config.retry_count):
            try:
                if self.config.method.upper() == "POST":
                    response = requests.post(
                        self.config.url,
                        json=payload,
                        headers=headers,
                        timeout=self.config.timeout
                    )
                elif self.config.method.upper() == "PUT":
                    response = requests.put(
                        self.config.url,
                        json=payload,
                        headers=headers,
                        timeout=self.config.timeout
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {self.config.method}")
                
                response.raise_for_status()
                self.logger.info(f"Webhook alert sent for event {alert_event.event_id}")
                return
                
            except Exception as e:
                self.logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                if attempt == self.config.retry_count - 1:
                    self.logger.error(f"All webhook attempts failed for event {alert_event.event_id}")
                    raise
    
    def _create_webhook_payload(self, alert_event: AlertEvent) -> Dict[str, Any]:
        """Create webhook payload"""
        return {
            "event_id": alert_event.event_id,
            "rule_id": alert_event.rule_id,
            "alert_type": alert_event.alert_type,
            "message": alert_event.message,
            "severity": alert_event.severity,
            "timestamp": alert_event.timestamp,
            "metric_name": alert_event.metric_name,
            "current_value": alert_event.current_value,
            "threshold": alert_event.threshold,
            "model_name": alert_event.model_name,
            "framework": "LLM Observability Framework"
        }


@dataclass  
class DiscordConfig:
    """Discord configuration"""
    webhook_url: str
    username: Optional[str] = "LLM Observability"


class DiscordNotifier(NotificationHandler):
    """Discord notification handler"""
    
    def __init__(self, config: DiscordConfig):
        """
        Initialize DiscordNotifier
        
        Args:
            config: DiscordConfig object with webhook settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def send_notification(self, alert_event: AlertEvent) -> None:
        """Send Discord notification"""
        try:
            message = self._create_discord_message(alert_event)
            
            response = requests.post(
                self.config.webhook_url,
                json=message,
                timeout=30
            )
            response.raise_for_status()
            
            self.logger.info(f"Discord alert sent for event {alert_event.event_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Discord alert: {e}")
            raise
    
    def _create_discord_message(self, alert_event: AlertEvent) -> Dict[str, Any]:
        """Create Discord message payload"""
        severity_color = {
            'critical': 16711680,  # Red
            'warning': 16753920,   # Orange  
            'medium': 16776960,    # Yellow
            'low': 65280          # Green
        }.get(alert_event.severity, 8421504)  # Gray
        
        severity_emoji = {
            'critical': '🚨',
            'warning': '⚠️',
            'medium': '🟡',
            'low': '🟢'
        }.get(alert_event.severity, '⚪')
        
        return {
            "username": self.config.username,
            "embeds": [{
                "title": f"{severity_emoji} {alert_event.severity.upper()} Alert: {alert_event.alert_type}",
                "description": alert_event.message,
                "color": severity_color,
                "fields": [
                    {
                        "name": "Model",
                        "value": alert_event.model_name,
                        "inline": True
                    },
                    {
                        "name": "Metric", 
                        "value": alert_event.metric_name,
                        "inline": True
                    },
                    {
                        "name": "Current Value",
                        "value": str(alert_event.current_value),
                        "inline": True
                    },
                    {
                        "name": "Threshold",
                        "value": str(alert_event.threshold),
                        "inline": True
                    },
                    {
                        "name": "Timestamp",
                        "value": alert_event.timestamp,
                        "inline": False
                    }
                ],
                "footer": {
                    "text": "LLM Observability Framework"
                }
            }]
        }


class LogNotifier(NotificationHandler):
    """Simple log-based notification handler for testing"""
    
    def __init__(self, log_level: str = "WARNING"):
        """
        Initialize LogNotifier
        
        Args:
            log_level: Log level for alerts (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(__name__)
        self.log_level = getattr(logging, log_level.upper())
    
    def send_notification(self, alert_event: AlertEvent) -> None:
        """Log the alert event"""
        message = (
            f"ALERT [{alert_event.severity.upper()}] {alert_event.message} "
            f"| Model: {alert_event.model_name} "
            f"| Metric: {alert_event.metric_name} "
            f"| Value: {alert_event.current_value} "
            f"| Threshold: {alert_event.threshold} "
            f"| ID: {alert_event.event_id}"
        )
        
        self.logger.log(self.log_level, message)