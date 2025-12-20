"""
Alert System Module

This module handles the alert and notification system for the passenger counting application.
It monitors passenger counts and triggers alerts when occupancy thresholds are exceeded,
enabling transportation services to respond to overcrowding situations.

Key Features:
- Multi-level alert system (Normal, Warning, Critical)
- Alert cooldown to prevent spam
- Multiple notification channels (console, file, webhook-ready)
- Alert history tracking
- Configurable thresholds

Author: Head Count Project Team
Date: 2025-12-20
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path
from enum import Enum
import json

from config import AlertConfig, LoggingConfig

# Set up logging
logging.basicConfig(
    level=getattr(logging, LoggingConfig.LOG_LEVEL),
    format=LoggingConfig.LOG_FORMAT,
    datefmt=LoggingConfig.DATE_FORMAT
)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Enumeration for alert severity levels"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertSystem:
    """
    Alert and notification system for passenger occupancy monitoring.
    
    This class monitors passenger counts and triggers appropriate alerts
    when thresholds are exceeded. It supports multiple notification channels
    and implements alert cooldown to prevent spam.
    """
    
    def __init__(
        self,
        normal_threshold: int = None,
        warning_threshold: int = None,
        critical_threshold: int = None,
        cooldown_seconds: int = None,
        enable_console: bool = None,
        enable_file: bool = None,
        enable_webhook: bool = None
    ):
        """
        Initialize the alert system.
        
        Args:
            normal_threshold: Maximum passengers for normal operation
            warning_threshold: Threshold for warning alerts
            critical_threshold: Threshold for critical alerts
            cooldown_seconds: Minimum time between alerts
            enable_console: Enable console output
            enable_file: Enable file logging
            enable_webhook: Enable webhook notifications
        """
        # Load thresholds from config or use provided values
        self.normal_threshold = normal_threshold or AlertConfig.NORMAL_THRESHOLD
        self.warning_threshold = warning_threshold or AlertConfig.WARNING_THRESHOLD
        self.critical_threshold = critical_threshold or AlertConfig.CRITICAL_THRESHOLD
        
        # Alert cooldown
        self.cooldown_seconds = cooldown_seconds or AlertConfig.ALERT_COOLDOWN
        self.last_alert_time = None
        self.last_alert_level = None
        
        # Notification channels
        self.enable_console = enable_console if enable_console is not None else AlertConfig.ENABLE_CONSOLE_ALERTS
        self.enable_file = enable_file if enable_file is not None else AlertConfig.ENABLE_FILE_ALERTS
        self.enable_webhook = enable_webhook if enable_webhook is not None else AlertConfig.ENABLE_WEBHOOK
        
        # Alert history
        self.alert_history: List[Dict] = []
        
        # Initialize file logging if enabled
        if self.enable_file:
            self._init_file_logging()
        
        logger.info(f"AlertSystem initialized with thresholds: Normal={self.normal_threshold}, "
                   f"Warning={self.warning_threshold}, Critical={self.critical_threshold}")
    
    def _init_file_logging(self):
        """Initialize file logging for alerts."""
        try:
            AlertConfig.ALERT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file if it doesn't exist and write header
            if not AlertConfig.ALERT_LOG_FILE.exists():
                with open(AlertConfig.ALERT_LOG_FILE, 'w') as f:
                    f.write("timestamp,alert_level,passenger_count,message\n")
            
            logger.info(f"Alert file logging initialized: {AlertConfig.ALERT_LOG_FILE}")
        except Exception as e:
            logger.error(f"Failed to initialize alert file logging: {e}")
            self.enable_file = False
    
    def _determine_alert_level(self, passenger_count: int) -> AlertLevel:
        """
        Determine the appropriate alert level based on passenger count.
        
        Args:
            passenger_count: Current number of passengers
        
        Returns:
            AlertLevel enum value
        """
        if passenger_count >= self.critical_threshold:
            return AlertLevel.CRITICAL
        elif passenger_count >= self.warning_threshold:
            return AlertLevel.WARNING
        else:
            return AlertLevel.NORMAL
    
    def _is_cooldown_active(self) -> bool:
        """
        Check if alert cooldown is currently active.
        
        Returns:
            True if in cooldown period, False otherwise
        """
        if self.last_alert_time is None:
            return False
        
        time_since_last_alert = (datetime.now() - self.last_alert_time).total_seconds()
        return time_since_last_alert < self.cooldown_seconds
    
    def _format_alert_message(self, alert_level: AlertLevel, passenger_count: int) -> str:
        """
        Format alert message based on level and count.
        
        Args:
            alert_level: Alert severity level
            passenger_count: Current passenger count
        
        Returns:
            Formatted alert message
        """
        template = AlertConfig.ALERT_MESSAGES.get(alert_level.value, "Alert: {count} passengers")
        return template.format(count=passenger_count)
    
    def _send_console_alert(self, message: str, alert_level: AlertLevel):
        """Send alert to console."""
        if alert_level == AlertLevel.CRITICAL:
            logger.critical(message)
        elif alert_level == AlertLevel.WARNING:
            logger.warning(message)
        else:
            logger.info(message)
    
    def _send_file_alert(self, message: str, alert_level: AlertLevel, passenger_count: int):
        """Send alert to log file."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"{timestamp},{alert_level.value},{passenger_count},\"{message}\"\n"
            
            with open(AlertConfig.ALERT_LOG_FILE, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
    
    def _send_webhook_alert(self, message: str, alert_level: AlertLevel, passenger_count: int):
        """
        Send alert via webhook (placeholder for future implementation).
        
        Args:
            message: Alert message
            alert_level: Alert severity level
            passenger_count: Current passenger count
        """
        # Placeholder for webhook implementation
        # In production, this would send HTTP POST request to configured endpoint
        payload = {
            'timestamp': datetime.now().isoformat(),
            'alert_level': alert_level.value,
            'passenger_count': passenger_count,
            'message': message,
            'thresholds': {
                'warning': self.warning_threshold,
                'critical': self.critical_threshold
            }
        }
        
        logger.debug(f"Webhook payload (not sent - webhook disabled): {json.dumps(payload)}")
        
        # TODO: Implement actual webhook sending
        # import requests
        # try:
        #     response = requests.post(AlertConfig.WEBHOOK_URL, json=payload, timeout=5)
        #     response.raise_for_status()
        # except Exception as e:
        #     logger.error(f"Failed to send webhook alert: {e}")
    
    def check_and_alert(self, passenger_count: int, force: bool = False) -> Optional[Dict]:
        """
        Check passenger count and trigger alerts if necessary.
        
        Args:
            passenger_count: Current number of passengers detected
            force: Force alert even if in cooldown period
        
        Returns:
            Alert information dict if alert was triggered, None otherwise
        """
        alert_level = self._determine_alert_level(passenger_count)
        
        # Check if we should send an alert
        should_alert = False
        
        if force:
            should_alert = True
        elif alert_level == AlertLevel.CRITICAL:
            # Always alert on critical (even during cooldown for escalation)
            should_alert = True
        elif alert_level == AlertLevel.WARNING:
            # Alert on warning if not in cooldown
            should_alert = not self._is_cooldown_active()
        elif alert_level == AlertLevel.NORMAL and self.last_alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
            # Alert when returning to normal from elevated state
            should_alert = True
        
        if not should_alert:
            return None
        
        # Generate alert
        message = self._format_alert_message(alert_level, passenger_count)
        
        # Send to enabled channels
        if self.enable_console:
            self._send_console_alert(message, alert_level)
        
        if self.enable_file:
            self._send_file_alert(message, alert_level, passenger_count)
        
        if self.enable_webhook:
            self._send_webhook_alert(message, alert_level, passenger_count)
        
        # Update alert state
        self.last_alert_time = datetime.now()
        self.last_alert_level = alert_level
        
        # Record in history
        alert_info = {
            'timestamp': self.last_alert_time.isoformat(),
            'alert_level': alert_level.value,
            'passenger_count': passenger_count,
            'message': message
        }
        self.alert_history.append(alert_info)
        
        return alert_info
    
    def get_alert_history(self, limit: int = None) -> List[Dict]:
        """
        Get alert history.
        
        Args:
            limit: Maximum number of recent alerts to return (None for all)
        
        Returns:
            List of alert information dictionaries
        """
        if limit is None:
            return self.alert_history.copy()
        return self.alert_history[-limit:]
    
    def clear_history(self):
        """Clear alert history."""
        self.alert_history.clear()
        logger.info("Alert history cleared")
    
    def get_status(self) -> Dict:
        """
        Get current status of the alert system.
        
        Returns:
            Dictionary with system status information
        """
        return {
            'thresholds': {
                'normal': self.normal_threshold,
                'warning': self.warning_threshold,
                'critical': self.critical_threshold
            },
            'cooldown_seconds': self.cooldown_seconds,
            'last_alert': {
                'time': self.last_alert_time.isoformat() if self.last_alert_time else None,
                'level': self.last_alert_level.value if self.last_alert_level else None
            },
            'cooldown_active': self._is_cooldown_active(),
            'total_alerts': len(self.alert_history),
            'channels': {
                'console': self.enable_console,
                'file': self.enable_file,
                'webhook': self.enable_webhook
            }
        }


def test_alert_system():
    """Test function for the alert system."""
    print("Testing AlertSystem...")
    print("=" * 60)
    
    # Create alert system with test thresholds
    alert_system = AlertSystem(
        normal_threshold=30,
        warning_threshold=50,
        critical_threshold=70,
        cooldown_seconds=5
    )
    
    # Test different passenger counts
    test_counts = [25, 55, 75, 60, 20]
    
    for count in test_counts:
        print(f"\nTesting with {count} passengers:")
        result = alert_system.check_and_alert(count)
        if result:
            print(f"  Alert triggered: {result['alert_level']} - {result['message']}")
        else:
            print("  No alert (cooldown or normal level)")
    
    # Print status
    print("\n" + "=" * 60)
    print("Alert System Status:")
    status = alert_system.get_status()
    print(json.dumps(status, indent=2))
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_alert_system()
