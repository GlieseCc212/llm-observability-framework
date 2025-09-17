"""
AlertManager for LLM Observability Framework

Manages alerts and notifications for LLM performance issues including:
- Alert routing and escalation
- Notification filtering and throttling
- Alert history and analytics
- Integration with multiple notification channels
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import sqlite3
from pathlib import Path

from monitoring.performance_tracker import PerformanceAlert


@dataclass
class AlertRule:
    """Configuration for alert rules"""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True
    models: Optional[List[str]] = None  # Apply to specific models only
    cooldown_minutes: int = 5  # Minimum time between similar alerts
    description: Optional[str] = None


@dataclass
class AlertEvent:
    """Alert event data"""
    event_id: str
    rule_id: str
    alert_type: str
    message: str
    severity: str
    timestamp: str
    metric_name: str
    current_value: float
    threshold: float
    model_name: str
    acknowledged: bool = False
    resolved: bool = False
    ack_timestamp: Optional[str] = None
    resolve_timestamp: Optional[str] = None
    notification_sent: bool = False


class AlertManager:
    """Manages LLM performance alerts and notifications"""
    
    def __init__(self, db_path: str = "data/alerts.db"):
        """
        Initialize AlertManager
        
        Args:
            db_path: Path to SQLite database for storing alerts
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Alert rules and handlers
        self.rules: Dict[str, AlertRule] = {}
        self.notification_handlers = []
        
        # Alert throttling and cooldown
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Load default alert rules
        self._load_default_rules()
    
    def _init_database(self):
        """Initialize SQLite database for alert storage"""
        with sqlite3.connect(self.db_path) as conn:
            # Alert events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_events (
                    event_id TEXT PRIMARY KEY,
                    rule_id TEXT,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    timestamp TEXT,
                    metric_name TEXT,
                    current_value REAL,
                    threshold REAL,
                    model_name TEXT,
                    acknowledged BOOLEAN,
                    resolved BOOLEAN,
                    ack_timestamp TEXT,
                    resolve_timestamp TEXT,
                    notification_sent BOOLEAN
                )
            ''')
            
            # Alert rules table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT,
                    metric_name TEXT,
                    condition TEXT,
                    threshold REAL,
                    severity TEXT,
                    enabled BOOLEAN,
                    models TEXT,
                    cooldown_minutes INTEGER,
                    description TEXT
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alert_events(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alert_severity ON alert_events(severity)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alert_resolved ON alert_events(resolved)')
    
    def _load_default_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="response_time_critical",
                name="Critical Response Time",
                metric_name="response_time_ms",
                condition="gt",
                threshold=15000,
                severity="critical",
                cooldown_minutes=2,
                description="Alert when response time exceeds 15 seconds"
            ),
            AlertRule(
                rule_id="response_time_warning",
                name="High Response Time",
                metric_name="response_time_ms",
                condition="gt",
                threshold=5000,
                severity="warning",
                cooldown_minutes=5,
                description="Alert when response time exceeds 5 seconds"
            ),
            AlertRule(
                rule_id="success_rate_critical",
                name="Critical Success Rate",
                metric_name="success_rate",
                condition="lt",
                threshold=0.85,
                severity="critical",
                cooldown_minutes=3,
                description="Alert when success rate drops below 85%"
            ),
            AlertRule(
                rule_id="success_rate_warning",
                name="Low Success Rate",
                metric_name="success_rate",
                condition="lt",
                threshold=0.95,
                severity="warning",
                cooldown_minutes=10,
                description="Alert when success rate drops below 95%"
            ),
            AlertRule(
                rule_id="quality_score_critical",
                name="Critical Quality Score",
                metric_name="quality_score",
                condition="lt",
                threshold=0.5,
                severity="critical",
                cooldown_minutes=5,
                description="Alert when quality score drops below 50%"
            ),
            AlertRule(
                rule_id="cost_spike_warning",
                name="Cost Spike Warning",
                metric_name="cost_per_token",
                condition="gt",
                threshold=0.001,
                severity="warning",
                cooldown_minutes=15,
                description="Alert when cost per token exceeds $0.001"
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        Add or update an alert rule
        
        Args:
            rule: AlertRule object
        """
        with self._lock:
            self.rules[rule.rule_id] = rule
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                models_json = json.dumps(rule.models) if rule.models else None
                conn.execute('''
                    INSERT OR REPLACE INTO alert_rules 
                    (rule_id, name, metric_name, condition, threshold, severity, 
                     enabled, models, cooldown_minutes, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rule.rule_id, rule.name, rule.metric_name, rule.condition,
                    rule.threshold, rule.severity, rule.enabled, models_json,
                    rule.cooldown_minutes, rule.description
                ))
        
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> None:
        """
        Remove an alert rule
        
        Args:
            rule_id: ID of the rule to remove
        """
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM alert_rules WHERE rule_id = ?', (rule_id,))
                
                self.logger.info(f"Removed alert rule: {rule_id}")
    
    def process_performance_alert(self, alert: PerformanceAlert) -> None:
        """
        Process a performance alert from the monitoring system
        
        Args:
            alert: PerformanceAlert object from performance tracker
        """
        # Check if alert matches any rules and should be processed
        matching_rules = self._find_matching_rules(alert)
        
        for rule in matching_rules:
            if self._should_send_alert(rule, alert):
                event = self._create_alert_event(rule, alert)
                self._store_alert_event(event)
                self._send_notifications(event)
    
    def _find_matching_rules(self, alert: PerformanceAlert) -> List[AlertRule]:
        """Find alert rules that match the given performance alert"""
        matching_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
                
            # Check metric name match
            if rule.metric_name != alert.metric_name:
                continue
                
            # Check model filter
            if rule.models and alert.model_name not in rule.models:
                continue
                
            # Check condition
            if self._check_condition(alert.current_value, rule.condition, rule.threshold):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if value meets the condition threshold"""
        if condition == 'gt':
            return value > threshold
        elif condition == 'lt':
            return value < threshold
        elif condition == 'eq':
            return value == threshold
        elif condition == 'ne':
            return value != threshold
        else:
            return False
    
    def _should_send_alert(self, rule: AlertRule, alert: PerformanceAlert) -> bool:
        """Check if alert should be sent based on cooldown and throttling"""
        cooldown_key = f"{rule.rule_id}_{alert.model_name}"
        
        # Check cooldown
        if cooldown_key in self.alert_cooldowns:
            cooldown_end = self.alert_cooldowns[cooldown_key] + timedelta(minutes=rule.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return False
        
        # Update cooldown
        self.alert_cooldowns[cooldown_key] = datetime.now()
        return True
    
    def _create_alert_event(self, rule: AlertRule, alert: PerformanceAlert) -> AlertEvent:
        """Create an AlertEvent from rule and performance alert"""
        import uuid
        
        return AlertEvent(
            event_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            alert_type=alert.alert_type,
            message=f"[{rule.name}] {alert.message}",
            severity=rule.severity,
            timestamp=alert.timestamp,
            metric_name=alert.metric_name,
            current_value=alert.current_value,
            threshold=alert.threshold,
            model_name=alert.model_name
        )
    
    def _store_alert_event(self, event: AlertEvent) -> None:
        """Store alert event in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                event_dict = asdict(event)
                placeholders = ', '.join(['?' for _ in event_dict])
                columns = ', '.join(event_dict.keys())
                
                conn.execute(
                    f'INSERT INTO alert_events ({columns}) VALUES ({placeholders})',
                    list(event_dict.values())
                )
            
            self.logger.info(f"Stored alert event: {event.event_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing alert event: {e}")
    
    def _send_notifications(self, event: AlertEvent) -> None:
        """Send notifications for alert event"""
        if not self.notification_handlers:
            self.logger.warning("No notification handlers configured")
            return
        
        for handler in self.notification_handlers:
            try:
                handler.send_notification(event)
                
                # Mark notification as sent
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        'UPDATE alert_events SET notification_sent = ? WHERE event_id = ?',
                        (True, event.event_id)
                    )
                
            except Exception as e:
                self.logger.error(f"Error sending notification: {e}")
    
    def add_notification_handler(self, handler) -> None:
        """
        Add a notification handler
        
        Args:
            handler: Notification handler object (EmailNotifier, SlackNotifier, etc.)
        """
        self.notification_handlers.append(handler)
        self.logger.info(f"Added notification handler: {type(handler).__name__}")
    
    def acknowledge_alert(self, event_id: str, user: str = "system") -> None:
        """
        Acknowledge an alert
        
        Args:
            event_id: ID of the alert event
            user: User who acknowledged the alert
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE alert_events 
                    SET acknowledged = ?, ack_timestamp = ?
                    WHERE event_id = ?
                ''', (True, datetime.now().isoformat(), event_id))
            
            self.logger.info(f"Alert {event_id} acknowledged by {user}")
            
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
    
    def resolve_alert(self, event_id: str, user: str = "system") -> None:
        """
        Resolve an alert
        
        Args:
            event_id: ID of the alert event
            user: User who resolved the alert
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE alert_events 
                    SET resolved = ?, resolve_timestamp = ?
                    WHERE event_id = ?
                ''', (True, datetime.now().isoformat(), event_id))
            
            self.logger.info(f"Alert {event_id} resolved by {user}")
            
        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
    
    def get_active_alerts(self, 
                         severity: Optional[str] = None,
                         model_name: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get active (unresolved) alerts
        
        Args:
            severity: Filter by severity level
            model_name: Filter by model name
            limit: Maximum number of alerts to return
            
        Returns:
            List of active alert dictionaries
        """
        query = "SELECT * FROM alert_events WHERE resolved = 0"
        params = []
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            return []
    
    def get_alert_history(self, 
                         start_time: Optional[str] = None,
                         end_time: Optional[str] = None,
                         limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get alert history
        
        Args:
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert event dictionaries
        """
        query = "SELECT * FROM alert_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        
        except Exception as e:
            self.logger.error(f"Error getting alert history: {e}")
            return []
    
    def get_alert_statistics(self, 
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Get alert statistics
        
        Args:
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
            
        Returns:
            Dictionary with alert statistics
        """
        base_query = "FROM alert_events WHERE 1=1"
        params = []
        
        if start_time:
            base_query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            base_query += " AND timestamp <= ?"
            params.append(end_time)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Overall stats
                stats_query = f"""
                    SELECT 
                        COUNT(*) as total_alerts,
                        SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved_alerts,
                        SUM(CASE WHEN acknowledged = 1 THEN 1 ELSE 0 END) as acknowledged_alerts
                    {base_query}
                """
                cursor = conn.execute(stats_query, params)
                stats = dict(cursor.fetchone())
                
                # Severity breakdown
                severity_query = f"""
                    SELECT severity, COUNT(*) as count
                    {base_query}
                    GROUP BY severity
                    ORDER BY count DESC
                """
                cursor = conn.execute(severity_query, params)
                stats['severity_breakdown'] = [dict(row) for row in cursor.fetchall()]
                
                # Model breakdown
                model_query = f"""
                    SELECT model_name, COUNT(*) as count
                    {base_query}
                    GROUP BY model_name
                    ORDER BY count DESC
                """
                cursor = conn.execute(model_query, params)
                stats['model_breakdown'] = [dict(row) for row in cursor.fetchall()]
                
                return stats
        
        except Exception as e:
            self.logger.error(f"Error getting alert statistics: {e}")
            return {}
    
    def export_alerts(self, 
                     filepath: str,
                     format: str = 'json',
                     start_time: Optional[str] = None,
                     end_time: Optional[str] = None) -> None:
        """
        Export alerts to file
        
        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
        """
        alerts = self.get_alert_history(start_time=start_time, end_time=end_time, limit=10000)
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(alerts, f, indent=2)
        elif format.lower() == 'csv':
            import pandas as pd
            df = pd.DataFrame(alerts)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Exported {len(alerts)} alerts to {filepath}")