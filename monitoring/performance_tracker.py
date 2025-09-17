"""
PerformanceTracker for LLM Observability Framework

Real-time tracking and analysis of LLM performance with:
- Sliding window statistics
- Anomaly detection
- Performance trend analysis
- Real-time alerts
"""

import time
import threading
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import statistics
import logging
from dataclasses import dataclass
import json


@dataclass
class PerformanceWindow:
    """Sliding window for performance metrics"""
    response_times: deque = None
    token_counts: deque = None
    success_rates: deque = None
    costs: deque = None
    quality_scores: deque = None
    timestamps: deque = None
    
    def __post_init__(self):
        """Initialize deques with maxlen"""
        max_size = 1000  # Keep last 1000 measurements
        self.response_times = deque(maxlen=max_size)
        self.token_counts = deque(maxlen=max_size)
        self.success_rates = deque(maxlen=max_size)
        self.costs = deque(maxlen=max_size)
        self.quality_scores = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)


@dataclass
class PerformanceAlert:
    """Performance alert data"""
    alert_type: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: str
    metric_name: str
    current_value: float
    threshold: float
    model_name: str


class PerformanceTracker:
    """Real-time LLM performance tracker with anomaly detection"""
    
    def __init__(self, 
                 window_size_minutes: int = 15,
                 alert_callback: Optional[Callable[[PerformanceAlert], None]] = None):
        """
        Initialize PerformanceTracker
        
        Args:
            window_size_minutes: Size of sliding window in minutes
            alert_callback: Callback function for handling alerts
        """
        self.window_size = timedelta(minutes=window_size_minutes)
        self.alert_callback = alert_callback
        self.logger = logging.getLogger(__name__)
        
        # Performance windows per model
        self.windows: Dict[str, PerformanceWindow] = defaultdict(PerformanceWindow)
        
        # Thresholds for alerting
        self.thresholds = {
            'response_time_ms': {
                'warning': 5000,      # 5 seconds
                'critical': 15000     # 15 seconds
            },
            'success_rate': {
                'warning': 0.95,      # 95%
                'critical': 0.85      # 85%
            },
            'quality_score': {
                'warning': 0.7,       # 70%
                'critical': 0.5       # 50%
            },
            'cost_per_token': {
                'warning': 0.001,     # $0.001 per token
                'critical': 0.005     # $0.005 per token
            }
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_data, daemon=True)
        self._cleanup_thread.start()
    
    def track_request(self, 
                     model_name: str,
                     response_time_ms: float,
                     total_tokens: int,
                     success: bool,
                     cost_usd: float = 0.0,
                     quality_score: Optional[float] = None) -> None:
        """
        Track a single LLM request
        
        Args:
            model_name: Name of the LLM model
            response_time_ms: Response time in milliseconds
            total_tokens: Total tokens used
            success: Whether the request was successful
            cost_usd: Cost in USD
            quality_score: Quality score (0-1)
        """
        current_time = datetime.now()
        
        with self._lock:
            window = self.windows[model_name]
            
            # Add metrics to window
            window.response_times.append(response_time_ms)
            window.token_counts.append(total_tokens)
            window.success_rates.append(1.0 if success else 0.0)
            window.costs.append(cost_usd)
            if quality_score is not None:
                window.quality_scores.append(quality_score)
            window.timestamps.append(current_time)
            
            # Check for alerts
            self._check_alerts(model_name, response_time_ms, success, cost_usd, quality_score)
    
    def _check_alerts(self, 
                     model_name: str,
                     response_time_ms: float,
                     success: bool,
                     cost_usd: float,
                     quality_score: Optional[float]) -> None:
        """Check if any metrics trigger alerts"""
        current_time = datetime.now().isoformat()
        
        # Response time alerts
        if response_time_ms > self.thresholds['response_time_ms']['critical']:
            alert = PerformanceAlert(
                alert_type='response_time',
                message=f"Critical response time: {response_time_ms:.2f}ms",
                severity='critical',
                timestamp=current_time,
                metric_name='response_time_ms',
                current_value=response_time_ms,
                threshold=self.thresholds['response_time_ms']['critical'],
                model_name=model_name
            )
            self._send_alert(alert)
        elif response_time_ms > self.thresholds['response_time_ms']['warning']:
            alert = PerformanceAlert(
                alert_type='response_time',
                message=f"High response time: {response_time_ms:.2f}ms",
                severity='warning',
                timestamp=current_time,
                metric_name='response_time_ms',
                current_value=response_time_ms,
                threshold=self.thresholds['response_time_ms']['warning'],
                model_name=model_name
            )
            self._send_alert(alert)
        
        # Success rate alerts (check recent success rate)
        window = self.windows[model_name]
        if len(window.success_rates) >= 10:  # Need at least 10 samples
            recent_success_rate = statistics.mean(list(window.success_rates)[-10:])
            
            if recent_success_rate < self.thresholds['success_rate']['critical']:
                alert = PerformanceAlert(
                    alert_type='success_rate',
                    message=f"Critical success rate: {recent_success_rate:.2%}",
                    severity='critical',
                    timestamp=current_time,
                    metric_name='success_rate',
                    current_value=recent_success_rate,
                    threshold=self.thresholds['success_rate']['critical'],
                    model_name=model_name
                )
                self._send_alert(alert)
            elif recent_success_rate < self.thresholds['success_rate']['warning']:
                alert = PerformanceAlert(
                    alert_type='success_rate',
                    message=f"Low success rate: {recent_success_rate:.2%}",
                    severity='warning',
                    timestamp=current_time,
                    metric_name='success_rate',
                    current_value=recent_success_rate,
                    threshold=self.thresholds['success_rate']['warning'],
                    model_name=model_name
                )
                self._send_alert(alert)
        
        # Quality score alerts
        if quality_score is not None:
            if quality_score < self.thresholds['quality_score']['critical']:
                alert = PerformanceAlert(
                    alert_type='quality_score',
                    message=f"Critical quality score: {quality_score:.2f}",
                    severity='critical',
                    timestamp=current_time,
                    metric_name='quality_score',
                    current_value=quality_score,
                    threshold=self.thresholds['quality_score']['critical'],
                    model_name=model_name
                )
                self._send_alert(alert)
            elif quality_score < self.thresholds['quality_score']['warning']:
                alert = PerformanceAlert(
                    alert_type='quality_score',
                    message=f"Low quality score: {quality_score:.2f}",
                    severity='warning',
                    timestamp=current_time,
                    metric_name='quality_score',
                    current_value=quality_score,
                    threshold=self.thresholds['quality_score']['warning'],
                    model_name=model_name
                )
                self._send_alert(alert)
    
    def _send_alert(self, alert: PerformanceAlert) -> None:
        """Send alert via callback or log"""
        try:
            if self.alert_callback:
                self.alert_callback(alert)
            else:
                self.logger.warning(f"ALERT [{alert.severity.upper()}] {alert.message}")
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    def get_current_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current performance statistics
        
        Args:
            model_name: Filter by model name (if None, returns all models)
            
        Returns:
            Dictionary with current performance stats
        """
        with self._lock:
            if model_name:
                models = [model_name] if model_name in self.windows else []
            else:
                models = list(self.windows.keys())
            
            stats = {}
            
            for model in models:
                window = self.windows[model]
                model_stats = {}
                
                # Response time stats
                if window.response_times:
                    model_stats['response_time'] = {
                        'avg': statistics.mean(window.response_times),
                        'median': statistics.median(window.response_times),
                        'p95': self._percentile(window.response_times, 95),
                        'p99': self._percentile(window.response_times, 99),
                        'min': min(window.response_times),
                        'max': max(window.response_times)
                    }
                
                # Token stats
                if window.token_counts:
                    model_stats['tokens'] = {
                        'avg': statistics.mean(window.token_counts),
                        'total': sum(window.token_counts),
                        'min': min(window.token_counts),
                        'max': max(window.token_counts)
                    }
                
                # Success rate
                if window.success_rates:
                    model_stats['success_rate'] = statistics.mean(window.success_rates)
                
                # Cost stats
                if window.costs:
                    total_cost = sum(window.costs)
                    total_tokens = sum(window.token_counts) if window.token_counts else 1
                    model_stats['cost'] = {
                        'total': total_cost,
                        'avg_per_request': statistics.mean(window.costs),
                        'cost_per_token': total_cost / total_tokens
                    }
                
                # Quality scores
                if window.quality_scores:
                    model_stats['quality'] = {
                        'avg': statistics.mean(window.quality_scores),
                        'min': min(window.quality_scores),
                        'max': max(window.quality_scores)
                    }
                
                # Request count and time range
                model_stats['request_count'] = len(window.timestamps)
                if window.timestamps:
                    model_stats['time_range'] = {
                        'start': min(window.timestamps).isoformat(),
                        'end': max(window.timestamps).isoformat()
                    }
                
                stats[model] = model_stats
            
            return stats
    
    def get_trend_analysis(self, model_name: str, minutes: int = 60) -> Dict[str, Any]:
        """
        Analyze performance trends over time
        
        Args:
            model_name: Name of the model to analyze
            minutes: Number of minutes to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if model_name not in self.windows:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        window = self.windows[model_name]
        
        # Filter data to the specified time range
        filtered_data = []
        for i, timestamp in enumerate(window.timestamps):
            if timestamp >= cutoff_time:
                filtered_data.append({
                    'timestamp': timestamp,
                    'response_time': window.response_times[i] if i < len(window.response_times) else None,
                    'tokens': window.token_counts[i] if i < len(window.token_counts) else None,
                    'success': window.success_rates[i] if i < len(window.success_rates) else None,
                    'cost': window.costs[i] if i < len(window.costs) else None,
                    'quality': window.quality_scores[i] if i < len(window.quality_scores) else None
                })
        
        if not filtered_data:
            return {}
        
        # Calculate trends
        trends = {
            'data_points': len(filtered_data),
            'time_range_minutes': minutes
        }
        
        # Response time trend
        response_times = [d['response_time'] for d in filtered_data if d['response_time'] is not None]
        if len(response_times) > 1:
            trends['response_time_trend'] = self._calculate_trend(response_times)
        
        # Success rate trend
        success_rates = [d['success'] for d in filtered_data if d['success'] is not None]
        if len(success_rates) > 1:
            trends['success_rate_trend'] = self._calculate_trend(success_rates)
        
        # Cost trend
        costs = [d['cost'] for d in filtered_data if d['cost'] is not None]
        if len(costs) > 1:
            trends['cost_trend'] = self._calculate_trend(costs)
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and magnitude"""
        if len(values) < 2:
            return {'direction': 'stable', 'change_percent': 0}
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if first_avg == 0:
            change_percent = 0
        else:
            change_percent = ((second_avg - first_avg) / first_avg) * 100
        
        if abs(change_percent) < 5:
            direction = 'stable'
        elif change_percent > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'change_percent': change_percent,
            'first_half_avg': first_avg,
            'second_half_avg': second_avg
        }
    
    def _percentile(self, values: deque, percentile: int) -> float:
        """Calculate percentile of values"""
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        else:
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    
    def _cleanup_old_data(self) -> None:
        """Background thread to clean up old data"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                cutoff_time = datetime.now() - self.window_size
                
                with self._lock:
                    for model_name, window in self.windows.items():
                        # Remove old timestamps and corresponding data
                        while (window.timestamps and 
                               window.timestamps[0] < cutoff_time):
                            window.timestamps.popleft()
                            if window.response_times:
                                window.response_times.popleft()
                            if window.token_counts:
                                window.token_counts.popleft()
                            if window.success_rates:
                                window.success_rates.popleft()
                            if window.costs:
                                window.costs.popleft()
                            if window.quality_scores:
                                window.quality_scores.popleft()
                
            except Exception as e:
                self.logger.error(f"Error in cleanup thread: {e}")
    
    def set_thresholds(self, thresholds: Dict[str, Dict[str, float]]) -> None:
        """
        Update alert thresholds
        
        Args:
            thresholds: Dictionary of threshold configurations
        """
        self.thresholds.update(thresholds)
        self.logger.info("Updated alert thresholds")
    
    def export_current_data(self, filepath: str) -> None:
        """
        Export current performance data to file
        
        Args:
            filepath: Output file path
        """
        stats = self.get_current_stats()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.logger.info(f"Exported performance data to {filepath}")