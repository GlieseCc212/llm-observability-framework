"""
MetricsCollector for LLM Observability Framework

Collects and stores various LLM performance metrics including:
- Response times
- Token usage
- API costs
- Error rates
- Quality scores
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import sqlite3
import threading
from pathlib import Path


@dataclass
class LLMMetrics:
    """Data class for storing LLM metrics"""
    timestamp: str
    request_id: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time_ms: float
    cost_usd: float
    success: bool
    error_message: Optional[str] = None
    quality_score: Optional[float] = None
    user_feedback: Optional[str] = None
    context_length: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class MetricsCollector:
    """Collects and stores LLM performance metrics"""
    
    def __init__(self, db_path: str = "data/llm_metrics.db"):
        """
        Initialize MetricsCollector
        
        Args:
            db_path: Path to SQLite database for storing metrics
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with metrics table"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS llm_metrics (
                    timestamp TEXT,
                    request_id TEXT PRIMARY KEY,
                    model_name TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    response_time_ms REAL,
                    cost_usd REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    quality_score REAL,
                    user_feedback TEXT,
                    context_length INTEGER,
                    temperature REAL,
                    max_tokens INTEGER
                )
            ''')
            
            # Create indexes for better query performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON llm_metrics(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON llm_metrics(model_name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_success ON llm_metrics(success)')
    
    def collect_metrics(self, metrics: LLMMetrics) -> None:
        """
        Store metrics in the database
        
        Args:
            metrics: LLMMetrics object containing the metrics to store
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    metrics_dict = asdict(metrics)
                    placeholders = ', '.join(['?' for _ in metrics_dict])
                    columns = ', '.join(metrics_dict.keys())
                    
                    conn.execute(
                        f'INSERT OR REPLACE INTO llm_metrics ({columns}) VALUES ({placeholders})',
                        list(metrics_dict.values())
                    )
                    
                self.logger.info(f"Metrics collected for request {metrics.request_id}")
            except Exception as e:
                self.logger.error(f"Error storing metrics: {e}")
    
    def get_metrics(self, 
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   model_name: Optional[str] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve metrics from the database
        
        Args:
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format) 
            model_name: Filter by model name
            limit: Maximum number of records to return
            
        Returns:
            List of metrics dictionaries
        """
        query = "SELECT * FROM llm_metrics WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
            
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_summary_stats(self, 
                         start_time: Optional[str] = None,
                         end_time: Optional[str] = None,
                         model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for LLM metrics
        
        Args:
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
            model_name: Filter by model name
            
        Returns:
            Dictionary containing summary statistics
        """
        base_query = "FROM llm_metrics WHERE 1=1"
        params = []
        
        if start_time:
            base_query += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            base_query += " AND timestamp <= ?"
            params.append(end_time)
            
        if model_name:
            base_query += " AND model_name = ?"
            params.append(model_name)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Get basic stats
            stats_query = f"""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                    AVG(response_time_ms) as avg_response_time,
                    MIN(response_time_ms) as min_response_time,
                    MAX(response_time_ms) as max_response_time,
                    AVG(total_tokens) as avg_tokens,
                    SUM(total_tokens) as total_tokens_used,
                    SUM(cost_usd) as total_cost,
                    AVG(quality_score) as avg_quality_score
                {base_query}
            """
            
            cursor = conn.execute(stats_query, params)
            row = cursor.fetchone()
            stats = dict(row) if row else {}
            
            # Calculate success rate
            if stats['total_requests'] > 0:
                stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            else:
                stats['success_rate'] = 0
            
            # Get model breakdown
            model_query = f"""
                SELECT 
                    model_name,
                    COUNT(*) as requests,
                    AVG(response_time_ms) as avg_response_time,
                    SUM(cost_usd) as total_cost
                {base_query}
                GROUP BY model_name
                ORDER BY requests DESC
            """
            
            cursor = conn.execute(model_query, params)
            stats['model_breakdown'] = [dict(row) for row in cursor.fetchall()]
            
            return stats
    
    def get_error_analysis(self, 
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze errors and failures
        
        Args:
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
            
        Returns:
            Dictionary containing error analysis
        """
        base_query = "FROM llm_metrics WHERE success = 0"
        params = []
        
        if start_time:
            base_query += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            base_query += " AND timestamp <= ?"
            params.append(end_time)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get error counts by message
            error_query = f"""
                SELECT 
                    error_message,
                    COUNT(*) as count,
                    model_name
                {base_query}
                GROUP BY error_message, model_name
                ORDER BY count DESC
            """
            
            cursor = conn.execute(error_query, params)
            error_breakdown = [dict(row) for row in cursor.fetchall()]
            
            # Get total error count
            total_query = f"SELECT COUNT(*) as total_errors {base_query}"
            cursor = conn.execute(total_query, params)
            total_errors = cursor.fetchone()[0]
            
            return {
                'total_errors': total_errors,
                'error_breakdown': error_breakdown
            }
    
    def export_metrics(self, 
                      filepath: str,
                      format: str = 'json',
                      start_time: Optional[str] = None,
                      end_time: Optional[str] = None) -> None:
        """
        Export metrics to file
        
        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
        """
        metrics = self.get_metrics(start_time=start_time, end_time=end_time, limit=10000)
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
        elif format.lower() == 'csv':
            import pandas as pd
            df = pd.DataFrame(metrics)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Exported {len(metrics)} metrics to {filepath}")


# Context manager for easy metric collection
class MetricsCollectorContext:
    """Context manager for collecting LLM metrics"""
    
    def __init__(self, collector: MetricsCollector, request_id: str, model_name: str):
        self.collector = collector
        self.request_id = request_id
        self.model_name = model_name
        self.start_time = None
        self.metrics = LLMMetrics(
            timestamp="",
            request_id=request_id,
            model_name=model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_time_ms=0.0,
            cost_usd=0.0,
            success=True
        )
    
    def __enter__(self):
        self.start_time = time.time()
        return self.metrics
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.metrics.timestamp = datetime.now().isoformat()
        self.metrics.response_time_ms = (end_time - self.start_time) * 1000
        
        if exc_type is not None:
            self.metrics.success = False
            self.metrics.error_message = str(exc_val)
        
        self.collector.collect_metrics(self.metrics)