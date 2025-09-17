"""
Streamlit Dashboard for LLM Observability Framework

Interactive web dashboard for visualizing LLM performance metrics including:
- Real-time performance stats
- Historical trends
- Model comparisons
- Error analysis
- Cost tracking
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.metrics_collector import MetricsCollector
from monitoring.performance_tracker import PerformanceTracker


class StreamlitDashboard:
    """Streamlit-based dashboard for LLM observability"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.collector = None
        self.tracker = None
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="LLM Observability Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_components(self):
        """Initialize metrics collector and performance tracker"""
        if 'collector' not in st.session_state:
            st.session_state.collector = MetricsCollector()
        if 'tracker' not in st.session_state:
            st.session_state.tracker = PerformanceTracker()
        
        self.collector = st.session_state.collector
        self.tracker = st.session_state.tracker
    
    def render_sidebar(self):
        """Render sidebar with filters and controls"""
        st.sidebar.title("📊 LLM Observability")
        st.sidebar.markdown("---")
        
        # Time range selection
        st.sidebar.subheader("📅 Time Range")
        time_range = st.sidebar.selectbox(
            "Select time range:",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
        )
        
        # Convert to datetime range
        now = datetime.now()
        if time_range == "Last Hour":
            start_time = now - timedelta(hours=1)
        elif time_range == "Last 6 Hours":
            start_time = now - timedelta(hours=6)
        elif time_range == "Last 24 Hours":
            start_time = now - timedelta(days=1)
        elif time_range == "Last 7 Days":
            start_time = now - timedelta(days=7)
        else:  # Last 30 Days
            start_time = now - timedelta(days=30)
        
        # Model selection
        st.sidebar.subheader("🤖 Model Filter")
        
        # Get available models from database
        try:
            all_metrics = self.collector.get_metrics(limit=10000)
            if all_metrics:
                available_models = list(set([m['model_name'] for m in all_metrics if m['model_name']]))
            else:
                available_models = []
        except Exception as e:
            st.sidebar.error(f"Error loading models: {e}")
            available_models = []
        
        selected_models = st.sidebar.multiselect(
            "Select models to display:",
            available_models,
            default=available_models[:3] if available_models else []
        )
        
        # Refresh controls
        st.sidebar.subheader("🔄 Controls")
        if st.sidebar.button("Refresh Data"):
            st.experimental_rerun()
        
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            st.empty()  # Placeholder for auto-refresh
        
        return {
            'start_time': start_time,
            'end_time': now,
            'selected_models': selected_models,
            'auto_refresh': auto_refresh
        }
    
    def render_overview_metrics(self, filters: Dict[str, Any]):
        """Render overview metrics cards"""
        st.subheader("📈 Overview Metrics")
        
        # Get summary stats
        try:
            stats = self.collector.get_summary_stats(
                start_time=filters['start_time'].isoformat(),
                end_time=filters['end_time'].isoformat()
            )
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Total Requests",
                    f"{stats.get('total_requests', 0):,}",
                    delta=None
                )
            
            with col2:
                success_rate = stats.get('success_rate', 0) * 100
                st.metric(
                    "Success Rate",
                    f"{success_rate:.1f}%",
                    delta=None
                )
            
            with col3:
                avg_response_time = stats.get('avg_response_time', 0)
                st.metric(
                    "Avg Response Time",
                    f"{avg_response_time:.0f}ms",
                    delta=None
                )
            
            with col4:
                total_cost = stats.get('total_cost', 0)
                st.metric(
                    "Total Cost",
                    f"${total_cost:.4f}",
                    delta=None
                )
            
            with col5:
                avg_quality = stats.get('avg_quality_score', 0) or 0
                st.metric(
                    "Avg Quality Score",
                    f"{avg_quality:.2f}",
                    delta=None
                )
        
        except Exception as e:
            st.error(f"Error loading overview metrics: {e}")
    
    def render_performance_trends(self, filters: Dict[str, Any]):
        """Render performance trend charts"""
        st.subheader("📊 Performance Trends")
        
        try:
            # Get metrics data
            metrics_data = self.collector.get_metrics(
                start_time=filters['start_time'].isoformat(),
                end_time=filters['end_time'].isoformat(),
                limit=10000
            )
            
            if not metrics_data:
                st.info("No data available for the selected time range.")
                return
            
            df = pd.DataFrame(metrics_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by selected models
            if filters['selected_models']:
                df = df[df['model_name'].isin(filters['selected_models'])]
            
            if df.empty:
                st.info("No data available for selected models.")
                return
            
            # Response time trends
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Response Time Trends")
                fig_response = px.line(
                    df, 
                    x='timestamp', 
                    y='response_time_ms',
                    color='model_name',
                    title="Response Time Over Time",
                    labels={'response_time_ms': 'Response Time (ms)', 'timestamp': 'Time'}
                )
                fig_response.update_layout(height=400)
                st.plotly_chart(fig_response, use_container_width=True)
            
            with col2:
                st.subheader("Token Usage Trends")
                fig_tokens = px.line(
                    df,
                    x='timestamp',
                    y='total_tokens',
                    color='model_name',
                    title="Token Usage Over Time",
                    labels={'total_tokens': 'Total Tokens', 'timestamp': 'Time'}
                )
                fig_tokens.update_layout(height=400)
                st.plotly_chart(fig_tokens, use_container_width=True)
            
            # Success rate and cost trends
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Success Rate Trends")
                # Calculate success rate over time windows
                df_success = df.groupby(['model_name', df['timestamp'].dt.floor('H')])['success'].mean().reset_index()
                fig_success = px.line(
                    df_success,
                    x='timestamp',
                    y='success',
                    color='model_name',
                    title="Success Rate Over Time",
                    labels={'success': 'Success Rate', 'timestamp': 'Time'}
                )
                fig_success.update_yaxis(range=[0, 1])
                fig_success.update_layout(height=400)
                st.plotly_chart(fig_success, use_container_width=True)
            
            with col4:
                st.subheader("Cost Accumulation")
                df_cost = df.groupby(['model_name', df['timestamp'].dt.floor('H')])['cost_usd'].sum().reset_index()
                df_cost['cumulative_cost'] = df_cost.groupby('model_name')['cost_usd'].cumsum()
                fig_cost = px.line(
                    df_cost,
                    x='timestamp',
                    y='cumulative_cost',
                    color='model_name',
                    title="Cumulative Cost Over Time",
                    labels={'cumulative_cost': 'Cumulative Cost ($)', 'timestamp': 'Time'}
                )
                fig_cost.update_layout(height=400)
                st.plotly_chart(fig_cost, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error rendering performance trends: {e}")
    
    def render_model_comparison(self, filters: Dict[str, Any]):
        """Render model comparison charts"""
        st.subheader("🆚 Model Comparison")
        
        try:
            stats = self.collector.get_summary_stats(
                start_time=filters['start_time'].isoformat(),
                end_time=filters['end_time'].isoformat()
            )
            
            model_breakdown = stats.get('model_breakdown', [])
            if not model_breakdown:
                st.info("No model data available.")
                return
            
            df_models = pd.DataFrame(model_breakdown)
            
            # Filter by selected models
            if filters['selected_models']:
                df_models = df_models[df_models['model_name'].isin(filters['selected_models'])]
            
            if df_models.empty:
                st.info("No data for selected models.")
                return
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_requests = px.bar(
                    df_models,
                    x='model_name',
                    y='requests',
                    title="Requests by Model",
                    labels={'requests': 'Number of Requests', 'model_name': 'Model'}
                )
                st.plotly_chart(fig_requests, use_container_width=True)
            
            with col2:
                fig_response_time = px.bar(
                    df_models,
                    x='model_name',
                    y='avg_response_time',
                    title="Average Response Time by Model",
                    labels={'avg_response_time': 'Avg Response Time (ms)', 'model_name': 'Model'}
                )
                st.plotly_chart(fig_response_time, use_container_width=True)
            
            with col3:
                fig_cost = px.bar(
                    df_models,
                    x='model_name',
                    y='total_cost',
                    title="Total Cost by Model",
                    labels={'total_cost': 'Total Cost ($)', 'model_name': 'Model'}
                )
                st.plotly_chart(fig_cost, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error rendering model comparison: {e}")
    
    def render_error_analysis(self, filters: Dict[str, Any]):
        """Render error analysis"""
        st.subheader("🚨 Error Analysis")
        
        try:
            error_data = self.collector.get_error_analysis(
                start_time=filters['start_time'].isoformat(),
                end_time=filters['end_time'].isoformat()
            )
            
            total_errors = error_data.get('total_errors', 0)
            error_breakdown = error_data.get('error_breakdown', [])
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Total Errors", f"{total_errors:,}")
                
                if error_breakdown:
                    st.subheader("Top Error Types")
                    for error in error_breakdown[:5]:
                        st.text(f"• {error['error_message']}: {error['count']}")
            
            with col2:
                if error_breakdown:
                    df_errors = pd.DataFrame(error_breakdown)
                    fig_errors = px.bar(
                        df_errors.head(10),
                        x='count',
                        y='error_message',
                        color='model_name',
                        orientation='h',
                        title="Error Distribution",
                        labels={'count': 'Number of Errors', 'error_message': 'Error Type'}
                    )
                    fig_errors.update_layout(height=400)
                    st.plotly_chart(fig_errors, use_container_width=True)
                else:
                    st.info("No errors found in the selected time range.")
                    
        except Exception as e:
            st.error(f"Error rendering error analysis: {e}")
    
    def render_real_time_stats(self):
        """Render real-time performance statistics"""
        st.subheader("⚡ Real-Time Performance")
        
        try:
            current_stats = self.tracker.get_current_stats()
            
            if not current_stats:
                st.info("No real-time data available. Start tracking requests to see live metrics.")
                return
            
            for model_name, stats in current_stats.items():
                with st.expander(f"📊 {model_name}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'response_time' in stats:
                            st.metric("Avg Response Time", f"{stats['response_time']['avg']:.0f}ms")
                            st.metric("P95 Response Time", f"{stats['response_time']['p95']:.0f}ms")
                    
                    with col2:
                        if 'success_rate' in stats:
                            st.metric("Success Rate", f"{stats['success_rate']:.2%}")
                        if 'request_count' in stats:
                            st.metric("Request Count", f"{stats['request_count']:,}")
                    
                    with col3:
                        if 'tokens' in stats:
                            st.metric("Avg Tokens", f"{stats['tokens']['avg']:.0f}")
                            st.metric("Total Tokens", f"{stats['tokens']['total']:,}")
                    
                    with col4:
                        if 'cost' in stats:
                            st.metric("Total Cost", f"${stats['cost']['total']:.4f}")
                            st.metric("Cost per Token", f"${stats['cost']['cost_per_token']:.6f}")
                            
        except Exception as e:
            st.error(f"Error rendering real-time stats: {e}")
    
    def render_data_export(self, filters: Dict[str, Any]):
        """Render data export section"""
        st.subheader("📁 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Metrics (JSON)"):
                try:
                    self.collector.export_metrics(
                        "data/exported_metrics.json",
                        format='json',
                        start_time=filters['start_time'].isoformat(),
                        end_time=filters['end_time'].isoformat()
                    )
                    st.success("Metrics exported to data/exported_metrics.json")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        with col2:
            if st.button("Export Metrics (CSV)"):
                try:
                    self.collector.export_metrics(
                        "data/exported_metrics.csv",
                        format='csv',
                        start_time=filters['start_time'].isoformat(),
                        end_time=filters['end_time'].isoformat()
                    )
                    st.success("Metrics exported to data/exported_metrics.csv")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        with col3:
            if st.button("Export Performance Data"):
                try:
                    self.tracker.export_current_data("data/performance_data.json")
                    st.success("Performance data exported to data/performance_data.json")
                except Exception as e:
                    st.error(f"Export failed: {e}")
    
    def run(self):
        """Main dashboard application"""
        self.setup_page_config()
        self.initialize_components()
        
        # Render sidebar
        filters = self.render_sidebar()
        
        # Main content
        st.title("🔍 LLM Observability Dashboard")
        st.markdown("Monitor and analyze your LLM performance in real-time")
        st.markdown("---")
        
        # Render different sections
        self.render_overview_metrics(filters)
        st.markdown("---")
        
        self.render_real_time_stats()
        st.markdown("---")
        
        self.render_performance_trends(filters)
        st.markdown("---")
        
        self.render_model_comparison(filters)
        st.markdown("---")
        
        self.render_error_analysis(filters)
        st.markdown("---")
        
        self.render_data_export(filters)
        
        # Auto-refresh logic
        if filters.get('auto_refresh'):
            import time
            time.sleep(30)
            st.experimental_rerun()


def main():
    """Main entry point for the dashboard"""
    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()