"""
Simple live dashboard for SC Memory System metrics visualization.

This module provides a Streamlit-based dashboard for real-time monitoring
of system performance, token savings, and operational metrics.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    st = None
    go = None
    px = None
    make_subplots = None
    pd = None

from src.core.config import Settings
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)


class LiveDashboard:
    """Simple live dashboard for demo purposes."""
    
    def __init__(self, settings: Settings, metrics_collector: MetricsCollector):
        """
        Initialize live dashboard.
        
        Args:
            settings: Application settings
            metrics_collector: Metrics collector instance
        """
        if not DASHBOARD_AVAILABLE:
            raise ImportError("Dashboard dependencies not available. Install streamlit and plotly.")
        
        self.settings = settings
        self.metrics_collector = metrics_collector
        self.start_time = datetime.utcnow()
    
    def create_dashboard(self) -> None:
        """Create and configure Streamlit dashboard."""
        st.set_page_config(
            page_title="SC Memory System Dashboard",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧠 SC Memory System - Live Performance Dashboard")
        st.markdown("Real-time monitoring of memory consolidation system performance")
        
        # Sidebar controls
        self._create_sidebar()
        
        # Main dashboard content
        self._create_main_content()
        
        # Auto-refresh
        if st.sidebar.button("🔄 Refresh Data"):
            st.experimental_rerun()
    
    def _create_sidebar(self) -> None:
        """Create sidebar with controls and summary."""
        st.sidebar.header("📊 Dashboard Controls")
        
        # Refresh rate
        refresh_rate = st.sidebar.selectbox(
            "Refresh Rate",
            [5, 10, 30, 60],
            index=1,
            help="Dashboard refresh rate in seconds"
        )
        
        # Time window
        time_window = st.sidebar.selectbox(
            "Time Window",
            ["5 minutes", "15 minutes", "1 hour", "24 hours"],
            index=1,
            help="Time window for metrics aggregation"
        )
        
        st.sidebar.markdown("---")
        
        # System summary
        st.sidebar.header("⚡ System Summary")
        dashboard_data = self.metrics_collector.get_dashboard_data()
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric(
                "Requests/min",
                f"{dashboard_data['overview']['map_requests_per_minute']:.1f}",
                delta=None
            )
            st.metric(
                "Avg Response",
                f"{dashboard_data['overview']['avg_response_time_ms']:.0f}ms",
                delta=None
            )
        
        with col2:
            st.metric(
                "Token Savings",
                f"{dashboard_data['overview']['avg_token_savings']:.1%}",
                delta=None
            )
            st.metric(
                "Cache Hit Rate",
                f"{dashboard_data['overview']['cache_hit_rate']:.1%}",
                delta=None
            )
        
        # System resources
        st.sidebar.markdown("### 💻 System Resources")
        memory_usage = dashboard_data['system']['memory_usage_mb']
        cpu_usage = dashboard_data['system']['cpu_usage_percent']
        
        st.sidebar.progress(min(cpu_usage / 100, 1.0))
        st.sidebar.caption(f"CPU: {cpu_usage:.1f}%")
        
        st.sidebar.progress(min(memory_usage / 8192, 1.0))  # Assume 8GB system
        st.sidebar.caption(f"Memory: {memory_usage:.0f}MB")
    
    def _create_main_content(self) -> None:
        """Create main dashboard content."""
        # Get current metrics
        dashboard_data = self.metrics_collector.get_dashboard_data()
        
        # Key metrics row
        self._create_key_metrics_row(dashboard_data)
        
        # Performance charts
        self._create_performance_charts(dashboard_data)
        
        # System metrics
        self._create_system_charts(dashboard_data)
        
        # Token savings analysis
        self._create_token_analysis(dashboard_data)
    
    def _create_key_metrics_row(self, dashboard_data: Dict[str, Any]) -> None:
        """Create key metrics display row."""
        st.markdown("### 📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            response_time = dashboard_data['overview']['avg_response_time_ms']
            delta_color = "normal" if response_time <= 200 else "inverse"
            st.metric(
                "Avg Response Time",
                f"{response_time:.0f}ms",
                delta=f"Target: 200ms",
                delta_color=delta_color
            )
        
        with col2:
            token_savings = dashboard_data['overview']['avg_token_savings']
            delta_color = "normal" if token_savings >= 0.7 else "inverse"
            st.metric(
                "Token Savings",
                f"{token_savings:.1%}",
                delta=f"Target: 70%",
                delta_color=delta_color
            )
        
        with col3:
            total_requests = dashboard_data['performance']['total_requests']
            st.metric(
                "Total Requests",
                f"{total_requests:,}",
                delta=None
            )
        
        with col4:
            cache_hit_rate = dashboard_data['overview']['cache_hit_rate']
            delta_color = "normal" if cache_hit_rate >= 80 else "inverse"
            st.metric(
                "Cache Hit Rate",
                f"{cache_hit_rate:.1%}",
                delta=f"Target: 80%",
                delta_color=delta_color
            )
    
    def _create_performance_charts(self, dashboard_data: Dict[str, Any]) -> None:
        """Create performance monitoring charts."""
        st.markdown("### 🚀 Performance Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time chart (simulated data)
            response_times = self._generate_sample_timeseries("response_time", 50, 200, 50)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(response_times))),
                y=response_times,
                mode='lines+markers',
                name='Response Time',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_hline(y=200, line_dash="dash", line_color="red", 
                         annotation_text="200ms Target")
            
            fig.update_layout(
                title="Response Time Over Time",
                xaxis_title="Time",
                yaxis_title="Response Time (ms)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Throughput chart (simulated data)
            throughput = self._generate_sample_timeseries("throughput", 10, 50, 15)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(throughput))),
                y=throughput,
                mode='lines+markers',
                name='Requests/min',
                line=dict(color='green', width=2),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title="Request Throughput",
                xaxis_title="Time",
                yaxis_title="Requests/min",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_system_charts(self, dashboard_data: Dict[str, Any]) -> None:
        """Create system resource monitoring charts."""
        st.markdown("### 💻 System Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Memory usage gauge
            memory_usage = dashboard_data['system']['memory_usage_mb']
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = memory_usage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Memory Usage (MB)"},
                delta = {'reference': 4096},
                gauge = {
                    'axis': {'range': [None, 8192]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 4096], 'color': "lightgray"},
                        {'range': [4096, 6144], 'color': "yellow"},
                        {'range': [6144, 8192], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 6144
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # CPU usage gauge
            cpu_usage = dashboard_data['system']['cpu_usage_percent']
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = cpu_usage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "CPU Usage (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_token_analysis(self, dashboard_data: Dict[str, Any]) -> None:
        """Create token usage analysis charts."""
        st.markdown("### 🎯 Token Savings Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Token savings distribution (simulated data)
            savings_data = self._generate_token_savings_data()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=savings_data['scenarios'],
                    y=savings_data['baseline'],
                    name='Baseline Tokens',
                    marker_color='lightcoral'
                ),
                go.Bar(
                    x=savings_data['scenarios'],
                    y=savings_data['compressed'],
                    name='Compressed Tokens',
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="Token Usage: Baseline vs Compressed",
                xaxis_title="Scenarios",
                yaxis_title="Token Count",
                barmode='group',
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost savings pie chart
            baseline_cost = 120.50
            compressed_cost = 15.25
            savings = baseline_cost - compressed_cost
            
            fig = go.Figure(data=[go.Pie(
                labels=['Costs Saved', 'Remaining Costs'],
                values=[savings, compressed_cost],
                hole=0.4
            )])
            
            fig.update_layout(
                title="Cost Savings Analysis",
                annotations=[dict(text=f'${savings:.0f}<br>Saved', x=0.5, y=0.5, font_size=16, showarrow=False)],
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.markdown("### 📋 Performance Summary")
        
        summary_data = {
            "Metric": [
                "Average Token Savings",
                "Average Response Time", 
                "P95 Response Time",
                "Cache Hit Rate",
                "Total Requests Processed",
                "System Uptime"
            ],
            "Value": [
                f"{dashboard_data['overview']['avg_token_savings']:.1%}",
                f"{dashboard_data['overview']['avg_response_time_ms']:.0f}ms",
                f"{dashboard_data['overview']['p95_response_time_ms']:.0f}ms",
                f"{dashboard_data['overview']['cache_hit_rate']:.1%}",
                f"{dashboard_data['performance']['total_requests']:,}",
                str(datetime.utcnow() - self.start_time).split('.')[0]
            ],
            "Target": [
                "≥70%",
                "≤200ms",
                "≤200ms", 
                "≥80%",
                "-",
                "-"
            ],
            "Status": [
                "✅" if dashboard_data['overview']['avg_token_savings'] >= 0.7 else "❌",
                "✅" if dashboard_data['overview']['avg_response_time_ms'] <= 200 else "❌",
                "✅" if dashboard_data['overview']['p95_response_time_ms'] <= 200 else "❌",
                "✅" if dashboard_data['overview']['cache_hit_rate'] >= 0.8 else "❌",
                "✅",
                "✅"
            ]
        }
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _generate_sample_timeseries(self, metric_type: str, base: float, target: float, noise: float) -> List[float]:
        """Generate sample timeseries data for demonstration."""
        import random
        
        data = []
        current_value = base
        
        for i in range(30):  # 30 data points
            # Add some trend and noise
            trend = (target - base) * (i / 30) * 0.3  # 30% of target trend
            random_noise = random.uniform(-noise/2, noise/2)
            
            current_value = base + trend + random_noise
            current_value = max(0, current_value)  # Ensure non-negative
            
            data.append(current_value)
        
        return data
    
    def _generate_token_savings_data(self) -> Dict[str, List]:
        """Generate sample token savings data."""
        return {
            "scenarios": ["Physics", "Code Review", "Medical", "General"],
            "baseline": [8500, 6200, 4800, 3200],
            "compressed": [1100, 950, 720, 580]
        }
    
    def run_dashboard(self, port: int = 8501) -> None:
        """Run the Streamlit dashboard."""
        if not DASHBOARD_AVAILABLE:
            logger.error("Dashboard dependencies not available")
            return
        
        logger.info(f"Starting dashboard on port {port}")
        
        # This would be called from a separate process or thread
        # streamlit run dashboard.py --server.port 8501


def create_static_report(
    metrics_collector: MetricsCollector, 
    output_path: Path
) -> None:
    """Create a static HTML report of current metrics."""
    try:
        dashboard_data = metrics_collector.get_dashboard_data()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SC Memory System - Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric-card {{ 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 20px; 
                    margin: 10px 0; 
                    background: #f9f9f9;
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2E7D32; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .success {{ color: #2E7D32; }}
                .warning {{ color: #F57C00; }}
                .error {{ color: #C62828; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
            </style>
        </head>
        <body>
            <h1>🧠 SC Memory System - Performance Report</h1>
            <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-value">{dashboard_data['overview']['avg_response_time_ms']:.0f}ms</div>
                    <div class="metric-label">Average Response Time</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value success">{dashboard_data['overview']['avg_token_savings']:.1%}</div>
                    <div class="metric-label">Token Savings</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{dashboard_data['overview']['cache_hit_rate']:.1%}</div>
                    <div class="metric-label">Cache Hit Rate</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{dashboard_data['performance']['total_requests']:,}</div>
                    <div class="metric-label">Total Requests</div>
                </div>
            </div>
            
            <h2>System Resources</h2>
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-value">{dashboard_data['system']['cpu_usage_percent']:.1f}%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{dashboard_data['system']['memory_usage_mb']:.0f}MB</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
            </div>
            
            <h2>Validation Results</h2>
            <table border="1" style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #f0f0f0;">
                    <th style="padding: 10px; text-align: left;">Metric</th>
                    <th style="padding: 10px; text-align: left;">Target</th>
                    <th style="padding: 10px; text-align: left;">Achieved</th>
                    <th style="padding: 10px; text-align: left;">Status</th>
                </tr>
                <tr>
                    <td style="padding: 10px;">Token Savings</td>
                    <td style="padding: 10px;">≥70%</td>
                    <td style="padding: 10px;">{dashboard_data['overview']['avg_token_savings']:.1%}</td>
                    <td style="padding: 10px;">{"✅" if dashboard_data['overview']['avg_token_savings'] >= 0.7 else "❌"}</td>
                </tr>
                <tr>
                    <td style="padding: 10px;">Response Time</td>
                    <td style="padding: 10px;">≤200ms</td>
                    <td style="padding: 10px;">{dashboard_data['overview']['avg_response_time_ms']:.0f}ms</td>
                    <td style="padding: 10px;">{"✅" if dashboard_data['overview']['avg_response_time_ms'] <= 200 else "❌"}</td>
                </tr>
                <tr>
                    <td style="padding: 10px;">Cache Hit Rate</td>
                    <td style="padding: 10px;">≥80%</td>
                    <td style="padding: 10px;">{dashboard_data['overview']['cache_hit_rate']:.1%}</td>
                    <td style="padding: 10px;">{"✅" if dashboard_data['overview']['cache_hit_rate'] >= 0.8 else "❌"}</td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Static report generated: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate static report: {e}")
        raise