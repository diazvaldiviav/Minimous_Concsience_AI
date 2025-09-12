"""
Comprehensive metrics collection and monitoring system.

This module provides detailed metrics collection, aggregation, and monitoring
capabilities for the SC Memory System, including real-time dashboards and
performance tracking.
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Deque
from enum import Enum
import json

import psutil

from src.core.config import Settings
from src.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """A time series of metric points."""
    name: str
    metric_type: MetricType
    description: str
    points: Deque[MetricPoint] = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Add a metric point."""
        point_labels = {**self.labels, **(labels or {})}
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=point_labels
        )
        self.points.append(point)
    
    def get_latest_value(self) -> Optional[float]:
        """Get the latest metric value."""
        return self.points[-1].value if self.points else None
    
    def get_average(self, time_window: Optional[timedelta] = None) -> Optional[float]:
        """Get average value over time window."""
        if not self.points:
            return None
        
        if time_window:
            cutoff = datetime.utcnow() - time_window
            relevant_points = [p for p in self.points if p.timestamp >= cutoff]
        else:
            relevant_points = list(self.points)
        
        if not relevant_points:
            return None
        
        return sum(p.value for p in relevant_points) / len(relevant_points)
    
    def get_percentile(self, percentile: float, time_window: Optional[timedelta] = None) -> Optional[float]:
        """Get percentile value over time window."""
        if not self.points:
            return None
        
        if time_window:
            cutoff = datetime.utcnow() - time_window
            relevant_points = [p for p in self.points if p.timestamp >= cutoff]
        else:
            relevant_points = list(self.points)
        
        if not relevant_points:
            return None
        
        values = sorted([p.value for p in relevant_points])
        index = int(len(values) * percentile / 100.0)
        return values[min(index, len(values) - 1)]


class MetricsCollector:
    """Core metrics collection and aggregation system."""
    
    def __init__(self, settings: Settings):
        """
        Initialize metrics collector.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self._metrics: Dict[str, MetricSeries] = {}
        self._lock = threading.RLock()
        
        # System monitoring
        self._monitor_system = True
        self._system_monitor_task: Optional[asyncio.Task] = None
        self._system_monitor_interval = 5.0  # seconds
        
        # Performance counters
        self.map_request_count = 0
        self.mep_request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize core metrics
        self._initialize_core_metrics()
    
    def _initialize_core_metrics(self) -> None:
        """Initialize core system metrics."""
        core_metrics = [
            # MAP API metrics
            ("map_requests_total", MetricType.COUNTER, "Total MAP API requests"),
            ("map_request_duration_ms", MetricType.HISTOGRAM, "MAP request duration in milliseconds"),
            ("map_response_tokens", MetricType.HISTOGRAM, "Number of tokens in MAP responses"),
            ("map_token_savings_ratio", MetricType.HISTOGRAM, "Token savings ratio for MAP responses"),
            
            # MEP API metrics
            ("mep_proposals_total", MetricType.COUNTER, "Total MEP proposals received"),
            ("mep_processing_duration_ms", MetricType.HISTOGRAM, "MEP processing duration in milliseconds"),
            ("mep_success_rate", MetricType.GAUGE, "MEP processing success rate"),
            
            # Cache metrics
            ("cache_hits_total", MetricType.COUNTER, "Total cache hits"),
            ("cache_misses_total", MetricType.COUNTER, "Total cache misses"),
            ("cache_hit_rate", MetricType.GAUGE, "Cache hit rate percentage"),
            
            # System metrics
            ("system_memory_usage_mb", MetricType.GAUGE, "System memory usage in MB"),
            ("system_cpu_usage_percent", MetricType.GAUGE, "System CPU usage percentage"),
            ("system_disk_usage_percent", MetricType.GAUGE, "System disk usage percentage"),
            
            # Model metrics
            ("model_load_duration_ms", MetricType.HISTOGRAM, "Model loading duration in milliseconds"),
            ("adapter_load_duration_ms", MetricType.HISTOGRAM, "Adapter loading duration in milliseconds"),
            ("active_adapters_count", MetricType.GAUGE, "Number of active loaded adapters"),
            
            # Processing metrics
            ("batch_processing_items", MetricType.HISTOGRAM, "Items processed in batch operations"),
            ("batch_processing_duration_ms", MetricType.HISTOGRAM, "Batch processing duration in milliseconds"),
            ("context_generation_tokens", MetricType.HISTOGRAM, "Tokens generated in context creation"),
        ]
        
        for name, metric_type, description in core_metrics:
            self._register_metric(name, metric_type, description)
    
    def _register_metric(
        self, 
        name: str, 
        metric_type: MetricType, 
        description: str,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Register a new metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MetricSeries(
                    name=name,
                    metric_type=metric_type,
                    description=description,
                    labels=labels or {}
                )
    
    def increment_counter(
        self, 
        metric_name: str, 
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        with self._lock:
            if metric_name in self._metrics:
                current_value = self._metrics[metric_name].get_latest_value() or 0.0
                self._metrics[metric_name].add_point(current_value + value, labels)
    
    def set_gauge(
        self, 
        metric_name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value."""
        with self._lock:
            if metric_name in self._metrics:
                self._metrics[metric_name].add_point(value, labels)
    
    def observe_histogram(
        self, 
        metric_name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Observe a value in a histogram metric."""
        with self._lock:
            if metric_name in self._metrics:
                self._metrics[metric_name].add_point(value, labels)
    
    def time_function(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator to time function execution."""
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        duration_ms = (time.time() - start_time) * 1000
                        self.observe_histogram(metric_name, duration_ms, labels)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        duration_ms = (time.time() - start_time) * 1000
                        self.observe_histogram(metric_name, duration_ms, labels)
                return sync_wrapper
        return decorator
    
    async def start_system_monitoring(self) -> None:
        """Start system resource monitoring."""
        if self._system_monitor_task:
            return
        
        logger.info("Starting system monitoring")
        self._monitor_system = True
        self._system_monitor_task = asyncio.create_task(self._system_monitor_loop())
    
    async def stop_system_monitoring(self) -> None:
        """Stop system resource monitoring."""
        self._monitor_system = False
        if self._system_monitor_task:
            self._system_monitor_task.cancel()
            try:
                await self._system_monitor_task
            except asyncio.CancelledError:
                pass
            self._system_monitor_task = None
        logger.info("System monitoring stopped")
    
    async def _system_monitor_loop(self) -> None:
        """System monitoring loop."""
        while self._monitor_system:
            try:
                # Collect system metrics
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                disk = psutil.disk_usage('/')
                
                # Update metrics
                self.set_gauge("system_memory_usage_mb", memory.used / (1024 * 1024))
                self.set_gauge("system_cpu_usage_percent", cpu_percent)
                self.set_gauge("system_disk_usage_percent", disk.percent)
                
                # Calculate derived metrics
                self._update_derived_metrics()
                
                await asyncio.sleep(self._system_monitor_interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self._system_monitor_interval)
    
    def _update_derived_metrics(self) -> None:
        """Update derived metrics like rates and ratios."""
        try:
            # Cache hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            if total_cache_requests > 0:
                hit_rate = (self.cache_hits / total_cache_requests) * 100
                self.set_gauge("cache_hit_rate", hit_rate)
            
        except Exception as e:
            logger.error(f"Failed to update derived metrics: {e}")
    
    def track_map_request(self, duration_ms: float, response_tokens: int, token_savings_ratio: float) -> None:
        """Track MAP API request metrics."""
        self.map_request_count += 1
        self.increment_counter("map_requests_total")
        self.observe_histogram("map_request_duration_ms", duration_ms)
        self.observe_histogram("map_response_tokens", response_tokens)
        self.observe_histogram("map_token_savings_ratio", token_savings_ratio)
    
    def track_mep_proposal(self, duration_ms: float, success: bool) -> None:
        """Track MEP proposal metrics."""
        self.mep_request_count += 1
        self.increment_counter("mep_proposals_total")
        self.observe_histogram("mep_processing_duration_ms", duration_ms)
        
        # Update success rate
        if hasattr(self, '_mep_successes'):
            if success:
                self._mep_successes += 1
        else:
            self._mep_successes = 1 if success else 0
        
        success_rate = (self._mep_successes / self.mep_request_count) * 100
        self.set_gauge("mep_success_rate", success_rate)
    
    def track_cache_hit(self) -> None:
        """Track cache hit."""
        self.cache_hits += 1
        self.increment_counter("cache_hits_total")
    
    def track_cache_miss(self) -> None:
        """Track cache miss."""
        self.cache_misses += 1
        self.increment_counter("cache_misses_total")
    
    def track_model_load(self, duration_ms: float, adapter: bool = False) -> None:
        """Track model/adapter loading metrics."""
        metric_name = "adapter_load_duration_ms" if adapter else "model_load_duration_ms"
        self.observe_histogram(metric_name, duration_ms)
    
    def track_batch_processing(self, items_count: int, duration_ms: float) -> None:
        """Track batch processing metrics."""
        self.observe_histogram("batch_processing_items", items_count)
        self.observe_histogram("batch_processing_duration_ms", duration_ms)
    
    def get_metrics_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {}
            }
            
            for name, metric_series in self._metrics.items():
                if not metric_series.points:
                    continue
                
                metric_data = {
                    "type": metric_series.metric_type,
                    "description": metric_series.description,
                    "latest_value": metric_series.get_latest_value(),
                }
                
                if metric_series.metric_type == MetricType.HISTOGRAM:
                    metric_data.update({
                        "avg": metric_series.get_average(time_window),
                        "p50": metric_series.get_percentile(50, time_window),
                        "p95": metric_series.get_percentile(95, time_window),
                        "p99": metric_series.get_percentile(99, time_window),
                    })
                
                summary["metrics"][name] = metric_data
            
            # Add request counts
            summary["request_counts"] = {
                "map_requests": self.map_request_count,
                "mep_requests": self.mep_request_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            }
            
            return summary
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display."""
        five_minutes = timedelta(minutes=5)
        summary = self.get_metrics_summary(five_minutes)
        
        # Format for dashboard
        dashboard_data = {
            "timestamp": summary["timestamp"],
            "overview": {
                "map_requests_per_minute": summary["metrics"].get("map_requests_total", {}).get("avg", 0) * 12,  # 5min avg * 12
                "avg_response_time_ms": summary["metrics"].get("map_request_duration_ms", {}).get("avg", 0),
                "p95_response_time_ms": summary["metrics"].get("map_request_duration_ms", {}).get("p95", 0),
                "avg_token_savings": summary["metrics"].get("map_token_savings_ratio", {}).get("avg", 0),
                "cache_hit_rate": summary["metrics"].get("cache_hit_rate", {}).get("latest_value", 0),
            },
            "system": {
                "memory_usage_mb": summary["metrics"].get("system_memory_usage_mb", {}).get("latest_value", 0),
                "cpu_usage_percent": summary["metrics"].get("system_cpu_usage_percent", {}).get("latest_value", 0),
                "disk_usage_percent": summary["metrics"].get("system_disk_usage_percent", {}).get("latest_value", 0),
            },
            "performance": {
                "total_requests": summary["request_counts"]["map_requests"] + summary["request_counts"]["mep_requests"],
                "successful_requests": summary["request_counts"]["map_requests"],  # Simplified
                "avg_processing_time": summary["metrics"].get("mep_processing_duration_ms", {}).get("avg", 0),
            }
        }
        
        return dashboard_data
    
    async def export_metrics(self, output_path: str, format: str = "json") -> None:
        """Export metrics to file."""
        try:
            summary = self.get_metrics_summary()
            
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Metrics exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup metrics collector."""
        await self.stop_system_monitoring()
        logger.info("Metrics collector cleaned up")