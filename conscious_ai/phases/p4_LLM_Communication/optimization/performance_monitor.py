"""
Premium Performance Monitor for Phase 4 Layer 2
===============================================
Real-time monitoring with automatic optimization for 51GB RAM + 15GB VRAM.
Tracks memory usage, response times, and system stability with intelligent alerts.
"""

import logging
import time
import threading
import psutil
import torch
import queue
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics snapshot"""
    timestamp: float
    ram_usage_gb: float
    ram_usage_percent: float
    vram_usage_gb: float
    vram_usage_percent: float
    cpu_usage_percent: float
    gpu_utilization_percent: float
    active_requests: int
    avg_response_time_ms: float
    memory_efficiency_score: float
    thermal_status: str = "normal"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'ram_usage_gb': round(self.ram_usage_gb, 2),
            'ram_usage_percent': round(self.ram_usage_percent, 1),
            'vram_usage_gb': round(self.vram_usage_gb, 2),
            'vram_usage_percent': round(self.vram_usage_percent, 1),
            'cpu_usage_percent': round(self.cpu_usage_percent, 1),
            'gpu_utilization_percent': round(self.gpu_utilization_percent, 1),
            'active_requests': self.active_requests,
            'avg_response_time_ms': round(self.avg_response_time_ms, 1),
            'memory_efficiency_score': round(self.memory_efficiency_score, 1),
            'thermal_status': self.thermal_status
        }


@dataclass
class AlertConfiguration:
    """Configuration for performance alerts"""
    ram_warning_threshold: float = 0.80  # 80%
    ram_critical_threshold: float = 0.90  # 90%
    vram_warning_threshold: float = 0.80
    vram_critical_threshold: float = 0.90
    response_time_warning_ms: float = 8000  # 8 seconds
    response_time_critical_ms: float = 15000  # 15 seconds
    memory_efficiency_warning: float = 0.60  # 60%
    continuous_monitoring_seconds: int = 5
    alert_cooldown_minutes: int = 5


@dataclass
class OptimizationRecommendation:
    """Automated optimization recommendation"""
    category: str  # memory, performance, stability
    priority: int  # 1=low, 5=critical
    title: str
    description: str
    action_required: bool
    estimated_impact: str
    implementation_time: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category,
            'priority': self.priority,
            'title': self.title,
            'description': self.description,
            'action_required': self.action_required,
            'estimated_impact': self.estimated_impact,
            'implementation_time': self.implementation_time
        }


class PerformanceAlert:
    """Performance alert with severity and timing"""
    
    def __init__(self, alert_type: str, severity: str, message: str, metrics: PerformanceMetrics):
        self.alert_type = alert_type
        self.severity = severity  # info, warning, critical
        self.message = message
        self.timestamp = time.time()
        self.metrics_snapshot = metrics
        self.acknowledged = False
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'timestamp': self.timestamp,
            'acknowledged': self.acknowledged,
            'metrics': self.metrics_snapshot.to_dict()
        }


class MemoryAnalyzer:
    """Advanced memory analysis and optimization"""
    
    def __init__(self, target_ram_gb: float = 45.0, target_vram_gb: float = 13.0):
        self.target_ram_gb = target_ram_gb
        self.target_vram_gb = target_vram_gb
        self.fragmentation_history = deque(maxlen=100)
        
    def analyze_memory_usage(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Comprehensive memory usage analysis"""
        # Calculate efficiency scores
        ram_efficiency = 1.0 - abs(metrics.ram_usage_gb - self.target_ram_gb) / self.target_ram_gb
        vram_efficiency = 1.0 - abs(metrics.vram_usage_gb - self.target_vram_gb) / self.target_vram_gb
        
        # Detect memory fragmentation
        fragmentation_score = self._calculate_fragmentation()
        
        # Memory pressure analysis
        pressure_level = self._calculate_memory_pressure(metrics)
        
        return {
            'ram_efficiency': max(0, ram_efficiency),
            'vram_efficiency': max(0, vram_efficiency),
            'fragmentation_score': fragmentation_score,
            'pressure_level': pressure_level,
            'optimization_potential': self._calculate_optimization_potential(metrics),
            'recommended_actions': self._generate_memory_recommendations(metrics)
        }
    
    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation score"""
        try:
            # Use virtual memory statistics as proxy for fragmentation
            memory = psutil.virtual_memory()
            
            # Simple fragmentation estimation based on available vs free memory
            if memory.total > 0:
                fragmentation = 1.0 - (memory.available / memory.total)
                self.fragmentation_history.append(fragmentation)
                
                # Return average fragmentation over recent history
                return statistics.mean(self.fragmentation_history) if self.fragmentation_history else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Fragmentation calculation error: {e}")
            return 0.0
    
    def _calculate_memory_pressure(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall memory pressure level"""
        ram_pressure = metrics.ram_usage_percent / 100.0
        vram_pressure = metrics.vram_usage_percent / 100.0
        
        max_pressure = max(ram_pressure, vram_pressure)
        
        if max_pressure > 0.95:
            return "critical"
        elif max_pressure > 0.85:
            return "high"
        elif max_pressure > 0.70:
            return "moderate"
        else:
            return "low"
    
    def _calculate_optimization_potential(self, metrics: PerformanceMetrics) -> float:
        """Calculate potential for memory optimization (0-1)"""
        factors = []
        
        # RAM optimization potential
        if metrics.ram_usage_gb > self.target_ram_gb:
            factors.append((metrics.ram_usage_gb - self.target_ram_gb) / self.target_ram_gb)
        
        # VRAM optimization potential
        if metrics.vram_usage_gb > self.target_vram_gb:
            factors.append((metrics.vram_usage_gb - self.target_vram_gb) / self.target_vram_gb)
        
        # Fragmentation factor
        if self.fragmentation_history:
            factors.append(statistics.mean(self.fragmentation_history))
        
        return min(1.0, statistics.mean(factors)) if factors else 0.0
    
    def _generate_memory_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate specific memory optimization recommendations"""
        recommendations = []
        
        if metrics.ram_usage_percent > 85:
            recommendations.append("Consider reducing CPU model layers or batch size")
        
        if metrics.vram_usage_percent > 85:
            recommendations.append("Reduce GPU model layers or enable gradient checkpointing")
        
        if self.fragmentation_history and statistics.mean(self.fragmentation_history) > 0.3:
            recommendations.append("Memory fragmentation detected - consider restart")
        
        if metrics.memory_efficiency_score < 0.7:
            recommendations.append("Rebalance memory allocation between CPU and GPU")
        
        return recommendations


class PremiumPerformanceMonitor:
    """
    Premium performance monitor with real-time tracking and automatic optimization.
    Optimized for 51GB RAM + 15GB VRAM T4 configuration with intelligent alerts.
    """
    
    def __init__(self, config: Optional[AlertConfiguration] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or AlertConfiguration()
        self.monitoring_active = False
        
        # Data storage
        self.metrics_history = deque(maxlen=1000)  # ~1.4 hours at 5s intervals
        self.alert_history = deque(maxlen=100)
        self.active_alerts = []
        
        # Components
        self.memory_analyzer = MemoryAnalyzer()
        
        # Threading
        self._monitor_thread = None
        self._analytics_thread = None
        self._alert_queue = queue.Queue()
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.optimization_callbacks: List[Callable] = []
        
        # State tracking
        self.last_alert_times = {}
        self.optimization_recommendations = []
        
    def start_monitoring(self):
        """Start comprehensive performance monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        # Start analytics thread
        self._analytics_thread = threading.Thread(
            target=self._analytics_loop,
            daemon=True
        )
        self._analytics_thread.start()
        
        self.logger.info("📊 Premium performance monitoring started")
        self.logger.info(f"  RAM Target: {self.memory_analyzer.target_ram_gb:.1f}GB")
        self.logger.info(f"  VRAM Target: {self.memory_analyzer.target_vram_gb:.1f}GB")
        self.logger.info(f"  Alert Thresholds: RAM {self.config.ram_warning_threshold*100:.0f}%/{self.config.ram_critical_threshold*100:.0f}%, VRAM {self.config.vram_warning_threshold*100:.0f}%/{self.config.vram_critical_threshold*100:.0f}%")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        
        # Wait for threads to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
            
        if self._analytics_thread and self._analytics_thread.is_alive():
            self._analytics_thread.join(timeout=5.0)
        
        self.logger.info("📊 Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_performance_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Sleep until next collection
                time.sleep(self.config.continuous_monitoring_seconds)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _analytics_loop(self):
        """Analytics and optimization loop"""
        while self.monitoring_active:
            try:
                if len(self.metrics_history) >= 10:  # Need some data
                    # Generate optimization recommendations
                    self._generate_optimization_recommendations()
                    
                    # Cleanup old alerts
                    self._cleanup_old_alerts()
                
                time.sleep(60)  # Run analytics every minute
                
            except Exception as e:
                self.logger.error(f"Analytics loop error: {e}")
                time.sleep(60)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        timestamp = time.time()
        
        # RAM metrics
        memory = psutil.virtual_memory()
        ram_usage_gb = memory.used / (1024**3)
        ram_usage_percent = memory.percent
        
        # CPU metrics
        cpu_usage_percent = psutil.cpu_percent(interval=1)
        
        # GPU/VRAM metrics
        vram_usage_gb = 0.0
        vram_usage_percent = 0.0
        gpu_utilization_percent = 0.0
        
        if torch.cuda.is_available():
            try:
                # VRAM usage
                vram_allocated = torch.cuda.memory_allocated(0)
                vram_total = torch.cuda.get_device_properties(0).total_memory
                vram_usage_gb = vram_allocated / (1024**3)
                vram_usage_percent = (vram_allocated / vram_total) * 100
                
                # GPU utilization (basic estimation)
                gpu_utilization_percent = min(100, vram_usage_percent * 1.2)
                
            except Exception as e:
                self.logger.debug(f"GPU metrics collection error: {e}")
        
        # Calculate memory efficiency score
        memory_efficiency_score = self._calculate_memory_efficiency(
            ram_usage_gb, vram_usage_gb
        )
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            ram_usage_gb=ram_usage_gb,
            ram_usage_percent=ram_usage_percent,
            vram_usage_gb=vram_usage_gb,
            vram_usage_percent=vram_usage_percent,
            cpu_usage_percent=cpu_usage_percent,
            gpu_utilization_percent=gpu_utilization_percent,
            active_requests=0,  # TODO: Track from backend manager
            avg_response_time_ms=0.0,  # TODO: Track from backend manager
            memory_efficiency_score=memory_efficiency_score,
            thermal_status=self._get_thermal_status()
        )
        
        return metrics
    
    def _calculate_memory_efficiency(self, ram_usage_gb: float, vram_usage_gb: float) -> float:
        """Calculate overall memory efficiency score (0-100)"""
        target_ram = self.memory_analyzer.target_ram_gb
        target_vram = self.memory_analyzer.target_vram_gb
        
        # Efficiency is best when usage is close to targets
        ram_efficiency = 1.0 - abs(ram_usage_gb - target_ram) / target_ram
        vram_efficiency = 1.0 - abs(vram_usage_gb - target_vram) / target_vram
        
        # Weight RAM efficiency more heavily (more capacity available)
        overall_efficiency = (ram_efficiency * 0.6 + vram_efficiency * 0.4)
        
        return max(0, min(100, overall_efficiency * 100))
    
    def _get_thermal_status(self) -> str:
        """Get thermal status (placeholder - would need hardware-specific implementation)"""
        # TODO: Implement actual thermal monitoring
        # For now, estimate based on utilization
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                max_temp = max([temp.current for temp_list in temps.values() for temp in temp_list])
                if max_temp > 85:
                    return "hot"
                elif max_temp > 75:
                    return "warm"
                else:
                    return "normal"
        except:
            pass
        
        return "unknown"
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for alert conditions and generate alerts"""
        current_time = time.time()
        
        # RAM alerts
        self._check_memory_alerts("ram", metrics.ram_usage_percent, 
                                self.config.ram_warning_threshold * 100,
                                self.config.ram_critical_threshold * 100,
                                current_time, metrics)
        
        # VRAM alerts
        self._check_memory_alerts("vram", metrics.vram_usage_percent,
                                self.config.vram_warning_threshold * 100,
                                self.config.vram_critical_threshold * 100,
                                current_time, metrics)
        
        # Response time alerts (if data available)
        if metrics.avg_response_time_ms > 0:
            self._check_response_time_alerts(metrics.avg_response_time_ms, current_time, metrics)
        
        # Memory efficiency alerts
        if metrics.memory_efficiency_score < self.config.memory_efficiency_warning * 100:
            self._create_alert("memory_efficiency", "warning",
                             f"Memory efficiency low: {metrics.memory_efficiency_score:.1f}%",
                             current_time, metrics)
    
    def _check_memory_alerts(self, memory_type: str, usage_percent: float,
                           warning_threshold: float, critical_threshold: float,
                           current_time: float, metrics: PerformanceMetrics):
        """Check memory usage alerts"""
        if usage_percent >= critical_threshold:
            self._create_alert(f"{memory_type}_critical", "critical",
                             f"{memory_type.upper()} usage critical: {usage_percent:.1f}%",
                             current_time, metrics)
        elif usage_percent >= warning_threshold:
            self._create_alert(f"{memory_type}_warning", "warning",
                             f"{memory_type.upper()} usage high: {usage_percent:.1f}%",
                             current_time, metrics)
    
    def _check_response_time_alerts(self, response_time_ms: float, current_time: float, metrics: PerformanceMetrics):
        """Check response time alerts"""
        if response_time_ms >= self.config.response_time_critical_ms:
            self._create_alert("response_time_critical", "critical",
                             f"Response time critical: {response_time_ms:.0f}ms",
                             current_time, metrics)
        elif response_time_ms >= self.config.response_time_warning_ms:
            self._create_alert("response_time_warning", "warning",
                             f"Response time high: {response_time_ms:.0f}ms",
                             current_time, metrics)
    
    def _create_alert(self, alert_type: str, severity: str, message: str,
                     current_time: float, metrics: PerformanceMetrics):
        """Create alert with cooldown logic"""
        # Check cooldown
        last_alert_time = self.last_alert_times.get(alert_type, 0)
        cooldown_seconds = self.config.alert_cooldown_minutes * 60
        
        if current_time - last_alert_time < cooldown_seconds:
            return  # Still in cooldown
        
        # Create alert
        alert = PerformanceAlert(alert_type, severity, message, metrics)
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_times[alert_type] = current_time
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
        
        # Log alert
        log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
        self.logger.log(log_level, f"🚨 {message}")
    
    def _generate_optimization_recommendations(self):
        """Generate intelligent optimization recommendations"""
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 readings
        current_metrics = recent_metrics[-1]
        
        recommendations = []
        
        # Memory analysis
        memory_analysis = self.memory_analyzer.analyze_memory_usage(current_metrics)
        
        # High memory usage recommendations
        if current_metrics.ram_usage_percent > 85:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority=4,
                title="High RAM Usage Detected",
                description=f"RAM usage at {current_metrics.ram_usage_percent:.1f}%. Consider reducing model layers on CPU or implementing more aggressive quantization.",
                action_required=True,
                estimated_impact="15-25% memory reduction",
                implementation_time="5-10 minutes"
            ))
        
        if current_metrics.vram_usage_percent > 85:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority=4,
                title="High VRAM Usage Detected", 
                description=f"VRAM usage at {current_metrics.vram_usage_percent:.1f}%. Consider moving layers to CPU or enabling gradient checkpointing.",
                action_required=True,
                estimated_impact="10-20% VRAM reduction",
                implementation_time="2-5 minutes"
            ))
        
        # Performance recommendations
        avg_efficiency = statistics.mean([m.memory_efficiency_score for m in recent_metrics])
        if avg_efficiency < 60:
            recommendations.append(OptimizationRecommendation(
                category="performance",
                priority=3,
                title="Low Memory Efficiency",
                description=f"Average efficiency {avg_efficiency:.1f}%. Memory allocation may not be optimal for current workload.",
                action_required=False,
                estimated_impact="10-15% efficiency improvement",
                implementation_time="10-15 minutes"
            ))
        
        # Stability recommendations
        recent_ram_variance = statistics.variance([m.ram_usage_percent for m in recent_metrics])
        if recent_ram_variance > 100:  # High variance
            recommendations.append(OptimizationRecommendation(
                category="stability",
                priority=2,
                title="Memory Usage Instability",
                description="RAM usage showing high variance. May indicate memory leaks or inefficient allocation patterns.",
                action_required=False,
                estimated_impact="Improved stability",
                implementation_time="Investigation required"
            ))
        
        # Update recommendations
        self.optimization_recommendations = recommendations
        
        # Trigger optimization callbacks
        for callback in self.optimization_callbacks:
            try:
                callback(recommendations)
            except Exception as e:
                self.logger.error(f"Optimization callback error: {e}")
    
    def _cleanup_old_alerts(self):
        """Clean up acknowledged and old alerts"""
        current_time = time.time()
        cutoff_time = current_time - (24 * 3600)  # 24 hours
        
        # Remove old alerts from active list
        self.active_alerts = [
            alert for alert in self.active_alerts
            if not alert.acknowledged and alert.timestamp > cutoff_time
        ]
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent performance metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified time period"""
        if not self.metrics_history:
            return {'error': 'No monitoring data available'}
        
        # Filter metrics for time period
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            recent_metrics = [self.metrics_history[-1]]  # At least show current
        
        # Calculate statistics
        ram_usage = [m.ram_usage_percent for m in recent_metrics]
        vram_usage = [m.vram_usage_percent for m in recent_metrics]
        efficiency = [m.memory_efficiency_score for m in recent_metrics]
        
        return {
            'time_period_hours': hours,
            'samples_collected': len(recent_metrics),
            'ram_stats': {
                'current_percent': ram_usage[-1],
                'average_percent': statistics.mean(ram_usage),
                'peak_percent': max(ram_usage),
                'variance': statistics.variance(ram_usage) if len(ram_usage) > 1 else 0
            },
            'vram_stats': {
                'current_percent': vram_usage[-1],
                'average_percent': statistics.mean(vram_usage),
                'peak_percent': max(vram_usage),
                'variance': statistics.variance(vram_usage) if len(vram_usage) > 1 else 0
            },
            'efficiency_stats': {
                'current_score': efficiency[-1],
                'average_score': statistics.mean(efficiency),
                'trend': 'improving' if len(efficiency) > 5 and efficiency[-1] > statistics.mean(efficiency[:-5]) else 'stable'
            },
            'alert_summary': {
                'active_alerts': len(self.active_alerts),
                'critical_alerts': len([a for a in self.active_alerts if a.severity == 'critical']),
                'warning_alerts': len([a for a in self.active_alerts if a.severity == 'warning'])
            },
            'optimization_recommendations': len(self.optimization_recommendations)
        }
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def add_optimization_callback(self, callback: Callable[[List[OptimizationRecommendation]], None]):
        """Add callback for optimization recommendations"""
        self.optimization_callbacks.append(callback)
    
    def acknowledge_alert(self, alert_type: str):
        """Acknowledge an alert to stop notifications"""
        for alert in self.active_alerts:
            if alert.alert_type == alert_type:
                alert.acknowledged = True
        
        self.logger.info(f"Alert acknowledged: {alert_type}")
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time data for dashboard display"""
        current = self.get_current_metrics()
        if not current:
            return {'error': 'No current data'}
        
        return {
            'timestamp': current.timestamp,
            'memory': {
                'ram_usage_gb': current.ram_usage_gb,
                'ram_percent': current.ram_usage_percent,
                'vram_usage_gb': current.vram_usage_gb,
                'vram_percent': current.vram_usage_percent,
                'efficiency_score': current.memory_efficiency_score
            },
            'performance': {
                'cpu_percent': current.cpu_usage_percent,
                'gpu_percent': current.gpu_utilization_percent,
                'avg_response_time_ms': current.avg_response_time_ms,
                'active_requests': current.active_requests
            },
            'health': {
                'thermal_status': current.thermal_status,
                'active_alerts': len(self.active_alerts),
                'optimization_recommendations': len(self.optimization_recommendations)
            }
        }
    
    def detect_memory_fragmentation(self) -> Dict[str, Any]:
        """Detect memory fragmentation and provide optimization suggestions"""
        try:
            # Analyze RAM fragmentation
            memory = psutil.virtual_memory()
            fragmentation_info = {
                'ram_fragmentation': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'fragmentation_score': self._calculate_ram_fragmentation(),
                    'large_blocks_available': memory.available > (10 * 1024**3)  # >10GB blocks
                }
            }
            
            # Analyze VRAM fragmentation if available
            if torch.cuda.is_available():
                vram_info = {
                    'vram_fragmentation': {
                        'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                        'cached_gb': torch.cuda.memory_reserved() / (1024**3),
                        'fragmentation_score': self._calculate_vram_fragmentation(),
                        'needs_cleanup': torch.cuda.memory_reserved() > torch.cuda.memory_allocated() * 1.5
                    }
                }
                fragmentation_info.update(vram_info)
            
            # Generate optimization suggestions
            suggestions = self._generate_fragmentation_optimizations(fragmentation_info)
            fragmentation_info['optimization_suggestions'] = suggestions
            
            return fragmentation_info
            
        except Exception as e:
            logger.error(f"Memory fragmentation detection failed: {e}")
            return {'error': str(e)}
    
    def _calculate_ram_fragmentation(self) -> float:
        """Calculate RAM fragmentation score (0-100, higher = more fragmented)"""
        try:
            memory = psutil.virtual_memory()
            # Simple fragmentation heuristic based on available vs free memory
            if memory.total > 0:
                utilization = (memory.total - memory.available) / memory.total
                # High utilization with low available indicates fragmentation
                if utilization > 0.8 and memory.available < (5 * 1024**3):  # <5GB available
                    return min(95, 60 + (utilization - 0.8) * 175)  # 60-95 range
                else:
                    return max(0, utilization * 40)  # 0-40 range for normal usage
            return 0
        except:
            return 50  # Unknown, assume moderate fragmentation
    
    def _calculate_vram_fragmentation(self) -> float:
        """Calculate VRAM fragmentation score (0-100, higher = more fragmented)"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                
                if reserved > 0:
                    # Fragmentation occurs when reserved >> allocated
                    waste_ratio = (reserved - allocated) / reserved
                    return min(90, waste_ratio * 100)
            return 0
        except:
            return 0
    
    def _generate_fragmentation_optimizations(self, fragmentation_info: Dict) -> List[str]:
        """Generate memory fragmentation optimization suggestions"""
        suggestions = []
        
        # RAM optimizations
        ram_frag = fragmentation_info.get('ram_fragmentation', {})
        if ram_frag.get('fragmentation_score', 0) > 60:
            suggestions.append("RAM fragmentation high: Consider restarting model loading process")
        if not ram_frag.get('large_blocks_available', True):
            suggestions.append("Low RAM availability: Reduce model batch size or enable gradient checkpointing")
        
        # VRAM optimizations  
        vram_frag = fragmentation_info.get('vram_fragmentation', {})
        if vram_frag.get('fragmentation_score', 0) > 50:
            suggestions.append("VRAM fragmentation detected: Run torch.cuda.empty_cache()")
        if vram_frag.get('needs_cleanup', False):
            suggestions.append("VRAM cache cleanup recommended: Significant unused reserved memory")
            
        return suggestions
    
    def auto_optimize_layer_distribution(self, current_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-optimization of layer distribution based on usage patterns"""
        try:
            logger.info("🔍 Analyzing layer distribution for optimization...")
            
            # Analyze current performance metrics
            current_metrics = self.get_current_metrics()
            fragmentation_info = self.detect_memory_fragmentation()
            
            optimization_plan = {
                'current_state': current_distribution,
                'recommendations': [],
                'performance_impact': 'unknown',
                'confidence': 0.0
            }
            
            # RAM-based optimizations
            if current_metrics and current_metrics.ram_usage_percent > 85:
                optimization_plan['recommendations'].append({
                    'type': 'reduce_cpu_layers',
                    'reason': f'RAM usage high at {current_metrics.ram_usage_percent:.1f}%',
                    'action': 'Move 2-3 less critical layers to GPU if VRAM available'
                })
            
            # VRAM-based optimizations  
            if current_metrics and current_metrics.vram_usage_percent > 90:
                optimization_plan['recommendations'].append({
                    'type': 'reduce_gpu_layers', 
                    'reason': f'VRAM usage critical at {current_metrics.vram_usage_percent:.1f}%',
                    'action': 'Move MLP layers to CPU, keep attention on GPU'
                })
            elif current_metrics and current_metrics.vram_usage_percent < 60:
                optimization_plan['recommendations'].append({
                    'type': 'increase_gpu_layers',
                    'reason': f'VRAM underutilized at {current_metrics.vram_usage_percent:.1f}%', 
                    'action': 'Move more attention layers to GPU for better performance'
                })
            
            # Response time optimizations
            if current_metrics and current_metrics.avg_response_time_ms > 10000:  # >10s target
                optimization_plan['recommendations'].append({
                    'type': 'performance_tuning',
                    'reason': f'Response time slow at {current_metrics.avg_response_time_ms:.1f}ms',
                    'action': 'Increase GPU layer allocation, enable flash attention if not active'
                })
            
            # Calculate confidence based on data quality
            if current_metrics and len(self.metrics_history) > 10:
                optimization_plan['confidence'] = min(0.9, len(self.metrics_history) / 50.0)
            else:
                optimization_plan['confidence'] = 0.3
                
            # Set performance impact prediction
            if len(optimization_plan['recommendations']) == 0:
                optimization_plan['performance_impact'] = 'minimal'
            elif any('critical' in str(rec).lower() for rec in optimization_plan['recommendations']):
                optimization_plan['performance_impact'] = 'high'
            else:
                optimization_plan['performance_impact'] = 'moderate'
            
            logger.info(f"✅ Layer distribution analysis complete: {len(optimization_plan['recommendations'])} recommendations")
            return optimization_plan
            
        except Exception as e:
            logger.error(f"Layer distribution optimization failed: {e}")
            return {'error': str(e)}
    
    def get_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive analytics data for real-time dashboard"""
        try:
            current_metrics = self.get_current_metrics()
            if not current_metrics:
                return {'error': 'No current metrics available'}
            
            # Historical trends (last 60 data points)
            recent_metrics = list(self.metrics_history)[-60:] if len(self.metrics_history) > 0 else []
            
            dashboard_data = {
                'realtime': {
                    'timestamp': datetime.now().isoformat(),
                    'system_status': self._get_system_health_status(),
                    'current_metrics': current_metrics.to_dict()
                },
                'trends': {
                    'ram_usage_trend': [m.ram_usage_percent for m in recent_metrics],
                    'vram_usage_trend': [m.vram_usage_percent for m in recent_metrics],
                    'response_time_trend': [m.avg_response_time_ms for m in recent_metrics if m.avg_response_time_ms > 0],
                    'efficiency_trend': [m.memory_efficiency_score for m in recent_metrics]
                },
                'alerts': {
                    'active_count': len(self.active_alerts),
                    'recent_alerts': [
                        {
                            'type': alert.alert_type,
                            'level': alert.severity,
                            'message': alert.message,
                            'timestamp': alert.timestamp
                        } for alert in sorted(self.active_alerts, key=lambda x: x.timestamp, reverse=True)[:10]
                    ]
                },
                'optimizations': {
                    'recommendations_count': len(self.optimization_recommendations),
                    'recent_recommendations': [
                        {
                            'category': rec.category,
                            'title': rec.title,
                            'impact': rec.estimated_impact,
                            'timestamp': rec.timestamp
                        } for rec in self.optimization_recommendations[-5:]
                    ]
                },
                'hardware': {
                    'fragmentation_status': self.detect_memory_fragmentation(),
                    'thermal_status': current_metrics.thermal_status,
                    'uptime_hours': (time.time() - self.start_time) / 3600 if hasattr(self, 'start_time') else 0
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Analytics dashboard data generation failed: {e}")
            return {'error': str(e)}
    
    def _get_system_health_status(self) -> str:
        """Get overall system health status"""
        current_metrics = self.get_current_metrics()
        if not current_metrics:
            return 'unknown'
        
        # Check critical conditions
        if (current_metrics.ram_usage_percent > 95 or 
            current_metrics.vram_usage_percent > 95 or
            len([a for a in self.active_alerts if 'critical' in a.severity.lower()]) > 0):
            return 'critical'
        
        # Check warning conditions
        if (current_metrics.ram_usage_percent > 85 or 
            current_metrics.vram_usage_percent > 85 or
            current_metrics.avg_response_time_ms > 15000 or  # >15s
            len(self.active_alerts) > 3):
            return 'warning'
        
        # Check if performing well
        if (current_metrics.memory_efficiency_score > 80 and
            current_metrics.avg_response_time_ms < 8000 and  # <8s
            len(self.active_alerts) == 0):
            return 'excellent'
        
        return 'good'