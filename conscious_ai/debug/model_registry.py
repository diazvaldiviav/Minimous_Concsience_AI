"""
Model Usage Registry for Consciousness Pipeline
===============================================
Tracks and logs which models are used in each phase for debugging,
monitoring, and optimization purposes.

Features:
- Session-based usage tracking
- Token consumption monitoring  
- Phase-specific model logging
- JSON persistence
- Real-time usage statistics
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ModelUsage:
    """Records usage of a specific model in a specific phase"""
    session_id: str
    timestamp: datetime
    phase: str
    model_name: str
    input_tokens: int
    output_tokens: int
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'phase': self.phase,
            'model_name': self.model_name,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'processing_time_ms': self.processing_time_ms,
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class SessionSummary:
    """Summary of model usage for a complete session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_queries: int
    phase_usage: Dict[str, str]  # phase -> model_name
    token_counts: Dict[str, Dict[str, int]]  # model -> {input, output}
    success_rate: float
    total_cost_estimate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_queries': self.total_queries,
            'phase_usage': self.phase_usage,
            'token_counts': self.token_counts,
            'success_rate': self.success_rate,
            'total_cost_estimate': self.total_cost_estimate
        }


class ModelRegistry:
    """
    Central registry for tracking model usage across consciousness pipeline.
    
    Provides:
    - Real-time usage logging
    - Session management
    - Cost estimation
    - Performance analytics
    - JSON persistence
    """
    
    def __init__(self, debug_dir: str = "./debug"):
        """
        Initialize model registry.
        
        Args:
            debug_dir: Directory for storing debug files
        """
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        
        # Current session
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now()
        self.session_usage: List[ModelUsage] = []
        
        # Persistence
        self.usage_file = self.debug_dir / "model_usage.json"
        self.sessions_file = self.debug_dir / "sessions.json"
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Model costs (approximate USD per 1K tokens)
        self.model_costs = {
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'local_mistral': {'input': 0.0, 'output': 0.0},
            'local_fallback': {'input': 0.0, 'output': 0.0}
        }
        
        logger.info(f"🔍 Model Registry initialized - Session: {self.session_id}")
    
    def log_model_usage(self, 
                       phase: str,
                       model_name: str,
                       input_tokens: int,
                       output_tokens: int,
                       processing_time_ms: float = 0.0,
                       success: bool = True,
                       error_message: Optional[str] = None) -> None:
        """
        Log usage of a model in a specific phase.
        
        Args:
            phase: Phase name (e.g., "phase_4_thoughts", "phase_7_final")
            model_name: Name of model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            processing_time_ms: Processing time in milliseconds
            success: Whether the operation succeeded
            error_message: Error message if failed
        """
        with self._lock:
            usage = ModelUsage(
                session_id=self.session_id,
                timestamp=datetime.now(),
                phase=phase,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                processing_time_ms=processing_time_ms,
                success=success,
                error_message=error_message
            )
            
            self.session_usage.append(usage)
            
            # Log to file immediately
            self._append_to_usage_file(usage)
            
            logger.debug(f"📊 Model usage logged: {phase} -> {model_name} "
                        f"({input_tokens}→{output_tokens} tokens)")
    
    def get_current_session_summary(self) -> SessionSummary:
        """Get summary of current session"""
        with self._lock:
            # Calculate phase usage (most recent model per phase)
            phase_usage = {}
            for usage in reversed(self.session_usage):  # Most recent first
                if usage.phase not in phase_usage:
                    phase_usage[usage.phase] = usage.model_name
            
            # Calculate token counts by model
            token_counts = {}
            for usage in self.session_usage:
                if usage.model_name not in token_counts:
                    token_counts[usage.model_name] = {'input': 0, 'output': 0}
                
                token_counts[usage.model_name]['input'] += usage.input_tokens
                token_counts[usage.model_name]['output'] += usage.output_tokens
            
            # Calculate success rate
            if self.session_usage:
                successful = sum(1 for usage in self.session_usage if usage.success)
                success_rate = successful / len(self.session_usage)
            else:
                success_rate = 1.0
            
            # Estimate costs
            total_cost = self._calculate_session_cost()
            
            return SessionSummary(
                session_id=self.session_id,
                start_time=self.session_start,
                end_time=None,  # Session still active
                total_queries=len(set(usage.timestamp for usage in self.session_usage)),
                phase_usage=phase_usage,
                token_counts=token_counts,
                success_rate=success_rate,
                total_cost_estimate=total_cost
            )
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics for current session"""
        with self._lock:
            if not self.session_usage:
                return {
                    'session_id': self.session_id,
                    'session_duration_minutes': 0,
                    'total_usage_count': 0,
                    'unique_models': [],
                    'phase_breakdown': {},
                    'token_usage': {},
                    'cost_breakdown': {},
                    'performance_metrics': {}
                }
            
            # Session duration
            duration = (datetime.now() - self.session_start).total_seconds() / 60
            
            # Unique models
            unique_models = list(set(usage.model_name for usage in self.session_usage))
            
            # Phase breakdown
            phase_breakdown = {}
            for usage in self.session_usage:
                if usage.phase not in phase_breakdown:
                    phase_breakdown[usage.phase] = {
                        'count': 0,
                        'models_used': set(),
                        'total_tokens': 0,
                        'avg_processing_time': 0
                    }
                
                phase_breakdown[usage.phase]['count'] += 1
                phase_breakdown[usage.phase]['models_used'].add(usage.model_name)
                phase_breakdown[usage.phase]['total_tokens'] += usage.input_tokens + usage.output_tokens
            
            # Convert sets to lists for JSON serialization
            for phase_data in phase_breakdown.values():
                phase_data['models_used'] = list(phase_data['models_used'])
                if phase_data['count'] > 0:
                    phase_data['avg_processing_time'] = sum(
                        u.processing_time_ms for u in self.session_usage 
                        if u.phase == phase_data
                    ) / phase_data['count']
            
            # Token usage by model
            token_usage = {}
            for usage in self.session_usage:
                if usage.model_name not in token_usage:
                    token_usage[usage.model_name] = {
                        'total_input': 0,
                        'total_output': 0,
                        'usage_count': 0
                    }
                
                token_usage[usage.model_name]['total_input'] += usage.input_tokens
                token_usage[usage.model_name]['total_output'] += usage.output_tokens
                token_usage[usage.model_name]['usage_count'] += 1
            
            # Cost breakdown
            cost_breakdown = {}
            for model, tokens in token_usage.items():
                if model in self.model_costs:
                    input_cost = (tokens['total_input'] / 1000) * self.model_costs[model]['input']
                    output_cost = (tokens['total_output'] / 1000) * self.model_costs[model]['output']
                    cost_breakdown[model] = {
                        'input_cost': input_cost,
                        'output_cost': output_cost,
                        'total_cost': input_cost + output_cost
                    }
            
            # Performance metrics
            processing_times = [u.processing_time_ms for u in self.session_usage if u.processing_time_ms > 0]
            performance_metrics = {
                'avg_processing_time_ms': sum(processing_times) / len(processing_times) if processing_times else 0,
                'min_processing_time_ms': min(processing_times) if processing_times else 0,
                'max_processing_time_ms': max(processing_times) if processing_times else 0,
                'success_rate': sum(1 for u in self.session_usage if u.success) / len(self.session_usage)
            }
            
            return {
                'session_id': self.session_id,
                'session_duration_minutes': duration,
                'total_usage_count': len(self.session_usage),
                'unique_models': unique_models,
                'phase_breakdown': phase_breakdown,
                'token_usage': token_usage,
                'cost_breakdown': cost_breakdown,
                'performance_metrics': performance_metrics
            }
    
    def end_session(self) -> SessionSummary:
        """End current session and return final summary"""
        with self._lock:
            summary = self.get_current_session_summary()
            summary.end_time = datetime.now()
            
            # Save session summary
            self._save_session_summary(summary)
            
            logger.info(f"📊 Session {self.session_id} ended - "
                       f"{len(self.session_usage)} operations, "
                       f"${summary.total_cost_estimate:.4f} estimated cost")
            
            return summary
    
    def start_new_session(self) -> str:
        """Start a new tracking session"""
        # End current session
        if self.session_usage:
            self.end_session()
        
        # Start new session
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now()
        self.session_usage.clear()
        
        logger.info(f"🔍 New session started: {self.session_id}")
        return self.session_id
    
    def get_historical_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical session summaries"""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)
                
                # Return most recent sessions
                return sorted(sessions, 
                            key=lambda x: x['start_time'], 
                            reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Failed to load historical sessions: {e}")
        
        return []
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"session_{timestamp}_{unique_id}"
    
    def _calculate_session_cost(self) -> float:
        """Calculate estimated cost for current session"""
        total_cost = 0.0
        
        for usage in self.session_usage:
            if usage.model_name in self.model_costs:
                costs = self.model_costs[usage.model_name]
                input_cost = (usage.input_tokens / 1000) * costs['input']
                output_cost = (usage.output_tokens / 1000) * costs['output']
                total_cost += input_cost + output_cost
        
        return total_cost
    
    def _append_to_usage_file(self, usage: ModelUsage) -> None:
        """Append usage record to file"""
        try:
            # Load existing data
            usage_data = []
            if self.usage_file.exists():
                with open(self.usage_file, 'r') as f:
                    usage_data = json.load(f)
            
            # Append new usage
            usage_data.append(usage.to_dict())
            
            # Keep only last 1000 records to prevent file bloat
            if len(usage_data) > 1000:
                usage_data = usage_data[-1000:]
            
            # Save back to file
            with open(self.usage_file, 'w') as f:
                json.dump(usage_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save usage data: {e}")
    
    def _save_session_summary(self, summary: SessionSummary) -> None:
        """Save session summary to file"""
        try:
            # Load existing sessions
            sessions = []
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)
            
            # Add new session
            sessions.append(summary.to_dict())
            
            # Keep only last 50 sessions
            if len(sessions) > 50:
                sessions = sessions[-50:]
            
            # Save back to file
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save session summary: {e}")


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get global model registry instance (singleton)"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


def log_phase_model_usage(phase: str, 
                         model: str,
                         input_tokens: int = 0,
                         output_tokens: int = 0,
                         processing_time_ms: float = 0.0,
                         success: bool = True,
                         error: Optional[str] = None) -> None:
    """Convenience function for logging model usage"""
    registry = get_model_registry()
    registry.log_model_usage(
        phase=phase,
        model_name=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        processing_time_ms=processing_time_ms,
        success=success,
        error_message=error
    )


# Phase-specific logging helpers

def log_phase4_usage(model: str, input_tokens: int, output_tokens: int, 
                    processing_time_ms: float = 0.0, success: bool = True):
    """Log Phase 4 (LLM Communication) model usage"""
    log_phase_model_usage(
        phase="phase_4_thoughts",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        processing_time_ms=processing_time_ms,
        success=success
    )


def log_phase7_consciousness_usage(model: str, input_tokens: int, output_tokens: int,
                                  processing_time_ms: float = 0.0, success: bool = True):
    """Log Phase 7 consciousness processing model usage"""
    log_phase_model_usage(
        phase="phase_7_consciousness",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        processing_time_ms=processing_time_ms,
        success=success
    )


def log_phase7_final_usage(model: str, input_tokens: int, output_tokens: int,
                          processing_time_ms: float = 0.0, success: bool = True):
    """Log Phase 7 final response generation model usage"""
    log_phase_model_usage(
        phase="phase_7_final",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        processing_time_ms=processing_time_ms,
        success=success
    )