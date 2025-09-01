"""
Process Logger Module for Phase 5.5
===================================
Event capture system that intercepts key moments from each phase execution.
Implements a lightweight circular buffer tracking phase transitions and cognitive events.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

try:
    from ...config.narrative_config import EventType, NARRATIVE_CONFIG
except ImportError:
    from conscious_ai.config.narrative_config import EventType, NARRATIVE_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessEvent:
    """
    Represents a single consciousness event captured during pipeline execution.
    """
    event_type: EventType
    phase_name: str
    description: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_level: Optional[float] = None
    emotional_state: Optional[str] = None
    
    @property
    def age_ms(self) -> float:
        """Age of this event in milliseconds"""
        return (time.time() - self.timestamp) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_type': self.event_type.value,
            'phase_name': self.phase_name,
            'description': self.description,
            'timestamp': self.timestamp,
            'age_ms': self.age_ms,
            'metadata': self.metadata,
            'confidence_level': self.confidence_level,
            'emotional_state': self.emotional_state
        }
    
    def is_similar_to(self, other: 'ConsciousnessEvent', threshold: float = 0.8) -> bool:
        """Check if this event is similar to another (for deduplication)"""
        if not isinstance(other, ConsciousnessEvent):
            return False
            
        # Same event type and phase
        if self.event_type != other.event_type or self.phase_name != other.phase_name:
            return False
        
        # Similar descriptions (basic text similarity)
        desc_words1 = set(self.description.lower().split())
        desc_words2 = set(other.description.lower().split())
        
        if not desc_words1 or not desc_words2:
            return False
            
        overlap = len(desc_words1 & desc_words2) / len(desc_words1 | desc_words2)
        return overlap >= threshold


class ProcessLogger:
    """
    Lightweight event capture system for consciousness pipeline events.
    Uses circular buffer to maintain recent events with zero-blocking operation.
    """
    
    def __init__(self, max_events: int = None):
        """
        Initialize the process logger.
        
        Args:
            max_events: Maximum number of events to retain (defaults to config)
        """
        self.max_events = max_events or NARRATIVE_CONFIG.get('max_events', 20)
        self.events: Deque[ConsciousnessEvent] = deque(maxlen=self.max_events)
        self._last_confidence = None
        self._last_emotion = None
        self._session_start = time.time()
        self._event_count = 0
        
        # Performance tracking
        self._capture_times = []
        self._max_capture_times = 100
        
        logger.debug(f"ProcessLogger initialized with max_events={self.max_events}")
    
    def log_phase_transition(self, 
                           phase_name: str, 
                           description: str,
                           confidence: Optional[float] = None,
                           emotion: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a phase transition event.
        
        Args:
            phase_name: Name of the phase (e.g., "Phase 1", "Phase 2")
            description: Human-readable description of what happened
            confidence: Current confidence level (0-1)
            emotion: Current emotional state
            metadata: Additional contextual data
        """
        self._capture_event(
            event_type=EventType.PHASE_TRANSITION,
            phase_name=phase_name,
            description=description,
            confidence=confidence,
            emotion=emotion,
            metadata=metadata or {}
        )
    
    def log_decision_point(self,
                          phase_name: str,
                          decision_description: str,
                          confidence: Optional[float] = None,
                          alternatives: Optional[List[str]] = None,
                          reasoning: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a cognitive decision point.
        
        Args:
            phase_name: Phase where decision occurred
            decision_description: What decision was made
            confidence: Decision confidence level
            alternatives: Other options considered
            reasoning: Why this decision was made
            metadata: Additional context
        """
        meta = metadata or {}
        if alternatives:
            meta['alternatives'] = alternatives
        if reasoning:
            meta['reasoning'] = reasoning
            
        self._capture_event(
            event_type=EventType.DECISION_POINT,
            phase_name=phase_name,
            description=decision_description,
            confidence=confidence,
            metadata=meta
        )
    
    def log_confidence_change(self,
                            phase_name: str,
                            old_confidence: float,
                            new_confidence: float,
                            reason: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log significant confidence level changes.
        
        Args:
            phase_name: Phase where change occurred
            old_confidence: Previous confidence level
            new_confidence: New confidence level  
            reason: Why confidence changed
            metadata: Additional context
        """
        change_magnitude = abs(new_confidence - old_confidence)
        threshold = NARRATIVE_CONFIG.get('confidence_change_threshold', 0.20)
        
        if change_magnitude >= threshold:
            direction = "increased" if new_confidence > old_confidence else "decreased"
            description = f"Confidence {direction} from {old_confidence:.1%} to {new_confidence:.1%}"
            
            meta = metadata or {}
            meta.update({
                'old_confidence': old_confidence,
                'new_confidence': new_confidence,
                'change_magnitude': change_magnitude,
                'direction': direction
            })
            
            if reason:
                meta['reason'] = reason
                description += f": {reason}"
            
            self._capture_event(
                event_type=EventType.CONFIDENCE_CHANGE,
                phase_name=phase_name,
                description=description,
                confidence=new_confidence,
                metadata=meta
            )
    
    def log_emotional_shift(self,
                          phase_name: str,
                          old_emotion: str,
                          new_emotion: str,
                          trigger: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log emotional state transitions.
        
        Args:
            phase_name: Phase where shift occurred
            old_emotion: Previous emotional state
            new_emotion: New emotional state
            trigger: What caused the emotional shift
            metadata: Additional context
        """
        if old_emotion != new_emotion:
            description = f"Emotional state shifted from {old_emotion} to {new_emotion}"
            
            meta = metadata or {}
            meta.update({
                'old_emotion': old_emotion,
                'new_emotion': new_emotion
            })
            
            if trigger:
                meta['trigger'] = trigger
                description += f" due to {trigger}"
            
            self._capture_event(
                event_type=EventType.EMOTIONAL_SHIFT,
                phase_name=phase_name,
                description=description,
                emotion=new_emotion,
                metadata=meta
            )
    
    def log_metacognitive_moment(self,
                                phase_name: str,
                                observation: str,
                                recursion_level: int = 1,
                                confidence: Optional[float] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log metacognitive observations and self-reflection moments.
        
        Args:
            phase_name: Phase where observation occurred
            observation: What was observed about own thinking
            recursion_level: Depth of recursive self-observation
            confidence: Confidence in the observation
            metadata: Additional context
        """
        description = f"Metacognitive observation: {observation}"
        
        meta = metadata or {}
        meta['recursion_level'] = recursion_level
        meta['observation_type'] = 'self_reflection'
        
        self._capture_event(
            event_type=EventType.METACOGNITIVE_MOMENT,
            phase_name=phase_name,
            description=description,
            confidence=confidence,
            metadata=meta
        )
    
    def log_memory_retrieval(self,
                           phase_name: str,
                           memory_count: int,
                           relevance_score: Optional[float] = None,
                           memory_summary: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log memory access and retrieval events.
        
        Args:
            phase_name: Phase where memory was accessed
            memory_count: Number of memories retrieved
            relevance_score: Average relevance of retrieved memories
            memory_summary: Brief summary of memory content
            metadata: Additional context
        """
        description = f"Retrieved {memory_count} relevant memories"
        
        meta = metadata or {}
        meta['memory_count'] = memory_count
        
        if relevance_score is not None:
            meta['relevance_score'] = relevance_score
            description += f" (relevance: {relevance_score:.2f})"
        
        if memory_summary:
            meta['summary'] = memory_summary
            description += f": {memory_summary}"
        
        self._capture_event(
            event_type=EventType.MEMORY_RETRIEVAL,
            phase_name=phase_name,
            description=description,
            metadata=meta
        )
    
    def log_goal_evolution(self,
                          phase_name: str,
                          old_goal: str,
                          new_goal: str,
                          evolution_reason: Optional[str] = None,
                          confidence: Optional[float] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log goal changes and evolution.
        
        Args:
            phase_name: Phase where goal evolved
            old_goal: Previous goal
            new_goal: New goal
            evolution_reason: Why goal changed
            confidence: Confidence in new goal
            metadata: Additional context
        """
        description = f"Goal evolved from '{old_goal}' to '{new_goal}'"
        
        meta = metadata or {}
        meta.update({
            'old_goal': old_goal,
            'new_goal': new_goal
        })
        
        if evolution_reason:
            meta['evolution_reason'] = evolution_reason
            description += f" because {evolution_reason}"
        
        self._capture_event(
            event_type=EventType.GOAL_EVOLUTION,
            phase_name=phase_name,
            description=description,
            confidence=confidence,
            metadata=meta
        )
    
    def log_self_correction(self,
                          phase_name: str,
                          correction_description: str,
                          original_state: Optional[str] = None,
                          corrected_state: Optional[str] = None,
                          confidence: Optional[float] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log self-corrections and adjustments.
        
        Args:
            phase_name: Phase where correction occurred
            correction_description: What was corrected
            original_state: State before correction
            corrected_state: State after correction
            confidence: Confidence in correction
            metadata: Additional context
        """
        description = f"Self-correction: {correction_description}"
        
        meta = metadata or {}
        if original_state:
            meta['original_state'] = original_state
        if corrected_state:
            meta['corrected_state'] = corrected_state
        
        self._capture_event(
            event_type=EventType.SELF_CORRECTION,
            phase_name=phase_name,
            description=description,
            confidence=confidence,
            metadata=meta
        )
    
    def log_uncertainty_expression(self,
                                 phase_name: str,
                                 uncertainty_description: str,
                                 confidence: Optional[float] = None,
                                 doubt_level: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log expressions of uncertainty and doubt.
        
        Args:
            phase_name: Phase where uncertainty was expressed
            uncertainty_description: Description of uncertainty
            confidence: Current confidence level
            doubt_level: Level of doubt (low/medium/high)
            metadata: Additional context
        """
        description = f"Uncertainty: {uncertainty_description}"
        
        meta = metadata or {}
        if doubt_level:
            meta['doubt_level'] = doubt_level
            description += f" (doubt: {doubt_level})"
        
        self._capture_event(
            event_type=EventType.UNCERTAINTY_EXPRESSION,
            phase_name=phase_name,
            description=description,
            confidence=confidence,
            metadata=meta
        )
    
    def log_introspective_observation(self,
                                    phase_name: str,
                                    observation: str,
                                    insight_type: Optional[str] = None,
                                    confidence: Optional[float] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log introspective observations and insights.
        
        Args:
            phase_name: Phase where observation occurred
            observation: The introspective insight
            insight_type: Type of insight (pattern, connection, realization, etc.)
            confidence: Confidence in the insight
            metadata: Additional context
        """
        description = f"Introspection: {observation}"
        
        meta = metadata or {}
        if insight_type:
            meta['insight_type'] = insight_type
            description = f"{insight_type.title()}: {observation}"
        
        self._capture_event(
            event_type=EventType.INTROSPECTIVE_OBSERVATION,
            phase_name=phase_name,
            description=description,
            confidence=confidence,
            metadata=meta
        )
    
    def _capture_event(self,
                      event_type: EventType,
                      phase_name: str,
                      description: str,
                      confidence: Optional[float] = None,
                      emotion: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Internal method to capture and store events.
        Designed for zero-blocking operation to maintain pipeline performance.
        """
        capture_start = time.time()
        
        try:
            # Track confidence and emotion changes automatically
            if confidence is not None and self._last_confidence is not None:
                change = abs(confidence - self._last_confidence)
                if change >= NARRATIVE_CONFIG.get('confidence_change_threshold', 0.20):
                    # This will be logged separately by the calling method
                    pass
            
            if emotion and self._last_emotion and emotion != self._last_emotion:
                # This will be logged separately by the calling method
                pass
            
            # Create event
            event = ConsciousnessEvent(
                event_type=event_type,
                phase_name=phase_name,
                description=description,
                timestamp=time.time(),
                metadata=metadata or {},
                confidence_level=confidence,
                emotional_state=emotion
            )
            
            # Check for similar recent events (deduplication)
            if NARRATIVE_CONFIG.get('redundancy_elimination', True):
                for recent_event in list(self.events)[-3:]:  # Check last 3 events
                    if event.is_similar_to(recent_event):
                        logger.debug(f"Skipping duplicate event: {description[:50]}")
                        return
            
            # Add to circular buffer (automatically handles max size)
            self.events.append(event)
            self._event_count += 1
            
            # Update tracking variables
            self._last_confidence = confidence
            if emotion:
                self._last_emotion = emotion
            
            # Track performance
            capture_time = (time.time() - capture_start) * 1000
            self._capture_times.append(capture_time)
            if len(self._capture_times) > self._max_capture_times:
                self._capture_times.pop(0)
            
            if NARRATIVE_CONFIG.get('debug_mode', False):
                logger.debug(f"Captured event: {event_type.value} in {phase_name} - {description[:100]}")
                
        except Exception as e:
            # Never block the pipeline due to logging errors
            logger.error(f"Event capture failed: {e}")
    
    def get_events(self, 
                  phase_filter: Optional[str] = None,
                  event_type_filter: Optional[EventType] = None,
                  max_age_ms: Optional[float] = None) -> List[ConsciousnessEvent]:
        """
        Retrieve captured events with optional filtering.
        
        Args:
            phase_filter: Only events from this phase
            event_type_filter: Only events of this type
            max_age_ms: Only events newer than this age in milliseconds
            
        Returns:
            List of filtered consciousness events
        """
        events = list(self.events)
        
        # Apply filters
        if phase_filter:
            events = [e for e in events if e.phase_name == phase_filter]
        
        if event_type_filter:
            events = [e for e in events if e.event_type == event_type_filter]
        
        if max_age_ms is not None:
            events = [e for e in events if e.age_ms <= max_age_ms]
        
        return events
    
    def get_recent_events(self, count: int = 10) -> List[ConsciousnessEvent]:
        """Get the most recent events"""
        return list(self.events)[-count:] if self.events else []
    
    def clear_events(self) -> None:
        """Clear all captured events"""
        self.events.clear()
        logger.debug("Event buffer cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics and performance metrics"""
        avg_capture_time = sum(self._capture_times) / len(self._capture_times) if self._capture_times else 0
        
        event_counts = {}
        for event in self.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            'total_events_captured': self._event_count,
            'current_buffer_size': len(self.events),
            'max_buffer_size': self.max_events,
            'session_duration_ms': (time.time() - self._session_start) * 1000,
            'average_capture_time_ms': avg_capture_time,
            'event_type_counts': event_counts,
            'buffer_utilization': len(self.events) / self.max_events if self.max_events > 0 else 0
        }