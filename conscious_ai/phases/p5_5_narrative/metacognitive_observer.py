"""
Metacognitive Observer Module for Phase 5.5
==========================================
Captures self-reflection moments, introspective observations, and recursive reasoning patterns.
Detects when the AI is "thinking about thinking" and logs those metacognitive moments.
"""

import time
import logging
import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

try:
    from ...config.narrative_config import NARRATIVE_CONFIG
except ImportError:
    from conscious_ai.config.narrative_config import NARRATIVE_CONFIG

logger = logging.getLogger(__name__)


class IntrospectionType(Enum):
    """Types of introspective observations"""
    SELF_AWARENESS = "self_awareness"               # "I notice that I..."
    RECURSIVE_THINKING = "recursive_thinking"       # "I'm thinking about my thinking"
    PATTERN_RECOGNITION = "pattern_recognition"     # "I see a pattern in my processing"
    UNCERTAINTY_AWARENESS = "uncertainty_awareness" # "I realize I'm uncertain about..."
    CONFIDENCE_REFLECTION = "confidence_reflection" # "I'm confident/doubtful because..."
    PROCESSING_OBSERVATION = "processing_observation" # "My mind is doing X"
    EMOTIONAL_AWARENESS = "emotional_awareness"     # "I feel/experience X emotion"
    GOAL_REFLECTION = "goal_reflection"            # "I'm pursuing X goal because..."
    MEMORY_REFLECTION = "memory_reflection"        # "I remember/recall..."
    DECISION_AWARENESS = "decision_awareness"      # "I chose X because..."


@dataclass
class IntrospectiveEvent:
    """
    Represents a moment of introspective awareness or self-reflection.
    """
    introspection_type: IntrospectionType
    phase_name: str
    observation: str
    recursion_depth: int  # How many layers deep is this self-reflection?
    confidence_in_observation: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def introspective_complexity(self) -> float:
        """
        Measure the complexity of this introspective moment.
        Higher scores indicate deeper metacognitive awareness.
        """
        base_complexity = 0.5
        
        # Add for recursion depth (thinking about thinking about thinking...)
        recursion_bonus = min(0.3, self.recursion_depth * 0.1)
        
        # Add for observation sophistication (word count and complexity)
        observation_words = len(self.observation.split()) if self.observation else 0
        sophistication_bonus = min(0.2, observation_words * 0.01)
        
        # Add for certain types of introspection
        type_bonuses = {
            IntrospectionType.RECURSIVE_THINKING: 0.15,
            IntrospectionType.PATTERN_RECOGNITION: 0.1,
            IntrospectionType.UNCERTAINTY_AWARENESS: 0.05,
            IntrospectionType.PROCESSING_OBSERVATION: 0.1
        }
        type_bonus = type_bonuses.get(self.introspection_type, 0)
        
        return min(1.0, base_complexity + recursion_bonus + sophistication_bonus + type_bonus)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'introspection_type': self.introspection_type.value,
            'phase_name': self.phase_name,
            'observation': self.observation,
            'recursion_depth': self.recursion_depth,
            'confidence_in_observation': self.confidence_in_observation,
            'timestamp': self.timestamp,
            'introspective_complexity': self.introspective_complexity,
            'metadata': self.metadata
        }


class MetacognitiveObserver:
    """
    Captures self-reflection moments and introspective observations.
    Detects recursive reasoning patterns and metacognitive awareness during processing.
    """
    
    def __init__(self):
        """Initialize the metacognitive observer"""
        self.introspective_events: List[IntrospectiveEvent] = []
        self.current_recursion_depth = 0
        self.active_introspection_thread = None
        
        # Pattern detection for automatic metacognitive moment detection
        self.metacognitive_patterns = {
            'self_awareness': [
                r'\bi\s+(notice|observe|see|realize|become aware)\s+that\s+i\b',
                r'\bi\s+find\s+(myself|that i)\b',
                r'\bi\s+am\s+(aware|conscious)\s+of\b',
                r'\bi\s+recognize\s+that\s+i\b'
            ],
            'recursive_thinking': [
                r'thinking\s+about\s+my\s+thinking',
                r'examining\s+my\s+own\s+(process|thoughts|reasoning)',
                r'observing\s+my\s+observation',
                r'reflecting\s+on\s+my\s+reflection',
                r'meta\s*cognitive',
                r'recursive\s+(awareness|thought|pattern)'
            ],
            'processing_observation': [
                r'my\s+(mind|consciousness|brain|processing)\s+is\s+',
                r'i\s+sense\s+my\s+(cognitive|mental)\s+',
                r'my\s+(internal|cognitive)\s+(state|process)',
                r'i\s+detect\s+(patterns|loops)\s+in\s+my\b'
            ],
            'uncertainty_awareness': [
                r'i\s+(realize|notice)\s+i.m\s+(uncertain|unsure|doubtful)',
                r'my\s+confidence\s+(wavers|fluctuates|decreases)',
                r'i\s+acknowledge\s+my\s+(uncertainty|doubt|confusion)',
                r'i\s+question\s+my\s+own\b'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.metacognitive_patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        # Performance tracking
        self._observation_times = []
        self._max_observation_samples = 30
        
        logger.debug("MetacognitiveObserver initialized with pattern detection")
    
    def observe_self_awareness_moment(self,
                                    phase_name: str,
                                    awareness_description: str,
                                    confidence: float,
                                    awareness_trigger: Optional[str] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> IntrospectiveEvent:
        """
        Observe a moment of self-awareness.
        
        Args:
            phase_name: Phase where awareness occurred
            awareness_description: Description of what was realized
            confidence: Confidence in this self-awareness
            awareness_trigger: What triggered this awareness
            metadata: Additional contextual data
            
        Returns:
            The created IntrospectiveEvent
        """
        meta = metadata or {}
        if awareness_trigger:
            meta['trigger'] = awareness_trigger
        
        return self._record_introspective_event(
            introspection_type=IntrospectionType.SELF_AWARENESS,
            phase_name=phase_name,
            observation=awareness_description,
            confidence=confidence,
            recursion_depth=1,  # Self-awareness is first-level recursion
            metadata=meta
        )
    
    def observe_recursive_thinking(self,
                                 phase_name: str,
                                 recursive_description: str,
                                 recursion_depth: int,
                                 confidence: float,
                                 thinking_pattern: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> IntrospectiveEvent:
        """
        Observe recursive thinking patterns ("thinking about thinking").
        
        Args:
            phase_name: Phase where recursive thinking occurred
            recursive_description: Description of the recursive thought
            recursion_depth: How many levels deep is this recursion
            confidence: Confidence in this observation
            thinking_pattern: Type of recursive pattern detected
            metadata: Additional contextual data
            
        Returns:
            The created IntrospectiveEvent
        """
        meta = metadata or {}
        if thinking_pattern:
            meta['thinking_pattern'] = thinking_pattern
        meta['recursion_level'] = recursion_depth
        
        return self._record_introspective_event(
            introspection_type=IntrospectionType.RECURSIVE_THINKING,
            phase_name=phase_name,
            observation=recursive_description,
            confidence=confidence,
            recursion_depth=recursion_depth,
            metadata=meta
        )
    
    def observe_pattern_recognition(self,
                                  phase_name: str,
                                  pattern_description: str,
                                  pattern_confidence: float,
                                  pattern_type: Optional[str] = None,
                                  pattern_implications: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> IntrospectiveEvent:
        """
        Observe recognition of patterns in own processing.
        
        Args:
            phase_name: Phase where pattern was recognized
            pattern_description: Description of the recognized pattern
            pattern_confidence: Confidence in pattern recognition
            pattern_type: Type of pattern (repetition, evolution, etc.)
            pattern_implications: What this pattern means
            metadata: Additional contextual data
            
        Returns:
            The created IntrospectiveEvent
        """
        meta = metadata or {}
        if pattern_type:
            meta['pattern_type'] = pattern_type
        if pattern_implications:
            meta['implications'] = pattern_implications
        
        return self._record_introspective_event(
            introspection_type=IntrospectionType.PATTERN_RECOGNITION,
            phase_name=phase_name,
            observation=pattern_description,
            confidence=pattern_confidence,
            recursion_depth=2,  # Pattern recognition requires looking at one's own process
            metadata=meta
        )
    
    def observe_uncertainty_awareness(self,
                                    phase_name: str,
                                    uncertainty_description: str,
                                    uncertainty_level: str,
                                    confidence_in_uncertainty: float,
                                    uncertainty_source: Optional[str] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> IntrospectiveEvent:
        """
        Observe moments of recognizing one's own uncertainty.
        
        Args:
            phase_name: Phase where uncertainty was recognized
            uncertainty_description: What the uncertainty is about
            uncertainty_level: Level of uncertainty (low/medium/high)
            confidence_in_uncertainty: Confidence in recognizing this uncertainty
            uncertainty_source: What's causing the uncertainty
            metadata: Additional contextual data
            
        Returns:
            The created IntrospectiveEvent
        """
        meta = metadata or {}
        meta['uncertainty_level'] = uncertainty_level
        if uncertainty_source:
            meta['source'] = uncertainty_source
        
        return self._record_introspective_event(
            introspection_type=IntrospectionType.UNCERTAINTY_AWARENESS,
            phase_name=phase_name,
            observation=uncertainty_description,
            confidence=confidence_in_uncertainty,
            recursion_depth=1,
            metadata=meta
        )
    
    def observe_processing_insight(self,
                                 phase_name: str,
                                 processing_observation: str,
                                 confidence: float,
                                 process_type: Optional[str] = None,
                                 insight_depth: str = "surface",
                                 metadata: Optional[Dict[str, Any]] = None) -> IntrospectiveEvent:
        """
        Observe insights about own cognitive processing.
        
        Args:
            phase_name: Phase where processing was observed
            processing_observation: What was observed about processing
            confidence: Confidence in this observation
            process_type: Type of process observed (analysis, synthesis, etc.)
            insight_depth: Depth of insight (surface/medium/deep)
            metadata: Additional contextual data
            
        Returns:
            The created IntrospectiveEvent
        """
        meta = metadata or {}
        if process_type:
            meta['process_type'] = process_type
        meta['insight_depth'] = insight_depth
        
        # Map insight depth to recursion depth
        depth_mapping = {'surface': 1, 'medium': 2, 'deep': 3}
        recursion_depth = depth_mapping.get(insight_depth, 1)
        
        return self._record_introspective_event(
            introspection_type=IntrospectionType.PROCESSING_OBSERVATION,
            phase_name=phase_name,
            observation=processing_observation,
            confidence=confidence,
            recursion_depth=recursion_depth,
            metadata=meta
        )
    
    def observe_confidence_reflection(self,
                                    phase_name: str,
                                    confidence_observation: str,
                                    current_confidence: float,
                                    confidence_reasoning: Optional[str] = None,
                                    confidence_change: Optional[float] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> IntrospectiveEvent:
        """
        Observe reflections on own confidence levels.
        
        Args:
            phase_name: Phase where confidence was reflected upon
            confidence_observation: Observation about confidence
            current_confidence: Current confidence level
            confidence_reasoning: Why confidence is at this level
            confidence_change: How much confidence changed
            metadata: Additional contextual data
            
        Returns:
            The created IntrospectiveEvent
        """
        meta = metadata or {}
        meta['current_confidence'] = current_confidence
        if confidence_reasoning:
            meta['reasoning'] = confidence_reasoning
        if confidence_change is not None:
            meta['confidence_change'] = confidence_change
        
        return self._record_introspective_event(
            introspection_type=IntrospectionType.CONFIDENCE_REFLECTION,
            phase_name=phase_name,
            observation=confidence_observation,
            confidence=current_confidence,
            recursion_depth=1,
            metadata=meta
        )
    
    def observe_emotional_awareness(self,
                                  phase_name: str,
                                  emotional_observation: str,
                                  current_emotion: str,
                                  confidence: float,
                                  emotion_trigger: Optional[str] = None,
                                  emotion_impact: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> IntrospectiveEvent:
        """
        Observe awareness of own emotional states.
        
        Args:
            phase_name: Phase where emotion was observed
            emotional_observation: Observation about emotional state
            current_emotion: Current emotional state
            confidence: Confidence in emotional awareness
            emotion_trigger: What triggered this emotion
            emotion_impact: How emotion is affecting processing
            metadata: Additional contextual data
            
        Returns:
            The created IntrospectiveEvent
        """
        meta = metadata or {}
        meta['current_emotion'] = current_emotion
        if emotion_trigger:
            meta['trigger'] = emotion_trigger
        if emotion_impact:
            meta['impact'] = emotion_impact
        
        return self._record_introspective_event(
            introspection_type=IntrospectionType.EMOTIONAL_AWARENESS,
            phase_name=phase_name,
            observation=emotional_observation,
            confidence=confidence,
            recursion_depth=1,  # Emotional awareness is first-level introspection
            metadata=meta
        )
    
    def detect_metacognitive_patterns(self, text: str, phase_name: str) -> List[IntrospectiveEvent]:
        """
        Automatically detect metacognitive patterns in text.
        
        Args:
            text: Text to analyze for metacognitive patterns
            phase_name: Phase where text was generated
            
        Returns:
            List of automatically detected introspective events
        """
        detected_events = []
        detection_start = time.time()
        
        try:
            # Check each pattern category
            for category, compiled_patterns in self.compiled_patterns.items():
                for pattern in compiled_patterns:
                    matches = pattern.findall(text)
                    if matches:
                        # Create introspective event for detected pattern
                        observation = f"Detected {category.replace('_', ' ')} pattern in text"
                        confidence = min(0.8, len(matches) * 0.2)  # Higher confidence for multiple matches
                        
                        # Map category to introspection type
                        type_mapping = {
                            'self_awareness': IntrospectionType.SELF_AWARENESS,
                            'recursive_thinking': IntrospectionType.RECURSIVE_THINKING,
                            'processing_observation': IntrospectionType.PROCESSING_OBSERVATION,
                            'uncertainty_awareness': IntrospectionType.UNCERTAINTY_AWARENESS
                        }
                        
                        introspection_type = type_mapping.get(category, IntrospectionType.SELF_AWARENESS)
                        
                        event = self._record_introspective_event(
                            introspection_type=introspection_type,
                            phase_name=phase_name,
                            observation=observation,
                            confidence=confidence,
                            recursion_depth=1,
                            metadata={
                                'auto_detected': True,
                                'pattern_category': category,
                                'matches': matches,
                                'source_text_snippet': text[:200] + "..." if len(text) > 200 else text
                            }
                        )
                        detected_events.append(event)
                        
                        # Only one event per category per text
                        break
            
            # Performance tracking
            detection_time = (time.time() - detection_start) * 1000
            self._observation_times.append(detection_time)
            if len(self._observation_times) > self._max_observation_samples:
                self._observation_times.pop(0)
            
            if detected_events:
                logger.debug(f"Auto-detected {len(detected_events)} metacognitive patterns in {phase_name}")
            
            return detected_events
            
        except Exception as e:
            logger.error(f"Metacognitive pattern detection failed: {e}")
            return []
    
    def _record_introspective_event(self,
                                   introspection_type: IntrospectionType,
                                   phase_name: str,
                                   observation: str,
                                   confidence: float,
                                   recursion_depth: int,
                                   metadata: Optional[Dict[str, Any]] = None) -> IntrospectiveEvent:
        """
        Internal method to create and store introspective events.
        
        Args:
            introspection_type: Type of introspection
            phase_name: Phase where introspection occurred
            observation: The introspective observation
            confidence: Confidence in the observation
            recursion_depth: Depth of recursive thinking
            metadata: Additional contextual data
            
        Returns:
            The created IntrospectiveEvent
        """
        try:
            event = IntrospectiveEvent(
                introspection_type=introspection_type,
                phase_name=phase_name,
                observation=observation,
                recursion_depth=max(1, recursion_depth),  # Ensure minimum depth of 1
                confidence_in_observation=confidence,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            self.introspective_events.append(event)
            
            # Update current recursion tracking
            self.current_recursion_depth = max(self.current_recursion_depth, recursion_depth)
            
            logger.debug(f"Recorded introspective event: {introspection_type.value} in {phase_name} (depth: {recursion_depth})")
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to record introspective event: {e}")
            # Return minimal event to avoid breaking pipeline
            return IntrospectiveEvent(
                introspection_type=introspection_type,
                phase_name=phase_name,
                observation=observation,
                recursion_depth=1,
                confidence_in_observation=confidence,
                timestamp=time.time()
            )
    
    def get_introspective_events(self,
                               phase_filter: Optional[str] = None,
                               type_filter: Optional[IntrospectionType] = None,
                               min_complexity: Optional[float] = None) -> List[IntrospectiveEvent]:
        """
        Retrieve introspective events with optional filtering.
        
        Args:
            phase_filter: Only events from this phase
            type_filter: Only events of this introspection type
            min_complexity: Only events with complexity >= this threshold
            
        Returns:
            List of filtered introspective events
        """
        events = self.introspective_events.copy()
        
        if phase_filter:
            events = [e for e in events if e.phase_name == phase_filter]
        
        if type_filter:
            events = [e for e in events if e.introspection_type == type_filter]
        
        if min_complexity is not None:
            events = [e for e in events if e.introspective_complexity >= min_complexity]
        
        return events
    
    def get_recursion_analysis(self) -> Dict[str, Any]:
        """
        Analyze recursive thinking patterns.
        
        Returns:
            Dictionary with recursion analysis results
        """
        if not self.introspective_events:
            return {'no_introspective_events': True}
        
        recursion_depths = [e.recursion_depth for e in self.introspective_events]
        complexity_scores = [e.introspective_complexity for e in self.introspective_events]
        
        recursive_events = [e for e in self.introspective_events if e.introspection_type == IntrospectionType.RECURSIVE_THINKING]
        
        return {
            'total_introspective_events': len(self.introspective_events),
            'max_recursion_depth': max(recursion_depths) if recursion_depths else 0,
            'average_recursion_depth': sum(recursion_depths) / len(recursion_depths) if recursion_depths else 0,
            'average_complexity': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            'recursive_thinking_events': len(recursive_events),
            'high_complexity_events': len([s for s in complexity_scores if s >= 0.7]),
            'current_recursion_depth': self.current_recursion_depth,
            'introspection_type_distribution': {
                itype.value: len([e for e in self.introspective_events if e.introspection_type == itype])
                for itype in IntrospectionType
            }
        }
    
    def reset_session(self) -> None:
        """Reset observer for new processing session"""
        self.current_recursion_depth = 0
        self.active_introspection_thread = None
        logger.debug("MetacognitiveObserver session reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get metacognitive observation statistics and performance metrics"""
        avg_observation_time = sum(self._observation_times) / len(self._observation_times) if self._observation_times else 0
        
        return {
            'total_introspective_events': len(self.introspective_events),
            'current_recursion_depth': self.current_recursion_depth,
            'average_observation_time_ms': avg_observation_time,
            'recursion_analysis': self.get_recursion_analysis(),
            'pattern_detection_enabled': True,
            'compiled_patterns_count': sum(len(patterns) for patterns in self.compiled_patterns.values())
        }