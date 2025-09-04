"""
Decision Tracker Module for Phase 5.5
====================================
Monitors cognitive choice points and tracks confidence fluctuations during decision-making.
Captures the "I considered X but selected Y because..." moments in consciousness processing.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

try:
    from ...config.narrative_config import NARRATIVE_CONFIG
except ImportError:
    from conscious_ai.config.narrative_config import NARRATIVE_CONFIG

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of cognitive decisions tracked"""
    GOAL_SELECTION = "goal_selection"
    RESPONSE_STRATEGY = "response_strategy"
    INFORMATION_INTEGRATION = "information_integration"
    MEMORY_UTILIZATION = "memory_utilization"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    EMOTIONAL_REGULATION = "emotional_regulation"
    PROCESSING_PATH = "processing_path"
    SELF_CORRECTION = "self_correction"


@dataclass
class CognitiveChoice:
    """
    Represents a specific cognitive decision point with alternatives and reasoning.
    """
    decision_type: DecisionType
    phase_name: str
    selected_option: str
    alternatives: List[str]
    reasoning: str
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def decision_quality_score(self) -> float:
        """
        Assess the quality of this decision based on confidence and reasoning depth.
        Returns a score from 0 to 1.
        """
        # Base score from confidence
        base_score = self.confidence
        
        # Boost for having multiple alternatives considered
        alternatives_boost = min(0.2, len(self.alternatives) * 0.05)
        
        # Boost for reasoning depth (based on word count and content)
        reasoning_words = len(self.reasoning.split()) if self.reasoning else 0
        reasoning_boost = min(0.2, reasoning_words * 0.02)
        
        # Penalty for very quick decisions (may indicate insufficient consideration)
        if hasattr(self, 'deliberation_time_ms') and self.deliberation_time_ms < 10:
            quick_penalty = 0.1
        else:
            quick_penalty = 0
        
        return min(1.0, base_score + alternatives_boost + reasoning_boost - quick_penalty)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'decision_type': self.decision_type.value,
            'phase_name': self.phase_name,
            'selected_option': self.selected_option,
            'alternatives': self.alternatives,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'decision_quality_score': self.decision_quality_score,
            'metadata': self.metadata
        }


class DecisionTracker:
    """
    Monitors cognitive choice points during consciousness processing.
    Tracks decision-making patterns, alternative considerations, and reasoning quality.
    """
    
    def __init__(self, max_decisions_per_cycle: int = None):
        """
        Initialize the decision tracker.
        
        Args:
            max_decisions_per_cycle: Maximum decisions to track per processing cycle
        """
        self.max_decisions = max_decisions_per_cycle or NARRATIVE_CONFIG.get('max_decision_points', 5)
        self.decisions: List[CognitiveChoice] = []
        self.current_cycle_decisions: List[CognitiveChoice] = []
        self.confidence_history: List[tuple] = []  # (timestamp, confidence, phase)
        
        # Decision pattern tracking
        self._decision_patterns = {
            'frequent_reconsiderations': 0,
            'confidence_driven_changes': 0,
            'alternative_rich_decisions': 0,
            'quick_decisive_moments': 0
        }
        
        # Performance tracking
        self._tracking_overhead_times = []
        self._max_overhead_samples = 50
        
        logger.debug(f"DecisionTracker initialized with max_decisions={self.max_decisions}")
    
    def track_decision(self,
                      decision_type: DecisionType,
                      phase_name: str,
                      selected_option: str,
                      alternatives: List[str],
                      reasoning: str,
                      confidence: float,
                      metadata: Optional[Dict[str, Any]] = None) -> CognitiveChoice:
        """
        Track a cognitive decision point.
        
        Args:
            decision_type: Type of decision being made
            phase_name: Phase where decision occurred
            selected_option: What was chosen
            alternatives: Other options that were considered
            reasoning: Why this choice was made
            confidence: Confidence in the decision (0-1)
            metadata: Additional contextual data
            
        Returns:
            The created CognitiveChoice object
        """
        track_start = time.time()
        
        try:
            # Create the cognitive choice
            choice = CognitiveChoice(
                decision_type=decision_type,
                phase_name=phase_name,
                selected_option=selected_option,
                alternatives=alternatives or [],
                reasoning=reasoning,
                confidence=confidence,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            # Add deliberation time if previous confidence point exists
            if self.confidence_history:
                last_timestamp = self.confidence_history[-1][0]
                choice.metadata['deliberation_time_ms'] = (choice.timestamp - last_timestamp) * 1000
            
            # Store the decision
            self.decisions.append(choice)
            self.current_cycle_decisions.append(choice)
            
            # Maintain cycle limit
            if len(self.current_cycle_decisions) > self.max_decisions:
                self.current_cycle_decisions.pop(0)
            
            # Update decision patterns
            self._update_decision_patterns(choice)
            
            # Track confidence for this decision
            self.track_confidence_fluctuation(phase_name, confidence, "decision_point")
            
            # Performance tracking
            tracking_time = (time.time() - track_start) * 1000
            self._tracking_overhead_times.append(tracking_time)
            if len(self._tracking_overhead_times) > self._max_overhead_samples:
                self._tracking_overhead_times.pop(0)
            
            logger.debug(f"Tracked decision: {decision_type.value} in {phase_name} - chose '{selected_option}' over {len(alternatives)} alternatives")
            
            return choice
            
        except Exception as e:
            logger.error(f"Decision tracking failed: {e}")
            # Return a minimal choice object to avoid breaking the pipeline
            return CognitiveChoice(
                decision_type=decision_type,
                phase_name=phase_name,
                selected_option=selected_option,
                alternatives=[],
                reasoning="tracking_failed",
                confidence=confidence,
                timestamp=time.time()
            )
    
    def track_goal_selection(self,
                           phase_name: str,
                           selected_goal: str,
                           considered_goals: List[str],
                           selection_reasoning: str,
                           confidence: float,
                           context: Optional[Dict[str, Any]] = None) -> CognitiveChoice:
        """
        Track goal selection decisions.
        
        Args:
            phase_name: Phase where goal was selected
            selected_goal: The chosen goal
            considered_goals: Other goals that were considered
            selection_reasoning: Why this goal was selected
            confidence: Confidence in goal selection
            context: Additional context about the goal selection
            
        Returns:
            The created CognitiveChoice object
        """
        return self.track_decision(
            decision_type=DecisionType.GOAL_SELECTION,
            phase_name=phase_name,
            selected_option=selected_goal,
            alternatives=considered_goals,
            reasoning=selection_reasoning,
            confidence=confidence,
            metadata={'context': context or {}, 'decision_category': 'goal_management'}
        )
    
    def track_response_strategy(self,
                              phase_name: str,
                              selected_strategy: str,
                              alternative_strategies: List[str],
                              strategy_reasoning: str,
                              confidence: float,
                              expected_effectiveness: Optional[float] = None) -> CognitiveChoice:
        """
        Track response strategy selection.
        
        Args:
            phase_name: Phase where strategy was selected
            selected_strategy: The chosen response strategy
            alternative_strategies: Other strategies considered
            strategy_reasoning: Why this strategy was chosen
            confidence: Confidence in strategy choice
            expected_effectiveness: Expected effectiveness of chosen strategy (0-1)
            
        Returns:
            The created CognitiveChoice object
        """
        metadata = {'decision_category': 'response_generation'}
        if expected_effectiveness is not None:
            metadata['expected_effectiveness'] = expected_effectiveness
            
        return self.track_decision(
            decision_type=DecisionType.RESPONSE_STRATEGY,
            phase_name=phase_name,
            selected_option=selected_strategy,
            alternatives=alternative_strategies,
            reasoning=strategy_reasoning,
            confidence=confidence,
            metadata=metadata
        )
    
    def track_information_integration(self,
                                   phase_name: str,
                                   integration_approach: str,
                                   alternative_approaches: List[str],
                                   integration_reasoning: str,
                                   confidence: float,
                                   information_sources: Optional[List[str]] = None) -> CognitiveChoice:
        """
        Track information integration decisions.
        
        Args:
            phase_name: Phase where integration occurred
            integration_approach: How information was integrated
            alternative_approaches: Other integration methods considered
            integration_reasoning: Why this approach was chosen
            confidence: Confidence in integration quality
            information_sources: Sources of information being integrated
            
        Returns:
            The created CognitiveChoice object
        """
        metadata = {'decision_category': 'information_processing'}
        if information_sources:
            metadata['information_sources'] = information_sources
            
        return self.track_decision(
            decision_type=DecisionType.INFORMATION_INTEGRATION,
            phase_name=phase_name,
            selected_option=integration_approach,
            alternatives=alternative_approaches,
            reasoning=integration_reasoning,
            confidence=confidence,
            metadata=metadata
        )
    
    def track_confidence_fluctuation(self,
                                   phase_name: str,
                                   confidence: float,
                                   trigger: str,
                                   previous_confidence: Optional[float] = None) -> None:
        """
        Track confidence level changes during processing.
        
        Args:
            phase_name: Phase where confidence changed
            confidence: New confidence level
            trigger: What caused the confidence change
            previous_confidence: Previous confidence level
        """
        timestamp = time.time()
        
        # Store confidence history
        self.confidence_history.append((timestamp, confidence, phase_name, trigger))
        
        # Keep history manageable
        if len(self.confidence_history) > 50:
            self.confidence_history = self.confidence_history[-50:]
        
        # Detect significant fluctuations
        if previous_confidence is not None:
            change_magnitude = abs(confidence - previous_confidence)
            threshold = NARRATIVE_CONFIG.get('confidence_change_threshold', 0.20)
            
            if change_magnitude >= threshold:
                self._decision_patterns['confidence_driven_changes'] += 1
                
                logger.debug(f"Significant confidence change in {phase_name}: {previous_confidence:.2f} -> {confidence:.2f} due to {trigger}")
    
    def track_memory_utilization_decision(self,
                                        phase_name: str,
                                        utilization_strategy: str,
                                        available_memories: int,
                                        selected_memories: int,
                                        selection_reasoning: str,
                                        confidence: float) -> CognitiveChoice:
        """
        Track decisions about memory utilization.
        
        Args:
            phase_name: Phase where memory decision occurred
            utilization_strategy: How memories were utilized
            available_memories: Total memories available
            selected_memories: Number of memories actually used
            selection_reasoning: Why these memories were chosen
            confidence: Confidence in memory selection
            
        Returns:
            The created CognitiveChoice object
        """
        alternatives = []
        if available_memories > selected_memories:
            alternatives.append(f"use_all_{available_memories}_memories")
            alternatives.append("use_minimal_memory")
            alternatives.append("weighted_memory_selection")
        
        metadata = {
            'available_memories': available_memories,
            'selected_memories': selected_memories,
            'utilization_efficiency': selected_memories / max(1, available_memories),
            'decision_category': 'memory_management'
        }
        
        return self.track_decision(
            decision_type=DecisionType.MEMORY_UTILIZATION,
            phase_name=phase_name,
            selected_option=utilization_strategy,
            alternatives=alternatives,
            reasoning=selection_reasoning,
            confidence=confidence,
            metadata=metadata
        )
    
    def track_self_correction(self,
                            phase_name: str,
                            correction_type: str,
                            original_choice: str,
                            corrected_choice: str,
                            correction_reasoning: str,
                            confidence: float) -> CognitiveChoice:
        """
        Track self-correction decisions.
        
        Args:
            phase_name: Phase where correction occurred
            correction_type: Type of correction made
            original_choice: What was originally chosen
            corrected_choice: What it was corrected to
            correction_reasoning: Why the correction was made
            confidence: Confidence in the correction
            
        Returns:
            The created CognitiveChoice object
        """
        metadata = {
            'original_choice': original_choice,
            'correction_type': correction_type,
            'decision_category': 'self_regulation'
        }
        
        return self.track_decision(
            decision_type=DecisionType.SELF_CORRECTION,
            phase_name=phase_name,
            selected_option=corrected_choice,
            alternatives=[original_choice],
            reasoning=correction_reasoning,
            confidence=confidence,
            metadata=metadata
        )
    
    def _update_decision_patterns(self, choice: CognitiveChoice) -> None:
        """Update decision pattern statistics"""
        try:
            # Track rich alternative consideration
            if len(choice.alternatives) >= 3:
                self._decision_patterns['alternative_rich_decisions'] += 1
            
            # Track quick decisions
            if choice.metadata.get('deliberation_time_ms', float('inf')) < 50:
                self._decision_patterns['quick_decisive_moments'] += 1
            
            # Track reconsiderations (decisions that override previous ones)
            if choice.decision_type == DecisionType.SELF_CORRECTION:
                self._decision_patterns['frequent_reconsiderations'] += 1
                
        except Exception as e:
            logger.warning(f"Pattern update failed: {e}")
    
    def get_cycle_decisions(self) -> List[CognitiveChoice]:
        """Get decisions made in the current processing cycle"""
        return self.current_cycle_decisions.copy()
    
    def get_recent_decisions(self, count: int = 10) -> List[CognitiveChoice]:
        """Get the most recent decisions"""
        return self.decisions[-count:] if self.decisions else []
    
    def get_decisions_by_type(self, decision_type: DecisionType) -> List[CognitiveChoice]:
        """Get all decisions of a specific type"""
        return [d for d in self.decisions if d.decision_type == decision_type]
    
    def get_decisions_by_phase(self, phase_name: str) -> List[CognitiveChoice]:
        """Get all decisions from a specific phase"""
        return [d for d in self.decisions if d.phase_name == phase_name]
    
    def get_confidence_trajectory(self, phase_filter: Optional[str] = None) -> List[tuple]:
        """
        Get confidence changes over time.
        
        Args:
            phase_filter: Only include confidence points from this phase
            
        Returns:
            List of (timestamp, confidence, phase, trigger) tuples
        """
        if phase_filter:
            return [entry for entry in self.confidence_history if entry[2] == phase_filter]
        return self.confidence_history.copy()
    
    def analyze_decision_quality(self) -> Dict[str, Any]:
        """
        Analyze the quality of decisions made.
        
        Returns:
            Dictionary with decision quality metrics
        """
        if not self.decisions:
            return {'no_decisions': True}
        
        # Calculate quality metrics
        quality_scores = [d.decision_quality_score for d in self.decisions]
        alternative_counts = [len(d.alternatives) for d in self.decisions]
        reasoning_lengths = [len(d.reasoning.split()) for d in self.decisions if d.reasoning]
        
        # Confidence analysis
        confidences = [d.confidence for d in self.decisions]
        
        return {
            'total_decisions': len(self.decisions),
            'average_quality_score': sum(quality_scores) / len(quality_scores),
            'average_alternatives_considered': sum(alternative_counts) / len(alternative_counts),
            'average_reasoning_depth': sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0,
            'average_confidence': sum(confidences) / len(confidences),
            'decision_patterns': self._decision_patterns.copy(),
            'high_quality_decisions': len([s for s in quality_scores if s >= 0.8]),
            'low_confidence_decisions': len([c for c in confidences if c < 0.5]),
            'well_reasoned_decisions': len([d for d in self.decisions if len(d.reasoning.split()) >= 10])
        }
    
    def reset_cycle(self) -> None:
        """Reset tracking for a new processing cycle"""
        self.current_cycle_decisions.clear()
        logger.debug("DecisionTracker cycle reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decision tracking statistics and performance metrics"""
        avg_overhead = sum(self._tracking_overhead_times) / len(self._tracking_overhead_times) if self._tracking_overhead_times else 0
        
        return {
            'total_decisions_tracked': len(self.decisions),
            'current_cycle_decisions': len(self.current_cycle_decisions),
            'confidence_history_length': len(self.confidence_history),
            'decision_patterns': self._decision_patterns.copy(),
            'average_tracking_overhead_ms': avg_overhead,
            'decision_quality_analysis': self.analyze_decision_quality()
        }