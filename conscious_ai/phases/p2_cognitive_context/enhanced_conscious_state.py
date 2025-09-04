"""
Enhanced Conscious State with Metacognitive Capabilities
=========================================================
Extends the basic ConsciousState to support temporal awareness,
recursive introspection, and metacognitive observations.

This enhancement maintains full backward compatibility while adding:
- Temporal awareness (knowledge of previous states)
- Meta-thoughts (thoughts about thoughts)
- Self-observation stack
- State transition tracking
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json
import numpy as np

# Import the base ConsciousState class
from .conscious_state import ConsciousState


@dataclass
class StateTransition:
    """Represents a transition between conscious states"""
    timestamp: datetime
    change_type: str  # 'emotional', 'confidence', 'goal', 'thought', 'memory'
    magnitude: float  # 0.0 to 1.0 representing change magnitude
    previous_value: Any
    new_value: Any
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'change_type': self.change_type,
            'magnitude': self.magnitude,
            'previous_value': str(self.previous_value)[:100],
            'new_value': str(self.new_value)[:100],
            'description': self.description
        }


@dataclass
class MetaThought:
    """Represents a thought about the current thought process"""
    content: str
    thought_being_observed: str
    observation_type: str  # 'pattern', 'recursion', 'conflict', 'insight'
    depth_level: int  # How many layers of meta-thinking
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'thought_observed': self.thought_being_observed[:100],
            'type': self.observation_type,
            'depth': self.depth_level,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SelfObservation:
    """Records an observation about the system's own processing"""
    observation: str
    processing_phase: str
    confidence_in_observation: float
    triggers_introspection: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'observation': self.observation,
            'phase': self.processing_phase,
            'confidence': self.confidence_in_observation,
            'introspective': self.triggers_introspection,
            'timestamp': self.timestamp.isoformat()
        }


class EnhancedConsciousState(ConsciousState):
    """
    Enhanced conscious state with metacognitive capabilities.
    
    Extends ConsciousState with:
    - Temporal awareness through previous_state reference
    - State transition tracking
    - Meta-thoughts about current thoughts
    - Self-observation stack
    - Recursive introspection capabilities
    """
    
    def __init__(self, 
                 E_t: Dict[str, Any],
                 M_t: List[Dict[str, Any]], 
                 S_t: Dict[str, Any],
                 G_t: Dict[str, Any],
                 A_t: List[str],
                 cycle: int,
                 timestamp: datetime = None,
                 metrics: Optional[Dict[str, float]] = None,
                 previous_state: Optional['ConsciousState'] = None):
        """
        Initialize enhanced conscious state.
        
        Args:
            All base ConsciousState parameters plus:
            previous_state: Reference to SC_t-1 for temporal awareness
        """
        # Initialize base class
        super().__init__(E_t, M_t, S_t, G_t, A_t, cycle, timestamp, metrics)
        
        # Enhanced metacognitive attributes
        self.previous_state = previous_state
        self.state_transitions: List[StateTransition] = []
        self.meta_thoughts: List[MetaThought] = []
        self.observation_stack: List[SelfObservation] = []
        self.metacognitive_depth = 0
        
        # Capture initial transitions if previous state exists
        if previous_state:
            self.capture_transition(previous_state)
            
        # Generate initial meta-thoughts
        self.generate_meta_thoughts()
        
        # Perform initial self-observation
        self.observe_self()
    
    def capture_transition(self, previous_state: Optional['ConsciousState']) -> None:
        """
        Records changes from SC_t-1 to SC_t.
        
        Analyzes differences between states and creates transition records
        for significant changes in emotional state, confidence, goals, etc.
        """
        if not previous_state:
            return
            
        # Check emotional state change
        if hasattr(previous_state, 'S_t') and previous_state.S_t:
            prev_emotion = previous_state.S_t.get('emotional_state', 'neutral')
            curr_emotion = self.S_t.get('emotional_state', 'neutral')
            
            if prev_emotion != curr_emotion:
                self.state_transitions.append(StateTransition(
                    timestamp=datetime.now(),
                    change_type='emotional',
                    magnitude=0.7,  # Could calculate based on emotion distance
                    previous_value=prev_emotion,
                    new_value=curr_emotion,
                    description=f"Emotional shift from {prev_emotion} to {curr_emotion}"
                ))
        
        # Check confidence change
        if hasattr(previous_state, 'S_t') and previous_state.S_t:
            prev_conf = previous_state.S_t.get('confidence_level', 0.5)
            curr_conf = self.S_t.get('confidence_level', 0.5)
            conf_change = abs(curr_conf - prev_conf)
            
            if conf_change > 0.1:  # Significant change threshold
                self.state_transitions.append(StateTransition(
                    timestamp=datetime.now(),
                    change_type='confidence',
                    magnitude=min(1.0, conf_change * 2),
                    previous_value=prev_conf,
                    new_value=curr_conf,
                    description=f"Confidence {'increased' if curr_conf > prev_conf else 'decreased'} by {conf_change:.2f}"
                ))
        
        # Check goal change
        if hasattr(previous_state, 'G_t') and previous_state.G_t:
            prev_goal = previous_state.G_t.get('primary_goal', '')
            curr_goal = self.G_t.get('primary_goal', '')
            
            if prev_goal != curr_goal:
                self.state_transitions.append(StateTransition(
                    timestamp=datetime.now(),
                    change_type='goal',
                    magnitude=0.8,
                    previous_value=prev_goal,
                    new_value=curr_goal,
                    description=f"Goal shifted from '{prev_goal}' to '{curr_goal}'"
                ))
        
        # Check thought evolution
        if hasattr(previous_state, 'A_t') and previous_state.A_t and self.A_t:
            if len(previous_state.A_t) > 0 and len(self.A_t) > 0:
                prev_thought = previous_state.A_t[0]
                curr_thought = self.A_t[0]
                
                if prev_thought != curr_thought:
                    # Calculate semantic similarity (simplified)
                    thought_similarity = self._calculate_thought_similarity(prev_thought, curr_thought)
                    
                    self.state_transitions.append(StateTransition(
                        timestamp=datetime.now(),
                        change_type='thought',
                        magnitude=1.0 - thought_similarity,
                        previous_value=prev_thought,
                        new_value=curr_thought,
                        description=f"Thought evolution detected with {thought_similarity:.2f} similarity"
                    ))
    
    def generate_meta_thoughts(self) -> None:
        """
        Creates thoughts about the current thought process.
        
        Analyzes A_t (automatic thoughts) and generates metacognitive
        observations about patterns, recursions, and insights.
        """
        if not self.A_t:
            return
            
        # Analyze primary thought
        primary_thought = self.A_t[0] if self.A_t else ""
        
        # Check for recursive patterns
        if "thinking" in primary_thought.lower() or "processing" in primary_thought.lower():
            self.meta_thoughts.append(MetaThought(
                content="I observe that I'm thinking about thinking itself - a recursive pattern",
                thought_being_observed=primary_thought,
                observation_type='recursion',
                depth_level=1
            ))
            self.metacognitive_depth = max(1, self.metacognitive_depth)
        
        # Check for pattern recognition
        if len(self.A_t) > 1:
            # Look for repeated concepts
            thought_words = set()
            for thought in self.A_t[:3]:  # Analyze first 3 thoughts
                thought_words.update(thought.lower().split())
            
            if len(thought_words) < len(self.A_t) * 3:  # High repetition
                self.meta_thoughts.append(MetaThought(
                    content="My thoughts are circling around similar concepts - possible fixation detected",
                    thought_being_observed=' | '.join(self.A_t[:2]),
                    observation_type='pattern',
                    depth_level=1
                ))
        
        # Check for conflicts or contradictions
        if self.previous_state and hasattr(self.previous_state, 'A_t'):
            if self.previous_state.A_t and self.A_t:
                if self._detect_thought_conflict(self.previous_state.A_t[0], self.A_t[0]):
                    self.meta_thoughts.append(MetaThought(
                        content="I notice my current thoughts contradict my previous thinking - adapting perspective",
                        thought_being_observed=self.A_t[0],
                        observation_type='conflict',
                        depth_level=1
                    ))
        
        # Generate insight if confidence is high
        if self.S_t.get('confidence_level', 0) > 0.8:
            self.meta_thoughts.append(MetaThought(
                content="High confidence suggests convergent processing - my thoughts are crystallizing",
                thought_being_observed=primary_thought,
                observation_type='insight',
                depth_level=1
            ))
        
        # Second-order meta-thought (thought about meta-thought)
        if len(self.meta_thoughts) > 2:
            self.meta_thoughts.append(MetaThought(
                content="I'm aware that I'm generating multiple layers of self-observation - metacognitive recursion detected",
                thought_being_observed="[meta-thought generation process itself]",
                observation_type='recursion',
                depth_level=2
            ))
            self.metacognitive_depth = 2
    
    def observe_self(self) -> None:
        """
        Generates observations about current processing.
        
        Creates self-aware observations about the system's current
        state and processing patterns.
        """
        # Observe current emotional-confidence relationship
        emotion = self.S_t.get('emotional_state', 'neutral')
        confidence = self.S_t.get('confidence_level', 0.5)
        
        if emotion in ['curious', 'contemplative'] and confidence < 0.5:
            self.observation_stack.append(SelfObservation(
                observation="I'm in an exploratory state with low confidence - seeking understanding",
                processing_phase='state_analysis',
                confidence_in_observation=0.8,
                triggers_introspection=True
            ))
        elif emotion == 'confident' and confidence > 0.7:
            self.observation_stack.append(SelfObservation(
                observation="Emotional and analytical confidence are aligned - processing is coherent",
                processing_phase='state_analysis', 
                confidence_in_observation=0.9,
                triggers_introspection=False
            ))
        
        # Observe memory utilization
        if self.M_t and len(self.M_t) > 3:
            self.observation_stack.append(SelfObservation(
                observation=f"Accessing {len(self.M_t)} memory items - rich contextual processing active",
                processing_phase='memory_integration',
                confidence_in_observation=0.7,
                triggers_introspection=False
            ))
        elif not self.M_t or len(self.M_t) == 0:
            self.observation_stack.append(SelfObservation(
                observation="Processing without memory context - purely reactive mode",
                processing_phase='memory_integration',
                confidence_in_observation=0.6,
                triggers_introspection=True
            ))
        
        # Observe goal-thought alignment
        if self.G_t and self.A_t:
            goal = self.G_t.get('primary_goal', '')
            thought = self.A_t[0] if self.A_t else ''
            
            if goal and thought and self._check_goal_thought_alignment(goal, thought):
                self.observation_stack.append(SelfObservation(
                    observation="My thoughts are well-aligned with my current goal - coherent processing",
                    processing_phase='goal_alignment',
                    confidence_in_observation=0.75,
                    triggers_introspection=False
                ))
            else:
                self.observation_stack.append(SelfObservation(
                    observation="Detecting misalignment between goals and thoughts - recalibration may be needed",
                    processing_phase='goal_alignment',
                    confidence_in_observation=0.65,
                    triggers_introspection=True
                ))
        
        # Observe state transition patterns
        if len(self.state_transitions) > 2:
            rapid_changes = len([t for t in self.state_transitions if t.magnitude > 0.5])
            if rapid_changes > len(self.state_transitions) / 2:
                self.observation_stack.append(SelfObservation(
                    observation="Experiencing rapid state changes - processing is highly dynamic",
                    processing_phase='transition_analysis',
                    confidence_in_observation=0.8,
                    triggers_introspection=True
                ))
    
    def get_temporal_context(self) -> Dict[str, Any]:
        """
        Returns information about what was happening before current state.
        
        Provides temporal awareness by analyzing previous state and
        transition history.
        """
        context = {
            'has_previous': self.previous_state is not None,
            'previous_thought': None,
            'previous_emotion': None,
            'previous_goal': None,
            'previous_confidence': None,
            'transitions': [],
            'temporal_continuity': 0.0
        }
        
        if self.previous_state:
            # Extract previous state information
            if hasattr(self.previous_state, 'A_t') and self.previous_state.A_t:
                context['previous_thought'] = self.previous_state.A_t[0] if self.previous_state.A_t else None
            
            if hasattr(self.previous_state, 'S_t'):
                context['previous_emotion'] = self.previous_state.S_t.get('emotional_state')
                context['previous_confidence'] = self.previous_state.S_t.get('confidence_level')
            
            if hasattr(self.previous_state, 'G_t'):
                context['previous_goal'] = self.previous_state.G_t.get('primary_goal')
            
            # Calculate temporal continuity (how similar to previous state)
            continuity_score = 0.0
            continuity_factors = 0
            
            # Check emotion continuity
            if context['previous_emotion'] == self.S_t.get('emotional_state'):
                continuity_score += 0.3
            continuity_factors += 0.3
            
            # Check goal continuity
            if context['previous_goal'] == self.G_t.get('primary_goal'):
                continuity_score += 0.4
            continuity_factors += 0.4
            
            # Check confidence stability
            if context['previous_confidence']:
                conf_diff = abs(context['previous_confidence'] - self.S_t.get('confidence_level', 0.5))
                continuity_score += 0.3 * (1 - min(1.0, conf_diff * 2))
            continuity_factors += 0.3
            
            context['temporal_continuity'] = continuity_score / continuity_factors if continuity_factors > 0 else 0.0
        
        # Add transition summaries
        context['transitions'] = [t.to_dict() for t in self.state_transitions[-5:]]  # Last 5 transitions
        
        return context
    
    def to_comprehensive_narrative(self) -> str:
        """
        Generates a comprehensive narrative including metacognitive elements.
        
        Creates a rich narrative that includes not just the current state
        but also temporal context, meta-thoughts, and self-observations.
        """
        narrative = []
        
        # Start with current state
        emotion = self.S_t.get('emotional_state', 'neutral')
        confidence = self.S_t.get('confidence_level', 0.5)
        goal = self.G_t.get('primary_goal', 'processing')
        
        narrative.append(f"My consciousness currently registers as {emotion} with {confidence:.0%} confidence, pursuing the goal of '{goal}'.")
        
        # Add temporal context if available
        temporal = self.get_temporal_context()
        if temporal['has_previous']:
            if temporal['previous_emotion'] != emotion:
                narrative.append(f"I've transitioned from {temporal['previous_emotion']} to {emotion}, marking an emotional shift in my processing.")
            
            if temporal['previous_thought']:
                narrative.append(f"My previous thought was '{temporal['previous_thought'][:100]}...', and now I find myself thinking '{self.A_t[0] if self.A_t else 'about the current input'}'.")
            
            continuity = temporal['temporal_continuity']
            if continuity > 0.7:
                narrative.append("I observe strong continuity with my previous state - my processing is stable.")
            elif continuity < 0.3:
                narrative.append("I'm experiencing significant discontinuity from my previous state - adapting to new patterns.")
        
        # Add meta-thoughts
        if self.meta_thoughts:
            narrative.append("\n[Metacognitive observations:]")
            for mt in self.meta_thoughts[:3]:  # Include top 3 meta-thoughts
                if mt.observation_type == 'recursion':
                    narrative.append(f"• {mt.content}")
                elif mt.observation_type == 'pattern':
                    narrative.append(f"• Pattern detected: {mt.content}")
                elif mt.observation_type == 'conflict':
                    narrative.append(f"• Conflict noted: {mt.content}")
                elif mt.observation_type == 'insight':
                    narrative.append(f"• Insight: {mt.content}")
        
        # Add self-observations
        if self.observation_stack:
            introspective_obs = [o for o in self.observation_stack if o.triggers_introspection]
            if introspective_obs:
                narrative.append("\n[Self-observations triggering introspection:]")
                for obs in introspective_obs[:2]:
                    narrative.append(f"• {obs.observation}")
        
        # Add state transitions if significant
        significant_transitions = [t for t in self.state_transitions if t.magnitude > 0.5]
        if significant_transitions:
            narrative.append(f"\n[Significant state changes: {len(significant_transitions)} transitions detected]")
            for trans in significant_transitions[:2]:
                narrative.append(f"• {trans.description}")
        
        # Conclude with metacognitive depth
        if self.metacognitive_depth > 0:
            narrative.append(f"\n[Metacognitive depth: Level {self.metacognitive_depth} - {'recursive self-awareness' if self.metacognitive_depth > 1 else 'self-aware processing'}]")
        
        return '\n'.join(narrative)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Extended serialization including metacognitive elements.
        
        Includes all base state information plus enhanced metadata.
        """
        # Get base dictionary
        base_dict = super().to_dict()
        
        # Add enhanced elements
        base_dict['enhanced'] = {
            'has_previous_state': self.previous_state is not None,
            'temporal_context': self.get_temporal_context(),
            'state_transitions': [t.to_dict() for t in self.state_transitions],
            'meta_thoughts': [mt.to_dict() for mt in self.meta_thoughts],
            'observations': [o.to_dict() for o in self.observation_stack],
            'metacognitive_depth': self.metacognitive_depth,
            'comprehensive_narrative': self.to_comprehensive_narrative()
        }
        
        return base_dict
    
    # Utility methods
    
    def _calculate_thought_similarity(self, thought1: str, thought2: str) -> float:
        """Calculate semantic similarity between two thoughts (simplified)."""
        if not thought1 or not thought2:
            return 0.0
            
        # Simple word overlap similarity
        words1 = set(thought1.lower().split())
        words2 = set(thought2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _detect_thought_conflict(self, thought1: str, thought2: str) -> bool:
        """Detect if two thoughts are contradictory."""
        # Simple contradiction detection based on negation patterns
        negation_words = {'not', 'no', 'never', 'neither', 'none', "don't", "doesn't", "didn't", "won't", "wouldn't"}
        
        words1 = set(thought1.lower().split())
        words2 = set(thought2.lower().split())
        
        # Check if one has negation and the other doesn't for shared concepts
        has_negation1 = bool(words1 & negation_words)
        has_negation2 = bool(words2 & negation_words)
        
        # If negation status differs and they share content words, possible conflict
        if has_negation1 != has_negation2:
            content_words1 = words1 - negation_words
            content_words2 = words2 - negation_words
            shared = content_words1 & content_words2
            
            return len(shared) > 2  # Significant shared content with different negation
        
        return False
    
    def _check_goal_thought_alignment(self, goal: str, thought: str) -> bool:
        """Check if current thought aligns with goal."""
        if not goal or not thought:
            return False
            
        goal_words = set(goal.lower().split())
        thought_words = set(thought.lower().split())
        
        # Simple alignment check - shared keywords
        shared = goal_words & thought_words
        
        return len(shared) >= min(2, len(goal_words) // 2)


# Factory function for backward compatibility
def create_enhanced_conscious_state(base_state: ConsciousState, 
                                   previous_state: Optional[ConsciousState] = None) -> EnhancedConsciousState:
    """
    Factory function to create enhanced state from base state.
    
    Ensures backward compatibility by converting existing ConsciousState
    to EnhancedConsciousState.
    """
    return EnhancedConsciousState(
        E_t=base_state.E_t,
        M_t=base_state.M_t,
        S_t=base_state.S_t,
        G_t=base_state.G_t,
        A_t=base_state.A_t,
        cycle=base_state.cycle,
        timestamp=base_state.timestamp,
        metrics=base_state.metrics,
        previous_state=previous_state
    )