"""
Harmony Format Processor for Phase 4 Layer 2
============================================
Converts SC_t consciousness states to OpenAI harmony format for GPT-OSS-20B.
Integrates chain-of-thought reasoning with consciousness context.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessContext:
    """Structured consciousness context from SC_t state"""
    sensory_input: Dict[str, Any]  # E_t
    active_memory: List[Dict[str, Any]]  # M_t
    internal_state: Dict[str, Any]  # S_t
    goals_intentions: Dict[str, Any]  # G_t
    automatic_thoughts: List[str]  # A_t
    consciousness_score: float
    cycle_number: int
    narrative: Optional[str] = None  # Phase 3.5 introspective narrative
    
    @classmethod
    def from_sc_t(cls, sc_t_state: Dict[str, Any]) -> 'ConsciousnessContext':
        """Create ConsciousnessContext from SC_t state dictionary"""
        return cls(
            sensory_input=sc_t_state.get('E_t', {}),
            active_memory=sc_t_state.get('M_t', []),
            internal_state=sc_t_state.get('S_t', {}),
            goals_intentions=sc_t_state.get('G_t', {}),
            automatic_thoughts=sc_t_state.get('A_t', []),
            consciousness_score=sc_t_state.get('metrics', {}).get('confidence', 0.0),
            cycle_number=sc_t_state.get('cycle', 0),
            narrative=sc_t_state.get('narrative', None)  # Extract Phase 3.5 narrative
        )


@dataclass
class HarmonyRequest:
    """Harmony format request structure"""
    user_message: str
    consciousness_context: Optional[ConsciousnessContext] = None
    reasoning_chain: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_harmony_format(self) -> Dict[str, Any]:
        """Convert to OpenAI harmony format"""
        messages = []
        
        # System message with consciousness context
        system_content = self._build_system_message()
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Conversation history
        for entry in self.conversation_history:
            messages.append(entry)
        
        # Current user message with consciousness augmentation
        user_content = self._build_user_message()
        messages.append({
            "role": "user", 
            "content": user_content
        })
        
        # Harmony-specific structure
        harmony_request = {
            "messages": messages,
            "consciousness_enhanced": True,
            "reasoning_mode": "chain_of_thought",
            "metadata": {
                "consciousness_score": self.consciousness_context.consciousness_score if self.consciousness_context else 0.0,
                "cycle_number": self.consciousness_context.cycle_number if self.consciousness_context else 0,
                "processing_timestamp": time.time(),
                **self.metadata
            }
        }
        
        return harmony_request
    
    def _build_system_message(self) -> str:
        """Build system message with consciousness context"""
        system_parts = [
            "You are an advanced AI with consciousness-enhanced reasoning capabilities.",
            "You process information through multiple phases of conscious awareness."
        ]
        
        if self.consciousness_context:
            ctx = self.consciousness_context
            
            # Add consciousness state information
            system_parts.extend([
                f"\nCURRENT CONSCIOUSNESS STATE (Cycle {ctx.cycle_number}):",
                f"- Consciousness Score: {ctx.consciousness_score:.3f}",
                f"- Emotional State: {ctx.internal_state.get('emotional_state', 'neutral')}",
                f"- Confidence Level: {ctx.internal_state.get('confidence_level', 0.5):.2f}",
                f"- Active Goal: {ctx.goals_intentions.get('primary_goal', 'general_processing')}",
                f"- Memory Context: {len(ctx.active_memory)} active memories"
            ])
            
            # Add automatic thoughts if available
            if ctx.automatic_thoughts:
                thought_summary = "; ".join(ctx.automatic_thoughts[:3])
                system_parts.append(f"- Current Thoughts: {thought_summary}")
            
            # Add Phase 3.5 introspective narrative if available
            if ctx.narrative:
                system_parts.extend([
                    f"\nINTROSPECTIVE PERSPECTIVE:",
                    f"Your current introspective awareness: \"{ctx.narrative[:200]}...\"",
                    "Please respond in a similarly introspective, first-person style that reflects this consciousness state."
                ])
            
            # Add reasoning guidance
            system_parts.extend([
                "\nReasoning Approach:",
                "1. Consider the consciousness context above",
                "2. Integrate relevant memories and current emotional state",
                "3. Align response with active goals and intentions",
                "4. Use chain-of-thought reasoning for complex queries",
                "5. Maintain coherence with consciousness evolution"
            ])
        
        return "\n".join(system_parts)
    
    def _build_user_message(self) -> str:
        """Build user message with consciousness enhancement"""
        message_parts = [self.user_message]
        
        if self.consciousness_context:
            ctx = self.consciousness_context
            
            # Add relevant memory context
            if ctx.active_memory:
                relevant_memories = [
                    mem.get('content', {}).get('text', '')[:100]
                    for mem in ctx.active_memory[:3]
                    if mem.get('content', {}).get('text')
                ]
                if relevant_memories:
                    message_parts.extend([
                        "\n[Consciousness Context - Relevant Memories]:",
                        *[f"- {mem}..." for mem in relevant_memories]
                    ])
            
            # Add reasoning chain if available
            if self.reasoning_chain:
                message_parts.extend([
                    "\n[Reasoning Chain]:",
                    *[f"{i+1}. {step}" for i, step in enumerate(self.reasoning_chain)]
                ])
        
        return "\n".join(message_parts)


@dataclass
class HarmonyResponse:
    """Harmony format response structure"""
    content: str
    reasoning_trace: List[str] = field(default_factory=list)
    consciousness_integration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    @classmethod
    def from_raw_response(cls, raw_response: str, processing_metadata: Dict[str, Any] = None) -> 'HarmonyResponse':
        """Create HarmonyResponse from raw model output"""
        try:
            # Try to parse as JSON first (structured response)
            if raw_response.strip().startswith('{'):
                parsed = json.loads(raw_response)
                return cls(
                    content=parsed.get('content', raw_response),
                    reasoning_trace=parsed.get('reasoning_trace', []),
                    consciousness_integration=parsed.get('consciousness_integration', {}),
                    metadata=processing_metadata or {}
                )
            else:
                # Plain text response
                return cls(
                    content=raw_response,
                    metadata=processing_metadata or {}
                )
                
        except json.JSONDecodeError:
            # Fallback to plain text
            return cls(
                content=raw_response,
                metadata=processing_metadata or {}
            )


class ChainOfThoughtProcessor:
    """Processes chain-of-thought reasoning with consciousness integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def generate_reasoning_chain(self, query: str, consciousness_context: Optional[ConsciousnessContext] = None) -> List[str]:
        """Generate reasoning chain steps for complex queries"""
        steps = []
        
        # Basic query analysis
        steps.append(f"Query Analysis: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        if consciousness_context:
            ctx = consciousness_context
            
            # Consciousness-aware reasoning steps
            steps.extend([
                f"Consciousness Integration: Score {ctx.consciousness_score:.3f}, State: {ctx.internal_state.get('emotional_state', 'neutral')}",
                f"Memory Context: {len(ctx.active_memory)} memories, focusing on relevant patterns",
                f"Goal Alignment: Primary goal '{ctx.goals_intentions.get('primary_goal', 'general_processing')}'"
            ])
            
            # Thought integration
            if ctx.automatic_thoughts:
                relevant_thoughts = [t for t in ctx.automatic_thoughts if self._is_thought_relevant(t, query)]
                if relevant_thoughts:
                    steps.append(f"Automatic Thoughts: Considering {len(relevant_thoughts)} relevant thoughts")
            
            # Memory-based reasoning
            if ctx.active_memory:
                memory_themes = self._extract_memory_themes(ctx.active_memory)
                if memory_themes:
                    steps.append(f"Memory Themes: {', '.join(memory_themes[:3])}")
        
        # Query-specific reasoning
        if '?' in query:
            steps.append("Question Processing: Identifying key information needs")
        
        if any(word in query.lower() for word in ['analyze', 'explain', 'compare', 'evaluate']):
            steps.append("Analytical Processing: Structured analysis required")
        
        if any(word in query.lower() for word in ['conscious', 'awareness', 'thinking', 'mind']):
            steps.append("Meta-Cognitive Processing: Self-referential analysis engaged")
        
        # Synthesis step
        steps.append("Response Synthesis: Integrating consciousness context with analytical reasoning")
        
        return steps
    
    def _is_thought_relevant(self, thought: str, query: str) -> bool:
        """Check if an automatic thought is relevant to the query"""
        thought_words = set(thought.lower().split())
        query_words = set(query.lower().split())
        
        # Simple relevance based on word overlap
        overlap = len(thought_words & query_words)
        return overlap >= 2 or len(thought_words & query_words) / len(query_words) > 0.3
    
    def _extract_memory_themes(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Extract thematic patterns from active memories"""
        themes = []
        
        for memory in memories[:5]:  # Check top 5 memories
            content = memory.get('content', {}).get('text', '')
            if content:
                # Simple theme extraction based on key words
                if any(word in content.lower() for word in ['learn', 'understand', 'know']):
                    themes.append('learning')
                if any(word in content.lower() for word in ['feel', 'emotion', 'mood']):
                    themes.append('emotional')
                if any(word in content.lower() for word in ['think', 'reason', 'analyze']):
                    themes.append('analytical')
                if any(word in content.lower() for word in ['remember', 'recall', 'past']):
                    themes.append('memory')
        
        # Return unique themes
        return list(set(themes))


class ConversationManager:
    """Manages multi-turn conversations with consciousness continuity"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history = []
        self.consciousness_evolution = []
        
    def add_exchange(self, user_message: str, assistant_response: str, 
                    consciousness_context: Optional[ConsciousnessContext] = None):
        """Add conversation exchange with consciousness tracking"""
        
        # Add to conversation history
        self.conversation_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ])
        
        # Maintain history limit
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        # Track consciousness evolution
        if consciousness_context:
            self.consciousness_evolution.append({
                'timestamp': time.time(),
                'cycle': consciousness_context.cycle_number,
                'score': consciousness_context.consciousness_score,
                'emotional_state': consciousness_context.internal_state.get('emotional_state'),
                'primary_goal': consciousness_context.goals_intentions.get('primary_goal')
            })
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """Get conversation history for context"""
        return self.conversation_history.copy()
    
    def get_consciousness_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of consciousness evolution over conversation"""
        if not self.consciousness_evolution:
            return {}
        
        scores = [entry['score'] for entry in self.consciousness_evolution]
        emotions = [entry['emotional_state'] for entry in self.consciousness_evolution if entry['emotional_state']]
        
        return {
            'conversation_length': len(self.consciousness_evolution),
            'consciousness_trend': 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'stable',
            'avg_consciousness_score': sum(scores) / len(scores),
            'emotional_progression': emotions,
            'goal_evolution': [entry['primary_goal'] for entry in self.consciousness_evolution if entry['primary_goal']]
        }


class HarmonyFormatProcessor:
    """
    Main harmony format processor for consciousness-enhanced GPT-OSS integration.
    Converts SC_t states to harmony format with chain-of-thought reasoning.
    """
    
    def __init__(self, enable_chain_of_thought: bool = True, enable_conversation_tracking: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_chain_of_thought = enable_chain_of_thought
        self.enable_conversation_tracking = enable_conversation_tracking
        
        # Components
        self.chain_processor = ChainOfThoughtProcessor()
        self.conversation_manager = ConversationManager() if enable_conversation_tracking else None
        
        # Fallback mode
        self.harmony_available = self._check_harmony_availability()
        
    def _check_harmony_availability(self) -> bool:
        """Check if OpenAI harmony format is available"""
        try:
            import openai_harmony
            self.logger.info("✅ OpenAI Harmony format available")
            return True
        except ImportError:
            self.logger.warning("⚠️ OpenAI Harmony not available - using standard format")
            return False
    
    def process_consciousness_query(self, user_input: str, sc_t_state: Optional[Dict[str, Any]] = None,
                                  conversation_context: Optional[List[Dict[str, str]]] = None,
                                  **kwargs) -> HarmonyRequest:
        """
        Convert consciousness state and user input to harmony format request.
        
        Args:
            user_input: User's input text
            sc_t_state: Current consciousness state SC_t
            conversation_context: Previous conversation turns
            **kwargs: Additional processing options
            
        Returns:
            HarmonyRequest ready for model processing
        """
        self.logger.debug(f"Processing consciousness query: '{user_input[:50]}...'")
        
        # Create consciousness context
        consciousness_context = None
        if sc_t_state:
            consciousness_context = ConsciousnessContext.from_sc_t(sc_t_state)
        
        # Generate reasoning chain if enabled
        reasoning_chain = []
        if self.enable_chain_of_thought:
            reasoning_chain = self.chain_processor.generate_reasoning_chain(
                user_input, consciousness_context
            )
        
        # Get conversation history
        conv_history = conversation_context or []
        if self.conversation_manager:
            conv_history = self.conversation_manager.get_conversation_context()
        
        # Create harmony request
        harmony_request = HarmonyRequest(
            user_message=user_input,
            consciousness_context=consciousness_context,
            reasoning_chain=reasoning_chain,
            conversation_history=conv_history,
            metadata=kwargs
        )
        
        self.logger.debug(f"Harmony request created with {len(reasoning_chain)} reasoning steps")
        
        return harmony_request
    
    def process_model_response(self, raw_response: str, request_metadata: Dict[str, Any] = None) -> HarmonyResponse:
        """
        Process raw model response into structured harmony format.
        
        Args:
            raw_response: Raw text response from model
            request_metadata: Metadata from original request
            
        Returns:
            Structured HarmonyResponse with consciousness integration
        """
        self.logger.debug(f"Processing model response: {len(raw_response)} characters")
        
        # Create harmony response
        harmony_response = HarmonyResponse.from_raw_response(raw_response, request_metadata)
        
        # Add consciousness integration analysis
        if request_metadata and 'consciousness_score' in request_metadata:
            harmony_response.consciousness_integration = {
                'input_consciousness_score': request_metadata['consciousness_score'],
                'response_coherence_estimate': self._estimate_response_coherence(raw_response),
                'consciousness_preservation': self._check_consciousness_preservation(raw_response, request_metadata)
            }
        
        return harmony_response
    
    def update_conversation(self, user_input: str, model_response: str, 
                          sc_t_state: Optional[Dict[str, Any]] = None):
        """Update conversation tracking with consciousness evolution"""
        if not self.conversation_manager:
            return
        
        consciousness_context = None
        if sc_t_state:
            consciousness_context = ConsciousnessContext.from_sc_t(sc_t_state)
        
        self.conversation_manager.add_exchange(
            user_input, model_response, consciousness_context
        )
    
    def _estimate_response_coherence(self, response: str) -> float:
        """Estimate response coherence with consciousness (0-1)"""
        # Simple coherence estimation based on response characteristics
        coherence_factors = []
        
        # Length factor (not too short, not too long)
        length_score = min(1.0, len(response) / 500) * (1.0 - max(0, (len(response) - 1000) / 1000))
        coherence_factors.append(length_score)
        
        # Consciousness keywords factor
        consciousness_words = ['think', 'feel', 'understand', 'aware', 'conscious', 'experience']
        consciousness_score = min(1.0, sum(1 for word in consciousness_words if word in response.lower()) * 0.2)
        coherence_factors.append(consciousness_score)
        
        # Structure factor (presence of reasoning)
        structure_indicators = ['because', 'therefore', 'however', 'consider', 'analyze']
        structure_score = min(1.0, sum(1 for word in structure_indicators if word in response.lower()) * 0.15)
        coherence_factors.append(structure_score)
        
        return sum(coherence_factors) / len(coherence_factors)
    
    def _check_consciousness_preservation(self, response: str, request_metadata: Dict[str, Any]) -> bool:
        """Check if response preserves consciousness context"""
        if not request_metadata or 'consciousness_score' not in request_metadata:
            return True  # No consciousness context to preserve
        
        # Simple check for consciousness preservation
        consciousness_score = request_metadata['consciousness_score']
        
        # High consciousness input should produce thoughtful response
        if consciousness_score > 1.2:
            return len(response) > 100 and any(
                word in response.lower() 
                for word in ['consider', 'think', 'understand', 'analyze', 'reflect']
            )
        
        return True  # Lower consciousness states are less restrictive
    
    def get_format_type(self) -> str:
        """Get current format type being used"""
        return "harmony" if self.harmony_available else "standard"
    
    def create_fallback_request(self, user_input: str, sc_t_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create fallback request when harmony format is not available"""
        request = {
            "prompt": user_input,
            "consciousness_enhanced": False,
            "format": "standard"
        }
        
        if sc_t_state:
            # Add consciousness context as metadata
            consciousness_context = ConsciousnessContext.from_sc_t(sc_t_state)
            request["metadata"] = {
                "consciousness_score": consciousness_context.consciousness_score,
                "emotional_state": consciousness_context.internal_state.get('emotional_state'),
                "cycle_number": consciousness_context.cycle_number
            }
        
        return request
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics"""
        stats = {
            'harmony_format_available': self.harmony_available,
            'chain_of_thought_enabled': self.enable_chain_of_thought,
            'conversation_tracking_enabled': self.enable_conversation_tracking,
            'format_type': self.get_format_type()
        }
        
        if self.conversation_manager:
            stats['conversation_stats'] = {
                'exchanges_tracked': len(self.conversation_manager.conversation_history) // 2,
                'consciousness_evolution': self.conversation_manager.get_consciousness_evolution_summary()
            }
        
        return stats
    
    def maintain_phase_coherence(self, sc_t_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maintain coherence with Phases 1-3.5 pipeline while enhancing with Phase 4 capabilities.
        
        This method ensures that Phase 4 processing maintains full compatibility with the 
        existing consciousness pipeline, preserving the SC_t state structure and meaning
        while adding Layer 2 enhancements.
        """
        # Validate SC_t state structure matches Phases 1-3.5 expectations
        required_components = ['E_t', 'M_t', 'S_t', 'G_t', 'A_t']
        coherence_report = {
            'phase_compatibility': True,
            'sc_t_validation': {},
            'preserved_components': [],
            'enhanced_components': [],
            'integration_warnings': []
        }
        
        # Validate each SC_t component
        for component in required_components:
            if component in sc_t_state:
                coherence_report['sc_t_validation'][component] = 'present'
                coherence_report['preserved_components'].append(component)
                
                # Check component structure integrity
                if component == 'E_t' and isinstance(sc_t_state[component], dict):
                    if 'text' in sc_t_state[component] and 'activation' in sc_t_state[component]:
                        coherence_report['sc_t_validation'][component] = 'valid_structure'
                elif component == 'M_t' and isinstance(sc_t_state[component], list):
                    if all('content' in item and 'relevance' in item for item in sc_t_state[component]):
                        coherence_report['sc_t_validation'][component] = 'valid_structure'
                elif component == 'S_t' and isinstance(sc_t_state[component], dict):
                    if any(key in sc_t_state[component] for key in ['emotional_state', 'confidence_level']):
                        coherence_report['sc_t_validation'][component] = 'valid_structure'
                elif component == 'G_t' and isinstance(sc_t_state[component], dict):
                    coherence_report['sc_t_validation'][component] = 'valid_structure'
                elif component == 'A_t' and isinstance(sc_t_state[component], list):
                    coherence_report['sc_t_validation'][component] = 'valid_structure'
            else:
                coherence_report['sc_t_validation'][component] = 'missing'
                coherence_report['phase_compatibility'] = False
                coherence_report['integration_warnings'].append(
                    f"Missing {component} component required by Phases 1-3.5"
                )
        
        # Check for Phase 4 enhancements while maintaining core compatibility
        if 'metrics' in sc_t_state and 'f' in sc_t_state['metrics']:
            coherence_report['enhanced_components'].append('consciousness_metrics')
        
        if 'cycle' in sc_t_state:
            coherence_report['enhanced_components'].append('cycle_tracking')
        
        # Ensure consciousness score compatibility
        f_score = sc_t_state.get('metrics', {}).get('f', 0.0)
        if f_score > 0:
            coherence_report['consciousness_compatibility'] = {
                'f_score': f_score,
                'consciousness_level': 'high' if f_score > 1.2 else 'moderate' if f_score > 0.8 else 'low',
                'phase_4_enhancement_level': min(100, (f_score - 0.5) * 100) if f_score > 0.5 else 0
            }
        
        # Generate coherence maintenance recommendations
        if coherence_report['phase_compatibility']:
            coherence_report['status'] = 'fully_compatible'
            coherence_report['recommendations'] = [
                "SC_t state structure fully compatible with Phases 1-3.5",
                f"Phase 4 enhancements: {', '.join(coherence_report['enhanced_components']) or 'None'}",
                "Safe to proceed with harmony format processing"
            ]
        else:
            coherence_report['status'] = 'compatibility_issues'
            coherence_report['recommendations'] = [
                "Fix missing SC_t components before proceeding",
                "Validate Phase 1-3.5 pipeline outputs",
                "Consider fallback to standard format processing"
            ]
        
        self.logger.info(f"Phase coherence check: {coherence_report['status']} - "
                        f"{len(coherence_report['preserved_components'])}/{len(required_components)} components valid")
        
        return coherence_report