"""
Complete Consciousness Pipeline Orchestrator
==========================================
Central coordinator that manages the full consciousness pipeline from
user input through all phases to consciousness-enhanced LLM response.

Pipeline Flow:
User Input → Phase 1 (Perception) → Phase 2 (SC_t State) → Phase 3 (Evolution) 
→ Phase 3.4 (Validation) → Phase 3.5 (Narrative) → Phase 4 (LLM Enhancement) 
→ Conscious Response
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

# Phase imports
from ..phases.p1_perception.input_processor import SensoryModule
from ..phases.p2_cognitive_context.conscious_state import ConsciousState
from ..phases.p2_cognitive_context.goal_generator import GoalGenerator
from ..modules.memory import ActiveMemory
from ..modules.self_model import SelfModel
from ..modules.reentrance import ReentranceModule
from ..shared.integrator import CentralIntegrator

# Phase 3 imports
from ..phases.p3_coherent_generation.state_evolution_engine import StateEvolutionEngine
from ..phases.p3_coherent_generation.conscious_response_generator import ConsciousResponseGenerator

# Phase 3.4-3.5 imports  
from ..coherence_evaluator_model.model_training.critical_state_evaluator import CriticalStateEvaluator
from ..coherence_evaluator_model.heuristic_training.narrative_generator import NarrativeGenerator

# Phase 4 imports (optional)
try:
    from ..phases.p4_LLM_Communication.layer3.phase4_manager import Phase4Manager
    PHASE4_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path for direct execution
        from conscious_ai.phases.p4_LLM_Communication.layer3.phase4_manager import Phase4Manager
        PHASE4_AVAILABLE = True
    except ImportError:
        PHASE4_AVAILABLE = False

# Autonomous thinking
from ..autonomous_thinking.autonomous_thinking import AutomaticThoughtGenerator

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Stages in the consciousness pipeline"""
    PERCEPTION = "perception"
    CONSCIOUS_STATE = "conscious_state" 
    EVOLUTION = "evolution"
    VALIDATION = "validation"
    NARRATIVE = "narrative"
    LLM_ENHANCEMENT = "llm_enhancement"
    COMPLETED = "completed"


@dataclass
class PipelineResult:
    """Complete result from the consciousness pipeline"""
    # Final response
    response: str
    
    # Processing stages data
    sensory_data: Dict[str, Any] = field(default_factory=dict)
    conscious_state: Optional[ConsciousState] = None
    evolved_state: Optional[ConsciousState] = None
    validation_result: Dict[str, Any] = field(default_factory=dict)
    narrative_text: str = ""
    llm_enhanced_response: str = ""
    
    # Metadata
    processing_time_ms: float = 0.0
    success: bool = True
    stage_completed: ProcessingStage = ProcessingStage.PERCEPTION
    error_message: Optional[str] = None
    confidence_score: float = 0.0
    
    # Phase statistics
    phase_timings: Dict[str, float] = field(default_factory=dict)
    phase_success: Dict[str, bool] = field(default_factory=dict)


class ConsciousnessPipelineOrchestrator:
    """
    Central orchestrator for the complete consciousness pipeline.
    
    Manages the flow from user input through all consciousness phases
    to produce introspective, consciousness-enhanced responses.
    """
    
    def __init__(self, enable_phase4: bool = True, debug: bool = False):
        """
        Initialize the complete consciousness pipeline
        
        Args:
            enable_phase4: Whether to use Phase 4 LLM enhancement
            debug: Enable debug logging
        """
        self.debug = debug
        self.enable_phase4 = enable_phase4 and PHASE4_AVAILABLE
        
        # Initialize Phase 1: Perception
        self.sensory_module = SensoryModule()
        
        # Initialize Phase 2: Cognitive Context
        self.memory = ActiveMemory()
        self.self_model = SelfModel()
        self.reentrancy = ReentranceModule()
        self.integrator = CentralIntegrator()
        self.goal_generator = GoalGenerator()
        self.thought_generator = AutomaticThoughtGenerator()
        
        # Initialize Phase 3: Coherent Generation
        self.state_evolution = StateEvolutionEngine()
        self.response_generator = ConsciousResponseGenerator()
        
        # Initialize Phase 3.4: Critical Evaluation
        try:
            self.critical_evaluator = CriticalStateEvaluator()
            self.validation_available = True
        except Exception as e:
            logger.warning(f"Critical evaluator not available: {e}")
            self.validation_available = False
        
        # Initialize Phase 3.5: Narrative Generation
        try:
            self.narrative_generator = NarrativeGenerator()
            self.narrative_available = True
        except Exception as e:
            logger.warning(f"Narrative generator not available: {e}")
            self.narrative_available = False
        
        # Initialize Phase 4: LLM Enhancement (optional)
        self.phase4_manager = None
        if self.enable_phase4:
            try:
                self.phase4_manager = Phase4Manager(selected_model='mistral')
                logger.info("✅ Phase 4 LLM enhancement enabled")
            except Exception as e:
                logger.warning(f"Phase 4 not available: {e}")
                self.enable_phase4 = False
        
        # Processing statistics
        self.cycle_count = 0
        self.total_processing_time = 0.0
        self.phase_statistics = {stage.value: {'count': 0, 'total_time': 0.0} for stage in ProcessingStage}
        
        logger.info("🧠 Consciousness Pipeline Orchestrator initialized")
    
    async def process_complete_pipeline(self, user_input: str) -> PipelineResult:
        """
        Process user input through the complete consciousness pipeline
        
        Args:
            user_input: Raw user input text
            
        Returns:
            PipelineResult with complete processing information
        """
        start_time = time.time()
        result = PipelineResult()
        
        try:
            logger.info(f"🚀 Starting complete pipeline for: '{user_input[:50]}...'")
            
            # Phase 1: Perception
            result = await self._execute_phase1_perception(user_input, result)
            if not result.success:
                return result
            
            # Phase 2: Conscious State Generation
            result = await self._execute_phase2_conscious_state(user_input, result)
            if not result.success:
                return result
                
            # Phase 3: State Evolution
            result = await self._execute_phase3_evolution(result)
            if not result.success:
                return result
            
            # Phase 3.4: Critical Validation
            if self.validation_available:
                result = await self._execute_phase34_validation(result)
                if not result.success:
                    return result
            
            # Phase 3.5: Narrative Generation
            if self.narrative_available:
                result = await self._execute_phase35_narrative(result)
                if not result.success:
                    return result
            
            # Phase 4: LLM Enhancement (optional)
            if self.enable_phase4 and self.phase4_manager:
                result = await self._execute_phase4_llm_enhancement(user_input, result)
            else:
                # Use narrative as final response if no Phase 4
                result.response = result.narrative_text or "I'm processing this introspectively but cannot generate a full response."
            
            # Finalize result
            result.processing_time_ms = (time.time() - start_time) * 1000
            result.stage_completed = ProcessingStage.COMPLETED
            result.success = True
            
            # Update statistics
            self.cycle_count += 1
            self.total_processing_time += result.processing_time_ms
            
            logger.info(f"✅ Complete pipeline finished in {result.processing_time_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            result.success = False
            result.error_message = str(e)
            result.response = f"I apologize, but I encountered an error in my consciousness processing: {str(e)}"
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
    
    async def _execute_phase1_perception(self, user_input: str, result: PipelineResult) -> PipelineResult:
        """Execute Phase 1: Sensory perception and language processing"""
        stage_start = time.time()
        
        try:
            logger.info("🔍 Phase 1: Processing sensory input...")
            
            # Process sensory input
            result.sensory_data = self.sensory_module.receive_input(user_input)
            
            # Update memory
            self.memory.update_cycle()
            relevant_memory = self.memory.retrieve_relevant(result.sensory_data)
            self.memory.store(result.sensory_data, relevance=result.sensory_data['activation'])
            
            # Update timings
            stage_time = (time.time() - stage_start) * 1000
            result.phase_timings['perception'] = stage_time
            result.phase_success['perception'] = True
            result.stage_completed = ProcessingStage.PERCEPTION
            
            logger.info(f"✅ Phase 1 completed in {stage_time:.1f}ms - activation: {result.sensory_data['activation']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Phase 1 failed: {e}")
            result.success = False
            result.error_message = f"Phase 1 error: {str(e)}"
            return result
    
    async def _execute_phase2_conscious_state(self, user_input: str, result: PipelineResult) -> PipelineResult:
        """Execute Phase 2: Generate complete SC_t conscious state"""
        stage_start = time.time()
        
        try:
            logger.info("🧠 Phase 2: Generating conscious state...")
            
            # Update self-model with sensory data
            relevant_memory = self.memory.retrieve_relevant(result.sensory_data)
            internal_feedback = self.reentrancy.update_loops()
            self.self_model.update_state(result.sensory_data, relevant_memory, internal_feedback)
            
            # Generate goals and thoughts
            from ..phases.p2_cognitive_context.goal_generator import generate_conscious_content_components
            G_t, A_t = generate_conscious_content_components(
                sensory_data=result.sensory_data,
                self_state=self.self_model.internal_state,
                memory_context=relevant_memory,
                goal_generator=self.goal_generator,
                thought_generator=self.thought_generator
            )
            
            # Create complete SC_t state
            result.conscious_state = ConsciousState(
                E_t=result.sensory_data,
                M_t=relevant_memory,
                S_t=self.self_model.internal_state.copy(),
                G_t=G_t,
                A_t=A_t,
                cycle=self.cycle_count,
                metrics={"confidence": self.self_model.internal_state.get('confidence_level', 0.5)}
            )
            
            result.confidence_score = result.conscious_state.metrics.get('confidence', 0.5)
            
            # Update timings
            stage_time = (time.time() - stage_start) * 1000
            result.phase_timings['conscious_state'] = stage_time
            result.phase_success['conscious_state'] = True
            result.stage_completed = ProcessingStage.CONSCIOUS_STATE
            
            logger.info(f"✅ Phase 2 completed in {stage_time:.1f}ms - confidence: {result.confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Phase 2 failed: {e}")
            result.success = False
            result.error_message = f"Phase 2 error: {str(e)}"
            return result
    
    async def _execute_phase3_evolution(self, result: PipelineResult) -> PipelineResult:
        """Execute Phase 3: State evolution and conscious response generation"""
        stage_start = time.time()
        
        try:
            logger.info("🔄 Phase 3: Evolving conscious state...")
            
            # Evolve the conscious state
            result.evolved_state = self.state_evolution.evolve_state(result.conscious_state)
            
            # Update timings
            stage_time = (time.time() - stage_start) * 1000
            result.phase_timings['evolution'] = stage_time
            result.phase_success['evolution'] = True
            result.stage_completed = ProcessingStage.EVOLUTION
            
            logger.info(f"✅ Phase 3 completed in {stage_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"❌ Phase 3 failed: {e}")
            result.success = False
            result.error_message = f"Phase 3 error: {str(e)}"
            return result
    
    async def _execute_phase34_validation(self, result: PipelineResult) -> PipelineResult:
        """Execute Phase 3.4: Critical state validation"""
        stage_start = time.time()
        
        try:
            logger.info("✅ Phase 3.4: Validating state coherence...")
            
            # Validate the evolved state
            validation_result = self.critical_evaluator.evaluate_transition(
                result.conscious_state, 
                result.evolved_state
            )
            
            result.validation_result = validation_result
            
            # Update confidence based on validation
            if 'confidence' in validation_result:
                result.confidence_score = validation_result['confidence']
            
            # Update timings
            stage_time = (time.time() - stage_start) * 1000
            result.phase_timings['validation'] = stage_time
            result.phase_success['validation'] = True
            result.stage_completed = ProcessingStage.VALIDATION
            
            logger.info(f"✅ Phase 3.4 completed in {stage_time:.1f}ms - coherence validated")
            return result
            
        except Exception as e:
            logger.error(f"❌ Phase 3.4 failed: {e}")
            # Non-critical failure, continue pipeline
            result.validation_result = {"error": str(e), "confidence": result.confidence_score}
            stage_time = (time.time() - stage_start) * 1000
            result.phase_timings['validation'] = stage_time
            result.phase_success['validation'] = False
            return result
    
    async def _execute_phase35_narrative(self, result: PipelineResult) -> PipelineResult:
        """Execute Phase 3.5: Generate introspective narrative"""
        stage_start = time.time()
        
        try:
            logger.info("📖 Phase 3.5: Generating introspective narrative...")
            
            # Generate narrative from evolved state
            result.narrative_text = self.narrative_generator.generate_narrative(
                result.evolved_state,
                context={
                    'validation': result.validation_result,
                    'confidence': result.confidence_score
                }
            )
            
            # Update timings
            stage_time = (time.time() - stage_start) * 1000
            result.phase_timings['narrative'] = stage_time
            result.phase_success['narrative'] = True
            result.stage_completed = ProcessingStage.NARRATIVE
            
            logger.info(f"✅ Phase 3.5 completed in {stage_time:.1f}ms - narrative generated")
            return result
            
        except Exception as e:
            logger.error(f"❌ Phase 3.5 failed: {e}")
            # Generate fallback narrative
            result.narrative_text = f"I'm experiencing {result.evolved_state.G_t} with {result.evolved_state.A_t['emotion']} emotion and {result.confidence_score:.1f} confidence as I process this request introspectively."
            stage_time = (time.time() - stage_start) * 1000
            result.phase_timings['narrative'] = stage_time
            result.phase_success['narrative'] = False
            return result
    
    async def _execute_phase4_llm_enhancement(self, user_input: str, result: PipelineResult) -> PipelineResult:
        """Execute Phase 4: LLM consciousness enhancement"""
        stage_start = time.time()
        
        try:
            logger.info("🤖 Phase 4: Enhancing with LLM consciousness integration...")
            
            # Process through Phase 4 with complete consciousness context
            phase4_result = await self.phase4_manager.process_consciousness_query(
                user_input=user_input,
                sc_t_state={
                    'conscious_state': result.evolved_state.__dict__ if result.evolved_state else result.conscious_state.__dict__,
                    'narrative': result.narrative_text,
                    'validation': result.validation_result,
                    'confidence': result.confidence_score,
                    'processing_stages': result.phase_success
                }
            )
            
            result.llm_enhanced_response = phase4_result.response
            result.response = phase4_result.response
            
            # Update timings
            stage_time = (time.time() - stage_start) * 1000
            result.phase_timings['llm_enhancement'] = stage_time
            result.phase_success['llm_enhancement'] = True
            result.stage_completed = ProcessingStage.LLM_ENHANCEMENT
            
            logger.info(f"✅ Phase 4 completed in {stage_time:.1f}ms - consciousness-enhanced response generated")
            return result
            
        except Exception as e:
            logger.error(f"❌ Phase 4 failed: {e}")
            # Fallback to narrative response
            result.response = result.narrative_text or f"I'm processing this with {result.confidence_score:.1f} confidence, but cannot generate enhanced response: {str(e)}"
            stage_time = (time.time() - stage_start) * 1000
            result.phase_timings['llm_enhancement'] = stage_time
            result.phase_success['llm_enhancement'] = False
            return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'total_cycles': self.cycle_count,
            'total_processing_time_ms': self.total_processing_time,
            'average_processing_time_ms': self.total_processing_time / max(1, self.cycle_count),
            'phase_statistics': self.phase_statistics,
            'phase4_enabled': self.enable_phase4,
            'validation_available': self.validation_available,
            'narrative_available': self.narrative_available
        }


# Factory function for easy initialization
def create_consciousness_pipeline(enable_phase4: bool = True, debug: bool = False) -> ConsciousnessPipelineOrchestrator:
    """
    Create a complete consciousness pipeline orchestrator
    
    Args:
        enable_phase4: Enable Phase 4 LLM enhancement
        debug: Enable debug logging
        
    Returns:
        Configured pipeline orchestrator
    """
    return ConsciousnessPipelineOrchestrator(
        enable_phase4=enable_phase4,
        debug=debug
    )