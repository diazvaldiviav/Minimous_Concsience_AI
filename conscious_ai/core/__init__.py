"""
Core consciousness components - centralized imports for key modules
"""

# Phase 1: Perception
from ..phases.p1_perception.input_processor import SensoryModule

# Phase 2: Cognitive Context  
from ..phases.p2_cognitive_context.conscious_state import ConsciousState, ConsciousStateHistory
from ..phases.p2_cognitive_context.goal_generator import GoalGenerator, AutomaticThoughtGenerator
from ..phases.p2_cognitive_context.memory_integration import ActiveMemory

# Phase 3: Coherent Generation
from ..phases.p3_coherent_generation.state_evolution_engine import StateEvolutionEngine
from ..phases.p3_coherent_generation.conscious_response_generator import ConsciousResponseGenerator

# Phase 3.4: Critical Evaluation
from ..coherence_evaluator_model.model_training.critical_state_evaluator import CriticalStateEvaluator
from ..coherence_evaluator_model.heuristic_training.coherence_evaluator import CoherenceEvaluator

# Phase 3.5: Narrative Generation
from ..coherence_evaluator_model.heuristic_training.narrative_generator import NarrativeGenerator

# Autonomous Thinking - Import only base components to avoid circular dependencies
from ..autonomous_thinking.autonomous_thinking import AutonomousThoughtGenerator
# Note: AutonomousConsciousAI and EnhancedAutonomousConsciousAI moved to avoid circular imports
# Import these directly when needed: 
# from conscious_ai.autonomous_thinking.autonomous_integration import AutonomousConsciousAI
# from conscious_ai.autonomous_thinking.enhanced_autonomous_integration import EnhancedAutonomousConsciousAI

# Shared utilities
from ..shared.metrics import ConsciousnessMetrics
from ..shared.integrator import CentralIntegrator
from ..modules.self_model import SelfModel
from ..modules.reentrance import ReentranceModule
from ..modules.sensitivity_analysis import sensibilidad_metricas

__all__ = [
    # Phase 1
    'SensoryModule',
    
    # Phase 2
    'ConsciousState', 'ConsciousStateHistory', 'GoalGenerator', 
    'AutomaticThoughtGenerator', 'ActiveMemory',
    
    # Phase 3
    'StateEvolutionEngine', 'ConsciousResponseGenerator',
    
    # Phase 3.4
    'CriticalStateEvaluator', 'CoherenceEvaluator',
    
    # Phase 3.5
    'NarrativeGenerator',
    
    # Autonomous
    'AutonomousThoughtGenerator',
    
    # Shared
    'ConsciousnessMetrics', 'CentralIntegrator', 'SelfModel', 'ReentranceModule',
    'sensibilidad_metricas'
]