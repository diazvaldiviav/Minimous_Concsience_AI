"""
Phase 3.4: Critical Evaluation
==============================
Validates transition SC_t→SC_t+1, corrects inconsistencies.

Components:
- critical_state_evaluator: Main critical evaluation orchestrator
- coherence_evaluator: Heuristic-based coherence evaluation  
- model_based_coherence_evaluator: ML/semantic coherence evaluation
- transition_validator: Validates state transitions (future)

This phase is crucial for maintaining consciousness coherence and was recently
optimized to achieve 95%+ accuracy in coherent/incoherent detection.
"""

from .critical_state_evaluator import *
from .coherence_evaluator import *
from .model_based_coherence_evaluator import *

__all__ = [
    # Critical evaluation
    'CriticalStateEvaluator',
    'EvaluationResult',
    'EvaluationStrategy',
    
    # Coherence evaluation
    'CoherenceEvaluator', 
    'CoherenceVerdict',
    'TransitionAnalysis',
    
    # Model-based evaluation
    'ModelBasedCoherenceEvaluator',
]