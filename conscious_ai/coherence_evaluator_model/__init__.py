"""
Coherence Evaluator Model Package
=================================
Contains Phase 3.4 (Critical State Evaluation) and Phase 3.5 (Internal Conscious Translation) components.

This package provides:
- Hybrid coherence evaluation with semantic similarity and rule-based analysis
- Critical state evaluation with correction loops
- Internal conscious translation to introspective narratives
"""

# Import main components for easier access
try:
    from .model_training.critical_state_evaluator import (
        CriticalStateEvaluator, 
        create_critical_evaluator,
        CoherenceVerdict,
        EvaluationResult
    )
except ImportError:
    pass  # Graceful degradation if components are missing

try:
    from .heuristic_training.narrative_generator import (
        NarrativeGenerator,
        create_narrative_generator,
        NarrativeConfig,
        NarrativeModel
    )
except ImportError:
    pass  # Graceful degradation if components are missing

__version__ = "3.4.1"
__author__ = "Minimal Consciousness AI Team"