"""
Phase 5.5: Narrative Recording of Consciousness
==============================================
Captures and synthesizes the entire mental journey into human-readable narratives 
that provide transparency about the AI's reasoning process throughout all phases.

This module transforms technical consciousness states (SC_t) and pipeline events 
into coherent, accessible narratives that bridge the gap between computational 
consciousness and user understanding.
"""

from .process_logger import ProcessLogger, ConsciousnessEvent
from .decision_tracker import DecisionTracker, CognitiveChoice
from .metacognitive_observer import MetacognitiveObserver, IntrospectiveEvent
from .narrative_synthesizer import NarrativeSynthesizer, NarrativeResult

__all__ = [
    'ProcessLogger',
    'ConsciousnessEvent',
    'DecisionTracker', 
    'CognitiveChoice',
    'MetacognitiveObserver',
    'IntrospectiveEvent',
    'NarrativeSynthesizer',
    'NarrativeResult'
]

__version__ = "1.0.0"
__phase__ = "5.5"
__description__ = "Narrative Recording of Consciousness - Transparency through introspective narrative generation"