"""
Phase 3: Coherent SC_t Generation
=================================
Produces new self-conscious states using models and heuristics.

Components:
- state_evolution_engine: Core state evolution and generation logic
- conscious_response_generator: Generates conscious responses
- heuristic_generator: Heuristic-based state generation (future)
- model_based_generator: Model-based state generation (future)
"""

from .state_evolution_engine import *
from .conscious_response_generator import *

__all__ = [
    # State evolution
    'StateEvolutionEngine',
    'evolve_state',
    
    # Response generation
    'ConsciousResponseGenerator',
    'generate_response',
]