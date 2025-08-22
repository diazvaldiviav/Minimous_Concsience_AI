"""
Phase 4 Layer 3: Complete Integration Pipeline
============================================
End-to-end consciousness-enhanced query processing from SC_t states to generated responses.
"""

from .phase4_manager import Phase4Manager, QueryComplexity, create_phase4_manager
from .integration_layer import IntegrationBridge, ProcessingResult
from .phase4_examples import Phase4ExampleRunner
from .phase4_cli import Phase4CLI

__all__ = [
    'Phase4Manager',
    'QueryComplexity', 
    'create_phase4_manager',
    'IntegrationBridge',
    'ProcessingResult',
    'Phase4ExampleRunner',
    'Phase4CLI'
]

__version__ = "4.3.0"
__layer__ = "Phase 4 Layer 3"