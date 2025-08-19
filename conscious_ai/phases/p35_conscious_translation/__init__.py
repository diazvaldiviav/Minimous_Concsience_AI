"""
Phase 3.5: Internal Conscious Translation
=========================================
Transforms SC_t into natural introspective narrative.

Components:
- narrative_generator: Generates natural language narratives from conscious states
- introspection_builder: Builds introspective narratives (future)
- self_reflection_engine: Processes self-reflection (future)
"""

from .narrative_generator import *

__all__ = [
    # Narrative generation
    'NarrativeGenerator',
    'NarrativeConfig',
    'NarrativeModel',
    'generate_narrative',
]