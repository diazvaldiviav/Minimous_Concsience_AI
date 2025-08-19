"""
Phase 2: Cognitive Context (SC_t)
=================================
Construction of functional conscious state (goal, emotion, confidence, thought, memory).

Components:
- autonomous_thinking: Core autonomous thinking logic
- conscious_state: Conscious state data structures and management
- goal_generator: Generates and manages conscious goals
- memory_integration: Memory systems and integration
"""

from .autonomous_thinking import *
from .conscious_state import *
from .goal_generator import *
from .memory_integration import *

__all__ = [
    # Autonomous thinking
    'AutonomousThinking',
    'generate_conscious_state',
    
    # Conscious state
    'ConsciousState', 
    'StateComponent',
    
    # Goal generation
    'GoalThoughtGenerator',
    'generate_goal',
    
    # Memory integration
    'ActiveMemory',
    'MemoryIntegration',
]