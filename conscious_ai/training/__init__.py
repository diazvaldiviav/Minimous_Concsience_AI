"""
Training Pipelines and Model Setup
==================================
Centralized training functionality for all phases.

Components:
- phase1_training: Phase 1 language detection training
- phase2_training: Phase 2 autonomous thinking training  
- phase3_training: Phase 3 coherence evaluation training
- dataset_creators/: Dataset creation utilities
- model_trainers/: Specialized model training components
"""

from .phase1_training import *
from .phase2_training import *

__all__ = [
    # Phase training
    'train_phase1',
    'train_phase2', 
    'train_phase3',
]