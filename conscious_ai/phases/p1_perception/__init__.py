"""
Phase 1: Perception & User Input
================================
Handles input normalization, detection of language/topic/intent/urgency.

Components:
- language_detector: Detects language and linguistic patterns
- input_processor: Processes sensory input and normalizes data
- intent_detector: Analyzes user intent (future component)
- urgency_analyzer: Determines urgency levels (future component)
"""

from .language_detector import *
from .input_processor import *

__all__ = [
    # Language detection
    'LanguageDetector',
    'detect_language',
    
    # Input processing  
    'SensoryModule',
    'process_input',
]