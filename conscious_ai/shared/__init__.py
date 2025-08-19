"""
Shared Components
================
Common utilities and components used across multiple phases.

Components:
- metrics: Consciousness metrics and measurement tools
- integrator: Central integration functionality  
- analysis_tools: Analysis and diagnostic tools
"""

from .metrics import *
from .integrator import *
from .analysis_tools import *

__all__ = [
    # Metrics
    'ConsciousnessMetrics',
    'override_phi',
    
    # Integration
    'CentralIntegrator',
    'integrate_components',
    
    # Analysis tools
    'ConsciousAnalysisTools',
    'analyze_consciousness',
]