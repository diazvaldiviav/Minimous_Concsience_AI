"""
Phase 4 Core Components
======================
Core hardware profiling and backend management for GPT-OSS-20B integration.
"""

from .hardware_profiler import PremiumHardwareProfiler, HardwareConfiguration
from .backend_manager import PremiumBackendManager, BackendType, QueryContext

__all__ = [
    'PremiumHardwareProfiler',
    'HardwareConfiguration', 
    'PremiumBackendManager',
    'BackendType',
    'QueryContext'
]