"""
Phase 4: LLM Communication Layer
===============================
GPT-OSS-20B integration with hybrid CPU+GPU architecture for premium hardware.
Supports 51GB RAM + 15GB VRAM T4 configuration with intelligent load balancing.
"""

from .core.hardware_profiler import PremiumHardwareProfiler
from .core.backend_manager import PremiumBackendManager
from .models.gpt_oss_loader import HybridGPTOSSLoader
from .formatters.harmony_processor import HarmonyFormatProcessor
from .optimization.performance_monitor import PremiumPerformanceMonitor

__all__ = [
    'PremiumHardwareProfiler',
    'PremiumBackendManager', 
    'HybridGPTOSSLoader',
    'HarmonyFormatProcessor',
    'PremiumPerformanceMonitor'
]

__version__ = "4.2.0"
__phase__ = "Phase 4 Layer 2"