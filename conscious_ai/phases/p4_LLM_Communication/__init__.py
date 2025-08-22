"""
Phase 4: LLM Communication Layer
===============================
GPT-OSS-20B integration with hybrid CPU+GPU architecture for premium hardware.
Supports 51GB RAM + 15GB VRAM T4 configuration with intelligent load balancing.
"""

# Import with fallbacks for missing dependencies
try:
    from .core.hardware_profiler import PremiumHardwareProfiler
except ImportError:
    PremiumHardwareProfiler = None

try:
    from .core.backend_manager import PremiumBackendManager
except ImportError:
    PremiumBackendManager = None

try:
    from .models.gpt_oss_loader import HybridGPTOSSLoader
except ImportError:
    HybridGPTOSSLoader = None

try:
    from .formatters.harmony_processor import HarmonyFormatProcessor
except ImportError:
    HarmonyFormatProcessor = None

try:
    from .optimization.performance_monitor import PremiumPerformanceMonitor
except ImportError:
    PremiumPerformanceMonitor = None

__all__ = [
    'PremiumHardwareProfiler',
    'PremiumBackendManager', 
    'HybridGPTOSSLoader',
    'HarmonyFormatProcessor',
    'PremiumPerformanceMonitor'
]

__version__ = "4.2.0"
__phase__ = "Phase 4 Layer 2"