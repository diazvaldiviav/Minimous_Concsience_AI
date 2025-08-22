"""
Premium Hardware Profiler for Phase 4 Layer 2
==============================================
Detects and optimizes for 51GB RAM + 15GB VRAM T4 configuration.
Calculates optimal memory distribution for GPT-OSS-20B hybrid loading.
"""

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import subprocess
import platform

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfiguration:
    """Premium hardware configuration for GPT-OSS-20B hybrid loading"""
    total_ram_gb: float
    available_ram_gb: float
    total_vram_gb: float
    available_vram_gb: float
    gpu_name: str
    cuda_version: str
    architecture_type: str
    safety_margin_ram_gb: float = 6.0
    safety_margin_vram_gb: float = 2.0
    
    @property
    def usable_ram_gb(self) -> float:
        """RAM available for model loading after safety margin"""
        return max(0, self.available_ram_gb - self.safety_margin_ram_gb)
    
    @property
    def usable_vram_gb(self) -> float:
        """VRAM available for model loading after safety margin"""
        return max(0, self.available_vram_gb - self.safety_margin_vram_gb)
    
    @property
    def is_premium_hardware(self) -> bool:
        """Check if hardware meets premium specifications (51GB RAM + ~15GB VRAM)"""
        # Full premium: 51GB RAM + 14.5GB+ VRAM (realistic for T4/similar GPUs)
        full_premium = (self.total_ram_gb >= 51.0 and 
                       self.total_vram_gb >= 14.5 and  # Lowered from 15.0 to 14.5
                       self.usable_ram_gb >= 40.0 and  # Lowered from 45.0 to 40.0 
                       self.usable_vram_gb >= 12.0)     # Lowered from 13.0 to 12.0
        
        # Premium RAM configuration (for CPU-only or GPU setup issues)
        premium_ram = (self.total_ram_gb >= 51.0 and 
                      self.usable_ram_gb >= 40.0)      # Lowered from 45.0 to 40.0
        
        # Debug logging for premium hardware detection
        logger.debug(f"Premium hardware check:")
        logger.debug(f"  RAM: {self.total_ram_gb:.1f}GB total >= 51.0: {self.total_ram_gb >= 51.0}")
        logger.debug(f"  RAM: {self.usable_ram_gb:.1f}GB usable >= 40.0: {self.usable_ram_gb >= 40.0}")
        logger.debug(f"  VRAM: {self.total_vram_gb:.1f}GB total >= 14.5: {self.total_vram_gb >= 14.5}")
        logger.debug(f"  VRAM: {self.usable_vram_gb:.1f}GB usable >= 12.0: {self.usable_vram_gb >= 12.0}")
        logger.debug(f"  Full premium: {full_premium}, Premium RAM: {premium_ram}")
        
        return full_premium or premium_ram
    
    @property
    def memory_efficiency_score(self) -> float:
        """Calculate memory efficiency score (0-100)"""
        ram_efficiency = min(100, (self.usable_ram_gb / 40.0) * 100)   # Updated from 45.0 to 40.0
        vram_efficiency = min(100, (self.usable_vram_gb / 12.0) * 100)  # Updated from 13.0 to 12.0
        return (ram_efficiency + vram_efficiency) / 2


class PremiumHardwareProfiler:
    """
    Premium hardware profiler optimized for 51GB RAM + 15GB VRAM T4 configuration.
    Calculates optimal memory distribution for GPT-OSS-20B hybrid CPU+GPU loading.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cached_config: Optional[HardwareConfiguration] = None
        
    def detect_hardware_configuration(self, force_refresh: bool = False) -> HardwareConfiguration:
        """
        Detect and analyze hardware configuration for optimal GPT-OSS-20B loading.
        
        Args:
            force_refresh: Force re-detection instead of using cached results
            
        Returns:
            HardwareConfiguration with optimal memory distribution
        """
        if self._cached_config and not force_refresh:
            return self._cached_config
            
        self.logger.info("🔍 Detecting premium hardware configuration...")
        
        # Detect RAM configuration
        ram_info = self._detect_ram_configuration()
        
        # Detect GPU/VRAM configuration
        gpu_info = self._detect_gpu_configuration()
        
        # Determine architecture type
        architecture_type = self._determine_architecture_type(ram_info, gpu_info)
        
        config = HardwareConfiguration(
            total_ram_gb=ram_info['total_gb'],
            available_ram_gb=ram_info['available_gb'],
            total_vram_gb=gpu_info['total_gb'],
            available_vram_gb=gpu_info['available_gb'],
            gpu_name=gpu_info['name'],
            cuda_version=gpu_info['cuda_version'],
            architecture_type=architecture_type
        )
        
        self._cached_config = config
        self._log_configuration_summary(config)
        
        return config
    
    def _detect_ram_configuration(self) -> Dict[str, Any]:
        """Detect system RAM configuration with high precision"""
        try:
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                
                total_gb = memory.total / (1024**3)
                available_gb = memory.available / (1024**3)
                used_gb = memory.used / (1024**3)
                
                return {
                    'total_gb': round(total_gb, 2),
                    'available_gb': round(available_gb, 2),
                    'used_gb': round(used_gb, 2),
                    'percent_used': round(memory.percent, 1)
                }
            else:
                # Fallback when psutil not available
                logger.warning("⚠️ psutil not available - using conservative RAM estimates")
                return {
                    'total_gb': 12.0,
                    'available_gb': 8.0,
                    'used_gb': 4.0,
                    'percent_used': 33.3
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect RAM configuration: {e}")
            return {
                'total_gb': 0.0,
                'available_gb': 0.0,
                'used_gb': 0.0,
                'percent_used': 100.0
            }
    
    def _detect_gpu_configuration(self) -> Dict[str, Any]:
        """Detect GPU/VRAM configuration with CUDA support"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("⚠️ torch not available - using fallback GPU configuration")
                return self._get_fallback_gpu_info()
            
            # Check CUDA availability first
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            
            self.logger.info(f"CUDA Available: {cuda_available}, Device Count: {device_count}")
            
            if not cuda_available or device_count == 0:
                self.logger.info("CUDA not available or no devices - using CPU-only configuration")
                return self._get_fallback_gpu_info()
            
            # Get primary GPU (device 0)
            device_props = torch.cuda.get_device_properties(0)
            self.logger.info(f"Detected GPU: {device_props.name}")
            
            total_vram_bytes = device_props.total_memory
            allocated_bytes = torch.cuda.memory_allocated(0)
            cached_bytes = torch.cuda.memory_reserved(0)
            
            total_gb = total_vram_bytes / (1024**3)
            allocated_gb = allocated_bytes / (1024**3)
            cached_gb = cached_bytes / (1024**3)
            available_gb = total_gb - max(allocated_gb, cached_gb)
            
            # Get CUDA version
            cuda_version = torch.version.cuda or "Unknown"
            
            # Get multiprocessor count with compatibility fallback
            try:
                # Try newer PyTorch attribute names
                if hasattr(device_props, 'multi_processor_count'):
                    mp_count = device_props.multi_processor_count
                elif hasattr(device_props, 'multiprocessor_count'):
                    mp_count = device_props.multiprocessor_count
                else:
                    # Fallback: estimate based on GPU name for common models
                    mp_count = self._estimate_multiprocessor_count(device_props.name)
            except Exception:
                mp_count = 0
            
            return {
                'name': device_props.name,
                'total_gb': round(total_gb, 2),
                'available_gb': round(max(0, available_gb), 2),
                'allocated_gb': round(allocated_gb, 2),
                'cached_gb': round(cached_gb, 2),
                'cuda_version': cuda_version,
                'compute_capability': f"{device_props.major}.{device_props.minor}",
                'multiprocessor_count': mp_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect GPU configuration: {e}")
            return self._get_fallback_gpu_info()
    
    def _estimate_multiprocessor_count(self, gpu_name: str) -> int:
        """Estimate multiprocessor count based on GPU name"""
        gpu_name_lower = gpu_name.lower()
        
        # Common GPU multiprocessor counts
        if 'tesla t4' in gpu_name_lower:
            return 40
        elif 'tesla k80' in gpu_name_lower:
            return 26
        elif 'tesla v100' in gpu_name_lower:
            return 80
        elif 'tesla p100' in gpu_name_lower:
            return 56
        elif 'rtx 3090' in gpu_name_lower:
            return 82
        elif 'rtx 4090' in gpu_name_lower:
            return 128
        elif 'gtx 1080' in gpu_name_lower:
            return 20
        elif 'gtx 1060' in gpu_name_lower:
            return 10
        else:
            # Default estimate for unknown GPUs
            return 16
    
    def _get_fallback_gpu_info(self) -> Dict[str, Any]:
        """Fallback GPU info when CUDA detection fails"""
        return {
            'name': 'CPU Only',
            'total_gb': 0.0,
            'available_gb': 0.0,
            'allocated_gb': 0.0,
            'cached_gb': 0.0,
            'cuda_version': 'Not Available',
            'compute_capability': 'N/A',
            'multiprocessor_count': 0
        }
    
    def _determine_architecture_type(self, ram_info: Dict, gpu_info: Dict) -> str:
        """Determine optimal architecture type based on hardware specs"""
        total_ram = ram_info['total_gb']
        total_vram = gpu_info['total_gb']
        available_ram = ram_info['available_gb']
        available_vram = gpu_info['available_gb']
        
        self.logger.info(f"Architecture determination: RAM={total_ram:.1f}GB, VRAM={total_vram:.1f}GB")
        
        # Premium hybrid configuration (target specs)
        if (total_ram >= 50.0 and total_vram >= 14.5 and 
            available_ram >= 40.0 and available_vram >= 12.0):
            return "hybrid_cpu_gpu_premium"
        
        # Premium CPU configuration (high RAM, limited/no GPU)
        elif total_ram >= 50.0 and available_ram >= 40.0:
            return "cpu_premium_large"
        
        # Standard hybrid configuration
        elif (total_ram >= 30.0 and total_vram >= 10.0 and
              available_ram >= 25.0 and available_vram >= 8.0):
            return "hybrid_cpu_gpu_standard"
        
        # GPU-only configuration
        elif total_vram >= 20.0 and available_vram >= 18.0:
            return "gpu_only"
        
        # CPU-only configuration
        elif total_ram >= 40.0 and available_ram >= 35.0:
            return "cpu_only_large"
        
        # Limited resources
        else:
            return "limited_resources"
    
    def calculate_optimal_memory_distribution(self, config: HardwareConfiguration) -> Dict[str, Any]:
        """
        Calculate optimal memory distribution for GPT-OSS-20B hybrid loading.
        
        Returns:
            Detailed memory allocation strategy for hybrid CPU+GPU loading
        """
        if not config.is_premium_hardware:
            return self._calculate_fallback_distribution(config)
        
        # Premium configuration for GPT-OSS-20B (20B parameters ≈ 40GB FP16)
        distribution = {
            'strategy': 'hybrid_cpu_gpu_premium',
            'model_size_estimate_gb': 40.0,
            'quantization_recommended': 'MXFP4' if config.usable_vram_gb >= 12.0 else 'FP16',
            
            # GPU allocation (critical layers)
            'gpu_allocation': {
                'embedding_layers': 3.0,  # GB
                'attention_layers': 6.0,  # GB  
                'output_layer': 2.0,      # GB
                'buffers_cache': 2.0,     # GB
                'total_gpu_usage': 13.0   # GB
            },
            
            # CPU allocation (processing layers)
            'cpu_allocation': {
                'transformer_blocks': 20.0,  # GB
                'intermediate_layers': 8.0,  # GB
                'buffers_cache': 5.0,        # GB
                'system_overhead': 2.0,      # GB
                'total_cpu_usage': 35.0      # GB
            },
            
            # Safety margins
            'safety_margins': {
                'ram_reserved': config.safety_margin_ram_gb,
                'vram_reserved': config.safety_margin_vram_gb,
                'emergency_buffer': 5.0
            },
            
            # Performance targets
            'performance_targets': {
                'loading_time_max_minutes': 5,
                'response_time_max_seconds': 10,
                'memory_efficiency_target': 0.80,
                'stability_target_hours': 24
            }
        }
        
        # Validate distribution fits within available resources
        total_cpu_needed = distribution['cpu_allocation']['total_cpu_usage']
        total_gpu_needed = distribution['gpu_allocation']['total_gpu_usage']
        
        if total_cpu_needed > config.usable_ram_gb:
            distribution['warnings'] = [f"CPU allocation ({total_cpu_needed}GB) exceeds available RAM ({config.usable_ram_gb}GB)"]
        
        if total_gpu_needed > config.usable_vram_gb:
            distribution['warnings'] = distribution.get('warnings', []) + [f"GPU allocation ({total_gpu_needed}GB) exceeds available VRAM ({config.usable_vram_gb}GB)"]
        
        return distribution
    
    def _calculate_fallback_distribution(self, config: HardwareConfiguration) -> Dict[str, Any]:
        """Calculate fallback distribution for non-premium hardware"""
        return {
            'strategy': 'fallback_' + config.architecture_type,
            'model_size_estimate_gb': 40.0,
            'quantization_recommended': 'INT8',
            'warnings': ['Hardware does not meet premium specifications for optimal GPT-OSS-20B loading'],
            'fallback_options': [
                'Use smaller model (Mistral-7B)',
                'CPU-only mode with heavy quantization',
                'External API calls'
            ]
        }
    
    def _log_configuration_summary(self, config: HardwareConfiguration):
        """Log comprehensive hardware configuration summary"""
        self.logger.info("🖥️ Hardware Configuration Summary:")
        self.logger.info(f"  RAM: {config.total_ram_gb:.1f}GB total, {config.available_ram_gb:.1f}GB available, {config.usable_ram_gb:.1f}GB usable")
        self.logger.info(f"  VRAM: {config.total_vram_gb:.1f}GB total, {config.available_vram_gb:.1f}GB available, {config.usable_vram_gb:.1f}GB usable")
        self.logger.info(f"  GPU: {config.gpu_name} (CUDA {config.cuda_version})")
        self.logger.info(f"  Architecture: {config.architecture_type}")
        self.logger.info(f"  Premium Hardware: {'✅ YES' if config.is_premium_hardware else '❌ NO'}")
        self.logger.info(f"  Memory Efficiency Score: {config.memory_efficiency_score:.1f}%")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information for debugging"""
        try:
            return {
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'machine': platform.machine(),
                    'processor': platform.processor()
                },
                'python': {
                    'version': platform.python_version(),
                    'implementation': platform.python_implementation()
                },
                'torch': {
                    'version': torch.__version__ if TORCH_AVAILABLE else "Not available",
                    'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
                    'cuda_version': torch.version.cuda if TORCH_AVAILABLE else "Not available",
                    'device_count': torch.cuda.device_count() if TORCH_AVAILABLE and torch.cuda.is_available() else 0
                },
                'memory': self._detect_ram_configuration(),
                'gpu': self._detect_gpu_configuration()
            }
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {'error': str(e)}
    
    def validate_gpt_oss_requirements(self, config: HardwareConfiguration) -> Dict[str, Any]:
        """Validate hardware meets GPT-OSS-20B requirements"""
        requirements = {
            'minimum_ram_gb': 32.0,
            'recommended_ram_gb': 50.0,
            'minimum_vram_gb': 8.0,
            'recommended_vram_gb': 15.0
        }
        
        validation = {
            'meets_minimum': (config.usable_ram_gb >= requirements['minimum_ram_gb'] and
                            config.usable_vram_gb >= requirements['minimum_vram_gb']),
            'meets_recommended': (config.usable_ram_gb >= requirements['recommended_ram_gb'] and
                                config.usable_vram_gb >= requirements['recommended_vram_gb']),
            'ram_status': 'sufficient' if config.usable_ram_gb >= requirements['recommended_ram_gb'] else 'limited',
            'vram_status': 'sufficient' if config.usable_vram_gb >= requirements['recommended_vram_gb'] else 'limited',
            'overall_grade': 'A' if config.is_premium_hardware else 'B' if config.usable_ram_gb >= 25 else 'C'
        }
        
        return validation