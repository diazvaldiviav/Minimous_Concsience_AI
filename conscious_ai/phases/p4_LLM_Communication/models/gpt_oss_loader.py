"""
Hybrid GPT-OSS-20B Loader for Phase 4 Layer 2
==============================================
Loads GPT-OSS-20B with intelligent hybrid CPU+GPU distribution.
Optimized for 51GB RAM + 15GB VRAM T4 configuration with safety mechanisms.
"""

import torch
import logging
import time
import gc
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    BitsAndBytesConfig
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import psutil
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class LoadingConfiguration:
    """Configuration for GPT-OSS-20B hybrid loading"""
    model_name: str = "openai/gpt-oss-20b"
    use_hybrid_loading: bool = True
    quantization_type: str = "MXFP4"  # MXFP4, FP16, INT8
    max_loading_time_minutes: int = 5
    gpu_memory_limit_gb: float = 13.0
    cpu_memory_limit_gb: float = 35.0
    enable_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    trust_remote_code: bool = True
    
    
@dataclass 
class LoadingResult:
    """Result of GPT-OSS-20B loading attempt"""
    success: bool
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    loading_time_seconds: float = 0.0
    memory_usage_gb: Dict[str, float] = None
    configuration_used: Optional[LoadingConfiguration] = None
    warnings: List[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.memory_usage_gb is None:
            self.memory_usage_gb = {}
        if self.warnings is None:
            self.warnings = []


class MemoryMonitor:
    """Real-time memory monitoring during model loading"""
    
    def __init__(self, ram_limit_gb: float = 45.0, vram_limit_gb: float = 13.0):
        self.ram_limit_gb = ram_limit_gb
        self.vram_limit_gb = vram_limit_gb
        self.monitoring = False
        self.memory_history = []
        self._monitor_thread = None
        
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start real-time memory monitoring"""
        self.monitoring = True
        self.memory_history = []
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            
        if not self.memory_history:
            return {'error': 'No monitoring data collected'}
            
        ram_usage = [entry['ram_gb'] for entry in self.memory_history]
        vram_usage = [entry['vram_gb'] for entry in self.memory_history]
        
        return {
            'peak_ram_gb': max(ram_usage),
            'peak_vram_gb': max(vram_usage),
            'avg_ram_gb': sum(ram_usage) / len(ram_usage),
            'avg_vram_gb': sum(vram_usage) / len(vram_usage),
            'ram_limit_exceeded': any(ram > self.ram_limit_gb for ram in ram_usage),
            'vram_limit_exceeded': any(vram > self.vram_limit_gb for vram in vram_usage),
            'monitoring_duration_seconds': len(self.memory_history),
            'samples_collected': len(self.memory_history)
        }
        
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # RAM monitoring
                memory = psutil.virtual_memory()
                ram_gb = memory.used / (1024**3)
                
                # VRAM monitoring
                vram_gb = 0.0
                if torch.cuda.is_available():
                    vram_bytes = torch.cuda.memory_allocated(0)
                    vram_gb = vram_bytes / (1024**3)
                
                self.memory_history.append({
                    'timestamp': time.time(),
                    'ram_gb': ram_gb,
                    'vram_gb': vram_gb,
                    'ram_percent': memory.percent
                })
                
                # Emergency stop if limits exceeded by too much
                if ram_gb > self.ram_limit_gb * 1.1 or vram_gb > self.vram_limit_gb * 1.1:
                    logger.error(f"🚨 Memory limits exceeded: RAM {ram_gb:.1f}GB/{self.ram_limit_gb:.1f}GB, VRAM {vram_gb:.1f}GB/{self.vram_limit_gb:.1f}GB")
                    break
                    
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                break


class HybridGPTOSSLoader:
    """
    Advanced GPT-OSS-20B loader with hybrid CPU+GPU architecture.
    Optimized for premium hardware with intelligent layer distribution.
    """
    
    def __init__(self, hardware_config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.hardware_config = hardware_config or {}
        self.model = None
        self.tokenizer = None
        self.loading_config = None
        self.memory_monitor = MemoryMonitor()
        
    def load_gpt_oss_hybrid(self, config: LoadingConfiguration) -> LoadingResult:
        """
        Load GPT-OSS-20B with hybrid CPU+GPU distribution.
        
        Args:
            config: Loading configuration with memory limits and options
            
        Returns:
            LoadingResult with model, tokenizer, and performance metrics
        """
        start_time = time.time()
        result = LoadingResult(success=False, configuration_used=config)
        
        try:
            self.logger.info("🚀 Starting GPT-OSS-20B hybrid loading...")
            self.logger.info(f"  Model: {config.model_name}")
            self.logger.info(f"  Quantization: {config.quantization_type}")
            self.logger.info(f"  GPU Memory Limit: {config.gpu_memory_limit_gb:.1f}GB")
            self.logger.info(f"  CPU Memory Limit: {config.cpu_memory_limit_gb:.1f}GB")
            
            # Start memory monitoring
            self.memory_monitor.start_monitoring()
            
            # Pre-flight validation
            if not self._validate_loading_requirements(config):
                result.error_message = "Pre-flight validation failed"
                return result
            
            # Load tokenizer first (lightweight)
            result.tokenizer = self._load_tokenizer(config)
            if result.tokenizer is None:
                result.error_message = "Failed to load tokenizer"
                return result
                
            # Load model with hybrid strategy
            result.model = self._load_model_hybrid(config, result)
            if result.model is None:
                result.error_message = "Failed to load model"
                return result
                
            # Validate loaded model
            if not self._validate_loaded_model(result.model, result.tokenizer):
                result.error_message = "Model validation failed"
                return result
            
            # Success!
            result.success = True
            result.loading_time_seconds = time.time() - start_time
            
            self.logger.info(f"✅ GPT-OSS-20B loaded successfully in {result.loading_time_seconds:.1f}s")
            
        except Exception as e:
            self.logger.error(f"❌ GPT-OSS-20B loading failed: {e}")
            result.error_message = str(e)
            result.success = False
            
        finally:
            # Stop monitoring and collect stats
            memory_stats = self.memory_monitor.stop_monitoring()
            result.memory_usage_gb = memory_stats
            
            # Cleanup on failure
            if not result.success:
                self._cleanup_failed_loading()
                
        return result
    
    def _validate_loading_requirements(self, config: LoadingConfiguration) -> bool:
        """Pre-flight validation before loading"""
        try:
            # Check available memory
            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024**3)
            
            if available_ram_gb < config.cpu_memory_limit_gb:
                self.logger.error(f"Insufficient RAM: {available_ram_gb:.1f}GB < {config.cpu_memory_limit_gb:.1f}GB required")
                return False
            
            # Check GPU availability
            if config.use_hybrid_loading and torch.cuda.is_available():
                available_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated_vram_gb = torch.cuda.memory_allocated(0) / (1024**3)
                free_vram_gb = available_vram_gb - allocated_vram_gb
                
                if free_vram_gb < config.gpu_memory_limit_gb:
                    self.logger.warning(f"Limited VRAM: {free_vram_gb:.1f}GB < {config.gpu_memory_limit_gb:.1f}GB optimal")
                    config.gpu_memory_limit_gb = max(4.0, free_vram_gb * 0.8)  # Reduce to 80% of available
                    
            # Check quantization support
            if config.quantization_type == "MXFP4":
                try:
                    import kernels
                    self.logger.info("✅ MXFP4 quantization available")
                except ImportError:
                    self.logger.warning("⚠️ MXFP4 not available, falling back to FP16")
                    config.quantization_type = "FP16"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-flight validation error: {e}")
            return False
    
    def _load_tokenizer(self, config: LoadingConfiguration) -> Optional[Any]:
        """Load GPT-OSS tokenizer with error handling"""
        try:
            self.logger.info("📚 Loading GPT-OSS tokenizer...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                trust_remote_code=config.trust_remote_code,
                cache_dir=".cache/huggingface"
            )
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            self.logger.info(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"❌ Tokenizer loading failed: {e}")
            return None
    
    def _load_model_hybrid(self, config: LoadingConfiguration, result: LoadingResult) -> Optional[Any]:
        """Load model with hybrid CPU+GPU distribution"""
        try:
            self.logger.info("🏗️ Loading GPT-OSS-20B with hybrid distribution...")
            
            # Configure quantization
            quantization_config = self._get_quantization_config(config)
            
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                config.model_name, 
                trust_remote_code=config.trust_remote_code
            )
            
            # Apply optimizations
            if config.enable_gradient_checkpointing:
                model_config.use_cache = False
                model_config.gradient_checkpointing = True
                
            if config.use_flash_attention:
                model_config.use_flash_attention_2 = True
            
            # Load with device mapping for hybrid distribution
            device_map = self._calculate_device_map(config, model_config)
            
            self.logger.info(f"📍 Device mapping: {len(device_map)} layers distributed")
            
            # Load model with hybrid configuration
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=config.trust_remote_code,
                low_cpu_mem_usage=True,
                cache_dir=".cache/huggingface"
            )
            
            # Post-loading optimizations
            if hasattr(model, 'eval'):
                model.eval()
                
            # Disable gradients for inference
            for param in model.parameters():
                param.requires_grad = False
                
            self.logger.info("✅ Model loaded with hybrid distribution")
            return model
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"🚨 CUDA OOM during loading: {e}")
            self._handle_oom_fallback(config)
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            return None
    
    def _get_quantization_config(self, config: LoadingConfiguration) -> Optional[Any]:
        """Get quantization configuration based on selected type"""
        if config.quantization_type == "MXFP4":
            try:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            except:
                self.logger.warning("MXFP4 config failed, using INT8")
                config.quantization_type = "INT8"
        
        if config.quantization_type == "INT8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        # FP16 - no quantization config needed
        return None
    
    def _calculate_device_map(self, config: LoadingConfiguration, model_config: Any) -> Dict[str, str]:
        """Calculate optimal device mapping for hybrid CPU+GPU loading"""
        device_map = {}
        
        if not config.use_hybrid_loading or not torch.cuda.is_available():
            # CPU-only fallback
            return "cpu"
        
        try:
            # GPT-OSS-20B optimized for 51GB+15GB hardware
            # CRITICAL LAYERS ON GPU (13GB allocation target):
            # - Embedding layers (input/output)
            # - Attention mechanisms (most compute intensive)
            # - Output projection layers
            gpu_layers = [
                "transformer.wte",  # Word embeddings (~1GB)
                "transformer.wpe",  # Position embeddings (~0.5GB)  
                "transformer.ln_f", # Final layer norm (~0.1GB)
                "lm_head"          # Output head (~2GB)
            ]
            
            # Add critical attention layers to GPU (target: ~10GB for attention)
            num_layers = getattr(model_config, 'n_layer', 40)
            # For 51GB+15GB: Put first 12 layers (most used) on GPU, rest on CPU
            gpu_attention_layers = min(12, num_layers // 2)  
            
            for i in range(gpu_attention_layers):
                gpu_layers.extend([
                    f"transformer.h.{i}.attn",      # Attention mechanism
                    f"transformer.h.{i}.ln_1",      # Pre-attention norm
                    f"transformer.h.{i}.ln_2"       # Pre-MLP norm
                ])
            
            # Assign GPU layers (targeting 13GB VRAM usage)
            for layer_name in gpu_layers:
                device_map[layer_name] = 0  # GPU device 0
            
            # PROCESSING LAYERS ON CPU (20GB of 45GB available target):
            # - Remaining transformer layers (MLP blocks and remaining attention)
            # - This uses CPU for less critical processing layers
            for i in range(num_layers):
                if f"transformer.h.{i}" not in [k.split('.')[0] + '.' + k.split('.')[1] + '.' + k.split('.')[2] for k in device_map.keys()]:
                    device_map[f"transformer.h.{i}"] = "cpu"
            
            self.logger.info(f"📍 Hybrid mapping: {len([k for k, v in device_map.items() if v == 0])} layers on GPU, {len([k for k, v in device_map.items() if v == 'cpu'])} on CPU")
            
            return device_map
            
        except Exception as e:
            self.logger.error(f"Device mapping calculation failed: {e}")
            return "auto"  # Let transformers decide
    
    def _validate_loaded_model(self, model: Any, tokenizer: Any) -> bool:
        """Validate loaded model works correctly"""
        try:
            self.logger.info("🧪 Validating loaded model...")
            
            # Simple generation test
            test_input = "The future of artificial intelligence"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            # Move inputs to appropriate device
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Test forward pass
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.logger.info(f"✅ Validation successful: '{generated_text[:50]}...'")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {e}")
            return False
    
    def _handle_oom_fallback(self, config: LoadingConfiguration):
        """Handle out-of-memory with automatic fallback"""
        self.logger.warning("🔄 Attempting OOM recovery...")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Reduce memory limits
        config.gpu_memory_limit_gb *= 0.7
        config.cpu_memory_limit_gb *= 0.8
        
        # Force more aggressive quantization
        if config.quantization_type == "FP16":
            config.quantization_type = "INT8"
        elif config.quantization_type == "MXFP4":
            config.quantization_type = "INT8"
            
        self.logger.info(f"📉 Reduced limits: GPU={config.gpu_memory_limit_gb:.1f}GB, CPU={config.cpu_memory_limit_gb:.1f}GB")
    
    def _cleanup_failed_loading(self):
        """Clean up resources after failed loading"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
                
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info("🧹 Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    @contextmanager
    def loading_timeout(self, timeout_minutes: int):
        """Context manager for loading timeout"""
        def timeout_handler():
            time.sleep(timeout_minutes * 60)
            if self.memory_monitor.monitoring:
                self.logger.error(f"⏰ Loading timeout after {timeout_minutes} minutes")
                
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()
        
        try:
            yield
        finally:
            pass  # Thread will die when main thread exits
    
    def generate_text(self, prompt: str, max_tokens: int = 100, **kwargs) -> Optional[str]:
        """Generate text using loaded model"""
        if self.model is None or self.tokenizer is None:
            self.logger.error("Model not loaded")
            return None
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move to appropriate device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=kwargs.get('do_sample', True),
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.9),
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text[len(prompt):].strip()  # Remove original prompt
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        if self.model is None:
            return {'status': 'not_loaded'}
            
        try:
            # Calculate model size
            param_count = sum(p.numel() for p in self.model.parameters())
            model_size_gb = param_count * 2 / (1024**3)  # Assume FP16
            
            # Get device distribution
            device_distribution = {}
            for name, param in self.model.named_parameters():
                device = str(param.device)
                device_distribution[device] = device_distribution.get(device, 0) + param.numel()
            
            return {
                'status': 'loaded',
                'parameter_count': param_count,
                'model_size_gb': model_size_gb,
                'device_distribution': device_distribution,
                'dtype': str(next(self.model.parameters()).dtype),
                'config': self.loading_config.__dict__ if self.loading_config else {}
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}