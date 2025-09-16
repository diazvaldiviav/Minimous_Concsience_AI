"""
Base model management for SC Memory System.

This module provides the BaseModelManager class for loading and managing
TinyLlama-1.1B model with proper error handling, device management, and
memory optimization. Designed for future adapter integration (LoRA).

CRITICAL IMPORT ORDER:
- Import this module AFTER importing sentence-transformers if both are used
- PyTorch imports must come before transformers to ensure proper device detection
"""

import asyncio
import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

from ..core.config import Settings, get_settings
from ..core.exceptions import (
    ModelLoadError,
    ModelNotLoadedError,
    TokenizationError,
    ConfigurationError,
    MemoryError,
    CompatibilityError,
)
from ..core.models import ModelInfo

logger = logging.getLogger(__name__)


class BaseModelManager:
    """
    Manages TinyLlama-1.1B model loading and tokenization.
    
    Handles device placement, memory optimization, and provides
    a clean interface for future adapter integration. Supports
    both CPU and GPU inference with automatic device detection.
    
    Features:
    - Automatic device detection and optimization
    - Memory-efficient model loading with quantization options
    - Async model loading for non-blocking initialization  
    - Comprehensive error handling and logging
    - Model caching and persistence
    - Future-ready for LoRA adapter integration
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        settings: Optional[Settings] = None
    ) -> None:
        """
        Initialize the base model manager.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Target device ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            settings: Application settings instance
        """
        self._settings = settings or get_settings()
        self._model_name = model_name or self._settings.model.name
        self._device = device or self._settings.model.device
        
        # Model components
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._pipeline: Optional[pipeline] = None
        
        # State tracking
        self._is_loaded = False
        self._load_start_time: Optional[float] = None
        self._load_end_time: Optional[float] = None
        self._memory_usage: Optional[float] = None
        
        # Device management
        self._resolved_device: Optional[str] = None
        self._device_type: Optional[str] = None
        
        # Initialize device detection
        self._detect_device()
        
        logger.info(
            f"Initialized BaseModelManager for {self._model_name} on {self._resolved_device}"
        )
    
    def _detect_device(self) -> None:
        """
        Detect and configure the optimal device for model inference.
        
        Performs automatic device detection based on availability and
        user preferences, with fallback to CPU if GPU is unavailable.
        """
        try:
            if self._device == "auto":
                if torch.cuda.is_available():
                    # Use the first available GPU
                    self._resolved_device = "cuda:0"
                    self._device_type = "cuda"
                    logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
                else:
                    self._resolved_device = "cpu"
                    self._device_type = "cpu"
                    logger.info("CUDA not available, using CPU")
            else:
                self._resolved_device = self._device
                self._device_type = self._device.split(":")[0]
            
            # Set default device for PyTorch operations
            if self._device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_device(self._resolved_device)
                # Clear GPU cache
                torch.cuda.empty_cache()
            
            logger.info(f"Using device: {self._resolved_device}")
            
        except Exception as e:
            logger.error(f"Device detection failed: {e}")
            # Fallback to CPU
            self._resolved_device = "cpu"
            self._device_type = "cpu"
            logger.warning("Falling back to CPU due to device detection error")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration based on device and settings.
        
        Returns:
            Configuration dictionary for model loading
        """
        config = {
            "cache_dir": str(self._settings.model.cache_dir),
            "low_cpu_mem_usage": self._settings.model.low_cpu_mem_usage,
            "trust_remote_code": self._settings.model.trust_remote_code,
            "use_safetensors": self._settings.model.use_safetensors,
        }
        
        # Device-specific configurations
        if self._device_type == "cuda":
            config.update({
                "device_map": "auto",
                "torch_dtype": self._get_torch_dtype(),
            })
            
            # Add quantization for memory efficiency if needed
            if self._should_use_quantization():
                config["quantization_config"] = self._get_quantization_config()
                
        else:  # CPU - Optimized for Ryzen AMD 7 + 16GB RAM
            config.update({
                "torch_dtype": torch.float32,  # Use float32 for CPU
                # Disable accelerate device mapping to avoid conflicts
                # Your Ryzen AMD 7 can handle direct CPU loading efficiently
            })
        
        return config
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get appropriate torch dtype based on settings and device."""
        dtype_map = {
            "auto": torch.float16 if self._device_type == "cuda" else torch.float32,
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        
        requested_dtype = self._settings.model.torch_dtype
        if requested_dtype in dtype_map:
            dtype = dtype_map[requested_dtype]
            
            # Verify dtype compatibility with device
            if dtype == torch.float16 and self._device_type == "cpu":
                logger.warning("float16 not optimal for CPU, using float32")
                return torch.float32
            
            return dtype
        else:
            logger.warning(f"Unknown dtype {requested_dtype}, using auto")
            return dtype_map["auto"]
    
    def _should_use_quantization(self) -> bool:
        """Determine if quantization should be used based on available memory."""
        if self._device_type != "cuda":
            return False
            
        try:
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_memory_gb = total_memory / (1024 ** 3)
            
            # Use quantization if GPU has less than 8GB memory
            return total_memory_gb < 8.0
            
        except Exception as e:
            logger.warning(f"Could not determine GPU memory: {e}")
            return False
    
    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """Get quantization configuration for memory optimization."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    async def load_model(self) -> None:
        """
        Load the TinyLlama model and tokenizer asynchronously.
        
        Performs model loading in a thread pool to avoid blocking
        the event loop. Includes comprehensive error handling and
        performance monitoring.
        
        Raises:
            ModelLoadError: If model loading fails
            ConfigurationError: If model configuration is invalid
            MemoryError: If insufficient memory is available
        """
        if self._is_loaded:
            logger.info("Model already loaded")
            return
        
        self._load_start_time = time.time()
        logger.info(f"Starting to load model: {self._model_name}")
        
        try:
            # Run model loading in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._load_model_sync
            )
            
            self._load_end_time = time.time()
            self._is_loaded = True
            
            # Record memory usage
            self._record_memory_usage()
            
            load_time = self._load_end_time - self._load_start_time
            logger.info(
                f"Model loaded successfully in {load_time:.2f}s "
                f"(Memory: {self._memory_usage:.1f}MB)"
            )
            
        except Exception as e:
            self._is_loaded = False
            self._model = None
            self._tokenizer = None
            
            error_msg = f"Failed to load model {self._model_name}: {str(e)}"
            logger.error(error_msg)
            
            # Categorize the error
            if "out of memory" in str(e).lower():
                raise MemoryError(
                    message=f"Insufficient memory to load {self._model_name}",
                    memory_limit_mb=self._settings.performance.max_memory_usage_gb * 1024,
                    details={"original_error": str(e)}
                )
            elif "configuration" in str(e).lower():
                raise ConfigurationError(
                    message=f"Invalid model configuration: {str(e)}",
                    config_key="model",
                    details={"model_name": self._model_name}
                )
            else:
                raise ModelLoadError(
                    message=error_msg,
                    model_name=self._model_name,
                    model_type="language_model",
                    details={"device": self._resolved_device, "error": str(e)}
                )
    
    def _load_model_sync(self) -> None:
        """Synchronous model loading implementation."""
        config = self._get_model_config()
        
        try:
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                cache_dir=config["cache_dir"],
                trust_remote_code=config["trust_remote_code"],
                use_fast=True,
            )
            
            # Set padding token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            logger.info("Loading model...")
            
            # Load model with configuration
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                **config
            )
            
            # Create pipeline for easier inference
            # Note: When model uses accelerate, don't specify device parameter
            pipeline_kwargs = {
                "task": "text-generation",
                "model": self._model,
                "tokenizer": self._tokenizer,
                "torch_dtype": self._get_torch_dtype(),
            }

            # Create pipeline optimized for Ryzen AMD 7 CPU
            # With proper config, accelerate conflicts should be resolved
            logger.info("Creating pipeline optimized for Ryzen AMD 7")
            self._pipeline = pipeline(**pipeline_kwargs)
            
        except Exception as e:
            # Clean up any partially loaded components
            self._model = None
            self._tokenizer = None
            self._pipeline = None
            raise e
    
    def _record_memory_usage(self) -> None:
        """Record current memory usage after model loading."""
        try:
            if self._device_type == "cuda" and torch.cuda.is_available():
                # GPU memory usage
                memory_bytes = torch.cuda.memory_allocated()
                self._memory_usage = memory_bytes / (1024 ** 2)  # Convert to MB
            else:
                # Estimate CPU memory usage (rough approximation)
                if self._model is not None:
                    param_count = sum(p.numel() for p in self._model.parameters())
                    # Assume 4 bytes per parameter (float32)
                    estimated_bytes = param_count * 4
                    self._memory_usage = estimated_bytes / (1024 ** 2)
                else:
                    self._memory_usage = 0.0
                    
        except Exception as e:
            logger.warning(f"Could not record memory usage: {e}")
            self._memory_usage = None
    
    def tokenize(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text using the loaded tokenizer.
        
        Args:
            text: Text to tokenize
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate long sequences
            return_tensors: Return tensor format ('pt' for PyTorch)
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary containing tokenized inputs
            
        Raises:
            ModelNotLoadedError: If model/tokenizer not loaded
            TokenizationError: If tokenization fails
        """
        if not self._is_loaded or self._tokenizer is None:
            raise ModelNotLoadedError(
                message="Tokenizer not loaded",
                model_name=self._model_name
            )
        
        if not isinstance(text, str):
            raise TokenizationError(
                message=f"Text must be a string, got {type(text)}",
                text_length=None
            )
        
        if len(text.strip()) == 0:
            raise TokenizationError(
                message="Text cannot be empty",
                text_length=0
            )
        
        try:
            max_length = max_length or self._settings.model.max_length
            
            # Tokenize with configuration
            tokens = self._tokenizer(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                **kwargs
            )
            
            # Move to appropriate device
            if return_tensors == "pt" and self._device_type == "cuda":
                tokens = {k: v.to(self._resolved_device) for k, v in tokens.items()}
            
            logger.debug(
                f"Tokenized text of length {len(text)} -> {tokens['input_ids'].shape[-1]} tokens"
            )
            
            return tokens
            
        except Exception as e:
            error_msg = f"Tokenization failed: {str(e)}"
            logger.error(error_msg)
            
            raise TokenizationError(
                message=error_msg,
                text_length=len(text),
                details={"max_length": max_length, "error": str(e)}
            )
    
    def get_model_info(self) -> ModelInfo:
        """
        Get comprehensive information about the loaded model.
        
        Returns:
            ModelInfo object with model details
        """
        try:
            parameters_count = None
            if self._model is not None:
                parameters_count = sum(p.numel() for p in self._model.parameters())
            
            load_time = None
            if self._load_start_time and self._load_end_time:
                load_time = self._load_end_time - self._load_start_time
            
            return ModelInfo(
                name=self._model_name,
                type="language_model",
                device=self._resolved_device or "unknown",
                memory_usage_mb=self._memory_usage,
                parameters_count=parameters_count,
                is_loaded=self._is_loaded,
                load_time=load_time,
                metadata={
                    "device_type": self._device_type,
                    "torch_dtype": str(self._get_torch_dtype()),
                    "quantized": self._should_use_quantization(),
                    "max_length": self._settings.model.max_length,
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            
            return ModelInfo(
                name=self._model_name,
                type="language_model", 
                device="unknown",
                is_loaded=self._is_loaded,
                metadata={"error": str(e)}
            )
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready for use.
        
        Returns:
            True if model and tokenizer are loaded
        """
        return self._is_loaded and self._model is not None and self._tokenizer is not None
    
    def get_device(self) -> str:
        """
        Get the device where the model is loaded.
        
        Returns:
            Device string (e.g., 'cpu', 'cuda:0')
        """
        return self._resolved_device or "unknown"
    
    def get_tokenizer(self) -> AutoTokenizer:
        """
        Get the loaded tokenizer instance.
        
        Returns:
            AutoTokenizer instance
            
        Raises:
            ModelNotLoadedError: If tokenizer not loaded
        """
        if not self._is_loaded or self._tokenizer is None:
            raise ModelNotLoadedError(
                message="Tokenizer not loaded",
                model_name=self._model_name
            )
        return self._tokenizer
    
    def get_model(self) -> AutoModelForCausalLM:
        """
        Get the loaded model instance.
        
        Returns:
            AutoModelForCausalLM instance
            
        Raises:
            ModelNotLoadedError: If model not loaded
        """
        if not self._is_loaded or self._model is None:
            raise ModelNotLoadedError(
                message="Model not loaded",
                model_name=self._model_name
            )
        return self._model
    
    def unload_model(self) -> None:
        """
        Unload the model and free memory.
        
        Useful for memory management when the model is no longer needed.
        """
        if self._model is not None:
            del self._model
            self._model = None
            
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            
        # Clear GPU cache if using CUDA
        if self._device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        self._is_loaded = False
        self._memory_usage = None
        
        logger.info(f"Model {self._model_name} unloaded")
    
    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            if self.is_loaded():
                self.unload_model()
        except Exception:
            pass  # Ignore cleanup errors during destruction


# Convenience function for creating model manager
def create_model_manager(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    settings: Optional[Settings] = None
) -> BaseModelManager:
    """
    Create and return a BaseModelManager instance.
    
    Args:
        model_name: HuggingFace model name or local path
        device: Target device
        settings: Application settings
        
    Returns:
        BaseModelManager instance
    """
    return BaseModelManager(
        model_name=model_name,
        device=device, 
        settings=settings
    )


# Export main classes and functions
__all__ = [
    "BaseModelManager",
    "create_model_manager",
]