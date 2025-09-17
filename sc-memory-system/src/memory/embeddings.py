"""
Embeddings generation and management for SC Memory System.

This module provides the EmbeddingsManager class for generating text embeddings
using sentence-transformers with proper device management and batch processing.

CRITICAL COMPATIBILITY NOTE:
This module MUST be imported BEFORE faiss to avoid segmentation faults.
The import order is: sentence-transformers -> numpy -> faiss

Example correct import order in consuming modules:
    from memory.embeddings import EmbeddingsManager
    import faiss  # AFTER sentence-transformers import
"""

import asyncio
import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# CRITICAL: Import sentence-transformers BEFORE any faiss imports
# GRACEFUL FALLBACK: Handle potential dependency conflicts
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"sentence-transformers import failed: {e}")
    logger.warning("Embeddings functionality will be disabled - install compatible versions")
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import torch

from ..core.config import Settings, get_settings
from ..core.exceptions import (
    EmbeddingError,
    ModelLoadError,
    ModelNotLoadedError,
    ConfigurationError,
    ValidationError,
    CompatibilityError,
)
from ..core.models import EmbeddingResult, ModelInfo

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """
    Manages embedding generation using sentence-transformers.
    
    Provides efficient text encoding with proper device management,
    batch processing capabilities, and memory optimization. Designed
    to work seamlessly with FAISS vector stores.
    
    Features:
    - Automatic device detection and optimization
    - Batch processing for efficient embedding generation
    - Memory-efficient processing with configurable batch sizes
    - Proper import order compatibility with FAISS
    - Comprehensive error handling and logging
    - Async support for non-blocking operations
    - Model caching and persistence
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        settings: Optional[Settings] = None
    ) -> None:
        """
        Initialize the embeddings manager.
        
        Args:
            model_name: Sentence transformers model name
            device: Target device ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            settings: Application settings instance
        """
        self._settings = settings or get_settings()
        self._model_name = model_name or self._settings.embeddings.model_name
        self._device = device or self._settings.embeddings.device
        
        # Model components
        self._model: Optional[SentenceTransformer] = None
        
        # State tracking
        self._is_loaded = False
        self._load_start_time: Optional[float] = None
        self._load_end_time: Optional[float] = None
        self._memory_usage: Optional[float] = None
        
        # Device management
        self._resolved_device: Optional[str] = None
        self._device_type: Optional[str] = None
        
        # Model metadata
        self._dimension: Optional[int] = None
        self._max_seq_length: Optional[int] = None
        
        # Initialize device detection
        self._detect_device()
        
        logger.info(
            f"Initialized EmbeddingsManager for {self._model_name} on {self._resolved_device}"
        )
    
    def _detect_device(self) -> None:
        """
        Detect and configure the optimal device for embeddings generation.
        
        Performs automatic device detection with fallback to CPU if GPU
        is unavailable or incompatible.
        """
        try:
            if self._device == "auto":
                if torch.cuda.is_available():
                    self._resolved_device = "cuda"
                    self._device_type = "cuda"
                    logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
                else:
                    self._resolved_device = "cpu"
                    self._device_type = "cpu"
                    logger.info("CUDA not available, using CPU")
            else:
                self._resolved_device = self._device
                self._device_type = self._device.split(":")[0]
            
            logger.info(f"Using device: {self._resolved_device}")
            
        except Exception as e:
            logger.error(f"Device detection failed: {e}")
            # Fallback to CPU
            self._resolved_device = "cpu"
            self._device_type = "cpu"
            logger.warning("Falling back to CPU due to device detection error")
    
    async def load_model(self) -> None:
        """
        Load the sentence transformer model asynchronously.

        Performs model loading in a thread pool to avoid blocking
        the event loop. Includes comprehensive error handling and
        compatibility checks.

        Raises:
            ModelLoadError: If model loading fails
            ConfigurationError: If model configuration is invalid
            CompatibilityError: If import order issues detected
        """
        # CRITICAL CHECK: Ensure sentence-transformers is available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ModelLoadError(
                message="sentence-transformers not available due to dependency conflicts",
                model_name=self._model_name,
                model_type="embeddings",
                details={"solution": "Install compatible huggingface_hub and datasets versions"}
            )

        if self._is_loaded:
            logger.info("Embeddings model already loaded")
            return

        self._load_start_time = time.time()
        logger.info(f"Starting to load embeddings model: {self._model_name}")

        try:
            # Check for potential import order issues
            self._check_compatibility()
            
            # Run model loading in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._load_model_sync
            )
            
            self._load_end_time = time.time()
            self._is_loaded = True
            
            # Record memory usage and model metadata
            self._record_memory_usage()
            self._extract_model_metadata()
            
            load_time = self._load_end_time - self._load_start_time
            logger.info(
                f"Embeddings model loaded successfully in {load_time:.2f}s "
                f"(Dimension: {self._dimension}, Memory: {self._memory_usage:.1f}MB)"
            )
            
        except Exception as e:
            self._is_loaded = False
            self._model = None
            
            error_msg = f"Failed to load embeddings model {self._model_name}: {str(e)}"
            logger.error(error_msg)
            
            # Categorize the error
            if "import" in str(e).lower() or "faiss" in str(e).lower():
                raise CompatibilityError(
                    message=f"Import order compatibility issue: {str(e)}",
                    component_a="sentence_transformers",
                    component_b="faiss",
                    details={"model_name": self._model_name}
                )
            elif "configuration" in str(e).lower() or "config" in str(e).lower():
                raise ConfigurationError(
                    message=f"Invalid embeddings configuration: {str(e)}",
                    config_key="embeddings",
                    details={"model_name": self._model_name}
                )
            else:
                raise ModelLoadError(
                    message=error_msg,
                    model_name=self._model_name,
                    model_type="embeddings",
                    details={"device": self._resolved_device, "error": str(e)}
                )
    
    def _check_compatibility(self) -> None:
        """
        Check for potential compatibility issues with FAISS imports.
        
        This method attempts to detect if FAISS has been imported before
        sentence-transformers, which can cause segmentation faults.
        """
        import sys
        
        # Check if faiss is already imported
        faiss_modules = [name for name in sys.modules.keys() if name.startswith('faiss')]
        
        if faiss_modules:
            logger.warning(
                f"FAISS modules detected before sentence-transformers import: {faiss_modules}. "
                "This may cause compatibility issues. Consider importing sentence-transformers first."
            )
    
    def _load_model_sync(self) -> None:
        """Synchronous model loading implementation."""
        try:
            cache_dir = str(self._settings.embeddings.cache_dir)
            
            # Ensure cache directory exists
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Loading sentence transformer model from cache: {cache_dir}")
            
            # Load model with device placement
            self._model = SentenceTransformer(
                self._model_name,
                device=self._resolved_device,
                cache_folder=cache_dir
            )
            
            # Configure model settings
            if hasattr(self._model, 'max_seq_length'):
                # Set maximum sequence length if configurable
                max_length = self._settings.embeddings.max_seq_length
                if max_length != 384:  # Only set if different from default
                    self._model.max_seq_length = max_length
                    logger.info(f"Set max sequence length to {max_length}")
            
        except Exception as e:
            # Clean up any partially loaded components
            self._model = None
            raise e
    
    def _record_memory_usage(self) -> None:
        """Record current memory usage after model loading."""
        try:
            if self._device_type == "cuda" and torch.cuda.is_available():
                # GPU memory usage
                memory_bytes = torch.cuda.memory_allocated()
                self._memory_usage = memory_bytes / (1024 ** 2)  # Convert to MB
            else:
                # Estimate CPU memory usage
                if self._model is not None:
                    # Rough estimation based on model parameters
                    param_count = sum(
                        p.numel() for p in self._model.parameters()
                        if hasattr(self._model, 'parameters')
                    )
                    if param_count > 0:
                        # Assume 4 bytes per parameter
                        estimated_bytes = param_count * 4
                        self._memory_usage = estimated_bytes / (1024 ** 2)
                    else:
                        # Default estimation for sentence transformers
                        self._memory_usage = 200.0  # ~200MB typical
                else:
                    self._memory_usage = 0.0
                    
        except Exception as e:
            logger.warning(f"Could not record memory usage: {e}")
            self._memory_usage = None
    
    def _extract_model_metadata(self) -> None:
        """Extract model metadata like dimension and max sequence length."""
        try:
            if self._model is not None:
                # Get embedding dimension
                if hasattr(self._model, 'get_sentence_embedding_dimension'):
                    self._dimension = self._model.get_sentence_embedding_dimension()
                else:
                    # Fallback: encode a test string to get dimension
                    test_embedding = self._model.encode("test", convert_to_numpy=True)
                    self._dimension = len(test_embedding)
                
                # Get max sequence length
                if hasattr(self._model, 'max_seq_length'):
                    self._max_seq_length = self._model.max_seq_length
                else:
                    self._max_seq_length = self._settings.embeddings.max_seq_length
                
                logger.info(f"Model metadata: dimension={self._dimension}, max_length={self._max_seq_length}")
                
        except Exception as e:
            logger.warning(f"Could not extract model metadata: {e}")
            # Use configured defaults
            self._dimension = self._settings.embeddings.dimension
            self._max_seq_length = self._settings.embeddings.max_seq_length
    
    async def encode_text(
        self,
        text: str,
        normalize_embeddings: Optional[bool] = None
    ) -> EmbeddingResult:
        """
        Encode a single text into embeddings.
        
        Args:
            text: Text to encode
            normalize_embeddings: Whether to normalize embeddings to unit vectors
            
        Returns:
            EmbeddingResult with the generated embedding
            
        Raises:
            ModelNotLoadedError: If model not loaded
            EmbeddingError: If encoding fails
            ValidationError: If input validation fails
        """
        if not self._is_loaded or self._model is None:
            raise ModelNotLoadedError(
                message="Embeddings model not loaded",
                model_name=self._model_name
            )
        
        # Input validation
        if not isinstance(text, str):
            raise ValidationError(
                message=f"Text must be a string, got {type(text)}",
                field_errors={"text": "must be string"}
            )
        
        if len(text.strip()) == 0:
            raise ValidationError(
                message="Text cannot be empty",
                field_errors={"text": "cannot be empty"}
            )
        
        try:
            start_time = time.time()
            
            # Run encoding in thread pool to avoid blocking
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self._encode_single_sync, text, normalize_embeddings
            )
            
            processing_time = time.time() - start_time
            
            logger.debug(f"Encoded text of length {len(text)} in {processing_time:.3f}s")
            
            return EmbeddingResult(
                text=text,
                embedding=embedding.tolist(),
                model_name=self._model_name,
                dimension=len(embedding),
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            logger.error(error_msg)
            
            raise EmbeddingError(
                message=error_msg,
                model_name=self._model_name,
                text_length=len(text),
                details={"error": str(e)}
            )
    
    def _encode_single_sync(
        self,
        text: str,
        normalize_embeddings: Optional[bool] = None
    ) -> np.ndarray:
        """Synchronous single text encoding."""
        normalize = normalize_embeddings
        if normalize is None:
            normalize = self._settings.embeddings.normalize_embeddings
        
        # Generate embedding
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embedding
    
    async def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize_embeddings: Optional[bool] = None
    ) -> List[EmbeddingResult]:
        """
        Encode multiple texts into embeddings efficiently.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            List of EmbeddingResult objects
            
        Raises:
            ModelNotLoadedError: If model not loaded
            EmbeddingError: If encoding fails
            ValidationError: If input validation fails
        """
        if not self._is_loaded or self._model is None:
            raise ModelNotLoadedError(
                message="Embeddings model not loaded",
                model_name=self._model_name
            )
        
        # Input validation
        if not isinstance(texts, list):
            raise ValidationError(
                message=f"Texts must be a list, got {type(texts)}",
                field_errors={"texts": "must be list"}
            )
        
        if len(texts) == 0:
            return []
        
        # Validate all texts are strings and non-empty
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValidationError(
                    message=f"Text at index {i} must be a string",
                    field_errors={f"texts[{i}]": "must be string"}
                )
            if len(text.strip()) == 0:
                raise ValidationError(
                    message=f"Text at index {i} cannot be empty",
                    field_errors={f"texts[{i}]": "cannot be empty"}
                )
        
        try:
            start_time = time.time()
            batch_size = batch_size or self._settings.embeddings.batch_size
            
            # Run batch encoding in thread pool
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self._encode_batch_sync, texts, batch_size, normalize_embeddings
            )
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Encoded {len(texts)} texts in {processing_time:.3f}s "
                f"(batch_size={batch_size}, avg={processing_time/len(texts):.4f}s/text)"
            )
            
            # Create results
            results = []
            for text, embedding in zip(texts, embeddings):
                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding.tolist(),
                    model_name=self._model_name,
                    dimension=len(embedding),
                    processing_time=processing_time / len(texts)  # Average time per text
                ))
            
            return results
            
        except Exception as e:
            error_msg = f"Batch embedding generation failed: {str(e)}"
            logger.error(error_msg)
            
            raise EmbeddingError(
                message=error_msg,
                model_name=self._model_name,
                text_length=sum(len(t) for t in texts),
                details={
                    "batch_size": batch_size,
                    "num_texts": len(texts),
                    "error": str(e)
                }
            )
    
    def _encode_batch_sync(
        self,
        texts: List[str],
        batch_size: int,
        normalize_embeddings: Optional[bool] = None
    ) -> np.ndarray:
        """Synchronous batch encoding implementation."""
        normalize = normalize_embeddings
        if normalize is None:
            normalize = self._settings.embeddings.normalize_embeddings
        
        # Generate embeddings in batches
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100  # Show progress for large batches
        )
        
        return embeddings
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            Embedding dimension
            
        Raises:
            ModelNotLoadedError: If model not loaded
        """
        if not self._is_loaded or self._dimension is None:
            if self._model is None:
                raise ModelNotLoadedError(
                    message="Model not loaded",
                    model_name=self._model_name
                )
            # Try to extract dimension if not cached
            self._extract_model_metadata()
        
        return self._dimension or self._settings.embeddings.dimension
    
    def get_max_seq_length(self) -> int:
        """
        Get the maximum sequence length.
        
        Returns:
            Maximum sequence length in tokens
        """
        return self._max_seq_length or self._settings.embeddings.max_seq_length
    
    def get_model_info(self) -> ModelInfo:
        """
        Get comprehensive information about the loaded embeddings model.
        
        Returns:
            ModelInfo object with model details
        """
        try:
            load_time = None
            if self._load_start_time and self._load_end_time:
                load_time = self._load_end_time - self._load_start_time
            
            return ModelInfo(
                name=self._model_name,
                type="embeddings",
                device=self._resolved_device or "unknown",
                memory_usage_mb=self._memory_usage,
                is_loaded=self._is_loaded,
                load_time=load_time,
                metadata={
                    "device_type": self._device_type,
                    "dimension": self._dimension,
                    "max_seq_length": self._max_seq_length,
                    "normalize_embeddings": self._settings.embeddings.normalize_embeddings,
                    "batch_size": self._settings.embeddings.batch_size,
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get embeddings model info: {e}")
            
            return ModelInfo(
                name=self._model_name,
                type="embeddings",
                device="unknown",
                is_loaded=self._is_loaded,
                metadata={"error": str(e)}
            )
    
    def is_loaded(self) -> bool:
        """
        Check if the embeddings model is loaded and ready.
        
        Returns:
            True if model is loaded
        """
        return self._is_loaded and self._model is not None
    
    def get_device(self) -> str:
        """
        Get the device where the embeddings model is loaded.
        
        Returns:
            Device string
        """
        return self._resolved_device or "unknown"
    
    def unload_model(self) -> None:
        """
        Unload the embeddings model and free memory.
        """
        if self._model is not None:
            del self._model
            self._model = None
        
        # Clear GPU cache if using CUDA
        if self._device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        self._is_loaded = False
        self._memory_usage = None
        self._dimension = None
        self._max_seq_length = None
        
        logger.info(f"Embeddings model {self._model_name} unloaded")
    
    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            if self.is_loaded():
                self.unload_model()
        except Exception:
            pass  # Ignore cleanup errors during destruction


# Convenience function for creating embeddings manager
def create_embeddings_manager(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    settings: Optional[Settings] = None
) -> EmbeddingsManager:
    """
    Create and return an EmbeddingsManager instance.
    
    Args:
        model_name: Sentence transformers model name
        device: Target device
        settings: Application settings
        
    Returns:
        EmbeddingsManager instance
    """
    return EmbeddingsManager(
        model_name=model_name,
        device=device,
        settings=settings
    )


# Export main classes and functions
__all__ = [
    "EmbeddingsManager",
    "create_embeddings_manager",
]