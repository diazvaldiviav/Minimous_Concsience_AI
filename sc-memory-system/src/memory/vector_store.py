"""
FAISS vector store implementation for SC Memory System.

This module provides the VectorStore class for similarity search and retrieval
using FAISS (Facebook AI Similarity Search) with persistence and metadata management.

CRITICAL IMPORT ORDER:
This module MUST be imported AFTER sentence-transformers to avoid segmentation faults.
Import order: sentence_transformers -> numpy -> faiss

Example correct usage:
    from memory.embeddings import EmbeddingsManager  # This imports sentence-transformers
    from memory.vector_store import VectorStore       # This imports faiss AFTER
"""

import asyncio
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

# CRITICAL: Import faiss AFTER sentence-transformers (should be imported by embeddings module)
try:
    import faiss
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"FAISS import failed: {e}")
    logger.error("Make sure to install faiss-cpu or faiss-gpu and import sentence-transformers first")
    raise

from ..core.config import Settings, get_settings
from ..core.exceptions import (
    VectorStoreError,
    VectorIndexError,
    SearchError,
    ConfigurationError,
    ValidationError,
    CompatibilityError,
)
from ..core.models import VectorSearchResult, VectorSearchQuery

logger = logging.getLogger(__name__)


class SearchResult:
    """Individual search result with metadata."""
    
    def __init__(
        self,
        id: str,
        score: float,
        text: str,
        metadata: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> None:
        self.id = id
        self.score = score
        self.text = text
        self.metadata = metadata
        self.embedding = embedding


class VectorStore:
    """
    FAISS-based vector store for semantic search and retrieval.
    
    Handles embedding storage, similarity search, and metadata management
    with persistence to disk for data retention. Supports multiple FAISS
    index types and provides efficient batch operations.
    
    Features:
    - Multiple FAISS index types (Flat, IVF, HNSW)
    - Persistent storage with automatic backup
    - Metadata management with JSON storage
    - Batch operations for efficient indexing
    - Comprehensive error handling and logging
    - Memory-efficient operations
    - Thread-safe operations with async support
    """
    
    def __init__(
        self,
        dimension: Optional[int] = None,
        index_path: Optional[str] = None,
        settings: Optional[Settings] = None
    ) -> None:
        """
        Initialize the vector store.
        
        Args:
            dimension: Vector dimension (must match embeddings)
            index_path: Path to store FAISS index
            settings: Application settings instance
        """
        self._settings = settings or get_settings()
        self._dimension = dimension or self._settings.vector_store.dimension
        self._index_path = Path(index_path or self._settings.vector_store.index_path)
        
        # FAISS components
        self._index: Optional[faiss.Index] = None
        self._index_type = self._settings.vector_store.index_type
        self._metric_type = self._settings.vector_store.metric_type
        
        # Metadata storage
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._metadata_path = Path(self._settings.vector_store.metadata_path)
        
        # State tracking
        self._is_initialized = False
        self._next_idx = 0
        self._total_vectors = 0
        
        # Performance tracking
        self._last_save_time: Optional[float] = None
        self._modification_count = 0
        
        # Ensure directories exist
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized VectorStore (dim={self._dimension}, "
            f"index_type={self._index_type}, path={self._index_path})"
        )
    
    def _create_index(self) -> faiss.Index:
        """
        Create a new FAISS index based on configuration.
        
        Returns:
            FAISS Index instance
            
        Raises:
            VectorIndexError: If index creation fails
            ConfigurationError: If index configuration is invalid
        """
        try:
            index_factory = self._settings.vector_store.index_factory
            
            # Map configuration to FAISS index creation
            if index_factory.lower() == "flat":
                if self._metric_type == "METRIC_L2":
                    index = faiss.IndexFlatL2(self._dimension)
                elif self._metric_type == "METRIC_INNER_PRODUCT":
                    index = faiss.IndexFlatIP(self._dimension)
                else:
                    raise ConfigurationError(
                        message=f"Unsupported metric type for Flat index: {self._metric_type}",
                        config_key="vector_store.metric_type"
                    )
                    
            elif index_factory.lower().startswith("ivf"):
                # IVF index with configurable parameters
                ncentroids = 100  # Default number of centroids
                
                # Create quantizer
                if self._metric_type == "METRIC_L2":
                    quantizer = faiss.IndexFlatL2(self._dimension)
                    index = faiss.IndexIVFFlat(quantizer, self._dimension, ncentroids)
                elif self._metric_type == "METRIC_INNER_PRODUCT":
                    quantizer = faiss.IndexFlatIP(self._dimension)
                    index = faiss.IndexIVFFlat(quantizer, self._dimension, ncentroids)
                else:
                    raise ConfigurationError(
                        message=f"Unsupported metric type for IVF index: {self._metric_type}",
                        config_key="vector_store.metric_type"
                    )
                
                # Set nprobe for search
                index.nprobe = self._settings.vector_store.nprobe
                
            elif index_factory.lower().startswith("hnsw"):
                # HNSW index for fast approximate search
                index = faiss.IndexHNSWFlat(
                    self._dimension,
                    self._settings.vector_store.m
                )
                index.hnsw.efConstruction = self._settings.vector_store.efconstruction
                
            else:
                # Use factory string directly
                try:
                    index = faiss.index_factory(
                        self._dimension,
                        index_factory,
                        faiss.METRIC_L2 if self._metric_type == "METRIC_L2" else faiss.METRIC_INNER_PRODUCT
                    )
                except Exception as e:
                    raise ConfigurationError(
                        message=f"Invalid FAISS index factory string: {index_factory}",
                        config_key="vector_store.index_factory",
                        details={"error": str(e)}
                    )
            
            logger.info(f"Created FAISS index: {index}")
            return index
            
        except Exception as e:
            error_msg = f"Failed to create FAISS index: {str(e)}"
            logger.error(error_msg)
            
            raise VectorIndexError(
                message=error_msg,
                index_path=str(self._index_path),
                details={
                    "dimension": self._dimension,
                    "index_factory": index_factory,
                    "metric_type": self._metric_type,
                    "error": str(e)
                }
            )
    
    async def initialize(self) -> None:
        """
        Initialize the vector store.
        
        Attempts to load existing index from disk, creates new index if
        none exists. Loads metadata and performs consistency checks.
        
        Raises:
            VectorStoreError: If initialization fails
        """
        if self._is_initialized:
            logger.info("Vector store already initialized")
            return
        
        try:
            # Try to load existing index
            if await self.load_index():
                logger.info("Loaded existing FAISS index from disk")
            else:
                # Create new index
                self._index = self._create_index()
                logger.info("Created new FAISS index")
            
            # Load metadata
            await self._load_metadata()
            
            # Perform consistency checks
            self._validate_consistency()
            
            self._is_initialized = True
            
            logger.info(
                f"Vector store initialized successfully "
                f"({self._total_vectors} vectors, {len(self._metadata)} metadata entries)"
            )
            
        except Exception as e:
            error_msg = f"Vector store initialization failed: {str(e)}"
            logger.error(error_msg)
            
            raise VectorStoreError(
                message=error_msg,
                operation="initialize",
                index_type=self._index_type,
                details={"error": str(e)}
            )
    
    def _validate_consistency(self) -> None:
        """
        Validate consistency between index and metadata.
        
        Raises:
            VectorStoreError: If inconsistencies are found
        """
        if self._index is None:
            return
        
        index_size = self._index.ntotal
        metadata_size = len(self._metadata)
        
        if index_size != metadata_size:
            logger.warning(
                f"Inconsistency detected: index has {index_size} vectors, "
                f"metadata has {metadata_size} entries"
            )
            
            # Try to repair by truncating metadata to match index
            if index_size < metadata_size:
                # Remove excess metadata
                excess_indices = list(range(index_size, metadata_size))
                for idx in excess_indices:
                    if idx in self._metadata:
                        vector_id = self._idx_to_id.get(idx)
                        if vector_id:
                            del self._id_to_idx[vector_id]
                        del self._idx_to_id[idx]
                        del self._metadata[idx]
                
                logger.info(f"Repaired metadata: removed {len(excess_indices)} excess entries")
        
        self._total_vectors = index_size
        self._next_idx = max(self._idx_to_id.keys()) + 1 if self._idx_to_id else 0
    
    async def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add embeddings to the vector store with metadata.
        
        Args:
            embeddings: Array of embeddings to add (shape: [n, dimension])
            metadata: List of metadata dictionaries for each embedding
            
        Returns:
            List of unique IDs for the added embeddings
            
        Raises:
            VectorStoreError: If adding embeddings fails
            ValidationError: If input validation fails
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Input validation
        if not isinstance(embeddings, np.ndarray):
            raise ValidationError(
                message="Embeddings must be a numpy array",
                field_errors={"embeddings": "must be numpy array"}
            )
        
        if len(embeddings.shape) != 2:
            raise ValidationError(
                message=f"Embeddings must be 2D array, got shape {embeddings.shape}",
                field_errors={"embeddings": "must be 2D array"}
            )
        
        if embeddings.shape[1] != self._dimension:
            raise ValidationError(
                message=f"Embedding dimension {embeddings.shape[1]} doesn't match store dimension {self._dimension}",
                field_errors={"embeddings": f"dimension must be {self._dimension}"}
            )
        
        if len(metadata) != embeddings.shape[0]:
            raise ValidationError(
                message=f"Metadata length {len(metadata)} doesn't match embeddings count {embeddings.shape[0]}",
                field_errors={"metadata": "length must match embeddings count"}
            )
        
        try:
            # Run in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._add_embeddings_sync, embeddings, metadata
            )
            
            logger.info(f"Added {len(result)} embeddings to vector store")
            return result
            
        except Exception as e:
            error_msg = f"Failed to add embeddings: {str(e)}"
            logger.error(error_msg)
            
            raise VectorStoreError(
                message=error_msg,
                operation="add_embeddings",
                details={
                    "count": embeddings.shape[0],
                    "dimension": embeddings.shape[1],
                    "error": str(e)
                }
            )
    
    def _add_embeddings_sync(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """Synchronous implementation of embedding addition."""
        if self._index is None:
            raise VectorStoreError(
                message="Index not initialized",
                operation="add_embeddings"
            )
        
        # Ensure embeddings are in correct dtype
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Generate unique IDs
        vector_ids = [str(uuid4()) for _ in range(embeddings.shape[0])]
        
        # Add to FAISS index
        start_idx = self._next_idx
        
        # For IVF indexes, train if necessary
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            if self._total_vectors + embeddings.shape[0] >= 256:  # Minimum for IVF training
                combined_embeddings = embeddings
                if self._total_vectors > 0:
                    # Get existing embeddings for training
                    existing_embeddings = self._index.reconstruct_n(0, self._total_vectors)
                    combined_embeddings = np.vstack([existing_embeddings, embeddings])
                
                logger.info("Training IVF index...")
                self._index.train(combined_embeddings)
                logger.info("IVF index training completed")
        
        # Add embeddings to index
        self._index.add(embeddings)
        
        # Store metadata
        for i, (vector_id, meta) in enumerate(zip(vector_ids, metadata)):
            idx = start_idx + i
            
            # Add timestamp if not present
            if 'timestamp' not in meta:
                meta['timestamp'] = time.time()
            
            self._metadata[idx] = meta
            self._id_to_idx[vector_id] = idx
            self._idx_to_id[idx] = vector_id
        
        # Update counters
        self._next_idx += embeddings.shape[0]
        self._total_vectors = self._index.ntotal
        self._modification_count += 1
        
        # Auto-save if configured
        if self._modification_count % 100 == 0:  # Save every 100 modifications
            asyncio.create_task(self._auto_save())
        
        return vector_ids
    
    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[SearchResult]:
        """
        Search for similar vectors in the store.
        
        Args:
            query_embedding: Query vector to search for
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Metadata filters to apply
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            List of SearchResult objects
            
        Raises:
            SearchError: If search fails
            ValidationError: If input validation fails
        """
        if not self._is_initialized or self._index is None:
            raise SearchError(
                message="Vector store not initialized",
                k=k
            )
        
        # Input validation
        if not isinstance(query_embedding, np.ndarray):
            raise ValidationError(
                message="Query embedding must be a numpy array",
                field_errors={"query_embedding": "must be numpy array"}
            )
        
        if query_embedding.shape != (self._dimension,):
            raise ValidationError(
                message=f"Query embedding shape {query_embedding.shape} doesn't match dimension {self._dimension}",
                field_errors={"query_embedding": f"shape must be ({self._dimension},)"}
            )
        
        if k <= 0 or k > 1000:
            raise ValidationError(
                message=f"k must be between 1 and 1000, got {k}",
                field_errors={"k": "must be between 1 and 1000"}
            )
        
        if self._total_vectors == 0:
            logger.warning("Vector store is empty, returning empty results")
            return []
        
        try:
            # Run search in thread pool
            results = await asyncio.get_event_loop().run_in_executor(
                None, self._search_sync, query_embedding, k, score_threshold, filters, include_embeddings
            )
            
            logger.debug(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            logger.error(error_msg)
            
            raise SearchError(
                message=error_msg,
                k=k,
                details={"error": str(e)}
            )
    
    def _search_sync(
        self,
        query_embedding: np.ndarray,
        k: int,
        score_threshold: Optional[float],
        filters: Optional[Dict[str, Any]],
        include_embeddings: bool
    ) -> List[SearchResult]:
        """Synchronous search implementation."""
        # Ensure query embedding is in correct dtype and shape
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Perform FAISS search
        scores, indices = self._index.search(query_embedding, min(k, self._total_vectors))
        
        # Convert to list format
        scores = scores[0].tolist()
        indices = indices[0].tolist()
        
        results = []
        for score, idx in zip(scores, indices):
            # Skip invalid indices
            if idx == -1 or idx >= self._total_vectors:
                continue
            
            # Apply score threshold
            if score_threshold is not None and score > score_threshold:
                continue
            
            # Get metadata
            metadata = self._metadata.get(idx, {})
            
            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            # Get vector ID and text
            vector_id = self._idx_to_id.get(idx, f"idx_{idx}")
            text = metadata.get('text', '')
            
            # Get embedding if requested
            embedding = None
            if include_embeddings:
                try:
                    embedding = self._index.reconstruct(idx)
                except Exception as e:
                    logger.warning(f"Could not reconstruct embedding for index {idx}: {e}")
            
            results.append(SearchResult(
                id=vector_id,
                score=float(score),
                text=text,
                metadata=metadata.copy(),
                embedding=embedding
            ))
        
        return results
    
    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches the given filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            metadata_value = metadata[key]
            
            # Handle different filter types
            if isinstance(value, dict):
                # Range queries: {"field": {"gte": 10, "lte": 20}}
                if "gte" in value and metadata_value < value["gte"]:
                    return False
                if "gt" in value and metadata_value <= value["gt"]:
                    return False
                if "lte" in value and metadata_value > value["lte"]:
                    return False
                if "lt" in value and metadata_value >= value["lt"]:
                    return False
            elif isinstance(value, list):
                # List membership: {"field": ["value1", "value2"]}
                if metadata_value not in value:
                    return False
            else:
                # Exact match
                if metadata_value != value:
                    return False
        
        return True
    
    async def save_index(self) -> None:
        """
        Save the FAISS index and metadata to disk.
        
        Raises:
            VectorStoreError: If saving fails
        """
        if not self._is_initialized or self._index is None:
            raise VectorStoreError(
                message="Vector store not initialized",
                operation="save_index"
            )
        
        try:
            # Run in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_index_sync
            )
            
            self._last_save_time = time.time()
            logger.info(f"Saved vector store to {self._index_path}")
            
        except Exception as e:
            error_msg = f"Failed to save vector store: {str(e)}"
            logger.error(error_msg)
            
            raise VectorStoreError(
                message=error_msg,
                operation="save_index",
                details={"index_path": str(self._index_path), "error": str(e)}
            )
    
    def _save_index_sync(self) -> None:
        """Synchronous index saving implementation."""
        # Create backup of existing index
        if self._index_path.exists():
            backup_path = self._index_path.with_suffix('.index.backup')
            if backup_path.exists():
                backup_path.unlink()
            self._index_path.rename(backup_path)
        
        # Save FAISS index
        faiss.write_index(self._index, str(self._index_path))
        
        # Save metadata
        self._save_metadata_sync()
    
    async def load_index(self) -> bool:
        """
        Load FAISS index from disk if it exists.
        
        Returns:
            True if index was loaded successfully, False if not found
            
        Raises:
            VectorIndexError: If loading fails
        """
        if not self._index_path.exists():
            logger.info(f"Index file not found: {self._index_path}")
            return False
        
        try:
            # Run in thread pool to avoid blocking
            success = await asyncio.get_event_loop().run_in_executor(
                None, self._load_index_sync
            )
            
            if success:
                logger.info(f"Loaded FAISS index from {self._index_path}")
            
            return success
            
        except Exception as e:
            error_msg = f"Failed to load vector index: {str(e)}"
            logger.error(error_msg)
            
            raise VectorIndexError(
                message=error_msg,
                index_path=str(self._index_path),
                details={"error": str(e)}
            )
    
    def _load_index_sync(self) -> bool:
        """Synchronous index loading implementation."""
        try:
            # Load FAISS index
            self._index = faiss.read_index(str(self._index_path))
            
            # Verify dimension matches
            if hasattr(self._index, 'd') and self._index.d != self._dimension:
                raise VectorIndexError(
                    message=f"Index dimension {self._index.d} doesn't match configured dimension {self._dimension}",
                    index_path=str(self._index_path)
                )
            
            self._total_vectors = self._index.ntotal
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._index = None
            return False
    
    async def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if not self._metadata_path.exists():
            logger.info("Metadata file not found, starting with empty metadata")
            return
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._load_metadata_sync
            )
            
            logger.info(f"Loaded {len(self._metadata)} metadata entries")
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            # Continue with empty metadata
            self._metadata = {}
            self._id_to_idx = {}
            self._idx_to_id = {}
    
    def _load_metadata_sync(self) -> None:
        """Synchronous metadata loading implementation."""
        with open(self._metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._metadata = {int(k): v for k, v in data.get('metadata', {}).items()}
        self._id_to_idx = data.get('id_to_idx', {})
        self._idx_to_id = {int(k): v for k, v in data.get('idx_to_id', {}).items()}
        
        # Update next index counter
        self._next_idx = max(self._idx_to_id.keys()) + 1 if self._idx_to_id else 0
    
    def _save_metadata_sync(self) -> None:
        """Save metadata to disk."""
        # Create backup
        if self._metadata_path.exists():
            backup_path = self._metadata_path.with_suffix('.json.backup')
            if backup_path.exists():
                backup_path.unlink()
            self._metadata_path.rename(backup_path)
        
        # Prepare data for JSON serialization
        data = {
            'metadata': {str(k): v for k, v in self._metadata.items()},
            'id_to_idx': self._id_to_idx,
            'idx_to_id': {str(k): v for k, v in self._idx_to_id.items()},
            'total_vectors': self._total_vectors,
            'dimension': self._dimension,
            'last_save_time': time.time(),
        }
        
        # Write to file
        with open(self._metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def _auto_save(self) -> None:
        """Auto-save index and metadata if modifications exceed threshold."""
        try:
            await self.save_index()
            self._modification_count = 0
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index.
        
        Returns:
            Dictionary with index information
        """
        info = {
            "is_initialized": self._is_initialized,
            "total_vectors": self._total_vectors,
            "dimension": self._dimension,
            "index_type": self._index_type,
            "metric_type": self._metric_type,
            "index_path": str(self._index_path),
            "metadata_path": str(self._metadata_path),
            "metadata_entries": len(self._metadata),
            "modification_count": self._modification_count,
        }
        
        if self._index is not None:
            info.update({
                "index_class": self._index.__class__.__name__,
                "is_trained": getattr(self._index, 'is_trained', True),
            })
            
            # Add index-specific information
            if hasattr(self._index, 'nprobe'):
                info["nprobe"] = self._index.nprobe
            if hasattr(self._index, 'hnsw'):
                info["hnsw_m"] = self._index.hnsw.M
                info["hnsw_ef"] = self._index.hnsw.efSearch
        
        if self._last_save_time:
            info["last_save_time"] = self._last_save_time
        
        return info
    
    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            # Force garbage collection to clean up FAISS resources
            gc.collect()
        except Exception:
            pass  # Ignore cleanup errors during destruction


# Convenience function for creating vector store
def create_vector_store(
    dimension: Optional[int] = None,
    index_path: Optional[str] = None,
    settings: Optional[Settings] = None
) -> VectorStore:
    """
    Create and return a VectorStore instance.
    
    Args:
        dimension: Vector dimension
        index_path: Path to store index
        settings: Application settings
        
    Returns:
        VectorStore instance
    """
    return VectorStore(
        dimension=dimension,
        index_path=index_path,
        settings=settings
    )


# Export main classes and functions
__all__ = [
    "VectorStore",
    "SearchResult",
    "create_vector_store",
]