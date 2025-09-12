"""
Performance optimization system for SC Memory System.

This module provides system-wide performance optimizations including
model caching, batch operations, query optimization, and memory management
for MVP demonstration readiness.
"""

import asyncio
import logging
import time
import gc
import psutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import threading
from pathlib import Path

import torch
import faiss
import numpy as np
from transformers import AutoTokenizer

from src.core.config import Settings
from src.core.exceptions import ConfigurationError, ModelLoadError

logger = logging.getLogger(__name__)


class ModelCacheManager:
    """Manages LRU model cache with warm-up capabilities."""
    
    def __init__(self, max_size: int = 5, warmup_models: List[str] = None):
        """
        Initialize model cache manager.
        
        Args:
            max_size: Maximum number of models to cache
            warmup_models: Models to warm up on startup
        """
        self.max_size = max_size
        self.warmup_models = warmup_models or []
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    async def get_model(self, model_key: str, loader_func = None) -> Optional[Any]:
        """Get model from cache or load if not cached."""
        with self._lock:
            if model_key in self._cache:
                # Cache hit
                self.hits += 1
                self._move_to_end(model_key)
                self._cache[model_key]['last_accessed'] = datetime.utcnow()
                return self._cache[model_key]['model']
            
            # Cache miss
            self.misses += 1
            
            if loader_func:
                model = await loader_func()
                if model:
                    await self._cache_model(model_key, model)
                    return model
            
            return None
    
    async def _cache_model(self, model_key: str, model: Any) -> None:
        """Add model to cache."""
        with self._lock:
            # Remove LRU if at capacity
            if len(self._cache) >= self.max_size and model_key not in self._cache:
                await self._evict_lru()
            
            self._cache[model_key] = {
                'model': model,
                'cached_at': datetime.utcnow(),
                'last_accessed': datetime.utcnow(),
                'access_count': 1
            }
            
            self._move_to_end(model_key)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used model."""
        if self._access_order:
            lru_key = self._access_order[0]
            await self.remove_model(lru_key)
            self.evictions += 1
    
    async def remove_model(self, model_key: str) -> None:
        """Remove model from cache."""
        with self._lock:
            if model_key in self._cache:
                # Cleanup model resources
                model_data = self._cache[model_key]
                if hasattr(model_data['model'], 'cpu'):
                    model_data['model'].cpu()
                
                del self._cache[model_key]
                
                if model_key in self._access_order:
                    self._access_order.remove(model_key)
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _move_to_end(self, model_key: str) -> None:
        """Move model key to end of access order."""
        if model_key in self._access_order:
            self._access_order.remove(model_key)
        self._access_order.append(model_key)
    
    async def warmup_cache(self) -> None:
        """Warm up cache with frequently used models."""
        try:
            logger.info(f"Warming up model cache with {len(self.warmup_models)} models")
            
            for model_key in self.warmup_models:
                try:
                    # This would be implemented with actual model loaders
                    logger.info(f"Warming up model: {model_key}")
                    # await self.get_model(model_key, some_loader_func)
                except Exception as e:
                    logger.warning(f"Failed to warm up model {model_key}: {e}")
            
            logger.info("Model cache warmup completed")
            
        except Exception as e:
            logger.error(f"Model cache warmup failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'cached_models': list(self._cache.keys())
        }


class BatchProcessor:
    """Optimizes batch operations for embeddings and similarity search."""
    
    def __init__(self, batch_size: int = 32, max_concurrency: int = 4):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Default batch size for operations
            max_concurrency: Maximum concurrent batches
        """
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
        # Performance metrics
        self.total_items_processed = 0
        self.total_batches_processed = 0
        self.total_processing_time = 0.0
    
    async def process_embeddings_batch(
        self,
        texts: List[str],
        embedding_func,
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """Process embeddings in optimized batches."""
        batch_size = batch_size or self.batch_size
        results = []
        
        start_time = time.time()
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                async with self.semaphore:
                    batch_embeddings = await embedding_func(batch_texts)
                    results.extend(batch_embeddings)
                    
                    self.total_batches_processed += 1
            
            self.total_items_processed += len(texts)
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.debug(f"Processed {len(texts)} embeddings in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch embedding processing failed: {e}")
            return []
    
    async def process_similarity_search_batch(
        self,
        queries: List[np.ndarray],
        search_func,
        batch_size: Optional[int] = None
    ) -> List[List[Tuple[int, float]]]:
        """Process similarity searches in optimized batches."""
        batch_size = batch_size or self.batch_size
        results = []
        
        start_time = time.time()
        
        try:
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                
                async with self.semaphore:
                    batch_results = await search_func(batch_queries)
                    results.extend(batch_results)
                    
                    self.total_batches_processed += 1
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.debug(f"Processed {len(queries)} searches in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch similarity search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        avg_processing_time = (
            self.total_processing_time / self.total_batches_processed
            if self.total_batches_processed > 0 else 0.0
        )
        
        return {
            'total_items_processed': self.total_items_processed,
            'total_batches_processed': self.total_batches_processed,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time_per_batch': avg_processing_time,
            'batch_size': self.batch_size,
            'max_concurrency': self.max_concurrency
        }


class QueryOptimizer:
    """Optimizes MAP queries for sub-200ms response times."""
    
    def __init__(self, cache_ttl: int = 300):
        """
        Initialize query optimizer.
        
        Args:
            cache_ttl: Cache TTL in seconds
        """
        self.cache_ttl = cache_ttl
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        
        # Performance tracking
        self.query_times: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def optimize_query(
        self,
        query_key: str,
        query_func,
        *args,
        **kwargs
    ) -> Any:
        """Optimize query execution with caching."""
        start_time = time.time()
        
        try:
            # Check cache
            cached_result = self._get_cached_result(query_key)
            if cached_result is not None:
                self.cache_hits += 1
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                return cached_result
            
            # Execute query
            self.cache_misses += 1
            result = await query_func(*args, **kwargs)
            
            # Cache result
            self._cache_result(query_key, result)
            
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            
            logger.debug(f"Query {query_key} completed in {query_time*1000:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Query optimization failed for {query_key}: {e}")
            raise
    
    def _get_cached_result(self, query_key: str) -> Any:
        """Get cached result if valid."""
        with self._cache_lock:
            if query_key not in self._query_cache:
                return None
            
            cache_entry = self._query_cache[query_key]
            
            # Check TTL
            if time.time() - cache_entry['cached_at'] > self.cache_ttl:
                del self._query_cache[query_key]
                return None
            
            return cache_entry['result']
    
    def _cache_result(self, query_key: str, result: Any) -> None:
        """Cache query result."""
        with self._cache_lock:
            self._query_cache[query_key] = {
                'result': result,
                'cached_at': time.time()
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        if not self.query_times:
            return {
                'avg_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'total_queries': 0,
                'cache_hit_rate': 0.0
            }
        
        sorted_times = sorted(self.query_times)
        avg_latency = sum(self.query_times) / len(self.query_times) * 1000
        p95_index = int(len(sorted_times) * 0.95)
        p95_latency = sorted_times[p95_index] * 1000 if p95_index < len(sorted_times) else 0.0
        
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'total_queries': len(self.query_times),
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._query_cache)
        }


class MemoryManager:
    """Manages memory optimization and garbage collection."""
    
    def __init__(self, memory_threshold: float = 0.8, gc_interval: int = 300):
        """
        Initialize memory manager.
        
        Args:
            memory_threshold: Memory usage threshold to trigger cleanup (0.0-1.0)
            gc_interval: Garbage collection interval in seconds
        """
        self.memory_threshold = memory_threshold
        self.gc_interval = gc_interval
        self._last_gc = time.time()
        self._gc_stats = defaultdict(int)
    
    async def monitor_memory(self) -> Dict[str, Any]:
        """Monitor system memory usage."""
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            
            # Get GPU memory if available
            gpu_memory = None
            if torch.cuda.is_available():
                gpu_memory = {
                    'allocated': torch.cuda.memory_allocated(),
                    'cached': torch.cuda.memory_reserved(),
                    'max_allocated': torch.cuda.max_memory_allocated()
                }
            
            memory_info = {
                'system_memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percentage': memory.percent
                },
                'gpu_memory': gpu_memory
            }
            
            # Trigger cleanup if needed
            if memory.percent > self.memory_threshold * 100:
                await self._trigger_cleanup()
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            return {}
    
    async def _trigger_cleanup(self) -> None:
        """Trigger memory cleanup."""
        try:
            logger.info("Memory threshold exceeded, triggering cleanup")
            
            # Force garbage collection
            collected = gc.collect()
            self._gc_stats['manual_collections'] += 1
            self._gc_stats['objects_collected'] += collected
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Cleanup completed, collected {collected} objects")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    async def periodic_cleanup(self) -> None:
        """Perform periodic cleanup if interval has passed."""
        current_time = time.time()
        
        if current_time - self._last_gc > self.gc_interval:
            await self._trigger_cleanup()
            self._last_gc = current_time
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        return {
            'gc_stats': dict(self._gc_stats),
            'memory_threshold': self.memory_threshold,
            'gc_interval': self.gc_interval,
            'last_gc': self._last_gc
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, settings: Settings):
        """
        Initialize performance optimizer.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Initialize optimization components
        self.model_cache = ModelCacheManager(
            max_size=getattr(settings, 'model_cache_size', 5),
            warmup_models=getattr(settings, 'warmup_models', [])
        )
        
        self.batch_processor = BatchProcessor(
            batch_size=getattr(settings, 'batch_size', 32),
            max_concurrency=getattr(settings, 'max_concurrency', 4)
        )
        
        self.query_optimizer = QueryOptimizer(
            cache_ttl=getattr(settings, 'query_cache_ttl', 300)
        )
        
        self.memory_manager = MemoryManager(
            memory_threshold=getattr(settings, 'memory_threshold', 0.8),
            gc_interval=getattr(settings, 'gc_interval', 300)
        )
    
    async def initialize(self) -> None:
        """Initialize performance optimizations."""
        try:
            logger.info("Initializing performance optimizations")
            
            # Warm up model cache
            await self.model_cache.warmup_cache()
            
            # Initial memory check
            await self.memory_manager.monitor_memory()
            
            logger.info("Performance optimizations initialized")
            
        except Exception as e:
            logger.error(f"Performance optimization initialization failed: {e}")
            raise ConfigurationError(f"Performance optimization failed: {e}")
    
    async def optimize_map_query(self, query_key: str, query_func, *args, **kwargs) -> Any:
        """Optimize MAP query execution."""
        return await self.query_optimizer.optimize_query(
            query_key, query_func, *args, **kwargs
        )
    
    async def periodic_maintenance(self) -> None:
        """Perform periodic maintenance tasks."""
        try:
            # Memory cleanup
            await self.memory_manager.periodic_cleanup()
            
            # Monitor memory
            await self.memory_manager.monitor_memory()
            
        except Exception as e:
            logger.error(f"Periodic maintenance failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'model_cache': self.model_cache.get_stats(),
            'batch_processor': self.batch_processor.get_stats(),
            'query_optimizer': self.query_optimizer.get_performance_stats(),
            'memory_manager': self.memory_manager.get_memory_stats(),
            'timestamp': datetime.utcnow().isoformat()
        }