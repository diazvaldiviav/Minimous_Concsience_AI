"""
AdapterManager for LoRA adapter discovery, loading, and management.

This module provides functionality for discovering, loading, and managing
LoRA adapters for the MAP API. It handles adapter caching, metadata management,
and integration with the underlying model system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import weakref
import threading
from collections import OrderedDict

from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

from src.core.config import Settings
from src.core.models import AdapterInfo, ConversationAdapter
from src.core.exceptions import (
    ModelLoadError,
    ConfigurationError,
    ValidationError
)

logger = logging.getLogger(__name__)


class AdapterCache:
    """LRU cache for loaded LoRA adapters with TTL support."""
    
    def __init__(self, max_size: int = 5, ttl_seconds: int = 3600):
        """
        Initialize adapter cache.
        
        Args:
            max_size: Maximum number of adapters to cache
            ttl_seconds: Time to live for cached adapters
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        
        # Weak references to prevent memory leaks
        self._model_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
    
    def get(self, adapter_id: str) -> Optional[PeftModel]:
        """Get adapter from cache if available and not expired."""
        with self._lock:
            if adapter_id not in self._cache:
                return None
            
            # Check TTL
            access_time = self._access_times.get(adapter_id)
            if access_time and datetime.utcnow() - access_time > timedelta(seconds=self.ttl_seconds):
                self._remove(adapter_id)
                return None
            
            # Move to end (most recently used)
            cache_entry = self._cache[adapter_id]
            del self._cache[adapter_id]
            self._cache[adapter_id] = cache_entry
            self._access_times[adapter_id] = datetime.utcnow()
            
            return cache_entry.get('model')
    
    def put(self, adapter_id: str, model: PeftModel, adapter_info: AdapterInfo) -> None:
        """Add adapter to cache."""
        with self._lock:
            # Remove least recently used if at capacity
            if len(self._cache) >= self.max_size and adapter_id not in self._cache:
                self._remove_lru()
            
            self._cache[adapter_id] = {
                'model': model,
                'info': adapter_info,
                'loaded_at': datetime.utcnow()
            }
            self._access_times[adapter_id] = datetime.utcnow()
            self._model_refs[adapter_id] = model
            
            logger.info(f"Cached adapter {adapter_id}")
    
    def _remove_lru(self) -> None:
        """Remove least recently used adapter."""
        if not self._cache:
            return
        
        lru_key = next(iter(self._cache))
        self._remove(lru_key)
    
    def _remove(self, adapter_id: str) -> None:
        """Remove adapter from cache."""
        if adapter_id in self._cache:
            del self._cache[adapter_id]
        if adapter_id in self._access_times:
            del self._access_times[adapter_id]
        if adapter_id in self._model_refs:
            del self._model_refs[adapter_id]
        
        logger.debug(f"Removed adapter {adapter_id} from cache")
    
    def clear(self) -> None:
        """Clear all cached adapters."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._model_refs.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'adapters': list(self._cache.keys()),
                'hit_rate': getattr(self, '_hit_rate', 0.0)
            }


class AdapterManager:
    """Manages loading and searching LoRA adapters."""
    
    def __init__(self, settings: Settings):
        """
        Initialize adapter manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.cache = AdapterCache(
            max_size=getattr(settings, 'adapter_cache_size', 5),
            ttl_seconds=getattr(settings, 'adapter_cache_ttl', 3600)
        )
        self._adapter_registry: Dict[str, AdapterInfo] = {}
        self._base_model: Optional[AutoModelForCausalLM] = None
        self._registry_lock = threading.RLock()
        
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        
    async def initialize(self) -> None:
        """Initialize the adapter manager."""
        try:
            logger.info("Initializing AdapterManager")
            await self._load_base_model()
            await self._scan_adapters()
            logger.info(f"AdapterManager initialized with {len(self._adapter_registry)} adapters")
            
        except Exception as e:
            logger.error(f"Failed to initialize AdapterManager: {e}", exc_info=True)
            raise ConfigurationError(f"AdapterManager initialization failed: {e}")
    
    async def _load_base_model(self) -> None:
        """Load the base model for adapter attachment."""
        try:
            from src.memory.base_model import BaseModelManager
            
            model_manager = BaseModelManager(self.settings)
            await model_manager.initialize()
            
            self._base_model = model_manager.model
            logger.info("Base model loaded for adapter management")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}", exc_info=True)
            raise ModelLoadError(f"Base model loading failed: {e}")
    
    async def _scan_adapters(self) -> None:
        """Scan for available adapters and build registry."""
        try:
            adapters_dir = Path(self.settings.data_dir) / "adapters"
            if not adapters_dir.exists():
                logger.warning(f"Adapters directory not found: {adapters_dir}")
                return
            
            adapter_count = 0
            for adapter_path in adapters_dir.glob("**/adapter_config.json"):
                try:
                    adapter_info = await self._load_adapter_info(adapter_path.parent)
                    if adapter_info:
                        self._adapter_registry[adapter_info.adapter_id] = adapter_info
                        adapter_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to load adapter info from {adapter_path}: {e}")
            
            logger.info(f"Scanned {adapter_count} adapters")
            
        except Exception as e:
            logger.error(f"Failed to scan adapters: {e}", exc_info=True)
    
    async def _load_adapter_info(self, adapter_path: Path) -> Optional[AdapterInfo]:
        """Load adapter information from directory."""
        try:
            config_path = adapter_path / "adapter_config.json"
            metadata_path = adapter_path / "adapter_metadata.json"
            
            if not config_path.exists():
                return None
            
            # Load adapter configuration
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Load metadata if available
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Create AdapterInfo
            adapter_info = AdapterInfo(
                adapter_id=metadata.get('adapter_id', adapter_path.name),
                conversation_id=metadata.get('conversation_id', ''),
                provider=metadata.get('provider', 'unknown'),
                external_user_id=metadata.get('external_user_id', ''),
                external_chat_id=metadata.get('external_chat_id'),
                adapter_path=str(adapter_path),
                topic=metadata.get('topic'),
                turn_range=metadata.get('turn_range', {}),
                quality_score=metadata.get('quality_score', 0.0),
                metadata=metadata
            )
            
            return adapter_info
            
        except Exception as e:
            logger.error(f"Failed to load adapter info from {adapter_path}: {e}")
            return None
    
    async def find_user_adapters(
        self,
        provider: str,
        user_id: str,
        chat_id: Optional[str] = None
    ) -> List[AdapterInfo]:
        """
        Find relevant adapters for user/chat.
        
        Args:
            provider: Provider identifier (anthropic/openai)
            user_id: External user identifier
            chat_id: Optional external chat identifier
            
        Returns:
            List of relevant AdapterInfo objects
        """
        try:
            with self._registry_lock:
                adapters = []
                
                for adapter_info in self._adapter_registry.values():
                    # Match provider and user
                    if (adapter_info.provider == provider and 
                        adapter_info.external_user_id == user_id):
                        
                        # Filter by chat if specified
                        if chat_id is None or adapter_info.external_chat_id == chat_id:
                            adapters.append(adapter_info)
                
                # Sort by quality score (descending) and creation time (newest first)
                adapters.sort(
                    key=lambda x: (x.quality_score, x.created_at),
                    reverse=True
                )
                
                logger.debug(f"Found {len(adapters)} adapters for {provider}:{user_id}:{chat_id}")
                return adapters
                
        except Exception as e:
            logger.error(f"Failed to find adapters: {e}", exc_info=True)
            return []
    
    async def load_adapter(self, adapter_info: AdapterInfo) -> Optional[PeftModel]:
        """
        Load a specific adapter.
        
        Args:
            adapter_info: Adapter information
            
        Returns:
            Loaded PeftModel or None if failed
        """
        try:
            # Check cache first
            cached_model = self.cache.get(adapter_info.adapter_id)
            if cached_model:
                self._cache_hits += 1
                adapter_info.last_accessed = datetime.utcnow()
                logger.debug(f"Cache hit for adapter {adapter_info.adapter_id}")
                return cached_model
            
            self._cache_misses += 1
            
            # Load adapter from disk
            logger.info(f"Loading adapter {adapter_info.adapter_id} from {adapter_info.adapter_path}")
            
            if not self._base_model:
                raise ModelLoadError("Base model not initialized")
            
            adapter_path = Path(adapter_info.adapter_path)
            if not adapter_path.exists():
                raise ModelLoadError(f"Adapter path not found: {adapter_path}")
            
            # Load PEFT model
            model = PeftModel.from_pretrained(
                self._base_model,
                adapter_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Cache the loaded model
            self.cache.put(adapter_info.adapter_id, model, adapter_info)
            adapter_info.last_accessed = datetime.utcnow()
            
            logger.info(f"Successfully loaded adapter {adapter_info.adapter_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_info.adapter_id}: {e}", exc_info=True)
            return None
    
    async def get_adapter_by_id(self, adapter_id: str) -> Optional[AdapterInfo]:
        """Get adapter info by ID."""
        with self._registry_lock:
            return self._adapter_registry.get(adapter_id)
    
    async def register_adapter(self, adapter_info: AdapterInfo) -> None:
        """Register a new adapter."""
        with self._registry_lock:
            self._adapter_registry[adapter_info.adapter_id] = adapter_info
            logger.info(f"Registered adapter {adapter_info.adapter_id}")
    
    async def unload_adapter(self, adapter_id: str) -> None:
        """Unload adapter from cache."""
        self.cache._remove(adapter_id)
        logger.debug(f"Unloaded adapter {adapter_id}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_info': self.cache.get_stats(),
            'registry_size': len(self._adapter_registry)
        }
    
    async def warmup_cache(self, adapter_ids: List[str]) -> None:
        """Warm up cache with frequently used adapters."""
        try:
            logger.info(f"Warming up cache with {len(adapter_ids)} adapters")
            
            for adapter_id in adapter_ids:
                adapter_info = await self.get_adapter_by_id(adapter_id)
                if adapter_info:
                    await self.load_adapter(adapter_info)
                    
            logger.info("Cache warmup completed")
            
        except Exception as e:
            logger.error(f"Cache warmup failed: {e}", exc_info=True)
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.cache.clear()
            logger.info("AdapterManager cleanup completed")
        except Exception as e:
            logger.error(f"AdapterManager cleanup failed: {e}", exc_info=True)