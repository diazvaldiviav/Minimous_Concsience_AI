"""
Tests for MAP API AdapterManager.

This module tests the LoRA adapter discovery, loading, and management
functionality of the MAP API.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import json
import tempfile
import shutil

from src.api.map.adapter_manager import AdapterManager, AdapterCache
from src.core.config import Settings
from src.core.models import AdapterInfo
from src.core.exceptions import ModelLoadError, ConfigurationError


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.data_dir = "/tmp/test_data"
    settings.model = Mock()
    settings.model.name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    return settings


@pytest.fixture
def temp_adapters_dir():
    """Create temporary adapters directory for testing."""
    temp_dir = tempfile.mkdtemp()
    adapters_dir = Path(temp_dir) / "adapters"
    adapters_dir.mkdir(parents=True)
    
    # Create test adapter structure
    test_adapter_dir = adapters_dir / "test_adapter_1"
    test_adapter_dir.mkdir()
    
    # Create adapter config
    config = {
        "base_model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "peft_type": "LORA",
        "r": 4,
        "lora_alpha": 32
    }
    
    with open(test_adapter_dir / "adapter_config.json", 'w') as f:
        json.dump(config, f)
    
    # Create adapter metadata
    metadata = {
        "adapter_id": "test_adapter_1",
        "conversation_id": "conv_123",
        "provider": "anthropic",
        "external_user_id": "user_456",
        "external_chat_id": "chat_789",
        "topic": "physics",
        "turn_range": {"from_turn": 1, "to_turn": 50},
        "quality_score": 0.92
    }
    
    with open(test_adapter_dir / "adapter_metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestAdapterCache:
    """Test adapter cache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initialization with default parameters."""
        cache = AdapterCache()
        
        assert cache.max_size == 5
        assert cache.ttl_seconds == 3600
        assert len(cache._cache) == 0
    
    def test_cache_put_and_get(self):
        """Test putting and getting items from cache."""
        cache = AdapterCache(max_size=2)
        
        # Create mock model and adapter info
        mock_model = Mock()
        adapter_info = AdapterInfo(
            adapter_id="test_1",
            conversation_id="conv_1",
            provider="anthropic",
            external_user_id="user_1",
            adapter_path="/path/to/adapter"
        )
        
        # Put item in cache
        cache.put("test_1", mock_model, adapter_info)
        
        # Get item from cache
        cached_model = cache.get("test_1")
        
        assert cached_model == mock_model
        assert len(cache._cache) == 1
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = AdapterCache(max_size=2)
        
        mock_model_1 = Mock()
        mock_model_2 = Mock()
        mock_model_3 = Mock()
        
        adapter_info_1 = AdapterInfo(
            adapter_id="test_1",
            conversation_id="conv_1", 
            provider="anthropic",
            external_user_id="user_1",
            adapter_path="/path/1"
        )
        
        adapter_info_2 = AdapterInfo(
            adapter_id="test_2",
            conversation_id="conv_2",
            provider="anthropic", 
            external_user_id="user_2",
            adapter_path="/path/2"
        )
        
        adapter_info_3 = AdapterInfo(
            adapter_id="test_3",
            conversation_id="conv_3",
            provider="anthropic",
            external_user_id="user_3", 
            adapter_path="/path/3"
        )
        
        # Fill cache to capacity
        cache.put("test_1", mock_model_1, adapter_info_1)
        cache.put("test_2", mock_model_2, adapter_info_2)
        
        # Access first item to make it most recently used
        cache.get("test_1")
        
        # Add third item, should evict test_2 (least recently used)
        cache.put("test_3", mock_model_3, adapter_info_3)
        
        assert cache.get("test_1") == mock_model_1
        assert cache.get("test_2") is None  # Evicted
        assert cache.get("test_3") == mock_model_3
        assert len(cache._cache) == 2
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = AdapterCache(max_size=2)
        
        mock_model = Mock()
        adapter_info = AdapterInfo(
            adapter_id="test_1",
            conversation_id="conv_1",
            provider="anthropic", 
            external_user_id="user_1",
            adapter_path="/path/1"
        )
        
        cache.put("test_1", mock_model, adapter_info)
        
        stats = cache.get_stats()
        
        assert stats['size'] == 1
        assert stats['max_size'] == 2
        assert 'test_1' in stats['adapters']


class TestAdapterManager:
    """Test adapter manager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_settings, temp_adapters_dir):
        """Test adapter manager initialization."""
        mock_settings.data_dir = temp_adapters_dir
        
        adapter_manager = AdapterManager(mock_settings)
        
        with patch('src.api.map.adapter_manager.BaseModelManager') as mock_base_model:
            mock_base_model_instance = Mock()
            mock_base_model_instance.model = Mock()
            mock_base_model.return_value = mock_base_model_instance
            mock_base_model_instance.initialize = AsyncMock()
            
            await adapter_manager.initialize()
            
            assert len(adapter_manager._adapter_registry) == 1
            assert "test_adapter_1" in adapter_manager._adapter_registry
    
    @pytest.mark.asyncio
    async def test_find_user_adapters(self, mock_settings, temp_adapters_dir):
        """Test finding adapters for specific user."""
        mock_settings.data_dir = temp_adapters_dir
        
        adapter_manager = AdapterManager(mock_settings)
        
        with patch('src.api.map.adapter_manager.BaseModelManager') as mock_base_model:
            mock_base_model_instance = Mock()
            mock_base_model_instance.model = Mock()
            mock_base_model.return_value = mock_base_model_instance
            mock_base_model_instance.initialize = AsyncMock()
            
            await adapter_manager.initialize()
            
            # Find adapters for test user
            adapters = await adapter_manager.find_user_adapters(
                provider="anthropic",
                user_id="user_456",
                chat_id="chat_789"
            )
            
            assert len(adapters) == 1
            assert adapters[0].adapter_id == "test_adapter_1"
            assert adapters[0].external_user_id == "user_456"
            assert adapters[0].topic == "physics"
    
    @pytest.mark.asyncio
    async def test_find_user_adapters_no_match(self, mock_settings, temp_adapters_dir):
        """Test finding adapters when no match exists."""
        mock_settings.data_dir = temp_adapters_dir
        
        adapter_manager = AdapterManager(mock_settings)
        
        with patch('src.api.map.adapter_manager.BaseModelManager') as mock_base_model:
            mock_base_model_instance = Mock()
            mock_base_model_instance.model = Mock()
            mock_base_model.return_value = mock_base_model_instance
            mock_base_model_instance.initialize = AsyncMock()
            
            await adapter_manager.initialize()
            
            # Find adapters for non-existent user
            adapters = await adapter_manager.find_user_adapters(
                provider="openai",
                user_id="nonexistent_user",
                chat_id=None
            )
            
            assert len(adapters) == 0
    
    @pytest.mark.asyncio
    async def test_load_adapter_success(self, mock_settings, temp_adapters_dir):
        """Test successful adapter loading."""
        mock_settings.data_dir = temp_adapters_dir
        
        adapter_manager = AdapterManager(mock_settings)
        
        with patch('src.api.map.adapter_manager.BaseModelManager') as mock_base_model, \
             patch('src.api.map.adapter_manager.PeftModel') as mock_peft_model:
            
            mock_base_model_instance = Mock()
            mock_base_model_instance.model = Mock()
            mock_base_model.return_value = mock_base_model_instance
            mock_base_model_instance.initialize = AsyncMock()
            
            mock_loaded_model = Mock()
            mock_peft_model.from_pretrained.return_value = mock_loaded_model
            
            await adapter_manager.initialize()
            
            adapter_info = adapter_manager._adapter_registry["test_adapter_1"]
            
            # Load adapter
            loaded_model = await adapter_manager.load_adapter(adapter_info)
            
            assert loaded_model == mock_loaded_model
            mock_peft_model.from_pretrained.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_adapter_cache_hit(self, mock_settings, temp_adapters_dir):
        """Test adapter loading with cache hit."""
        mock_settings.data_dir = temp_adapters_dir
        
        adapter_manager = AdapterManager(mock_settings)
        
        with patch('src.api.map.adapter_manager.BaseModelManager') as mock_base_model:
            mock_base_model_instance = Mock()
            mock_base_model_instance.model = Mock()
            mock_base_model.return_value = mock_base_model_instance
            mock_base_model_instance.initialize = AsyncMock()
            
            await adapter_manager.initialize()
            
            adapter_info = adapter_manager._adapter_registry["test_adapter_1"]
            
            # Put model in cache manually
            mock_cached_model = Mock()
            adapter_manager.cache.put("test_adapter_1", mock_cached_model, adapter_info)
            
            # Load adapter (should hit cache)
            loaded_model = await adapter_manager.load_adapter(adapter_info)
            
            assert loaded_model == mock_cached_model
            assert adapter_manager._cache_hits == 1
    
    @pytest.mark.asyncio
    async def test_load_adapter_failure(self, mock_settings, temp_adapters_dir):
        """Test adapter loading failure."""
        mock_settings.data_dir = temp_adapters_dir
        
        adapter_manager = AdapterManager(mock_settings)
        
        with patch('src.api.map.adapter_manager.BaseModelManager') as mock_base_model, \
             patch('src.api.map.adapter_manager.PeftModel') as mock_peft_model:
            
            mock_base_model_instance = Mock()
            mock_base_model_instance.model = Mock()
            mock_base_model.return_value = mock_base_model_instance
            mock_base_model_instance.initialize = AsyncMock()
            
            mock_peft_model.from_pretrained.side_effect = Exception("Load failed")
            
            await adapter_manager.initialize()
            
            adapter_info = adapter_manager._adapter_registry["test_adapter_1"]
            
            # Load adapter (should fail)
            loaded_model = await adapter_manager.load_adapter(adapter_info)
            
            assert loaded_model is None
            assert adapter_manager._cache_misses == 1
    
    def test_get_cache_stats(self, mock_settings):
        """Test getting cache statistics."""
        adapter_manager = AdapterManager(mock_settings)
        
        # Simulate some cache activity
        adapter_manager._cache_hits = 10
        adapter_manager._cache_misses = 5
        
        stats = adapter_manager.get_cache_stats()
        
        assert stats['cache_hits'] == 10
        assert stats['cache_misses'] == 5
        assert stats['hit_rate'] == 10/15  # 10/(10+5)
        assert 'cache_info' in stats
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, mock_settings):
        """Test adapter manager initialization failure."""
        adapter_manager = AdapterManager(mock_settings)
        
        with patch('src.api.map.adapter_manager.BaseModelManager') as mock_base_model:
            mock_base_model.side_effect = Exception("Init failed")
            
            with pytest.raises(ConfigurationError):
                await adapter_manager.initialize()
    
    @pytest.mark.asyncio
    async def test_warmup_cache(self, mock_settings, temp_adapters_dir):
        """Test cache warmup functionality."""
        mock_settings.data_dir = temp_adapters_dir
        
        adapter_manager = AdapterManager(mock_settings)
        
        with patch('src.api.map.adapter_manager.BaseModelManager') as mock_base_model:
            mock_base_model_instance = Mock()
            mock_base_model_instance.model = Mock()
            mock_base_model.return_value = mock_base_model_instance
            mock_base_model_instance.initialize = AsyncMock()
            
            await adapter_manager.initialize()
            
            # Test warmup with adapter IDs
            await adapter_manager.warmup_cache(["test_adapter_1", "nonexistent"])
            
            # Should handle both existing and non-existing adapters gracefully
            # No assertions needed as method should not raise exceptions
    
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_settings):
        """Test adapter manager cleanup."""
        adapter_manager = AdapterManager(mock_settings)
        
        # Add some items to cache
        mock_model = Mock()
        adapter_info = AdapterInfo(
            adapter_id="test_1",
            conversation_id="conv_1",
            provider="anthropic",
            external_user_id="user_1",
            adapter_path="/path/1"
        )
        
        adapter_manager.cache.put("test_1", mock_model, adapter_info)
        
        # Cleanup
        await adapter_manager.cleanup()
        
        # Cache should be empty after cleanup
        assert len(adapter_manager.cache._cache) == 0