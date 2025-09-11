"""
Test suite for VectorStore.

This module contains comprehensive tests for FAISS vector store operations,
including indexing, searching, persistence, and error handling.
"""

import asyncio
import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.config import Settings
from src.core.exceptions import (
    VectorStoreError,
    VectorIndexError,
    SearchError,
    ValidationError,
)
from src.memory.vector_store import VectorStore, SearchResult, create_vector_store


class TestVectorStore:
    """Test cases for VectorStore."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def settings(self, temp_dir):
        """Create test settings."""
        return Settings(
            vector_store=Settings.VectorStoreConfig(
                dimension=128,
                index_path=temp_dir / "test.index",
                metadata_path=temp_dir / "metadata.json",
                data_dir=temp_dir / "vector_store",
                index_factory="Flat",
                metric_type="METRIC_L2"
            )
        )
    
    @pytest.fixture
    def vector_store(self, settings):
        """Create VectorStore instance for testing."""
        return VectorStore(settings=settings)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)  # For reproducible tests
        return np.random.randn(10, 128).astype(np.float32)
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return [
            {"text": f"Sample text {i}", "category": "test", "importance": 0.5 + i * 0.1}
            for i in range(10)
        ]
    
    def test_init_default_settings(self):
        """Test initialization with default settings."""
        store = VectorStore()
        assert store._dimension is not None
        assert not store._is_initialized
        assert store._index is None
        assert isinstance(store._metadata, dict)
        assert isinstance(store._id_to_idx, dict)
        assert isinstance(store._idx_to_id, dict)
    
    def test_init_custom_parameters(self, temp_dir, settings):
        """Test initialization with custom parameters."""
        store = VectorStore(
            dimension=256,
            index_path=str(temp_dir / "custom.index"),
            settings=settings
        )
        assert store._dimension == 256
        assert store._index_path == temp_dir / "custom.index"
    
    @patch('faiss.IndexFlatL2')
    def test_create_index_flat_l2(self, mock_index_class, vector_store):
        """Test creation of Flat L2 index."""
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        
        vector_store._settings.vector_store.index_factory = "Flat"
        vector_store._settings.vector_store.metric_type = "METRIC_L2"
        
        result = vector_store._create_index()
        
        mock_index_class.assert_called_once_with(vector_store._dimension)
        assert result == mock_index
    
    @patch('faiss.IndexFlatIP')
    def test_create_index_flat_ip(self, mock_index_class, vector_store):
        """Test creation of Flat Inner Product index."""
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        
        vector_store._settings.vector_store.index_factory = "Flat"
        vector_store._settings.vector_store.metric_type = "METRIC_INNER_PRODUCT"
        
        result = vector_store._create_index()
        
        mock_index_class.assert_called_once_with(vector_store._dimension)
        assert result == mock_index
    
    @patch('faiss.IndexFlatL2')
    @patch('faiss.IndexIVFFlat')
    def test_create_index_ivf(self, mock_ivf_class, mock_flat_class, vector_store):
        """Test creation of IVF index."""
        mock_quantizer = Mock()
        mock_flat_class.return_value = mock_quantizer
        
        mock_ivf_index = Mock()
        mock_ivf_index.nprobe = 8
        mock_ivf_class.return_value = mock_ivf_index
        
        vector_store._settings.vector_store.index_factory = "IVF100,Flat"
        vector_store._settings.vector_store.metric_type = "METRIC_L2"
        
        result = vector_store._create_index()
        
        mock_flat_class.assert_called_once_with(vector_store._dimension)
        mock_ivf_class.assert_called_once_with(mock_quantizer, vector_store._dimension, 100)
        assert result == mock_ivf_index
        assert mock_ivf_index.nprobe == vector_store._settings.vector_store.nprobe
    
    @patch('faiss.IndexHNSWFlat')
    def test_create_index_hnsw(self, mock_hnsw_class, vector_store):
        """Test creation of HNSW index."""
        mock_index = Mock()
        mock_hnsw = Mock()
        mock_index.hnsw = mock_hnsw
        mock_hnsw_class.return_value = mock_index
        
        vector_store._settings.vector_store.index_factory = "HNSW32"
        
        result = vector_store._create_index()
        
        mock_hnsw_class.assert_called_once_with(
            vector_store._dimension,
            vector_store._settings.vector_store.m
        )
        assert mock_hnsw.efConstruction == vector_store._settings.vector_store.efconstruction
        assert result == mock_index
    
    @patch('faiss.index_factory')
    def test_create_index_factory_string(self, mock_factory, vector_store):
        """Test creation using factory string."""
        mock_index = Mock()
        mock_factory.return_value = mock_index
        
        vector_store._settings.vector_store.index_factory = "IVF1024,PQ64"
        
        result = vector_store._create_index()
        
        mock_factory.assert_called_once_with(
            vector_store._dimension,
            "IVF1024,PQ64",
            1  # METRIC_L2
        )
        assert result == mock_index
    
    def test_create_index_invalid_factory(self, vector_store):
        """Test index creation with invalid factory string."""
        vector_store._settings.vector_store.index_factory = "InvalidFactory"
        
        with patch('faiss.index_factory', side_effect=Exception("Invalid factory")):
            with pytest.raises(VectorIndexError) as exc_info:
                vector_store._create_index()
            
            assert "Invalid FAISS index factory string" in str(exc_info.value)
    
    @patch('src.memory.vector_store.VectorStore._create_index')
    @patch('src.memory.vector_store.VectorStore.load_index')
    @patch('src.memory.vector_store.VectorStore._load_metadata')
    @pytest.mark.asyncio
    async def test_initialize_new_index(self, mock_load_metadata, mock_load_index, mock_create_index, vector_store):
        """Test initialization with new index."""
        mock_load_index.return_value = False  # No existing index
        mock_index = Mock()
        mock_create_index.return_value = mock_index
        mock_load_metadata.return_value = None
        
        await vector_store.initialize()
        
        assert vector_store._is_initialized
        assert vector_store._index == mock_index
        mock_create_index.assert_called_once()
        mock_load_metadata.assert_called_once()
    
    @patch('src.memory.vector_store.VectorStore.load_index')
    @patch('src.memory.vector_store.VectorStore._load_metadata')
    @pytest.mark.asyncio
    async def test_initialize_existing_index(self, mock_load_metadata, mock_load_index, vector_store):
        """Test initialization with existing index."""
        mock_load_index.return_value = True  # Existing index found
        mock_load_metadata.return_value = None
        
        await vector_store.initialize()
        
        assert vector_store._is_initialized
        mock_load_metadata.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, vector_store):
        """Test initialization when already initialized."""
        vector_store._is_initialized = True
        
        await vector_store.initialize()
        
        # Should exit early without doing work
        assert vector_store._is_initialized
    
    def test_validate_consistency_matching_sizes(self, vector_store):
        """Test consistency validation with matching sizes."""
        vector_store._index = Mock()
        vector_store._index.ntotal = 5
        vector_store._metadata = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
        vector_store._idx_to_id = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
        
        vector_store._validate_consistency()
        
        assert vector_store._total_vectors == 5
        assert vector_store._next_idx == 5
    
    def test_validate_consistency_excess_metadata(self, vector_store):
        """Test consistency validation with excess metadata."""
        vector_store._index = Mock()
        vector_store._index.ntotal = 3
        vector_store._metadata = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
        vector_store._idx_to_id = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
        vector_store._id_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
        
        vector_store._validate_consistency()
        
        # Should have trimmed excess metadata
        assert len(vector_store._metadata) == 3
        assert len(vector_store._idx_to_id) == 3
        assert len(vector_store._id_to_idx) == 3
        assert vector_store._total_vectors == 3
    
    @pytest.mark.asyncio
    async def test_add_embeddings_not_initialized(self, vector_store, sample_embeddings, sample_metadata):
        """Test adding embeddings initializes store if needed."""
        with patch.object(vector_store, 'initialize') as mock_init:
            mock_init.return_value = None
            vector_store._is_initialized = False
            
            # Mock the sync method to avoid actual FAISS operations
            with patch.object(vector_store, '_add_embeddings_sync', return_value=['id1', 'id2']) as mock_sync:
                result = await vector_store.add_embeddings(sample_embeddings[:2], sample_metadata[:2])
                
                mock_init.assert_called_once()
                assert result == ['id1', 'id2']
    
    @pytest.mark.asyncio
    async def test_add_embeddings_validation_errors(self, vector_store):
        """Test add embeddings with various validation errors."""
        vector_store._is_initialized = True
        
        # Test non-numpy array
        with pytest.raises(ValidationError) as exc_info:
            await vector_store.add_embeddings([1, 2, 3], [{}])
        assert "must be a numpy array" in str(exc_info.value)
        
        # Test wrong dimensions
        bad_embeddings = np.random.randn(5, 3, 128).astype(np.float32)  # 3D array
        with pytest.raises(ValidationError) as exc_info:
            await vector_store.add_embeddings(bad_embeddings, [{} for _ in range(5)])
        assert "must be 2D array" in str(exc_info.value)
        
        # Test dimension mismatch
        bad_embeddings = np.random.randn(5, 64).astype(np.float32)  # Wrong dimension
        with pytest.raises(ValidationError) as exc_info:
            await vector_store.add_embeddings(bad_embeddings, [{} for _ in range(5)])
        assert "doesn't match store dimension" in str(exc_info.value)
        
        # Test metadata length mismatch
        embeddings = np.random.randn(5, 128).astype(np.float32)
        with pytest.raises(ValidationError) as exc_info:
            await vector_store.add_embeddings(embeddings, [{}, {}])  # Only 2 metadata items
        assert "doesn't match embeddings count" in str(exc_info.value)
    
    def test_add_embeddings_sync(self, vector_store, sample_embeddings, sample_metadata):
        """Test synchronous embedding addition."""
        # Setup mock index
        mock_index = Mock()
        mock_index.add = Mock()
        mock_index.ntotal = 0
        vector_store._index = mock_index
        vector_store._next_idx = 0
        vector_store._total_vectors = 0
        
        embeddings = sample_embeddings[:3]
        metadata = sample_metadata[:3]
        
        result = vector_store._add_embeddings_sync(embeddings, metadata)
        
        # Verify index.add was called
        mock_index.add.assert_called_once()
        added_embeddings = mock_index.add.call_args[0][0]
        assert added_embeddings.dtype == np.float32
        assert added_embeddings.shape == (3, 128)
        
        # Verify metadata was stored
        assert len(vector_store._metadata) == 3
        assert len(vector_store._id_to_idx) == 3
        assert len(vector_store._idx_to_id) == 3
        
        # Verify returned IDs
        assert len(result) == 3
        for vector_id in result:
            assert vector_id in vector_store._id_to_idx
        
        # Verify counters updated
        assert vector_store._next_idx == 3
        assert vector_store._modification_count == 1
    
    def test_add_embeddings_sync_with_training(self, vector_store, sample_embeddings, sample_metadata):
        """Test synchronous embedding addition with IVF training."""
        # Setup mock IVF index
        mock_index = Mock()
        mock_index.add = Mock()
        mock_index.train = Mock()
        mock_index.is_trained = False
        mock_index.ntotal = 0
        vector_store._index = mock_index
        vector_store._total_vectors = 0
        
        # Use enough embeddings to trigger training
        embeddings = np.random.randn(300, 128).astype(np.float32)
        metadata = [{"text": f"text_{i}"} for i in range(300)]
        
        result = vector_store._add_embeddings_sync(embeddings, metadata)
        
        # Verify training was called
        mock_index.train.assert_called_once()
        mock_index.add.assert_called_once()
        assert len(result) == 300
    
    def test_add_embeddings_sync_dtype_conversion(self, vector_store, sample_metadata):
        """Test embedding dtype conversion during addition."""
        mock_index = Mock()
        mock_index.add = Mock()
        mock_index.ntotal = 0
        vector_store._index = mock_index
        vector_store._next_idx = 0
        vector_store._total_vectors = 0
        
        # Use float64 embeddings (should be converted to float32)
        embeddings = np.random.randn(3, 128).astype(np.float64)
        metadata = sample_metadata[:3]
        
        vector_store._add_embeddings_sync(embeddings, metadata)
        
        # Verify dtype was converted
        added_embeddings = mock_index.add.call_args[0][0]
        assert added_embeddings.dtype == np.float32
    
    @pytest.mark.asyncio
    async def test_search_not_initialized(self, vector_store):
        """Test search when not initialized."""
        query_embedding = np.random.randn(128).astype(np.float32)
        
        with pytest.raises(SearchError) as exc_info:
            await vector_store.search(query_embedding)
        
        assert "not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_search_validation_errors(self, vector_store):
        """Test search with validation errors."""
        vector_store._is_initialized = True
        vector_store._index = Mock()
        
        # Test non-numpy array
        with pytest.raises(ValidationError) as exc_info:
            await vector_store.search([1, 2, 3])
        assert "must be a numpy array" in str(exc_info.value)
        
        # Test wrong dimension
        bad_query = np.random.randn(64).astype(np.float32)  # Wrong dimension
        with pytest.raises(ValidationError) as exc_info:
            await vector_store.search(bad_query)
        assert "doesn't match dimension" in str(exc_info.value)
        
        # Test invalid k
        query = np.random.randn(128).astype(np.float32)
        with pytest.raises(ValidationError) as exc_info:
            await vector_store.search(query, k=0)
        assert "must be between 1 and 1000" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_search_empty_store(self, vector_store):
        """Test search on empty store."""
        vector_store._is_initialized = True
        vector_store._index = Mock()
        vector_store._total_vectors = 0
        
        query_embedding = np.random.randn(128).astype(np.float32)
        
        result = await vector_store.search(query_embedding)
        
        assert result == []
    
    def test_search_sync(self, vector_store):
        """Test synchronous search implementation."""
        # Setup mock index with search results
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.3, 0.5]]),  # scores
            np.array([[0, 2, 1]])         # indices
        )
        vector_store._index = mock_index
        vector_store._total_vectors = 3
        
        # Setup metadata
        vector_store._metadata = {
            0: {"text": "first", "category": "A"},
            1: {"text": "second", "category": "B"},
            2: {"text": "third", "category": "A"}
        }
        vector_store._idx_to_id = {0: "id0", 1: "id1", 2: "id2"}
        
        query_embedding = np.random.randn(128).astype(np.float32)
        
        results = vector_store._search_sync(query_embedding, k=3, score_threshold=None, filters=None, include_embeddings=False)
        
        assert len(results) == 3
        
        # Verify results are ordered by score
        assert results[0].id == "id0"
        assert results[0].score == 0.1
        assert results[0].text == "first"
        assert results[0].metadata["category"] == "A"
        
        assert results[1].id == "id2"
        assert results[1].score == 0.3
        assert results[1].text == "third"
    
    def test_search_sync_with_filters(self, vector_store):
        """Test synchronous search with metadata filters."""
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.3, 0.5]]),
            np.array([[0, 1, 2]])
        )
        vector_store._index = mock_index
        vector_store._total_vectors = 3
        
        vector_store._metadata = {
            0: {"text": "first", "category": "A", "score": 0.8},
            1: {"text": "second", "category": "B", "score": 0.6},
            2: {"text": "third", "category": "A", "score": 0.9}
        }
        vector_store._idx_to_id = {0: "id0", 1: "id1", 2: "id2"}
        
        query_embedding = np.random.randn(128).astype(np.float32)
        
        # Filter by category
        results = vector_store._search_sync(
            query_embedding, k=3, score_threshold=None,
            filters={"category": "A"}, include_embeddings=False
        )
        
        # Should only return items with category "A"
        assert len(results) == 2
        assert all(result.metadata["category"] == "A" for result in results)
    
    def test_search_sync_with_score_threshold(self, vector_store):
        """Test synchronous search with score threshold."""
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.3, 0.8]]),  # Third score exceeds threshold
            np.array([[0, 1, 2]])
        )
        vector_store._index = mock_index
        vector_store._total_vectors = 3
        
        vector_store._metadata = {0: {"text": "first"}, 1: {"text": "second"}, 2: {"text": "third"}}
        vector_store._idx_to_id = {0: "id0", 1: "id1", 2: "id2"}
        
        query_embedding = np.random.randn(128).astype(np.float32)
        
        results = vector_store._search_sync(
            query_embedding, k=3, score_threshold=0.5,
            filters=None, include_embeddings=False
        )
        
        # Should only return results with score <= 0.5
        assert len(results) == 2
        assert all(result.score <= 0.5 for result in results)
    
    def test_search_sync_with_embeddings(self, vector_store):
        """Test synchronous search including embeddings."""
        mock_index = Mock()
        mock_index.search.return_value = (np.array([[0.1]]), np.array([[0]]))
        mock_index.reconstruct.return_value = np.random.randn(128).astype(np.float32)
        vector_store._index = mock_index
        vector_store._total_vectors = 1
        
        vector_store._metadata = {0: {"text": "first"}}
        vector_store._idx_to_id = {0: "id0"}
        
        query_embedding = np.random.randn(128).astype(np.float32)
        
        results = vector_store._search_sync(
            query_embedding, k=1, score_threshold=None,
            filters=None, include_embeddings=True
        )
        
        assert len(results) == 1
        assert results[0].embedding is not None
        assert isinstance(results[0].embedding, np.ndarray)
        mock_index.reconstruct.assert_called_once_with(0)
    
    def test_matches_filters(self, vector_store):
        """Test metadata filtering logic."""
        metadata = {
            "category": "A",
            "score": 0.8,
            "tags": ["important", "recent"],
            "count": 10
        }
        
        # Test exact match
        assert vector_store._matches_filters(metadata, {"category": "A"})
        assert not vector_store._matches_filters(metadata, {"category": "B"})
        
        # Test list membership
        assert vector_store._matches_filters(metadata, {"category": ["A", "B"]})
        assert not vector_store._matches_filters(metadata, {"category": ["C", "D"]})
        
        # Test range queries
        assert vector_store._matches_filters(metadata, {"score": {"gte": 0.7}})
        assert vector_store._matches_filters(metadata, {"score": {"lte": 0.9}})
        assert not vector_store._matches_filters(metadata, {"score": {"gt": 0.8}})
        assert not vector_store._matches_filters(metadata, {"score": {"lt": 0.8}})
        
        # Test missing field
        assert not vector_store._matches_filters(metadata, {"missing_field": "value"})
        
        # Test multiple filters (AND logic)
        assert vector_store._matches_filters(metadata, {"category": "A", "score": {"gte": 0.7}})
        assert not vector_store._matches_filters(metadata, {"category": "A", "score": {"gte": 0.9}})
    
    @patch('faiss.write_index')
    @pytest.mark.asyncio
    async def test_save_index(self, mock_write, vector_store):
        """Test index saving."""
        vector_store._is_initialized = True
        mock_index = Mock()
        vector_store._index = mock_index
        
        with patch.object(vector_store, '_save_metadata_sync') as mock_save_meta:
            await vector_store.save_index()
            
            mock_write.assert_called_once_with(mock_index, str(vector_store._index_path))
            mock_save_meta.assert_called_once()
            assert vector_store._last_save_time is not None
    
    @pytest.mark.asyncio
    async def test_save_index_not_initialized(self, vector_store):
        """Test saving index when not initialized."""
        with pytest.raises(VectorStoreError) as exc_info:
            await vector_store.save_index()
        
        assert "not initialized" in str(exc_info.value)
    
    def test_save_index_sync_with_backup(self, vector_store, temp_dir):
        """Test synchronous index saving with backup creation."""
        vector_store._index_path = temp_dir / "test.index"
        vector_store._metadata_path = temp_dir / "metadata.json"
        mock_index = Mock()
        vector_store._index = mock_index
        
        # Create existing files
        vector_store._index_path.write_text("existing index")
        vector_store._metadata_path.write_text('{"existing": "metadata"}')
        
        with patch('faiss.write_index') as mock_write:
            with patch.object(vector_store, '_save_metadata_sync') as mock_save_meta:
                vector_store._save_index_sync()
                
                # Verify backup was created
                backup_path = vector_store._index_path.with_suffix('.index.backup')
                assert backup_path.exists()
                assert backup_path.read_text() == "existing index"
                
                mock_write.assert_called_once_with(mock_index, str(vector_store._index_path))
                mock_save_meta.assert_called_once()
    
    @patch('faiss.read_index')
    @pytest.mark.asyncio
    async def test_load_index_success(self, mock_read, vector_store):
        """Test successful index loading."""
        vector_store._index_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store._index_path.write_text("dummy index file")
        
        mock_index = Mock()
        mock_index.d = vector_store._dimension  # Match dimension
        mock_index.ntotal = 100
        mock_read.return_value = mock_index
        
        result = await vector_store.load_index()
        
        assert result is True
        assert vector_store._index == mock_index
        assert vector_store._total_vectors == 100
        mock_read.assert_called_once_with(str(vector_store._index_path))
    
    @pytest.mark.asyncio
    async def test_load_index_not_found(self, vector_store):
        """Test loading index when file doesn't exist."""
        result = await vector_store.load_index()
        
        assert result is False
    
    @patch('faiss.read_index')
    @pytest.mark.asyncio
    async def test_load_index_dimension_mismatch(self, mock_read, vector_store):
        """Test loading index with dimension mismatch."""
        vector_store._index_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store._index_path.write_text("dummy index file")
        
        mock_index = Mock()
        mock_index.d = 256  # Different dimension
        mock_read.return_value = mock_index
        
        with pytest.raises(VectorIndexError) as exc_info:
            await vector_store.load_index()
        
        assert "dimension" in str(exc_info.value)
        assert "doesn't match" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_load_metadata_success(self, vector_store, temp_dir):
        """Test successful metadata loading."""
        metadata_file = temp_dir / "metadata.json"
        test_data = {
            "metadata": {"0": {"text": "first"}, "1": {"text": "second"}},
            "id_to_idx": {"id0": 0, "id1": 1},
            "idx_to_id": {"0": "id0", "1": "id1"},
            "total_vectors": 2,
            "dimension": 128
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(test_data, f)
        
        vector_store._metadata_path = metadata_file
        
        await vector_store._load_metadata()
        
        assert len(vector_store._metadata) == 2
        assert vector_store._metadata[0]["text"] == "first"
        assert vector_store._metadata[1]["text"] == "second"
        assert vector_store._id_to_idx == {"id0": 0, "id1": 1}
        assert vector_store._idx_to_id == {0: "id0", 1: "id1"}
        assert vector_store._next_idx == 2
    
    @pytest.mark.asyncio
    async def test_load_metadata_not_found(self, vector_store):
        """Test loading metadata when file doesn't exist."""
        await vector_store._load_metadata()
        
        # Should continue with empty metadata
        assert isinstance(vector_store._metadata, dict)
        assert len(vector_store._metadata) == 0
    
    def test_save_metadata_sync(self, vector_store, temp_dir):
        """Test synchronous metadata saving."""
        vector_store._metadata_path = temp_dir / "metadata.json"
        vector_store._metadata = {0: {"text": "first"}, 1: {"text": "second"}}
        vector_store._id_to_idx = {"id0": 0, "id1": 1}
        vector_store._idx_to_id = {0: "id0", 1: "id1"}
        vector_store._total_vectors = 2
        vector_store._dimension = 128
        
        vector_store._save_metadata_sync()
        
        assert vector_store._metadata_path.exists()
        
        with open(vector_store._metadata_path, 'r') as f:
            saved_data = json.load(f)
        
        assert "metadata" in saved_data
        assert "id_to_idx" in saved_data
        assert "idx_to_id" in saved_data
        assert saved_data["total_vectors"] == 2
        assert saved_data["dimension"] == 128
    
    def test_get_index_info(self, vector_store):
        """Test getting index information."""
        vector_store._is_initialized = True
        vector_store._total_vectors = 100
        vector_store._modification_count = 5
        vector_store._last_save_time = 1234567890.0
        
        mock_index = Mock()
        mock_index.__class__.__name__ = "IndexFlatL2"
        mock_index.is_trained = True
        vector_store._index = mock_index
        
        info = vector_store.get_index_info()
        
        assert info["is_initialized"] is True
        assert info["total_vectors"] == 100
        assert info["dimension"] == vector_store._dimension
        assert info["index_type"] == vector_store._index_type
        assert info["modification_count"] == 5
        assert info["last_save_time"] == 1234567890.0
        assert info["index_class"] == "IndexFlatL2"
        assert info["is_trained"] is True


class TestSearchResult:
    """Test cases for SearchResult class."""
    
    def test_search_result_creation(self):
        """Test creating SearchResult instance."""
        embedding = np.random.randn(128).astype(np.float32)
        metadata = {"text": "test", "category": "A"}
        
        result = SearchResult(
            id="test_id",
            score=0.5,
            text="test text",
            metadata=metadata,
            embedding=embedding
        )
        
        assert result.id == "test_id"
        assert result.score == 0.5
        assert result.text == "test text"
        assert result.metadata == metadata
        assert np.array_equal(result.embedding, embedding)
    
    def test_search_result_without_embedding(self):
        """Test creating SearchResult without embedding."""
        result = SearchResult(
            id="test_id",
            score=0.5,
            text="test text",
            metadata={"category": "A"}
        )
        
        assert result.id == "test_id"
        assert result.score == 0.5
        assert result.text == "test text"
        assert result.embedding is None


class TestCreateVectorStore:
    """Test cases for create_vector_store function."""
    
    def test_create_vector_store_defaults(self):
        """Test creating vector store with defaults."""
        store = create_vector_store()
        assert isinstance(store, VectorStore)
        assert store._dimension is not None
    
    def test_create_vector_store_custom(self, temp_dir):
        """Test creating vector store with custom parameters."""
        settings = Settings()
        store = create_vector_store(
            dimension=256,
            index_path=str(temp_dir / "custom.index"),
            settings=settings
        )
        
        assert isinstance(store, VectorStore)
        assert store._dimension == 256
        assert store._index_path == temp_dir / "custom.index"
        assert store._settings == settings


# Integration tests
class TestVectorStoreIntegration:
    """Integration tests for VectorStore."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    @patch('faiss.IndexFlatL2')
    @pytest.mark.asyncio
    async def test_full_vector_store_lifecycle(self, mock_index_class, temp_dir):
        """Test complete vector store lifecycle."""
        # Setup mock FAISS index
        mock_index = Mock()
        mock_index.add = Mock()
        mock_index.search = Mock(return_value=(np.array([[0.1, 0.3]]), np.array([[0, 1]])))
        mock_index.ntotal = 0
        mock_index_class.return_value = mock_index
        
        # Create vector store
        settings = Settings(
            vector_store=Settings.VectorStoreConfig(
                dimension=128,
                index_path=temp_dir / "test.index",
                metadata_path=temp_dir / "metadata.json",
                data_dir=temp_dir,
                index_factory="Flat"
            )
        )
        
        store = VectorStore(settings=settings)
        
        # Initialize
        await store.initialize()
        assert store._is_initialized
        
        # Add embeddings
        embeddings = np.random.randn(5, 128).astype(np.float32)
        metadata = [{"text": f"text_{i}", "category": "test"} for i in range(5)]
        
        # Mock the behavior of successful addition
        def mock_add_side_effect(emb):
            mock_index.ntotal = len(emb)
        
        mock_index.add.side_effect = mock_add_side_effect
        
        vector_ids = await store.add_embeddings(embeddings, metadata)
        
        assert len(vector_ids) == 5
        assert all(isinstance(vid, str) for vid in vector_ids)
        mock_index.add.assert_called_once()
        
        # Search
        query_embedding = np.random.randn(128).astype(np.float32)
        
        # Setup mock search to return valid indices
        mock_index.search.return_value = (np.array([[0.1, 0.3]]), np.array([[0, 1]]))
        store._total_vectors = 2  # Simulate that we have vectors
        
        results = await store.search(query_embedding, k=2)
        
        mock_index.search.assert_called()
        # Results might be empty due to mocking, but no errors should occur
        
        # Get info
        info = store.get_index_info()
        assert info["is_initialized"]
        assert info["dimension"] == 128
    
    @pytest.mark.performance
    @patch('faiss.IndexFlatL2')
    @pytest.mark.asyncio
    async def test_vector_store_performance(self, mock_index_class, temp_dir):
        """Test vector store performance with various operations."""
        # Setup mock index for performance testing
        mock_index = Mock()
        mock_index.add = Mock()
        mock_index.search = Mock()
        mock_index.ntotal = 0
        mock_index_class.return_value = mock_index
        
        settings = Settings(
            vector_store=Settings.VectorStoreConfig(
                dimension=384,
                index_path=temp_dir / "perf.index",
                metadata_path=temp_dir / "perf_metadata.json",
                data_dir=temp_dir
            )
        )
        
        store = VectorStore(settings=settings)
        await store.initialize()
        
        # Test batch addition performance
        batch_sizes = [10, 100, 500]
        
        for batch_size in batch_sizes:
            embeddings = np.random.randn(batch_size, 384).astype(np.float32)
            metadata = [{"text": f"text_{i}", "batch": batch_size} for i in range(batch_size)]
            
            # Mock successful addition
            def mock_add_side_effect(emb):
                mock_index.ntotal += len(emb)
            
            mock_index.add.side_effect = mock_add_side_effect
            
            start_time = asyncio.get_event_loop().time()
            vector_ids = await store.add_embeddings(embeddings, metadata)
            end_time = asyncio.get_event_loop().time()
            
            duration = end_time - start_time
            throughput = batch_size / duration if duration > 0 else float('inf')
            
            assert len(vector_ids) == batch_size
            assert duration < 1.0, f"Batch size {batch_size} took too long: {duration}s"
            
            # Log performance metrics
            print(f"Batch size {batch_size}: {duration:.3f}s, {throughput:.1f} vectors/s")


if __name__ == "__main__":
    pytest.main([__file__])