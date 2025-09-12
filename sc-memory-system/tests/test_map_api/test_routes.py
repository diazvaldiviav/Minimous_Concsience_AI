"""
Tests for MAP API routes.

This module tests the MAP API endpoints including GET and POST context
retrieval, response formatting, and error handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import status
from datetime import datetime

from src.api.main import create_app
from src.core.config import Settings
from src.core.models import MAPResponse, CompressedTurn, FilteredFact, AdapterInfo


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.app_name = "SC Memory Test"
    settings.app_version = "1.0.0"
    settings.auth_token = "test_token"
    settings.api = Mock()
    settings.api.mep_prefix = "/mep/v1"
    return settings


@pytest.fixture
def test_client(mock_settings):
    """Create test client with mocked dependencies."""
    with patch('src.api.main.get_settings') as mock_get_settings, \
         patch('src.api.map.routes.initialize_map_components') as mock_init, \
         patch('src.api.map.routes.cleanup_map_components') as mock_cleanup:
        
        mock_get_settings.return_value = mock_settings
        mock_init.return_value = AsyncMock()
        mock_cleanup.return_value = AsyncMock()
        
        app = create_app(mock_settings)
        return TestClient(app)


@pytest.fixture
def mock_adapter_manager():
    """Create mock adapter manager."""
    manager = Mock()
    manager.find_user_adapters = AsyncMock()
    manager.load_adapter = AsyncMock()
    manager.get_cache_stats.return_value = {
        'cache_hits': 10,
        'cache_misses': 2,
        'hit_rate': 0.83
    }
    return manager


@pytest.fixture
def mock_context_builder():
    """Create mock context builder."""
    builder = Mock()
    builder.token_budget_manager = Mock()
    builder.token_budget_manager.allocate_tokens.return_value = Mock(
        gist_tokens=100,
        turns_tokens=100,
        facts_tokens=50,
        metadata_tokens=20
    )
    builder.generate_gist = AsyncMock(return_value="Test gist content")
    builder.extract_turn_sketch = AsyncMock(return_value=[
        CompressedTurn(id="T1U", r="u", t="User question"),
        CompressedTurn(id="T2A", r="a", t="Assistant response")
    ])
    builder.compile_facts = AsyncMock(return_value=[
        FilteredFact(c="Test fact", p=0.92, s="mem")
    ])
    builder.estimate_response_tokens.return_value = 280
    return builder


@pytest.fixture
def sample_adapter_info():
    """Create sample adapter info."""
    return AdapterInfo(
        adapter_id="test_adapter",
        conversation_id="conv_123",
        provider="anthropic",
        external_user_id="user_456",
        adapter_path="/path/to/adapter",
        topic="physics",
        turn_range={"from_turn": 1, "to_turn": 50},
        quality_score=0.92
    )


class TestMAPRoutes:
    """Test MAP API route functionality."""
    
    def test_get_context_success(self, test_client, mock_adapter_manager, mock_context_builder, sample_adapter_info):
        """Test successful GET context request."""
        with patch('src.api.map.routes.get_adapter_manager') as mock_get_adapter_mgr, \
             patch('src.api.map.routes.get_context_builder') as mock_get_ctx_builder, \
             patch('src.api.map.routes._generate_map_response') as mock_generate_response:
            
            mock_get_adapter_mgr.return_value = mock_adapter_manager
            mock_get_ctx_builder.return_value = mock_context_builder
            
            # Mock response generation
            mock_response = MAPResponse(
                has_memory=True,
                topic="physics",
                span={"from_turn": 1, "to_turn": 50},
                gist="Test gist content",
                turns=[CompressedTurn(id="T1U", r="u", t="User question")],
                facts=[FilteredFact(c="Test fact", p=0.92, s="mem")],
                tokens_est=280,
                shard_hint="sh_physics_v1"
            )
            mock_generate_response.return_value = mock_response
            
            response = test_client.get(
                "/map/v1/context",
                params={
                    "provider": "anthropic",
                    "external_user_id": "user_123",
                    "query": "What is physics?"
                },
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["has_memory"] is True
            assert data["topic"] == "physics"
            assert data["gist"] == "Test gist content"
            assert len(data["turns"]) == 1
            assert len(data["facts"]) == 1
    
    def test_get_context_missing_auth(self, test_client):
        """Test GET context request without authentication."""
        response = test_client.get(
            "/map/v1/context",
            params={
                "provider": "anthropic", 
                "external_user_id": "user_123",
                "query": "What is physics?"
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_context_invalid_parameters(self, test_client):
        """Test GET context request with invalid parameters."""
        response = test_client.get(
            "/map/v1/context",
            params={
                "provider": "invalid_provider",  # Should be anthropic or openai
                "external_user_id": "user_123",
                "query": "What is physics?"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_context_compact_text_format(self, test_client, mock_adapter_manager, mock_context_builder):
        """Test GET context request with compact_text format."""
        with patch('src.api.map.routes.get_adapter_manager') as mock_get_adapter_mgr, \
             patch('src.api.map.routes.get_context_builder') as mock_get_ctx_builder, \
             patch('src.api.map.routes._generate_map_response') as mock_generate_response:
            
            mock_get_adapter_mgr.return_value = mock_adapter_manager
            mock_get_ctx_builder.return_value = mock_context_builder
            
            mock_response = MAPResponse(
                has_memory=True,
                gist="Test gist",
                turns=[CompressedTurn(id="T1U", r="u", t="User question")],
                facts=[FilteredFact(c="Test fact", p=0.92, s="mem")],
                tokens_est=100
            )
            mock_generate_response.return_value = mock_response
            
            response = test_client.get(
                "/map/v1/context",
                params={
                    "provider": "anthropic",
                    "external_user_id": "user_123", 
                    "query": "What is physics?",
                    "format": "compact_text"
                },
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            content = response.text
            assert "GIST:" in content
            assert "TURNS:" in content
            assert "FACTS:" in content
    
    def test_post_context_success(self, test_client, mock_adapter_manager, mock_context_builder):
        """Test successful POST context request."""
        with patch('src.api.map.routes.get_adapter_manager') as mock_get_adapter_mgr, \
             patch('src.api.map.routes.get_context_builder') as mock_get_ctx_builder, \
             patch('src.api.map.routes._generate_map_response') as mock_generate_response:
            
            mock_get_adapter_mgr.return_value = mock_adapter_manager
            mock_get_ctx_builder.return_value = mock_context_builder
            
            mock_response = MAPResponse(
                has_memory=True,
                topic="physics",
                gist="Test gist content",
                turns=[CompressedTurn(id="T1U", r="u", t="User question")],
                facts=[FilteredFact(c="Test fact", p=0.92, s="mem")],
                tokens_est=280
            )
            mock_generate_response.return_value = mock_response
            
            request_data = {
                "provider": "anthropic",
                "external_user_id": "user_123",
                "query": "What is physics?",
                "token_budget": 320,
                "min_truth": 0.75,
                "format": "json",
                "granularity": "mix",
                "scope": "user"
            }
            
            response = test_client.post(
                "/map/v1/context",
                json=request_data,
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["has_memory"] is True
            assert data["topic"] == "physics"
    
    def test_post_context_invalid_json(self, test_client):
        """Test POST context request with invalid JSON."""
        request_data = {
            "provider": "invalid_provider",
            "external_user_id": "user_123",
            "query": "What is physics?"
        }
        
        response = test_client.post(
            "/map/v1/context",
            json=request_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_health_check(self, test_client, mock_adapter_manager, mock_context_builder):
        """Test MAP health check endpoint."""
        with patch('src.api.map.routes.get_adapter_manager') as mock_get_adapter_mgr, \
             patch('src.api.map.routes.get_context_builder') as mock_get_ctx_builder:
            
            mock_get_adapter_mgr.return_value = mock_adapter_manager
            mock_get_ctx_builder.return_value = mock_context_builder
            
            response = test_client.get("/map/v1/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "healthy"
            assert "adapter_cache" in data
            assert "components" in data
    
    def test_health_check_failure(self, test_client):
        """Test MAP health check when components fail."""
        with patch('src.api.map.routes.get_adapter_manager') as mock_get_adapter_mgr:
            mock_get_adapter_mgr.side_effect = Exception("Component failed")
            
            response = test_client.get("/map/v1/health")
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            data = response.json()
            assert data["status"] == "unhealthy"


class TestResponseFormatting:
    """Test MAP response formatting functionality."""
    
    @pytest.mark.asyncio
    async def test_format_response_json(self):
        """Test JSON response formatting."""
        from src.api.map.routes import _format_response
        
        response = MAPResponse(
            has_memory=True,
            topic="physics",
            gist="Test gist",
            turns=[CompressedTurn(id="T1U", r="u", t="User question")],
            facts=[FilteredFact(c="Test fact", p=0.92, s="mem")],
            tokens_est=150
        )
        
        formatted = await _format_response(response, "json")
        
        assert isinstance(formatted, dict)
        assert formatted["has_memory"] is True
        assert formatted["topic"] == "physics"
        assert len(formatted["turns"]) == 1
    
    @pytest.mark.asyncio
    async def test_format_response_json_compact(self):
        """Test compact JSON response formatting."""
        from src.api.map.routes import _format_response
        
        response = MAPResponse(
            has_memory=True,
            topic="physics", 
            gist="Test gist",
            turns=[CompressedTurn(id="T1U", r="u", t="User question")],
            facts=[FilteredFact(c="Test fact", p=0.92, s="mem")],
            tokens_est=150
        )
        
        formatted = await _format_response(response, "json_compact")
        
        assert isinstance(formatted, dict)
        assert "mem" in formatted
        assert "topic" in formatted
        assert "gist" in formatted
        assert "turns" in formatted
        assert "facts" in formatted
    
    @pytest.mark.asyncio
    async def test_format_response_compact_text(self):
        """Test compact text response formatting."""
        from src.api.map.routes import _format_response
        
        response = MAPResponse(
            has_memory=True,
            gist="Test gist",
            turns=[CompressedTurn(id="T1U", r="u", t="User question")],
            facts=[FilteredFact(c="Test fact", p=0.92, s="mem")],
            tokens_est=150
        )
        
        formatted = await _format_response(response, "compact_text")
        
        assert isinstance(formatted, str)
        assert "GIST:" in formatted
        assert "TURNS:" in formatted
        assert "T1U USER:" in formatted
        assert "FACTS:" in formatted
        assert "Test fact" in formatted
    
    @pytest.mark.asyncio
    async def test_format_response_no_memory(self):
        """Test formatting when no memory is available."""
        from src.api.map.routes import _format_response
        
        response = MAPResponse(
            has_memory=False,
            gist="",
            turns=[],
            facts=[],
            tokens_est=0
        )
        
        formatted = await _format_response(response, "compact_text")
        
        assert isinstance(formatted, str)
        assert "No memory available" in formatted


class TestMAPResponseGeneration:
    """Test MAP response generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_map_response_with_memory(self, mock_adapter_manager, mock_context_builder, sample_adapter_info):
        """Test MAP response generation when memory is available."""
        from src.api.map.routes import _generate_map_response
        from src.core.models import MAPQuery
        
        # Setup mocks
        mock_adapter_manager.find_user_adapters.return_value = [sample_adapter_info]
        mock_model = Mock()
        mock_adapter_manager.load_adapter.return_value = mock_model
        
        map_query = MAPQuery(
            provider="anthropic",
            external_user_id="user_123",
            query="What is physics?",
            token_budget=320
        )
        
        response = await _generate_map_response(
            map_query, mock_adapter_manager, mock_context_builder
        )
        
        assert response.has_memory is True
        assert response.topic == "physics"
        assert response.tokens_est == 280
        
        # Verify methods were called
        mock_adapter_manager.find_user_adapters.assert_called_once()
        mock_adapter_manager.load_adapter.assert_called_once()
        mock_context_builder.generate_gist.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_map_response_no_adapters(self, mock_adapter_manager, mock_context_builder):
        """Test MAP response generation when no adapters are found."""
        from src.api.map.routes import _generate_map_response
        from src.core.models import MAPQuery
        
        # Setup mocks
        mock_adapter_manager.find_user_adapters.return_value = []
        
        map_query = MAPQuery(
            provider="anthropic",
            external_user_id="user_123",
            query="What is physics?"
        )
        
        response = await _generate_map_response(
            map_query, mock_adapter_manager, mock_context_builder
        )
        
        assert response.has_memory is False
        assert response.gist == ""
        assert len(response.turns) == 0
        assert len(response.facts) == 0
        assert response.tokens_est == 0
    
    @pytest.mark.asyncio
    async def test_generate_map_response_adapter_load_failure(self, mock_adapter_manager, mock_context_builder, sample_adapter_info):
        """Test MAP response generation when adapter loading fails."""
        from src.api.map.routes import _generate_map_response
        from src.core.models import MAPQuery
        
        # Setup mocks
        mock_adapter_manager.find_user_adapters.return_value = [sample_adapter_info]
        mock_adapter_manager.load_adapter.return_value = None  # Failed to load
        
        map_query = MAPQuery(
            provider="anthropic",
            external_user_id="user_123", 
            query="What is physics?"
        )
        
        response = await _generate_map_response(
            map_query, mock_adapter_manager, mock_context_builder
        )
        
        assert response.has_memory is False
        assert response.tokens_est == 0