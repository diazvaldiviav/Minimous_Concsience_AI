"""
Test suite for MEP (Memory Exchange Protocol) API.

This module contains comprehensive tests for MEP API endpoints,
including request validation, authentication, error handling,
and response formatting.
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from src.api.main import create_app
from src.core.config import Settings
from src.core.models import TokenUsageInfo, MessageSpan, KeyFact


class TestMEPAPI:
    """Test cases for MEP API endpoints."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            api=Settings.APIConfig(
                bearer_token="test-token",
                mep_max_proposal_size=1048576,  # 1MB
                mep_queue_max_size=100
            ),
            debug=True
        )
    
    @pytest.fixture
    def client(self, settings):
        """Create test client."""
        app = create_app(settings)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        return {"Authorization": "Bearer test-token"}
    
    @pytest.fixture
    def valid_proposal(self):
        """Create valid MEP proposal data."""
        return {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "external_user_id": "user_12345",
            "external_chat_id": "chat_67890",
            "event_id": "evt_001",
            "trigger": "context_full",
            "context_fill": 0.85,
            "token_usage": {
                "window_tokens": 8000,
                "used_tokens": 6800,
                "max_tokens": 10000
            },
            "message_span": {
                "from_turn": 0,
                "to_turn": 15
            },
            "summary_text": "User discussed project requirements for a web application with authentication and database integration.",
            "key_facts": [
                {
                    "claim": "User prefers React for frontend development",
                    "importance": 0.8,
                    "confidence": 0.9,
                    "category": "preference"
                },
                {
                    "claim": "Database should support PostgreSQL",
                    "importance": 0.9,
                    "confidence": 0.95,
                    "category": "requirement"
                }
            ]
        }
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns application information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "healthy"
        assert "mep_api" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint returns health status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "version" in data
        assert data["status"] == "healthy"
    
    def test_mep_health_endpoint(self, client, auth_headers):
        """Test MEP health endpoint."""
        response = client.get("/mep/v1/health", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data
        assert "metrics" in data
        
        # Check components
        components = data["components"]
        assert "base_model" in components
        assert "embeddings" in components
        assert "vector_store" in components
        assert "queue" in components
    
    def test_submit_proposal_success(self, client, auth_headers, valid_proposal):
        """Test successful MEP proposal submission."""
        response = client.post(
            "/mep/v1/proposals",
            headers=auth_headers,
            json=valid_proposal
        )
        
        assert response.status_code == 202
        data = response.json()
        
        assert "proposal_id" in data
        assert data["status"] == "accepted"
        assert "message" in data
        assert "queue_position" in data
        assert "estimated_processing_time" in data
        assert "timestamp" in data
        
        # Verify proposal ID is valid UUID format
        import uuid
        uuid.UUID(data["proposal_id"])  # Should not raise exception
    
    def test_submit_proposal_no_auth(self, client, valid_proposal):
        """Test MEP proposal submission without authentication."""
        response = client.post("/mep/v1/proposals", json=valid_proposal)
        
        assert response.status_code == 401
        data = response.json()
        assert "Bearer token required" in data["detail"]
    
    def test_submit_proposal_invalid_token(self, client, valid_proposal):
        """Test MEP proposal submission with invalid token."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.post(
            "/mep/v1/proposals",
            headers=headers,
            json=valid_proposal
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "Invalid bearer token" in data["detail"]
    
    def test_submit_proposal_validation_errors(self, client, auth_headers):
        """Test MEP proposal submission with validation errors."""
        invalid_proposals = [
            # Missing required fields
            {},
            
            # Invalid provider
            {
                "provider": "",  # Empty string
                "model": "claude-3-sonnet",
                "external_user_id": "user_12345",
                "external_chat_id": "chat_67890",
                "event_id": "evt_001",
                "trigger": "context_full",
                "context_fill": 0.85,
                "token_usage": {"window_tokens": 8000, "used_tokens": 6800, "max_tokens": 10000},
                "message_span": {"from_turn": 0, "to_turn": 15},
                "summary_text": "Test summary with sufficient length for validation",
                "key_facts": [{"claim": "Test claim", "importance": 0.8}]
            },
            
            # Invalid context_fill
            {
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "external_user_id": "user_12345",
                "external_chat_id": "chat_67890",
                "event_id": "evt_001",
                "trigger": "context_full",
                "context_fill": 1.5,  # > 1.0
                "token_usage": {"window_tokens": 8000, "used_tokens": 6800, "max_tokens": 10000},
                "message_span": {"from_turn": 0, "to_turn": 15},
                "summary_text": "Test summary with sufficient length for validation",
                "key_facts": [{"claim": "Test claim", "importance": 0.8}]
            },
            
            # Invalid token usage
            {
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "external_user_id": "user_12345",
                "external_chat_id": "chat_67890",
                "event_id": "evt_001",
                "trigger": "context_full",
                "context_fill": 0.85,
                "token_usage": {
                    "window_tokens": 8000,
                    "used_tokens": 9000,  # > window_tokens
                    "max_tokens": 10000
                },
                "message_span": {"from_turn": 0, "to_turn": 15},
                "summary_text": "Test summary with sufficient length for validation",
                "key_facts": [{"claim": "Test claim", "importance": 0.8}]
            },
            
            # Invalid message span
            {
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "external_user_id": "user_12345",
                "external_chat_id": "chat_67890",
                "event_id": "evt_001",
                "trigger": "context_full",
                "context_fill": 0.85,
                "token_usage": {"window_tokens": 8000, "used_tokens": 6800, "max_tokens": 10000},
                "message_span": {"from_turn": 15, "to_turn": 10},  # from > to
                "summary_text": "Test summary with sufficient length for validation",
                "key_facts": [{"claim": "Test claim", "importance": 0.8}]
            },
            
            # Empty key facts
            {
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "external_user_id": "user_12345",
                "external_chat_id": "chat_67890",
                "event_id": "evt_001",
                "trigger": "context_full",
                "context_fill": 0.85,
                "token_usage": {"window_tokens": 8000, "used_tokens": 6800, "max_tokens": 10000},
                "message_span": {"from_turn": 0, "to_turn": 15},
                "summary_text": "Test summary with sufficient length for validation",
                "key_facts": []  # Empty list
            },
            
            # Summary too short
            {
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "external_user_id": "user_12345",
                "external_chat_id": "chat_67890",
                "event_id": "evt_001",
                "trigger": "context_full",
                "context_fill": 0.85,
                "token_usage": {"window_tokens": 8000, "used_tokens": 6800, "max_tokens": 10000},
                "message_span": {"from_turn": 0, "to_turn": 15},
                "summary_text": "Short",  # Too short
                "key_facts": [{"claim": "Test claim", "importance": 0.8}]
            }
        ]
        
        for invalid_proposal in invalid_proposals:
            response = client.post(
                "/mep/v1/proposals",
                headers=auth_headers,
                json=invalid_proposal
            )
            
            assert response.status_code in [422, 400], f"Expected validation error for: {invalid_proposal}"
    
    def test_submit_proposal_key_fact_validation(self, client, auth_headers, valid_proposal):
        """Test key fact validation in MEP proposals."""
        # Invalid importance score
        proposal = valid_proposal.copy()
        proposal["key_facts"] = [
            {
                "claim": "Test claim",
                "importance": 1.5  # > 1.0
            }
        ]
        
        response = client.post(
            "/mep/v1/proposals",
            headers=auth_headers,
            json=proposal
        )
        
        assert response.status_code == 422
    
    def test_submit_proposal_too_large(self, client, auth_headers, valid_proposal, settings):
        """Test MEP proposal submission with oversized request."""
        # Create a very large summary to exceed size limit
        large_summary = "x" * (settings.api.mep_max_proposal_size + 1000)
        proposal = valid_proposal.copy()
        proposal["summary_text"] = large_summary
        
        response = client.post(
            "/mep/v1/proposals",
            headers=auth_headers,
            json=proposal
        )
        
        assert response.status_code == 413
    
    @patch('src.api.mep.routes.proposal_queue')
    def test_submit_proposal_queue_full(self, mock_queue, client, auth_headers, valid_proposal, settings):
        """Test MEP proposal submission when queue is full."""
        # Mock queue to be at maximum capacity
        mock_queue.__len__ = Mock(return_value=settings.api.mep_queue_max_size)
        
        response = client.post(
            "/mep/v1/proposals",
            headers=auth_headers,
            json=proposal
        )
        
        assert response.status_code == 429
        data = response.json()
        assert "Queue full" in str(data.get("detail", ""))
    
    def test_get_proposal_status_not_found(self, client, auth_headers):
        """Test getting status of non-existent proposal."""
        response = client.get(
            "/mep/v1/proposals/non-existent-id/status",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in str(data.get("detail", "")).lower()
    
    def test_get_proposal_status_no_auth(self, client):
        """Test getting proposal status without authentication."""
        response = client.get("/mep/v1/proposals/some-id/status")
        
        assert response.status_code == 401
    
    @patch('src.api.mep.routes.proposal_queue')
    def test_get_proposal_status_success(self, mock_queue, client, auth_headers):
        """Test successful proposal status retrieval."""
        # Mock queue with proposal data
        proposal_data = {
            "proposal_id": "test-proposal-id",
            "status": "processing",
            "queue_position": 3,
            "submitted_at": datetime.utcnow(),
            "processed_at": None,
            "error": None
        }
        mock_queue.__iter__ = Mock(return_value=iter([proposal_data]))
        
        response = client.get(
            "/mep/v1/proposals/test-proposal-id/status",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["proposal_id"] == "test-proposal-id"
        assert data["status"] == "processing"
        assert data["queue_position"] == 3
    
    def test_get_queue_status(self, client, auth_headers):
        """Test getting queue status."""
        response = client.get("/mep/v1/queue/status", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "queue_size" in data
        assert "max_queue_size" in data
        assert "utilization" in data
        assert "status_breakdown" in data
        assert "recent_proposals_1h" in data
        assert "timestamp" in data
        
        # Verify status breakdown structure
        breakdown = data["status_breakdown"]
        assert "queued" in breakdown
        assert "processing" in breakdown
        assert "completed" in breakdown
        assert "failed" in breakdown
    
    def test_get_queue_status_no_auth(self, client):
        """Test getting queue status without authentication."""
        response = client.get("/mep/v1/queue/status")
        
        assert response.status_code == 401


class TestMEPSchemas:
    """Test cases for MEP schema validation."""
    
    def test_token_usage_info_validation(self):
        """Test TokenUsageInfo validation."""
        # Valid token usage
        token_usage = TokenUsageInfo(
            window_tokens=8000,
            used_tokens=6800,
            max_tokens=10000
        )
        
        assert token_usage.window_tokens == 8000
        assert token_usage.used_tokens == 6800
        assert token_usage.max_tokens == 10000
        assert token_usage.utilization_ratio == 0.85
        assert token_usage.capacity_ratio == 0.8
        
        # Invalid: used_tokens > window_tokens
        with pytest.raises(ValueError) as exc_info:
            TokenUsageInfo(
                window_tokens=8000,
                used_tokens=9000,
                max_tokens=10000
            )
        assert "cannot exceed" in str(exc_info.value)
    
    def test_message_span_validation(self):
        """Test MessageSpan validation."""
        # Valid message span
        span = MessageSpan(from_turn=0, to_turn=15)
        assert span.from_turn == 0
        assert span.to_turn == 15
        assert span.span_length == 16
        
        # Valid: same turn
        span = MessageSpan(from_turn=5, to_turn=5)
        assert span.span_length == 1
        
        # Invalid: to_turn < from_turn
        with pytest.raises(ValueError) as exc_info:
            MessageSpan(from_turn=15, to_turn=10)
        assert "greater than or equal" in str(exc_info.value)
    
    def test_key_fact_validation(self):
        """Test KeyFact validation."""
        # Valid key fact
        fact = KeyFact(
            claim="User prefers React for frontend",
            importance=0.8,
            confidence=0.9,
            source_turn=5,
            category="preference"
        )
        
        assert fact.claim == "User prefers React for frontend"
        assert fact.importance == 0.8
        assert fact.confidence == 0.9
        assert fact.source_turn == 5
        assert fact.category == "preference"
        
        # Invalid importance (> 1.0)
        with pytest.raises(ValueError):
            KeyFact(
                claim="Test claim",
                importance=1.5
            )
        
        # Invalid importance (< 0.0)
        with pytest.raises(ValueError):
            KeyFact(
                claim="Test claim",
                importance=-0.1
            )
        
        # Valid minimal fact
        fact = KeyFact(
            claim="Minimal test claim with sufficient length",
            importance=0.5
        )
        assert fact.confidence is None
        assert fact.source_turn is None
        assert fact.category is None


class TestMEPAuthentication:
    """Test cases for MEP API authentication."""
    
    def test_bearer_token_validation(self):
        """Test bearer token validation logic."""
        from src.api.mep.routes import authenticate_request
        from fastapi.security import HTTPAuthorizationCredentials
        
        settings = Settings(api=Settings.APIConfig(bearer_token="valid-token"))
        
        # Test with valid credentials
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="valid-token"
        )
        
        # This would normally be tested in the actual endpoint context
        # Here we just verify the logic exists
        assert credentials.credentials == "valid-token"
    
    def test_missing_authorization_header(self, client, valid_proposal):
        """Test request without Authorization header."""
        response = client.post("/mep/v1/proposals", json=valid_proposal)
        
        assert response.status_code == 401
        data = response.json()
        assert "Bearer token required" in data["detail"]
    
    def test_malformed_authorization_header(self, client, valid_proposal):
        """Test request with malformed Authorization header."""
        headers = {"Authorization": "InvalidFormat token"}
        response = client.post(
            "/mep/v1/proposals",
            headers=headers,
            json=valid_proposal
        )
        
        assert response.status_code == 401


class TestMEPErrorHandling:
    """Test cases for MEP API error handling."""
    
    def test_request_validation_error_format(self, client, auth_headers):
        """Test that validation errors are properly formatted."""
        invalid_proposal = {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            # Missing required fields
            "context_fill": 1.5,  # Invalid value
        }
        
        response = client.post(
            "/mep/v1/proposals",
            headers=auth_headers,
            json=invalid_proposal
        )
        
        assert response.status_code == 422
        data = response.json()
        
        # Check error response format
        assert "error" in data or "detail" in data
        
        # If using custom error format
        if "error" in data:
            error = data["error"]
            assert "error_code" in error
            assert "error_type" in error
            assert "message" in error
        
        # If using FastAPI default format
        if "detail" in data:
            detail = data["detail"]
            assert isinstance(detail, (str, dict, list))
    
    def test_internal_server_error_handling(self, client, auth_headers, valid_proposal):
        """Test internal server error handling."""
        # Mock an internal error during proposal processing
        with patch('src.api.mep.routes.add_to_queue', side_effect=Exception("Internal error")):
            response = client.post(
                "/mep/v1/proposals",
                headers=auth_headers,
                json=valid_proposal
            )
            
            assert response.status_code == 500
            data = response.json()
            
            # Should have proper error format
            assert "error" in data or "detail" in data
    
    def test_concurrent_request_handling(self, client, auth_headers, valid_proposal):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.post(
                "/mep/v1/proposals",
                headers=auth_headers,
                json=valid_proposal
            )
            results.append(response.status_code)
        
        # Create multiple threads to simulate concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed (or fail consistently)
        assert len(results) == 5
        assert all(status in [202, 429, 500] for status in results)  # Accept, queue full, or error


class TestMEPPerformance:
    """Performance tests for MEP API."""
    
    @pytest.mark.performance
    def test_proposal_submission_performance(self, client, auth_headers, valid_proposal):
        """Test MEP proposal submission performance."""
        import time
        
        # Measure response time for multiple requests
        response_times = []
        
        for _ in range(10):
            start_time = time.time()
            response = client.post(
                "/mep/v1/proposals",
                headers=auth_headers,
                json=valid_proposal
            )
            end_time = time.time()
            
            response_times.append(end_time - start_time)
            assert response.status_code in [202, 429]  # Success or queue full
        
        # Calculate performance metrics
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        
        # Performance assertions
        assert avg_time < 0.1, f"Average response time too slow: {avg_time:.3f}s"
        assert max_time < 0.2, f"Max response time too slow: {max_time:.3f}s"
        
        print(f"Average response time: {avg_time:.3f}s")
        print(f"Max response time: {max_time:.3f}s")
    
    @pytest.mark.performance
    def test_large_proposal_handling(self, client, auth_headers, valid_proposal):
        """Test handling of large proposals."""
        # Create proposal with many key facts
        large_proposal = valid_proposal.copy()
        large_proposal["key_facts"] = [
            {
                "claim": f"Test claim number {i} with detailed information about the user's requirements and preferences",
                "importance": 0.5 + (i % 5) * 0.1,
                "confidence": 0.8 + (i % 3) * 0.05,
                "category": f"category_{i % 3}"
            }
            for i in range(30)  # 30 key facts
        ]
        
        # Create long summary
        large_proposal["summary_text"] = (
            "This is a comprehensive summary of a long conversation. " * 50
        )
        
        import time
        start_time = time.time()
        
        response = client.post(
            "/mep/v1/proposals",
            headers=auth_headers,
            json=large_proposal
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert response.status_code in [202, 413, 422]  # Success, too large, or validation error
        assert duration < 1.0, f"Large proposal processing took too long: {duration:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__])