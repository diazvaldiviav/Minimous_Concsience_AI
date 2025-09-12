"""
Test suite for TruthModel - Week 2 implementation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from src.memory.truth_model import TruthModel, create_truth_model
from src.core.models import KeyFact, ValidatedFact
from src.core.config import Settings
from src.core.exceptions import TruthValidationError


class TestTruthModel:
    """Test cases for TruthModel."""
    
    @pytest.fixture
    def sample_key_facts(self):
        """Create sample key facts for testing."""
        return [
            KeyFact(
                claim="User prefers React for frontend",
                importance=0.9,
                confidence=0.8,
                category="preference"
            ),
            KeyFact(
                claim="Database should use PostgreSQL",
                importance=0.8,
                confidence=0.9,
                category="requirement"
            ),
            KeyFact(
                claim="User dislikes Java programming",
                importance=0.6,
                confidence=0.7,
                category="preference"
            )
        ]
    
    @pytest.fixture
    def conversation_context(self):
        """Sample conversation context."""
        return (
            "User discussed building a web application. They mentioned wanting to use "
            "React for the frontend because of its component-based architecture. "
            "For the backend, they want to use PostgreSQL database for data storage. "
            "They also expressed that they prefer Python over other languages."
        )
    
    @pytest.fixture
    def truth_model(self):
        """Create TruthModel instance for testing."""
        settings = Settings()
        settings.truth_model.confidence_threshold = 0.7
        return TruthModel(settings=settings)
    
    def test_truth_model_initialization(self, truth_model):
        """Test TruthModel initialization."""
        assert truth_model._confidence_threshold == 0.7
        assert truth_model._enable_nli is True
        assert not truth_model._is_initialized
    
    def test_create_truth_model(self):
        """Test truth model creation function."""
        model = create_truth_model()
        assert isinstance(model, TruthModel)
        assert model._confidence_threshold == 0.7  # Default
    
    @pytest.mark.asyncio
    async def test_load_nli_model(self, truth_model):
        """Test NLI model loading."""
        with patch.object(truth_model, '_train_dummy_nli_model', new_callable=AsyncMock):
            await truth_model.load_nli_model()
            
            assert truth_model._is_initialized
            assert truth_model._vectorizer is not None
            assert truth_model._nli_classifier is not None
    
    @pytest.mark.asyncio
    async def test_validate_key_facts(self, truth_model, sample_key_facts, conversation_context):
        """Test key facts validation."""
        # Mock the initialization and validation methods
        truth_model._is_initialized = True
        truth_model._vectorizer = Mock()
        truth_model._nli_classifier = Mock()
        
        with patch.object(truth_model, '_check_fact_consistency', return_value=0.8):
            with patch.object(truth_model, '_apply_truth_threshold', return_value=True):
                
                validated_facts = await truth_model.validate_key_facts(
                    sample_key_facts, conversation_context
                )
                
                assert len(validated_facts) == len(sample_key_facts)
                
                for vf in validated_facts:
                    assert isinstance(vf, ValidatedFact)
                    assert vf.truth_score == 0.8
                    assert vf.is_validated is True
    
    @pytest.mark.asyncio
    async def test_fact_consistency_checking(self, truth_model, sample_key_facts, conversation_context):
        """Test fact consistency checking."""
        # Mock embeddings manager
        mock_embeddings = Mock()
        mock_embeddings.is_loaded.return_value = True
        mock_embeddings.generate_embedding = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
        
        truth_model._embeddings = mock_embeddings
        truth_model._enable_nli = False  # Disable NLI for simpler test
        
        fact = sample_key_facts[0]  # "User prefers React for frontend"
        
        confidence = await truth_model._check_fact_consistency(fact, conversation_context)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should have reasonable confidence
    
    def test_apply_truth_threshold(self, truth_model):
        """Test truth threshold application."""
        # Test with confidence above threshold
        assert truth_model._apply_truth_threshold(0.8) is True
        
        # Test with confidence below threshold  
        assert truth_model._apply_truth_threshold(0.6) is False
        
        # Test with confidence at threshold
        assert truth_model._apply_truth_threshold(0.7) is True
    
    def test_check_contradictions(self, truth_model):
        """Test contradiction checking."""
        # Test contradictory statements
        fact_claim = "User likes Python programming"
        contradictory_context = "User mentioned they don't like Python and prefer Java instead"
        
        penalty = truth_model._check_contradictions(fact_claim, contradictory_context)
        assert penalty > 0.0
        
        # Test consistent statements
        consistent_context = "User expressed enthusiasm for Python development"
        penalty = truth_model._check_contradictions(fact_claim, consistent_context)
        assert penalty == 0.0
    
    def test_text_overlap_calculation(self, truth_model):
        """Test simple text overlap calculation."""
        fact_claim = "User prefers React framework"
        context = "We discussed React development and user preferences"
        
        overlap = truth_model._calculate_text_overlap(fact_claim, context)
        assert 0.0 <= overlap <= 1.0
        assert overlap > 0.0  # Should have some overlap
        
        # Test with no overlap
        no_overlap_context = "Completely different discussion about cooking recipes"
        overlap = truth_model._calculate_text_overlap(fact_claim, no_overlap_context)
        assert overlap == 0.0
    
    def test_get_validation_reason(self, truth_model):
        """Test validation reason generation."""
        # High confidence, valid
        reason = truth_model._get_validation_reason(0.95, True)
        assert "High confidence" in reason
        
        # Low confidence, invalid
        reason = truth_model._get_validation_reason(0.3, False)
        assert "Low confidence" in reason
        
        # Below threshold
        reason = truth_model._get_validation_reason(0.65, False)
        assert "threshold" in reason.lower()
    
    def test_get_validation_stats(self, truth_model):
        """Test validation statistics."""
        stats = truth_model.get_validation_stats()
        
        expected_keys = [
            "is_initialized", "confidence_threshold", "nli_enabled",
            "max_facts_per_conversation", "has_embeddings", "embeddings_loaded"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats["confidence_threshold"] == 0.7
        assert stats["nli_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_fact_limit_enforcement(self, truth_model):
        """Test that fact count is limited correctly."""
        # Create more facts than the limit
        many_facts = [
            KeyFact(claim=f"Fact {i}", importance=0.5, confidence=0.5)
            for i in range(25)  # More than max_facts (20)
        ]
        
        truth_model._is_initialized = True
        
        with patch.object(truth_model, '_check_fact_consistency', return_value=0.8):
            validated_facts = await truth_model.validate_key_facts(
                many_facts, "test context"
            )
            
            assert len(validated_facts) == truth_model._max_facts
    
    @pytest.mark.asyncio
    async def test_nli_score_calculation(self, truth_model):
        """Test NLI score calculation."""
        # Mock the classifier components
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        
        mock_classifier = Mock()
        mock_classifier.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% entailment
        
        truth_model._vectorizer = mock_vectorizer
        truth_model._nli_classifier = mock_classifier
        
        score = truth_model._calculate_nli_score("premise", "hypothesis")
        
        assert score == 0.7
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, truth_model, sample_key_facts):
        """Test error handling during validation."""
        truth_model._is_initialized = True
        
        # Mock a method to raise an exception
        with patch.object(truth_model, '_check_fact_consistency', side_effect=Exception("Test error")):
            
            validated_facts = await truth_model.validate_key_facts(
                sample_key_facts, "test context"
            )
            
            # Should still return results, but with failed validations
            assert len(validated_facts) == len(sample_key_facts)
            
            for vf in validated_facts:
                assert vf.truth_score == 0.0
                assert vf.is_validated is False
                assert "Validation error" in vf.validation_reason