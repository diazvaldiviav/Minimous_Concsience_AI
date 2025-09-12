"""
Tests for MAP API ContextBuilder and TokenBudgetManager.

This module tests the compressed memory context generation functionality
including token budget management and response formatting.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.api.map.context_builder import ContextBuilder, TokenBudgetManager
from src.core.config import Settings
from src.core.models import (
    AdapterInfo, CompressedTurn, FilteredFact, TokenAllocation, 
    ConversationTurn, ValidatedFact
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.model = Mock()
    settings.model.name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    return settings


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 2
    tokenizer.decode.return_value = "decoded text"
    return tokenizer


class TestTokenBudgetManager:
    """Test token budget management functionality."""
    
    def test_count_tokens(self, mock_tokenizer):
        """Test token counting functionality."""
        budget_manager = TokenBudgetManager(mock_tokenizer)
        
        # Test normal text
        count = budget_manager.count_tokens("Hello world")
        assert count == 5  # Mock returns 5 tokens
        
        # Test empty text
        count = budget_manager.count_tokens("")
        assert count == 0
    
    def test_count_tokens_fallback(self, mock_tokenizer):
        """Test token counting fallback when tokenizer fails."""
        mock_tokenizer.encode.side_effect = Exception("Tokenizer failed")
        
        budget_manager = TokenBudgetManager(mock_tokenizer)
        
        # Should fall back to character-based estimation
        count = budget_manager.count_tokens("Hello world")  # 11 chars / 4 = ~2.75 -> 2
        assert count == 2
    
    def test_allocate_tokens(self, mock_tokenizer):
        """Test token allocation across response components."""
        budget_manager = TokenBudgetManager(mock_tokenizer)
        
        allocation = budget_manager.allocate_tokens(
            total_budget=1000,
            gist_ratio=0.4,
            turns_ratio=0.4,
            facts_ratio=0.15,
            metadata_ratio=0.05
        )
        
        assert allocation.total_budget == 1000
        assert allocation.gist_tokens == 400
        assert allocation.turns_tokens == 400
        assert allocation.facts_tokens == 150
        assert allocation.metadata_tokens == 50
        assert allocation.total_allocated == 1000
    
    def test_allocate_tokens_normalized_ratios(self, mock_tokenizer):
        """Test token allocation with unnormalized ratios."""
        budget_manager = TokenBudgetManager(mock_tokenizer)
        
        # Ratios that don't sum to 1.0
        allocation = budget_manager.allocate_tokens(
            total_budget=1000,
            gist_ratio=0.8,  # Total = 1.6, should be normalized
            turns_ratio=0.4,
            facts_ratio=0.3,
            metadata_ratio=0.1
        )
        
        assert allocation.total_budget == 1000
        assert allocation.total_allocated == 1000
    
    def test_compress_to_fit_no_compression_needed(self, mock_tokenizer):
        """Test compression when content already fits."""
        budget_manager = TokenBudgetManager(mock_tokenizer)
        
        content = "Short content"
        compressed = budget_manager.compress_to_fit(content, max_tokens=10)
        
        assert compressed == content  # Should be unchanged
    
    def test_compress_to_fit_with_compression(self, mock_tokenizer):
        """Test content compression when it exceeds budget."""
        # Mock tokenizer to return decreasing token counts for shorter text
        def mock_encode_with_length(text, **kwargs):
            return [1] * len(text)  # 1 token per character for simplicity
        
        mock_tokenizer.encode.side_effect = mock_encode_with_length
        
        budget_manager = TokenBudgetManager(mock_tokenizer)
        
        content = "This is a very long content that exceeds the token budget"
        compressed = budget_manager.compress_to_fit(content, max_tokens=20)
        
        # Should be compressed and end with ellipsis
        assert len(compressed) < len(content)
        assert compressed.endswith("...")
    
    def test_compress_to_fit_empty_content(self, mock_tokenizer):
        """Test compression with empty content."""
        budget_manager = TokenBudgetManager(mock_tokenizer)
        
        compressed = budget_manager.compress_to_fit("", max_tokens=100)
        assert compressed == ""


class TestContextBuilder:
    """Test context builder functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_settings):
        """Test context builder initialization."""
        context_builder = ContextBuilder(mock_settings)
        
        with patch('src.api.map.context_builder.AutoTokenizer') as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.return_value = Mock()
            mock_tokenizer_class.from_pretrained.return_value.pad_token = None
            mock_tokenizer_class.from_pretrained.return_value.eos_token = "</s>"
            
            await context_builder.initialize()
            
            assert context_builder.tokenizer is not None
            assert context_builder.token_budget_manager is not None
    
    @pytest.mark.asyncio
    async def test_generate_gist(self, mock_settings):
        """Test conversation gist generation."""
        context_builder = ContextBuilder(mock_settings)
        context_builder.tokenizer = Mock()
        context_builder.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        context_builder.tokenizer.decode.return_value = "Generated gist about physics"
        context_builder.tokenizer.eos_token_id = 2
        
        context_builder.token_budget_manager = Mock()
        context_builder.token_budget_manager.compress_to_fit.return_value = "Compressed gist"
        
        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7, 8]]
        
        adapter_info = AdapterInfo(
            adapter_id="test_1",
            conversation_id="conv_1",
            provider="anthropic",
            external_user_id="user_1",
            adapter_path="/path/1",
            topic="physics",
            turn_range={"from_turn": 1, "to_turn": 10}
        )
        
        with patch('torch.no_grad'):
            gist = await context_builder.generate_gist(
                query="What is physics?",
                model=mock_model,
                adapter_info=adapter_info,
                max_tokens=100
            )
        
        assert gist == "Compressed gist"
        mock_model.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_gist_fallback(self, mock_settings):
        """Test gist generation with fallback when model fails."""
        context_builder = ContextBuilder(mock_settings)
        context_builder.tokenizer = Mock()
        context_builder.token_budget_manager = Mock()
        
        mock_model = Mock()
        mock_model.generate.side_effect = Exception("Model failed")
        
        adapter_info = AdapterInfo(
            adapter_id="test_1",
            conversation_id="conv_1",
            provider="anthropic",
            external_user_id="user_1",
            adapter_path="/path/1",
            topic="physics"
        )
        
        gist = await context_builder.generate_gist(
            query="What is physics?",
            model=mock_model,
            adapter_info=adapter_info,
            max_tokens=100
        )
        
        # Should generate fallback gist
        assert "physics" in gist.lower()
        assert "What is physics?" in gist
    
    @pytest.mark.asyncio
    async def test_extract_turn_sketch(self, mock_settings):
        """Test conversation turn extraction."""
        context_builder = ContextBuilder(mock_settings)
        context_builder.token_budget_manager = Mock()
        context_builder.token_budget_manager.count_tokens.return_value = 15
        context_builder.token_budget_manager.compress_to_fit.return_value = "Compressed turn"
        
        conversation_turns = [
            ConversationTurn(
                turn_number=1,
                role="user",
                content="What is quantum physics?",
                timestamp=datetime.utcnow()
            ),
            ConversationTurn(
                turn_number=2,
                role="assistant", 
                content="Quantum physics is the study of matter and energy at the smallest scales...",
                timestamp=datetime.utcnow()
            )
        ]
        
        compressed_turns = await context_builder.extract_turn_sketch(
            conversation_turns=conversation_turns,
            max_tokens=100,
            query="quantum physics"
        )
        
        assert len(compressed_turns) == 2
        assert compressed_turns[0].r == "u"
        assert compressed_turns[1].r == "a"
        assert compressed_turns[0].id == "T1U"
        assert compressed_turns[1].id == "T2A"
    
    @pytest.mark.asyncio
    async def test_extract_turn_sketch_empty_input(self, mock_settings):
        """Test turn extraction with empty input."""
        context_builder = ContextBuilder(mock_settings)
        context_builder.token_budget_manager = Mock()
        
        compressed_turns = await context_builder.extract_turn_sketch(
            conversation_turns=[],
            max_tokens=100,
            query=""
        )
        
        assert len(compressed_turns) == 0
    
    @pytest.mark.asyncio
    async def test_compile_facts(self, mock_settings):
        """Test facts compilation and filtering."""
        context_builder = ContextBuilder(mock_settings)
        context_builder.token_budget_manager = Mock()
        context_builder.token_budget_manager.count_tokens.return_value = 10
        context_builder.token_budget_manager.compress_to_fit.return_value = "Compressed fact"
        
        validated_facts = [
            ValidatedFact(
                claim="Quantum mechanics is fundamental to physics",
                confidence=0.95,
                source="textbook",
                timestamp=datetime.utcnow()
            ),
            ValidatedFact(
                claim="Einstein developed relativity theory",
                confidence=0.85,
                source="encyclopedia",
                timestamp=datetime.utcnow()
            ),
            ValidatedFact(
                claim="Low confidence claim",
                confidence=0.60,
                source="blog",
                timestamp=datetime.utcnow()
            )
        ]
        
        filtered_facts = await context_builder.compile_facts(
            validated_facts=validated_facts,
            min_truth=0.75,
            max_tokens=100
        )
        
        # Should filter out low confidence fact
        assert len(filtered_facts) == 2
        assert all(fact.p >= 0.75 for fact in filtered_facts)
        assert filtered_facts[0].p == 0.95  # Should be sorted by confidence
        assert filtered_facts[1].p == 0.85
    
    @pytest.mark.asyncio
    async def test_compile_facts_empty_input(self, mock_settings):
        """Test facts compilation with empty input."""
        context_builder = ContextBuilder(mock_settings)
        context_builder.token_budget_manager = Mock()
        
        filtered_facts = await context_builder.compile_facts(
            validated_facts=[],
            min_truth=0.75,
            max_tokens=100
        )
        
        assert len(filtered_facts) == 0
    
    def test_estimate_response_tokens(self, mock_settings):
        """Test response token estimation."""
        context_builder = ContextBuilder(mock_settings)
        context_builder.token_budget_manager = Mock()
        context_builder.token_budget_manager.count_tokens.side_effect = lambda x: len(x.split())
        
        gist = "This is a test gist"  # 5 words
        turns = [
            CompressedTurn(id="T1U", r="u", t="User question"),  # 2 words
            CompressedTurn(id="T2A", r="a", t="Assistant response here")  # 3 words
        ]
        facts = [
            FilteredFact(c="Physics is science", p=0.95, s="mem"),  # 3 words
            FilteredFact(c="Math is fundamental", p=0.90, s="mem")  # 3 words
        ]
        
        estimated_tokens = context_builder.estimate_response_tokens(gist, turns, facts)
        
        # 5 (gist) + 2+3+5*2 (turns+overhead) + 3+3+3*2 (facts+overhead) + 20 (structure) = 41
        assert estimated_tokens == 41
    
    @pytest.mark.asyncio
    async def test_rank_turns_by_relevance(self, mock_settings):
        """Test turn ranking by query relevance."""
        context_builder = ContextBuilder(mock_settings)
        
        turns = [
            ConversationTurn(
                turn_number=1,
                role="user",
                content="What is the weather?",
                timestamp=datetime.utcnow()
            ),
            ConversationTurn(
                turn_number=2,
                role="user", 
                content="Tell me about quantum physics and mechanics",
                timestamp=datetime.utcnow()
            ),
            ConversationTurn(
                turn_number=3,
                role="user",
                content="How does quantum theory work?",
                timestamp=datetime.utcnow()
            )
        ]
        
        ranked_turns = await context_builder._rank_turns_by_relevance(
            turns=turns,
            query="quantum physics"
        )
        
        # Turn 2 should be first (has both "quantum" and "physics")
        # Turn 3 should be second (has "quantum")  
        # Turn 1 should be last (no matching words)
        assert ranked_turns[0].turn_number == 2
        assert ranked_turns[1].turn_number == 3
        assert ranked_turns[2].turn_number == 1
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, mock_settings):
        """Test context builder initialization failure."""
        context_builder = ContextBuilder(mock_settings)
        
        with patch('src.api.map.context_builder.AutoTokenizer') as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.side_effect = Exception("Load failed")
            
            with pytest.raises(Exception):  # Should raise ModelLoadError
                await context_builder.initialize()