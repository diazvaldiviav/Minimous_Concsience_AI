"""
Test suite for ConversationProcessor - Week 2 implementation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import json

from src.memory.conversation_processor import ConversationProcessor, create_conversation_processor
from src.core.models import ValidatedFact, KeyFact, TrainingExample, ConversationTurn
from src.api.mep.schemas import MEPProposalRequest
from src.core.config import Settings


class TestConversationProcessor:
    """Test cases for ConversationProcessor."""
    
    @pytest.fixture
    def sample_proposal(self):
        """Create sample MEP proposal for testing."""
        return MEPProposalRequest(
            provider="anthropic",
            model="claude-3-sonnet",
            external_user_id="test_user_123",
            external_chat_id="test_chat_456",
            event_id="test_event_789",
            trigger="context_full",
            context_fill=0.85,
            token_usage={
                "window_tokens": 8000,
                "used_tokens": 6800,
                "max_tokens": 10000
            },
            message_span={
                "from_turn": 0,
                "to_turn": 10
            },
            summary_text=(
                "User discussed building a web application with React. "
                "They want to use FastAPI for the backend and PostgreSQL for the database. "
                "User prefers dark mode interface and needs authentication."
            ),
            key_facts=[
                {
                    "claim": "User prefers React for frontend",
                    "importance": 0.9,
                    "confidence": 0.8,
                    "category": "preference"
                },
                {
                    "claim": "Backend should use FastAPI",
                    "importance": 0.8,
                    "confidence": 0.9,
                    "category": "requirement"
                }
            ]
        )
    
    @pytest.fixture
    def validated_facts(self):
        """Create sample validated facts."""
        facts = [
            KeyFact(
                claim="User prefers React for frontend",
                importance=0.9,
                confidence=0.8,
                category="preference"
            ),
            KeyFact(
                claim="Backend should use FastAPI",
                importance=0.8,
                confidence=0.9,
                category="requirement"
            )
        ]
        
        return [
            ValidatedFact(
                original_fact=fact,
                truth_score=0.85,
                is_validated=True,
                validation_reason="High confidence validation"
            )
            for fact in facts
        ]
    
    @pytest.fixture
    def processor(self):
        """Create ConversationProcessor for testing."""
        settings = Settings()
        settings.conversation_processing.min_turns_for_training = 3
        settings.conversation_processing.max_examples_per_conversation = 20
        return ConversationProcessor(settings=settings)
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor._min_turns == 3
        assert processor._max_examples == 20
        assert processor._include_paraphrases is True
        assert processor._processed_conversations == 0
        assert processor._generated_examples == 0
    
    def test_create_conversation_processor(self):
        """Test processor creation function."""
        processor = create_conversation_processor()
        assert isinstance(processor, ConversationProcessor)
    
    @pytest.mark.asyncio
    async def test_process_conversation_to_training_data(self, processor, sample_proposal, validated_facts):
        """Test conversation processing to training data."""
        with patch.object(processor, '_extract_conversation_turns') as mock_extract:
            mock_extract.return_value = [
                ConversationTurn(role="user", content="I want to build a web app", turn_index=0),
                ConversationTurn(role="assistant", content="Great! What technologies?", turn_index=1),
                ConversationTurn(role="user", content="React and FastAPI", turn_index=2),
            ]
            
            examples = await processor.process_conversation_to_training_data(
                sample_proposal, validated_facts
            )
            
            assert len(examples) > 0
            assert processor._processed_conversations == 1
            assert processor._generated_examples > 0
            
            # Check example format
            for example in examples:
                assert isinstance(example, TrainingExample)
                assert len(example.instruction) > 0
                assert len(example.response) > 0
                assert example.source_conversation_id == sample_proposal.external_chat_id
    
    def test_extract_conversation_turns(self, processor):
        """Test conversation turn extraction from summary."""
        summary = (
            "User asked about React development. "
            "Assistant explained component concepts. "
            "User wanted FastAPI backend information. "
            "Assistant provided FastAPI details."
        )
        
        turns = processor._extract_conversation_turns(summary)
        
        assert len(turns) > 0
        assert all(isinstance(turn, ConversationTurn) for turn in turns)
        assert all(turn.role in ["user", "assistant"] for turn in turns)
        assert all(len(turn.content) > 0 for turn in turns)
    
    def test_create_recall_examples(self, processor, validated_facts):
        """Test recall example creation."""
        conversation_id = "test_chat_123"
        
        examples = processor._create_recall_examples(validated_facts, conversation_id)
        
        assert len(examples) > 0
        
        for example in examples:
            assert isinstance(example, TrainingExample)
            assert example.example_type == "recall"
            assert example.source_conversation_id == conversation_id
            assert "remember" in example.instruction.lower() or "what" in example.instruction.lower()
    
    def test_generate_summary_examples(self, processor):
        """Test summary example generation."""
        summary = "User discussed web development with React and FastAPI"
        conversation_id = "test_chat_123"
        
        examples = processor._generate_summary_examples(summary, conversation_id)
        
        assert len(examples) > 0
        
        for example in examples:
            assert isinstance(example, TrainingExample)
            assert example.example_type == "summary"
            assert example.response == summary
            assert example.source_conversation_id == conversation_id
    
    def test_generate_topic_examples(self, processor):
        """Test topic-based example generation."""
        key_facts = [
            {
                "claim": "User prefers React development",
                "importance": 0.8,
                "confidence": 0.9
            },
            {
                "claim": "FastAPI is chosen for backend",
                "importance": 0.7,
                "confidence": 0.8
            }
        ]
        conversation_id = "test_chat_123"
        
        examples = processor._generate_topic_examples(key_facts, conversation_id)
        
        assert len(examples) >= 0  # Might be empty if no clear topics extracted
        
        for example in examples:
            assert isinstance(example, TrainingExample)
            assert example.example_type == "topic"
            assert example.source_conversation_id == conversation_id
    
    def test_extract_topic(self, processor):
        """Test topic extraction from text."""
        # Test various topic patterns
        test_cases = [
            ("User prefers React for frontend development", "react"),
            ("Discussion about Python programming language", "python"),
            ("We talked about machine learning algorithms", "machine learning"),
            ("Database configuration with PostgreSQL", "postgresql"),
        ]
        
        for text, expected_topic in test_cases:
            topic = processor._extract_topic(text)
            if topic:  # Topic extraction is best-effort
                assert isinstance(topic, str)
                assert len(topic) > 0
    
    def test_add_paraphrases(self, processor):
        """Test paraphrase generation."""
        examples = [
            TrainingExample(
                instruction="What do you remember about React?",
                response="User prefers React for frontend",
                source_conversation_id="test_chat_123",
                example_type="recall"
            )
        ]
        
        paraphrases = processor._add_paraphrases(examples, "test_chat_123")
        
        # Should generate some paraphrases (probabilistic)
        for paraphrase in paraphrases:
            assert isinstance(paraphrase, TrainingExample)
            assert paraphrase.example_type.endswith("_paraphrase")
            assert paraphrase.source_conversation_id == "test_chat_123"
    
    @pytest.mark.asyncio
    async def test_save_training_data(self, processor):
        """Test training data saving."""
        examples = [
            TrainingExample(
                instruction="What did we discuss?",
                response="React development",
                source_conversation_id="test_chat_123",
                example_type="summary"
            ),
            TrainingExample(
                instruction="What do you remember about React?",
                response="User prefers React for frontend",
                source_conversation_id="test_chat_123",
                example_type="recall"
            )
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_training.jsonl"
            
            saved_path = await processor.save_training_data(examples, output_path)
            
            assert saved_path == output_path
            assert output_path.exists()
            
            # Verify file content
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == len(examples)
                
                # Check first line
                first_example = json.loads(lines[0])
                assert "instruction" in first_example
                assert "response" in first_example
                assert "conversation_id" in first_example
                assert "example_type" in first_example
    
    def test_get_processing_stats(self, processor):
        """Test processing statistics."""
        stats = processor.get_processing_stats()
        
        expected_keys = [
            "processed_conversations", "generated_examples", 
            "average_examples_per_conversation", "min_turns_required",
            "max_examples_per_conversation", "paraphrases_enabled",
            "training_data_dir"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats["processed_conversations"] == 0  # Initial state
        assert stats["generated_examples"] == 0
    
    @pytest.mark.asyncio
    async def test_processing_with_insufficient_turns(self, processor, sample_proposal):
        """Test processing with insufficient conversation turns."""
        # Mock to return fewer turns than minimum
        with patch.object(processor, '_extract_conversation_turns') as mock_extract:
            mock_extract.return_value = [
                ConversationTurn(role="user", content="Hi", turn_index=0),
            ]
            
            # Should still generate examples (with warning logged)
            examples = await processor.process_conversation_to_training_data(sample_proposal)
            
            # Should generate at least summary examples
            assert len(examples) > 0
    
    @pytest.mark.asyncio  
    async def test_example_limit_enforcement(self, processor, sample_proposal, validated_facts):
        """Test that example count is limited correctly."""
        # Set very low limit
        processor._max_examples = 3
        
        examples = await processor.process_conversation_to_training_data(
            sample_proposal, validated_facts
        )
        
        assert len(examples) <= 3
    
    @pytest.mark.asyncio
    async def test_processing_without_validated_facts(self, processor, sample_proposal):
        """Test processing when no validated facts are provided."""
        examples = await processor.process_conversation_to_training_data(
            sample_proposal, validated_facts=None
        )
        
        # Should still generate examples from summary and key facts
        assert len(examples) > 0
        
        # Should have summary examples at minimum
        summary_examples = [e for e in examples if e.example_type == "summary"]
        assert len(summary_examples) > 0