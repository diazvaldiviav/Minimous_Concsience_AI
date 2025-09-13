"""Test hybrid memory implementation."""  # ENGLISH

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import components to test
from src.memory.consolidator import MemoryConsolidator
from src.api.map.routes import _load_conversation_turns, _load_validated_facts
from src.core.models import AdapterInfo, ConversationAdapter, ValidatedFact, KeyFact
from src.api.mep.schemas import MEPProposalRequest


class TestHybridMemory:
    """Test hybrid memory saving and loading."""  # ENGLISH
    
    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """Create temporary test directory."""  # ENGLISH
        data_dir = tmp_path / "data" / "conversations" / "test_adapter_123"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    @pytest.fixture
    def sample_proposal(self):
        """Create sample MEP proposal for testing."""  # ENGLISH
        return MEPProposalRequest(
            provider="anthropic",
            model="claude-3",
            external_user_id="user_123",
            external_chat_id="chat_456",
            event_id="evt_789",
            trigger="context_full",
            context_fill=0.85,
            token_usage={
                "window_tokens": 1000,
                "used_tokens": 850,
                "max_tokens": 2000
            },
            message_span={
                "from_turn": 0,
                "to_turn": 10
            },
            summary_text="Discussion about machine learning concepts",
            key_facts=[
                {
                    "claim": "Neural networks are inspired by the brain",
                    "importance": 0.9
                },
                {
                    "claim": "Backpropagation is used for training",
                    "importance": 0.8
                }
            ]
        )
    
    @pytest.fixture
    def sample_adapter(self):
        """Create sample adapter for testing."""  # ENGLISH
        return Mock(
            adapter_id="test_adapter_123",
            conversation_id="chat_456",
            topic="machine_learning",
            training_examples_count=20,
            training_date=datetime.utcnow(),
            performance_metrics={"loss": 0.1}
        )
    
    @pytest.mark.asyncio
    async def test_saves_conversation_data(self, test_data_dir, sample_proposal, sample_adapter):
        """Test that conversation data is saved correctly."""  # ENGLISH
        
        # Create consolidator
        consolidator = MemoryConsolidator()
        
        # Create validated facts
        validated_facts = [
            ValidatedFact(
                original_fact=KeyFact(
                    claim="Neural networks are inspired by the brain",
                    importance=0.9
                ),
                truth_score=0.85,
                is_validated=True,
                validation_reason="Verified through knowledge base"
            )
        ]
        
        # Mock the data directory path
        with patch('pathlib.Path.mkdir'):
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                # Call the save method
                await consolidator._save_conversation_data(
                    sample_adapter,
                    sample_proposal,
                    validated_facts
                )
                
                # Verify files were written
                assert mock_open.call_count >= 2  # conversation.json and validated_facts.json
    
    @pytest.mark.asyncio
    async def test_loads_real_conversation_data(self, test_data_dir):
        """Test loading real conversation data when it exists."""  # ENGLISH
        
        # Create test conversation data
        conversation_data = {
            "adapter_id": "test_adapter_123",
            "conversation_id": "chat_456",
            "topic": "machine_learning",
            "messages": [
                {
                    "turn_number": 0,
                    "role": "user",
                    "content": "What is machine learning?",
                    "timestamp": "2024-12-13T10:00:00"
                },
                {
                    "turn_number": 1,
                    "role": "assistant",
                    "content": "Machine learning is a subset of AI...",
                    "timestamp": "2024-12-13T10:00:30"
                }
            ],
            "summary": "Discussion about ML concepts",
            "metadata": {
                "provider": "anthropic",
                "user_id": "user_123"
            }
        }
        
        # Write test data to file
        conversation_file = test_data_dir / "conversation.json"
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2)
        
        # Create adapter info with data path
        adapter_info = AdapterInfo(
            adapter_id="test_adapter_123",
            conversation_id="chat_456",
            provider="anthropic",
            external_user_id="user_123",
            adapter_path="/models/adapters/test_adapter_123",
            data_path=str(test_data_dir)
        )
        
        # Load conversation turns
        turns = await _load_conversation_turns(adapter_info)
        
        # Verify loaded data
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[0].content == "What is machine learning?"
        assert turns[1].role == "assistant"
        assert "Machine learning" in turns[1].content
    
    @pytest.mark.asyncio
    async def test_loads_real_facts_data(self, test_data_dir):
        """Test loading real validated facts when they exist."""  # ENGLISH
        
        # Create test facts data
        facts_data = [
            {
                "claim": "Neural networks are inspired by the brain",
                "confidence": 0.85,
                "importance": 0.9,
                "category": "fundamental",
                "validation_reason": "Verified through knowledge base",
                "is_validated": True
            },
            {
                "claim": "Backpropagation is used for training",
                "confidence": 0.82,
                "importance": 0.8,
                "category": "technical",
                "validation_reason": "Common knowledge in ML",
                "is_validated": True
            }
        ]
        
        # Write test data to file
        facts_file = test_data_dir / "validated_facts.json"
        with open(facts_file, 'w', encoding='utf-8') as f:
            json.dump(facts_data, f, indent=2)
        
        # Create adapter info with data path
        adapter_info = AdapterInfo(
            adapter_id="test_adapter_123",
            conversation_id="chat_456",
            provider="anthropic",
            external_user_id="user_123",
            adapter_path="/models/adapters/test_adapter_123",
            data_path=str(test_data_dir)
        )
        
        # Load validated facts
        facts = await _load_validated_facts(adapter_info)
        
        # Verify loaded data
        assert len(facts) == 2
        assert facts[0].original_fact.claim == "Neural networks are inspired by the brain"
        assert facts[0].truth_score == 0.85
        assert facts[0].is_validated == True
        assert facts[1].original_fact.claim == "Backpropagation is used for training"
    
    @pytest.mark.asyncio
    async def test_fallback_to_simulation(self):
        """Test fallback to simulation when no real data exists."""  # ENGLISH
        
        # Create adapter info without data path
        adapter_info = AdapterInfo(
            adapter_id="test_adapter_no_data",
            conversation_id="chat_999",
            provider="anthropic",
            external_user_id="user_999",
            adapter_path="/models/adapters/test_adapter_no_data",
            topic="test_topic",
            data_path=None  # No data path
        )
        
        # Load conversation turns (should fallback to simulation)
        turns = await _load_conversation_turns(adapter_info)
        
        # Verify simulated data is returned
        assert len(turns) > 0
        assert any("test_topic" in turn.content for turn in turns)
        assert all(turn.metadata.get("source") == "simulated" for turn in turns)
        
        # Load facts (should fallback to simulation)
        facts = await _load_validated_facts(adapter_info)
        
        # Verify simulated facts are returned
        assert len(facts) > 0
        assert any("test_topic" in fact.original_fact.claim for fact in facts)
    
    def test_topic_extraction(self):
        """Test topic extraction from summary text."""  # ENGLISH
        
        consolidator = MemoryConsolidator()
        
        # Test normal summary
        topic = consolidator._extract_topic_from_summary(
            "Discussion about machine learning and neural networks. We covered basics."
        )
        assert topic == "discussion_about_machine_learning_and"
        
        # Test empty summary
        topic = consolidator._extract_topic_from_summary("")
        assert topic == "general_conversation"
        
        # Test special characters
        topic = consolidator._extract_topic_from_summary(
            "React.js & Node.js: Full-stack development!"
        )
        assert "react" in topic.lower()
        assert "js" in topic.lower()
    
    def test_message_extraction(self):
        """Test message extraction from proposal."""  # ENGLISH
        
        consolidator = MemoryConsolidator()
        
        # Create proposal with message span
        proposal = Mock(
            message_span=Mock(from_turn=0, to_turn=3),
            summary_text="Test summary"
        )
        
        # Extract messages
        messages = consolidator._extract_messages_from_proposal(proposal)
        
        # Verify message structure
        assert len(messages) == 5  # 4 turns + 1 summary
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[-1]["is_summary"] == True
        assert messages[-1]["content"] == "Test summary"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])