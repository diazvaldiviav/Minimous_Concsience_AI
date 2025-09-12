"""
Test suite for MemoryConsolidator - Week 2 implementation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from src.memory.consolidator import MemoryConsolidator, create_memory_consolidator
from src.core.models import ConsolidationResult, ValidatedFact, KeyFact
from src.api.mep.schemas import MEPProposalRequest
from src.core.config import Settings
from src.core.exceptions import ConsolidationError


class TestMemoryConsolidator:
    """Test cases for MemoryConsolidator."""
    
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
            summary_text="User discussed React development with FastAPI backend.",
            key_facts=[
                {
                    "claim": "User prefers React",
                    "importance": 0.8,
                    "confidence": 0.9,
                    "category": "preference"
                }
            ]
        )
    
    @pytest.fixture
    def consolidator(self):
        """Create MemoryConsolidator for testing."""
        return MemoryConsolidator()
    
    def test_consolidator_initialization(self, consolidator):
        """Test consolidator initialization."""
        assert consolidator._truth_model is not None
        assert consolidator._conversation_processor is not None
        assert len(consolidator._consolidation_history) == 0
        assert len(consolidator._active_consolidations) == 0
    
    def test_create_memory_consolidator(self):
        """Test consolidator creation function."""
        consolidator = create_memory_consolidator()
        assert isinstance(consolidator, MemoryConsolidator)
    
    @pytest.mark.asyncio
    async def test_consolidate_conversation_success(self, consolidator, sample_proposal):
        """Test successful conversation consolidation."""
        # Mock all the sub-components
        mock_validated_facts = [
            ValidatedFact(
                original_fact=KeyFact(claim="User prefers React", importance=0.8, confidence=0.9),
                truth_score=0.85,
                is_validated=True,
                validation_reason="Test validation"
            )
        ]
        
        mock_adapter = Mock()
        mock_adapter.adapter_id = "test_adapter_123"
        mock_adapter.training_examples_count = 15
        
        with patch.object(consolidator, '_validate_conversation_facts', 
                         return_value=mock_validated_facts) as mock_validate:
            with patch.object(consolidator, '_convert_to_training_data',
                             return_value=Path("/tmp/training.jsonl")) as mock_convert:
                with patch.object(consolidator, '_train_conversation_adapter',
                                 return_value=mock_adapter) as mock_train:
                    with patch.object(consolidator, '_store_adapter_metadata') as mock_store:
                        
                        result = await consolidator.consolidate_conversation(sample_proposal)
                        
                        # Verify result
                        assert isinstance(result, ConsolidationResult)
                        assert result.status == "success"
                        assert result.adapter_id == "test_adapter_123"
                        assert result.validated_facts_count == 1
                        assert result.training_examples_count == 15
                        assert result.training_time_seconds > 0
                        
                        # Verify all stages were called
                        mock_validate.assert_called_once_with(sample_proposal)
                        mock_convert.assert_called_once()
                        mock_train.assert_called_once()
                        mock_store.assert_called_once()
                        
                        # Verify history was updated
                        assert len(consolidator._consolidation_history) == 1
                        history_entry = consolidator._consolidation_history[0]
                        assert history_entry["status"] == "success"
                        assert history_entry["adapter_id"] == "test_adapter_123"
    
    @pytest.mark.asyncio
    async def test_consolidate_conversation_failure(self, consolidator, sample_proposal):
        """Test conversation consolidation failure handling."""
        # Mock validation to raise an exception
        with patch.object(consolidator, '_validate_conversation_facts',
                         side_effect=Exception("Validation failed")):
            
            result = await consolidator.consolidate_conversation(sample_proposal)
            
            # Verify failure result
            assert isinstance(result, ConsolidationResult)
            assert result.status == "failed"
            assert result.adapter_id == ""
            assert result.error_message == "Validation failed"
            
            # Verify history was updated
            assert len(consolidator._consolidation_history) == 1
            history_entry = consolidator._consolidation_history[0]
            assert history_entry["status"] == "failed"
            assert "error" in history_entry
    
    @pytest.mark.asyncio
    async def test_validate_conversation_facts(self, consolidator, sample_proposal):
        """Test conversation fact validation."""
        # Mock truth model
        mock_validated_facts = [
            ValidatedFact(
                original_fact=KeyFact(claim="Test claim", importance=0.8, confidence=0.9),
                truth_score=0.85,
                is_validated=True,
                validation_reason="Test"
            )
        ]
        
        consolidator._truth_model._is_initialized = True
        with patch.object(consolidator._truth_model, 'validate_key_facts',
                         return_value=mock_validated_facts) as mock_validate:
            
            result = await consolidator._validate_conversation_facts(sample_proposal)
            
            assert result == mock_validated_facts
            mock_validate.assert_called_once_with(
                sample_proposal.key_facts, sample_proposal.summary_text
            )
    
    @pytest.mark.asyncio
    async def test_convert_to_training_data(self, consolidator, sample_proposal):
        """Test training data conversion."""
        mock_validated_facts = [
            ValidatedFact(
                original_fact=KeyFact(claim="Test claim", importance=0.8, confidence=0.9),
                truth_score=0.85,
                is_validated=True,
                validation_reason="Test"
            )
        ]
        
        mock_training_examples = [Mock(), Mock()]  # Mock training examples
        mock_data_path = Path("/tmp/training.jsonl")
        
        with patch.object(consolidator._conversation_processor, 
                         'process_conversation_to_training_data',
                         return_value=mock_training_examples) as mock_process:
            with patch.object(consolidator._conversation_processor, 'save_training_data',
                             return_value=mock_data_path) as mock_save:
                
                result = await consolidator._convert_to_training_data(
                    sample_proposal, mock_validated_facts
                )
                
                assert result == mock_data_path
                mock_process.assert_called_once_with(sample_proposal, mock_validated_facts)
                mock_save.assert_called_once_with(mock_training_examples)
    
    @pytest.mark.asyncio
    async def test_convert_to_training_data_no_examples(self, consolidator, sample_proposal):
        """Test training data conversion when no examples are generated."""
        with patch.object(consolidator._conversation_processor,
                         'process_conversation_to_training_data',
                         return_value=[]):  # Empty list
            
            with pytest.raises(ConsolidationError) as exc_info:
                await consolidator._convert_to_training_data(sample_proposal, [])
            
            assert "No training examples generated" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_train_conversation_adapter(self, consolidator):
        """Test conversation adapter training."""
        training_data_path = Path("/tmp/training.jsonl")
        conversation_id = "test_chat_123"
        
        mock_adapter = Mock()
        mock_adapter.adapter_id = "test_adapter_456"
        
        # Mock base model and LoRA trainer
        mock_base_model = Mock()
        mock_lora_trainer = Mock()
        mock_lora_trainer.train_conversation_adapter = AsyncMock(return_value=mock_adapter)
        
        consolidator._base_model = mock_base_model
        consolidator._lora_trainer = mock_lora_trainer
        
        result = await consolidator._train_conversation_adapter(
            training_data_path, conversation_id
        )
        
        assert result == mock_adapter
        mock_lora_trainer.train_conversation_adapter.assert_called_once_with(
            training_data_path, conversation_id
        )
    
    @pytest.mark.asyncio
    async def test_store_adapter_metadata(self, consolidator, sample_proposal):
        """Test adapter metadata storage."""
        mock_adapter = Mock()
        mock_adapter.adapter_id = "test_adapter_123"
        mock_adapter.conversation_id = "test_chat_456"
        mock_adapter.adapter_path = Path("/tmp/adapter")
        mock_adapter.training_examples_count = 10
        mock_adapter.training_date.isoformat.return_value = "2023-01-01T00:00:00"
        mock_adapter.performance_metrics = {"loss": 0.5}
        
        # Should not raise an exception (metadata storage is not critical)
        await consolidator._store_adapter_metadata(mock_adapter, sample_proposal)
    
    @pytest.mark.asyncio
    async def test_update_consolidation_status(self, consolidator):
        """Test consolidation status updates."""
        proposal_id = "test_proposal_123"
        
        # Add initial tracking
        consolidator._active_consolidations[proposal_id] = {
            "stage": "starting",
            "start_time": 1234567890
        }
        
        # Update status
        await consolidator._update_consolidation_status(
            proposal_id, "validating", {"progress": 50}
        )
        
        # Verify update
        status = consolidator._active_consolidations[proposal_id]
        assert status["stage"] == "validating"
        assert status["progress"] == 50
        assert "last_update" in status
    
    def test_get_consolidation_status(self, consolidator):
        """Test getting consolidation status."""
        proposal_id = "test_proposal_123"
        
        # Test non-existent proposal
        status = consolidator.get_consolidation_status(proposal_id)
        assert status is None
        
        # Add proposal and test
        test_status = {"stage": "training", "progress": 75}
        consolidator._active_consolidations[proposal_id] = test_status
        
        status = consolidator.get_consolidation_status(proposal_id)
        assert status == test_status
    
    def test_get_active_consolidations(self, consolidator):
        """Test getting active consolidations."""
        # Test empty state
        active = consolidator.get_active_consolidations()
        assert len(active) == 0
        
        # Add some active consolidations
        consolidator._active_consolidations["prop1"] = {"stage": "validating"}
        consolidator._active_consolidations["prop2"] = {"stage": "training"}
        
        active = consolidator.get_active_consolidations()
        assert len(active) == 2
        assert "prop1" in active
        assert "prop2" in active
    
    def test_get_consolidation_history(self, consolidator):
        """Test getting consolidation history."""
        # Test empty state
        history = consolidator.get_consolidation_history()
        assert len(history) == 0
        
        # Add some history
        consolidator._consolidation_history.append({
            "proposal_id": "prop1",
            "status": "success",
            "processing_time": 30.5
        })
        
        history = consolidator.get_consolidation_history()
        assert len(history) == 1
        assert history[0]["proposal_id"] == "prop1"
    
    def test_get_consolidation_stats(self, consolidator):
        """Test getting consolidation statistics."""
        # Add some test data
        consolidator._consolidation_history = [
            {"status": "success", "processing_time": 30.0},
            {"status": "success", "processing_time": 40.0},
            {"status": "failed", "processing_time": 10.0},
        ]
        
        consolidator._active_consolidations = {
            "prop1": {"stage": "training"},
            "prop2": {"stage": "validating"}
        }
        
        stats = consolidator.get_consolidation_stats()
        
        expected_keys = [
            "total_consolidations", "successful_consolidations", 
            "failed_consolidations", "success_rate",
            "average_processing_time", "active_consolidations",
            "components_initialized"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats["total_consolidations"] == 3
        assert stats["successful_consolidations"] == 2
        assert stats["failed_consolidations"] == 1
        assert stats["success_rate"] == 2/3
        assert stats["average_processing_time"] == 35.0  # (30+40)/2
        assert stats["active_consolidations"] == 2