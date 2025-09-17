"""
Memory Consolidation Orchestrator for SC Memory System.

This module orchestrates the complete consolidation pipeline: MEP proposal → 
truth validation → training data generation → LoRA adapter training.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import Settings, get_settings
from ..core.exceptions import ConsolidationError
from ..core.models import ConsolidationResult, ValidatedFact
from ..api.mep.schemas import MEPProposalRequest
from .truth_model import TruthModel, create_truth_model
from .conversation_processor import ConversationProcessor, create_conversation_processor
from .lora_trainer import LoRATrainer, create_lora_trainer
from .base_model import BaseModelManager
from .embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Orchestrates memory consolidation pipeline.
    
    Coordinates truth validation, conversation processing, and LoRA training
    to convert specific conversations into trained adapters that enable
    the model to "remember" those conversations.
    
    Features:
    - Complete consolidation workflow orchestration
    - Status tracking and progress updates
    - Error handling and recovery
    - Performance monitoring
    - Adapter metadata management
    """
    
    def __init__(
        self,
        base_model: Optional[BaseModelManager] = None,
        embeddings_manager: Optional[EmbeddingsManager] = None,
        settings: Optional[Settings] = None
    ) -> None:
        """
        Initialize the memory consolidator.
        
        Args:
            base_model: Optional base model manager
            embeddings_manager: Optional embeddings manager
            settings: Application settings
        """
        self._settings = settings or get_settings()
        
        # Initialize components
        self._base_model = base_model
        self._embeddings = embeddings_manager
        self._truth_model = create_truth_model(
            embeddings_manager=self._embeddings,
            settings=self._settings
        )
        self._conversation_processor = create_conversation_processor(
            settings=self._settings
        )
        self._lora_trainer: Optional[LoRATrainer] = None  # Lazy initialization
        
        # State tracking
        self._consolidation_history: List[Dict[str, Any]] = []
        self._active_consolidations: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized MemoryConsolidator")
    
    async def consolidate_conversation(
        self,
        proposal: MEPProposalRequest
    ) -> ConsolidationResult:
        """
        Consolidate a conversation into a LoRA adapter.
        
        This is the main orchestration method that:
        1. Validates conversation facts with truth model
        2. Converts conversation to training data
        3. Trains LoRA adapter on conversation data
        4. Returns consolidation results
        
        Args:
            proposal: MEP proposal containing conversation data
            
        Returns:
            Consolidation results with adapter information
        """
        proposal_id = proposal.event_id
        conversation_id = proposal.external_chat_id
        start_time = time.time()
        
        # Track active consolidation
        self._active_consolidations[proposal_id] = {
            "stage": "starting",
            "start_time": start_time,
            "conversation_id": conversation_id,
        }
        
        try:
            logger.info(
                f"Starting consolidation for conversation {conversation_id} "
                f"(proposal {proposal_id})"
            )
            
            # Stage 1: Validate conversation facts
            await self._update_consolidation_status(
                proposal_id, "validating", {"stage_start": time.time()}
            )
            
            validated_facts = await self._validate_conversation_facts(proposal)
            
            logger.info(
                f"Validated {len([f for f in validated_facts if f.is_validated])}"
                f"/{len(validated_facts)} facts for conversation {conversation_id}"
            )
            
            # Stage 2: Convert to training data
            await self._update_consolidation_status(
                proposal_id, "building_dataset", {"validated_facts": len(validated_facts)}
            )
            
            training_data_path = await self._convert_to_training_data(
                proposal, validated_facts
            )
            
            # Stage 3: Train conversation adapter
            await self._update_consolidation_status(
                proposal_id, "training", {"training_data_path": str(training_data_path)}
            )
            
            adapter = await self._train_conversation_adapter(
                training_data_path, conversation_id, proposal
            )
            
            # Stage 4: Store adapter metadata
            await self._store_adapter_metadata(adapter, proposal)
            
            # Stage 5: Save conversation data for hybrid retrieval
            await self._save_conversation_data(adapter, proposal, validated_facts)
            
            # Complete consolidation
            processing_time = time.time() - start_time
            
            result = ConsolidationResult(
                adapter_id=adapter.adapter_id,
                status="success",
                validated_facts_count=len([f for f in validated_facts if f.is_validated]),
                training_examples_count=adapter.training_examples_count,
                training_time_seconds=processing_time
            )
            
            await self._update_consolidation_status(
                proposal_id, "completed", {
                    "adapter_id": adapter.adapter_id,
                    "processing_time": processing_time
                }
            )
            
            # Record in history
            self._consolidation_history.append({
                "proposal_id": proposal_id,
                "conversation_id": conversation_id,
                "adapter_id": adapter.adapter_id,
                "processing_time": processing_time,
                "validated_facts": len([f for f in validated_facts if f.is_validated]),
                "training_examples": adapter.training_examples_count,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "success"
            })
            
            # Clean up active tracking
            self._active_consolidations.pop(proposal_id, None)
            
            logger.info(
                f"Successfully consolidated conversation {conversation_id} "
                f"into adapter {adapter.adapter_id} in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            # Handle consolidation failure
            error_msg = f"Consolidation failed for conversation {conversation_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            processing_time = time.time() - start_time
            
            # Record failure
            self._consolidation_history.append({
                "proposal_id": proposal_id,
                "conversation_id": conversation_id,
                "processing_time": processing_time,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            
            await self._update_consolidation_status(
                proposal_id, "failed", {
                    "error": str(e),
                    "processing_time": processing_time
                }
            )
            
            # Clean up
            self._active_consolidations.pop(proposal_id, None)
            
            # Return failed result
            result = ConsolidationResult(
                adapter_id="",
                status="failed",
                validated_facts_count=0,
                training_examples_count=0,
                training_time_seconds=processing_time,
                error_message=str(e)
            )
            
            return result
    
    async def _validate_conversation_facts(
        self,
        proposal: MEPProposalRequest
    ) -> List[ValidatedFact]:
        """
        Validate conversation facts using truth model.
        
        Args:
            proposal: MEP proposal with conversation data
            
        Returns:
            List of validated facts
        """
        try:
            # Ensure truth model is loaded
            if not self._truth_model._is_initialized:
                await self._truth_model.load_nli_model()
            
            # Extract conversation context from summary
            conversation_context = proposal.summary_text
            
            # Validate key facts
            validated_facts = await self._truth_model.validate_key_facts(
                proposal.key_facts, conversation_context
            )
            
            return validated_facts
            
        except Exception as e:
            error_msg = f"Truth validation failed: {str(e)}"
            logger.error(error_msg)
            raise ConsolidationError(
                message=error_msg,
                proposal_id=proposal.event_id,
                consolidation_stage="truth_validation"
            )
    
    async def _convert_to_training_data(
        self,
        proposal: MEPProposalRequest,
        validated_facts: List[ValidatedFact]
    ) -> Path:
        """
        Convert conversation to training data.
        
        Args:
            proposal: MEP proposal with conversation data
            validated_facts: Validated facts from truth model
            
        Returns:
            Path to generated training data
        """
        try:
            # Generate training examples
            training_examples = await self._conversation_processor.process_conversation_to_training_data(
                proposal, validated_facts
            )
            
            if not training_examples:
                raise ConsolidationError(
                    message="No training examples generated from conversation",
                    proposal_id=proposal.event_id,
                    consolidation_stage="data_generation"
                )
            
            # Save training data
            training_data_path = await self._conversation_processor.save_training_data(
                training_examples
            )
            
            return training_data_path
            
        except Exception as e:
            error_msg = f"Training data conversion failed: {str(e)}"
            logger.error(error_msg)
            raise ConsolidationError(
                message=error_msg,
                proposal_id=proposal.event_id,
                consolidation_stage="data_conversion"
            )
    
    async def _train_conversation_adapter(
        self,
        training_data_path: Path,
        conversation_id: str,
        proposal: MEPProposalRequest
    ) -> Any:  # ConversationAdapter return type
        """
        Train LoRA adapter on conversation data.

        Args:
            training_data_path: Path to training data
            conversation_id: Conversation identifier
            proposal: MEP proposal with complete metadata

        Returns:
            Trained conversation adapter
        """
        try:
            # Initialize LoRA trainer if needed
            if self._lora_trainer is None:
                # Initialize base model if needed
                if self._base_model is None:
                    self._base_model = BaseModelManager(settings=self._settings)
                    await self._base_model.load_model()
                
                self._lora_trainer = create_lora_trainer(
                    base_model=self._base_model,
                    settings=self._settings
                )
            
            # Train adapter with complete metadata for proper retrieval
            adapter = await self._lora_trainer.train_conversation_adapter(
                training_data_path,
                conversation_id,
                provider=proposal.provider,  # Pass provider for MAP matching
                external_user_id=proposal.external_user_id,  # Pass user_id for MAP matching
                external_chat_id=proposal.external_chat_id,  # Pass chat_id for MAP matching
                proposal_id=proposal.event_id  # Pass proposal ID for tracking
            )
            
            return adapter
            
        except Exception as e:
            error_msg = f"LoRA training failed: {str(e)}"
            logger.error(error_msg)
            raise ConsolidationError(
                message=error_msg,
                consolidation_stage="lora_training"
            )
    
    async def _store_adapter_metadata(
        self,
        adapter: Any,  # ConversationAdapter
        proposal: MEPProposalRequest
    ) -> None:
        """
        Store adapter metadata for future retrieval.
        
        Args:
            adapter: Trained conversation adapter
            proposal: Original MEP proposal
        """
        try:
            # For MVP, metadata is already stored in the adapter directory
            # In production, would store in database for efficient querying
            
            # Create registry entry (simple JSON file for MVP)
            registry_path = Path("./data/adapter_registry.json")
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing registry
            registry = []
            if registry_path.exists():
                try:
                    import json
                    with open(registry_path, 'r') as f:
                        registry = json.load(f)
                except:
                    logger.warning("Could not load adapter registry, creating new one")
            
            # Add new adapter entry
            registry_entry = {
                "adapter_id": adapter.adapter_id,
                "conversation_id": adapter.conversation_id,
                "proposal_id": proposal.event_id,
                "user_id": proposal.external_user_id,
                "chat_id": proposal.external_chat_id,
                "adapter_path": str(adapter.adapter_path),
                "training_examples_count": adapter.training_examples_count,
                "training_date": adapter.training_date.isoformat(),
                "performance_metrics": adapter.performance_metrics,
            }
            
            registry.append(registry_entry)
            
            # Save updated registry
            with open(registry_path, 'w') as f:
                import json
                json.dump(registry, f, indent=2)
            
            logger.info(f"Stored adapter metadata for {adapter.adapter_id}")
            
        except Exception as e:
            logger.warning(f"Failed to store adapter metadata: {e}")
            # Not critical for MVP, just log warning
    
    async def _save_conversation_data(
        self,
        adapter: Any,  # ConversationAdapter
        proposal: MEPProposalRequest,
        validated_facts: List[ValidatedFact]
    ) -> None:
        """
        Save conversation data for hybrid retrieval.
        
        ALL COMMENTS MUST BE IN ENGLISH.
        
        Args:
            adapter: Trained adapter metadata
            proposal: Original MEP proposal
            validated_facts: Validated facts list
        """
        try:
            import json
            from pathlib import Path
            from datetime import datetime
            
            # Create data directory for this adapter (ENGLISH COMMENT)
            data_dir = Path(f"./data/conversations/{adapter.adapter_id}")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract messages from proposal - since MEPProposalRequest doesn't have messages field,
            # we'll construct them from the summary and message span (ENGLISH COMMENT)
            messages = self._extract_messages_from_proposal(proposal)
            
            # Save conversation data (ENGLISH COMMENT)
            conversation_data = {
                "adapter_id": adapter.adapter_id,
                "conversation_id": proposal.external_chat_id,
                "topic": getattr(adapter, 'topic', self._extract_topic_from_summary(proposal.summary_text)),
                "messages": messages,
                "summary": proposal.summary_text,
                "message_span": {
                    "from_turn": proposal.message_span.from_turn,
                    "to_turn": proposal.message_span.to_turn
                },
                "metadata": {
                    "created_at": datetime.utcnow().isoformat(),
                    "provider": proposal.provider,
                    "model": proposal.model,
                    "user_id": proposal.external_user_id,
                    "event_id": proposal.event_id,
                    "trigger": proposal.trigger,
                    "context_fill": proposal.context_fill,
                    "token_usage": {
                        "window_tokens": proposal.token_usage.window_tokens,
                        "used_tokens": proposal.token_usage.used_tokens,
                        "max_tokens": proposal.token_usage.max_tokens
                    }
                }
            }
            
            # Write conversation file (ENGLISH COMMENT)
            with open(data_dir / "conversation.json", 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            # Save validated facts (ENGLISH COMMENT)
            facts_data = [
                {
                    "claim": fact.original_fact.claim,
                    "confidence": fact.truth_score,
                    "importance": fact.original_fact.importance,
                    "category": getattr(fact.original_fact, 'category', None),
                    "source_turn": getattr(fact.original_fact, 'source_turn', None),
                    "validation_reason": fact.validation_reason,
                    "is_validated": fact.is_validated
                }
                for fact in validated_facts if fact.is_validated
            ]
            
            # Write validated facts file (ENGLISH COMMENT)
            with open(data_dir / "validated_facts.json", 'w', encoding='utf-8') as f:
                json.dump(facts_data, f, indent=2, ensure_ascii=False)
            
            # Log success in English
            logger.info(
                f"Saved conversation data for adapter {adapter.adapter_id}: "
                f"{len(messages)} messages, {len(facts_data)} validated facts"
            )
            
        except Exception as e:
            # Log error in English but don't fail the consolidation
            logger.error(f"Failed to save conversation data: {e}", exc_info=True)
    
    def _extract_messages_from_proposal(self, proposal: MEPProposalRequest) -> List[Dict[str, Any]]:
        """
        Extract messages from proposal.
        
        Since MEPProposalRequest doesn't have a messages field, we create a simplified
        representation from the summary and message span.
        
        Args:
            proposal: MEP proposal
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Create a simplified message structure from available data (ENGLISH COMMENT)
        # In a real implementation, messages would be passed in the proposal
        for turn in range(proposal.message_span.from_turn, proposal.message_span.to_turn + 1):
            # Alternate between user and assistant for demo (ENGLISH COMMENT)
            role = "user" if turn % 2 == 0 else "assistant"
            messages.append({
                "turn_number": turn,
                "role": role,
                "content": f"Turn {turn} content (extracted from summary)",
                "timestamp": None  # Would have real timestamp in production
            })
        
        # Add summary as final assistant message if not empty (ENGLISH COMMENT)
        if proposal.summary_text:
            messages.append({
                "turn_number": proposal.message_span.to_turn + 1,
                "role": "assistant",
                "content": proposal.summary_text,
                "timestamp": datetime.utcnow().isoformat(),
                "is_summary": True
            })
        
        return messages
    
    def _extract_topic_from_summary(self, summary: str) -> str:
        """
        Extract topic from summary text.
        
        Simple extraction - takes first few words or identifies key theme.
        
        Args:
            summary: Summary text
            
        Returns:
            Extracted topic string
        """
        if not summary:
            return "general_conversation"
        
        # Simple topic extraction - first sentence or first 50 chars (ENGLISH COMMENT)
        first_sentence = summary.split('.')[0] if '.' in summary else summary[:50]
        
        # Clean and simplify (ENGLISH COMMENT)
        topic = first_sentence.lower().strip()
        topic = ''.join(c if c.isalnum() or c.isspace() else '_' for c in topic)
        topic = '_'.join(topic.split()[:5])  # First 5 words max
        
        return topic or "general_conversation"
    
    async def _update_consolidation_status(
        self,
        proposal_id: str,
        status: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Update consolidation status.
        
        Args:
            proposal_id: Proposal identifier
            status: Current status
            details: Additional status details
        """
        if proposal_id in self._active_consolidations:
            self._active_consolidations[proposal_id].update({
                "stage": status,
                "last_update": time.time(),
                **details
            })
        
        logger.debug(f"Consolidation {proposal_id} status: {status}")
    
    def get_consolidation_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current consolidation status.
        
        Args:
            proposal_id: Proposal identifier
            
        Returns:
            Status information or None if not found
        """
        return self._active_consolidations.get(proposal_id)
    
    def get_active_consolidations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active consolidations.
        
        Returns:
            Dictionary of active consolidations
        """
        return self._active_consolidations.copy()
    
    def get_consolidation_history(self) -> List[Dict[str, Any]]:
        """
        Get consolidation history.
        
        Returns:
            List of historical consolidation records
        """
        return self._consolidation_history.copy()
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """
        Get consolidation statistics.
        
        Returns:
            Dictionary with consolidation stats
        """
        total_consolidations = len(self._consolidation_history)
        successful_consolidations = len([
            h for h in self._consolidation_history
            if h.get("status") == "success"
        ])
        
        avg_processing_time = 0.0
        if successful_consolidations > 0:
            total_time = sum(
                h.get("processing_time", 0)
                for h in self._consolidation_history
                if h.get("status") == "success"
            )
            avg_processing_time = total_time / successful_consolidations
        
        return {
            "total_consolidations": total_consolidations,
            "successful_consolidations": successful_consolidations,
            "failed_consolidations": total_consolidations - successful_consolidations,
            "success_rate": successful_consolidations / max(1, total_consolidations),
            "average_processing_time": avg_processing_time,
            "active_consolidations": len(self._active_consolidations),
            "components_initialized": {
                "base_model": self._base_model is not None,
                "embeddings": self._embeddings is not None,
                "truth_model": self._truth_model._is_initialized,
                "lora_trainer": self._lora_trainer is not None,
            }
        }


# Convenience function for creating consolidator
def create_memory_consolidator(
    base_model: Optional[BaseModelManager] = None,
    embeddings_manager: Optional[EmbeddingsManager] = None,
    settings: Optional[Settings] = None
) -> MemoryConsolidator:
    """
    Create and return a MemoryConsolidator instance.
    
    Args:
        base_model: Optional base model manager
        embeddings_manager: Optional embeddings manager
        settings: Application settings
        
    Returns:
        MemoryConsolidator instance
    """
    return MemoryConsolidator(
        base_model=base_model,
        embeddings_manager=embeddings_manager,
        settings=settings
    )


# Export main classes and functions
__all__ = [
    "MemoryConsolidator",
    "create_memory_consolidator",
]