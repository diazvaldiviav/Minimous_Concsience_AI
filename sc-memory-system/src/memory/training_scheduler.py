"""
Training Scheduler for Week 3 - Batch job scheduler for efficient LoRA training.

This module optimizes training of multiple conversations through:
- Intelligent batching of similar conversations
- Resource-aware scheduling based on GPU/CPU usage
- Priority-based job ordering
- Batch optimization for efficient resource utilization

Scale: Batch optimize 2-5 pending conversations for efficient LoRA training
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core.config import Settings, get_settings
from ..core.exceptions import ConsolidationError
from ..core.models import (
    BatchTrainingJob,
    ResourceUsage,
    MEPProposal
)
from ..utils.batch_utils import BatchGroup, BatchOptimizer, ResourceAwareBatchProcessor

logger = logging.getLogger(__name__)


class ConversationData:
    """Data structure for conversation information used in scheduling."""
    
    def __init__(
        self,
        conversation_id: str,
        proposal: MEPProposal,
        training_examples_count: int,
        priority: int = 1,
        estimated_training_time: float = 300.0
    ):
        """Initialize conversation data."""
        self.conversation_id = conversation_id
        self.proposal = proposal
        self.training_examples_count = training_examples_count
        self.priority = priority
        self.estimated_training_time = estimated_training_time
        self.created_at = datetime.utcnow()
        
        # Extract features for similarity comparison
        self.summary_text = getattr(proposal, 'summary_text', '')
        self.key_facts_text = ' '.join([
            fact.claim for fact in getattr(proposal, 'key_facts', [])
        ])
        self.combined_text = f"{self.summary_text} {self.key_facts_text}"


class ConversationSimilarityCalculator:
    """Calculates similarity between conversations for batching."""
    
    def __init__(self):
        """Initialize similarity calculator."""
        self.vectorizer = None
        
    async def calculate_similarity_matrix(
        self,
        conversations: List[ConversationData]
    ) -> np.ndarray:
        """
        Calculate similarity matrix for conversations.
        
        Args:
            conversations: List of conversation data
            
        Returns:
            Similarity matrix as numpy array
        """
        if len(conversations) <= 1:
            return np.array([[1.0]] if conversations else [[]])
        
        # Extract text features for similarity calculation
        conversation_texts = []
        for conv in conversations:
            # Combine summary and key facts for similarity
            text = f"{conv.summary_text} {conv.key_facts_text}"
            conversation_texts.append(text if text.strip() else "empty conversation")
        
        try:
            # Use TF-IDF for similarity calculation
            self.vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(conversation_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            logger.debug(f"Calculated similarity matrix for {len(conversations)} conversations")
            return similarity_matrix
            
        except ValueError as e:
            logger.warning(f"TF-IDF similarity calculation failed: {e}, using fallback")
            return await self._fallback_similarity(conversations)
    
    async def _fallback_similarity(
        self,
        conversations: List[ConversationData]
    ) -> np.ndarray:
        """Fallback similarity calculation using simple word overlap."""
        n = len(conversations)
        similarity_matrix = np.eye(n)  # Identity matrix (1.0 diagonal, 0.0 elsewhere)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Simple word overlap similarity
                words_i = set(conversations[i].combined_text.lower().split())
                words_j = set(conversations[j].combined_text.lower().split())
                
                intersection = len(words_i.intersection(words_j))
                union = len(words_i.union(words_j))
                
                similarity = intersection / union if union > 0 else 0.0
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix


class ResourceMonitor:
    """Monitors system resources for training scheduling."""
    
    def __init__(self, settings: Settings):
        """Initialize resource monitor."""
        self.settings = settings
        
    async def get_current_resources(self) -> ResourceUsage:
        """Get current system resource usage."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU monitoring if available
            gpu_percent = None
            gpu_memory_percent = None
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_percent = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100
            except ImportError:
                pass
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_io_write = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_percent=gpu_percent,
                gpu_memory_percent=gpu_memory_percent,
                disk_io_read=disk_io_read,
                disk_io_write=disk_io_write,
                active_workers=0  # Will be set by calling component
            )
            
        except ImportError:
            logger.warning("psutil not available, returning default resource usage")
            return ResourceUsage(
                cpu_percent=50.0,
                memory_percent=60.0,
                active_workers=0
            )
        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")
            return ResourceUsage(
                cpu_percent=100.0,  # Conservative estimate
                memory_percent=100.0,
                active_workers=0
            )
    
    async def can_handle_batch(self, batch_size: int) -> bool:
        """Check if system can handle a training batch of given size."""
        resources = await self.get_current_resources()
        
        # Check resource thresholds
        cpu_available = resources.cpu_percent < (self.settings.training_scheduler.max_cpu_usage * 100)
        memory_available = resources.memory_percent < (self.settings.training_scheduler.max_memory_usage * 100)
        
        gpu_available = True
        if resources.gpu_percent is not None:
            gpu_available = resources.gpu_percent < (self.settings.training_scheduler.max_gpu_usage * 100)
        
        # Estimate resource requirements for batch
        estimated_cpu_increase = batch_size * 10  # Rough estimate: 10% CPU per conversation
        estimated_memory_increase = batch_size * 15  # Rough estimate: 15% memory per conversation
        
        can_handle = (
            cpu_available and
            memory_available and 
            gpu_available and
            (resources.cpu_percent + estimated_cpu_increase) < 90 and
            (resources.memory_percent + estimated_memory_increase) < 90
        )
        
        logger.debug(f"Resource check for batch size {batch_size}: {can_handle}")
        return can_handle


class TrainingScheduler:
    """Main training scheduler for batch optimization of LoRA training."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize training scheduler."""
        self.settings = settings or get_settings()
        self.similarity_calculator = ConversationSimilarityCalculator()
        self.resource_monitor = ResourceMonitor(self.settings)
        self.batch_optimizer = BatchOptimizer(
            max_batch_size=self.settings.training_scheduler.batch_size,
            min_batch_size=1,
            similarity_threshold=0.6
        )
        
        # Active batch tracking
        self.active_batches: Dict[str, BatchTrainingJob] = {}
        self.batch_history: List[BatchTrainingJob] = []
        
    async def schedule_conversation_batch(
        self,
        pending_conversations: List[ConversationData]
    ) -> List[BatchTrainingJob]:
        """
        Group similar conversations for efficient batch training.
        
        Input: 2-5 pending conversations
        Output: Optimized batch training jobs
        
        Args:
            pending_conversations: List of conversations ready for training
            
        Returns:
            List of optimized batch training jobs
        """
        if not pending_conversations:
            return []
        
        logger.info(f"Scheduling batch training for {len(pending_conversations)} conversations")
        
        try:
            # Step 1: Group similar conversations
            conversation_groups = await self._group_similar_conversations(pending_conversations)
            
            # Step 2: Optimize resource allocation for each group
            batch_jobs = []
            for group in conversation_groups:
                batch_job = await self._create_batch_job(group)
                batch_jobs.append(batch_job)
            
            # Step 3: Prioritize batches
            batch_jobs = await self._prioritize_batches(batch_jobs)
            
            # Step 4: Validate resource requirements
            validated_batches = await self._validate_resource_requirements(batch_jobs)
            
            # Track active batches
            for batch in validated_batches:
                self.active_batches[batch.batch_id] = batch
            
            logger.info(f"Created {len(validated_batches)} optimized batch training jobs")
            return validated_batches
            
        except Exception as e:
            logger.error(f"Batch scheduling failed: {e}", exc_info=True)
            raise ConsolidationError(
                f"Training scheduling failed: {e}",
                consolidation_stage="batch_scheduling"
            )
    
    async def _group_similar_conversations(
        self,
        conversations: List[ConversationData]
    ) -> List[List[ConversationData]]:
        """Group conversations by similarity for batch efficiency."""
        if len(conversations) <= 1:
            return [conversations]
        
        # Calculate similarity matrix
        similarity_matrix = await self.similarity_calculator.calculate_similarity_matrix(conversations)
        
        # Use batch optimizer with similarity function
        def similarity_func(conv1: ConversationData, conv2: ConversationData) -> float:
            try:
                idx1 = conversations.index(conv1)
                idx2 = conversations.index(conv2)
                return float(similarity_matrix[idx1][idx2])
            except (ValueError, IndexError):
                return 0.0
        
        def priority_func(conv: ConversationData) -> int:
            return conv.priority
        
        # Optimize batches
        batch_groups = self.batch_optimizer.optimize_batches(
            conversations,
            similarity_func,
            priority_func
        )
        
        # Convert BatchGroup objects to lists of ConversationData
        conversation_groups = []
        for batch_group in batch_groups:
            conversation_groups.append(batch_group.items)
        
        logger.info(f"Grouped {len(conversations)} conversations into {len(conversation_groups)} batches")
        return conversation_groups
    
    async def _create_batch_job(self, conversations: List[ConversationData]) -> BatchTrainingJob:
        """Create a batch training job from a group of conversations."""
        batch_id = f"batch_{uuid4().hex[:8]}"
        conversation_ids = [conv.conversation_id for conv in conversations]
        
        # Calculate priority as average of conversation priorities
        average_priority = sum(conv.priority for conv in conversations) / len(conversations)
        
        # Estimate training time
        total_examples = sum(conv.training_examples_count for conv in conversations)
        base_time_per_example = 5.0  # seconds
        batch_overhead = 30.0  # seconds
        estimated_time = total_examples * base_time_per_example + batch_overhead
        
        # Determine resource requirements
        resource_requirements = {
            "conversations_count": len(conversations),
            "total_training_examples": total_examples,
            "estimated_gpu_memory_gb": len(conversations) * 2.0,  # 2GB per conversation
            "estimated_cpu_cores": min(len(conversations), 4),
            "estimated_training_time_seconds": estimated_time
        }
        
        batch_job = BatchTrainingJob(
            batch_id=batch_id,
            conversation_ids=conversation_ids,
            priority_level=int(average_priority),
            estimated_training_time=estimated_time,
            resource_requirements=resource_requirements,
            status="scheduled"
        )
        
        return batch_job
    
    async def _prioritize_batches(
        self,
        batch_jobs: List[BatchTrainingJob]
    ) -> List[BatchTrainingJob]:
        """Sort batch jobs by priority and optimization criteria."""
        def batch_score(batch: BatchTrainingJob) -> float:
            # Scoring factors:
            # 1. Priority level (higher is better)
            # 2. Batch efficiency (more conversations per batch is better)
            # 3. Resource efficiency (less resource per conversation is better)
            
            priority_score = batch.priority_level * 10
            
            conversations_count = len(batch.conversation_ids)
            efficiency_score = conversations_count * 5
            
            # Resource efficiency (lower resource per conversation is better)
            gpu_memory_per_conv = batch.resource_requirements.get("estimated_gpu_memory_gb", 2.0) / conversations_count
            resource_score = max(0, 10 - gpu_memory_per_conv)
            
            # Time efficiency (shorter batches get slight preference for faster turnaround)
            time_penalty = min(5, batch.estimated_training_time / 300.0)
            
            total_score = priority_score + efficiency_score + resource_score - time_penalty
            return total_score
        
        # Sort by score (highest first)
        sorted_batches = sorted(batch_jobs, key=batch_score, reverse=True)
        
        # Update scheduling metadata
        for i, batch in enumerate(sorted_batches):
            batch.resource_requirements["scheduling_priority"] = i + 1
            batch.resource_requirements["batch_score"] = batch_score(batch)
        
        return sorted_batches
    
    async def _validate_resource_requirements(
        self,
        batch_jobs: List[BatchTrainingJob]
    ) -> List[BatchTrainingJob]:
        """Validate that batch jobs can be executed given current resources."""
        validated_batches = []
        
        for batch in batch_jobs:
            conversations_count = len(batch.conversation_ids)
            
            # Check if system can handle this batch
            can_handle = await self.resource_monitor.can_handle_batch(conversations_count)
            
            if can_handle:
                batch.status = "ready"
                validated_batches.append(batch)
            else:
                # Split large batches or defer
                if conversations_count > 1:
                    # Split into smaller batches
                    split_batches = await self._split_batch(batch)
                    for split_batch in split_batches:
                        split_can_handle = await self.resource_monitor.can_handle_batch(
                            len(split_batch.conversation_ids)
                        )
                        if split_can_handle:
                            split_batch.status = "ready"
                            validated_batches.append(split_batch)
                        else:
                            split_batch.status = "deferred"
                            logger.warning(f"Batch {split_batch.batch_id} deferred due to resource constraints")
                else:
                    batch.status = "deferred"
                    logger.warning(f"Single conversation batch {batch.batch_id} deferred due to resource constraints")
        
        return validated_batches
    
    async def _split_batch(self, batch: BatchTrainingJob) -> List[BatchTrainingJob]:
        """Split a large batch into smaller ones."""
        conversation_ids = batch.conversation_ids
        
        if len(conversation_ids) <= 1:
            return [batch]
        
        # Split into pairs
        split_batches = []
        for i in range(0, len(conversation_ids), 2):
            split_ids = conversation_ids[i:i+2]
            
            split_batch = BatchTrainingJob(
                batch_id=f"{batch.batch_id}_split_{i//2 + 1}",
                conversation_ids=split_ids,
                priority_level=batch.priority_level,
                estimated_training_time=batch.estimated_training_time * len(split_ids) / len(conversation_ids),
                resource_requirements={
                    **batch.resource_requirements,
                    "conversations_count": len(split_ids),
                    "estimated_gpu_memory_gb": batch.resource_requirements.get("estimated_gpu_memory_gb", 4.0) * len(split_ids) / len(conversation_ids)
                },
                status="scheduled"
            )
            split_batches.append(split_batch)
        
        logger.info(f"Split batch {batch.batch_id} into {len(split_batches)} smaller batches")
        return split_batches
    
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status and metrics."""
        current_resources = await self.resource_monitor.get_current_resources()
        
        return {
            "active_batches": len(self.active_batches),
            "total_batches_processed": len(self.batch_history),
            "current_resources": current_resources.model_dump(),
            "scheduler_config": {
                "max_batch_size": self.settings.training_scheduler.batch_size,
                "max_cpu_usage": self.settings.training_scheduler.max_cpu_usage,
                "max_memory_usage": self.settings.training_scheduler.max_gpu_memory_usage,
                "resource_monitoring_interval": self.settings.training_scheduler.resource_monitoring_interval
            },
            "active_batch_details": {
                batch_id: {
                    "conversation_count": len(batch.conversation_ids),
                    "priority": batch.priority_level,
                    "status": batch.status,
                    "estimated_time": batch.estimated_training_time
                }
                for batch_id, batch in self.active_batches.items()
            }
        }
    
    async def mark_batch_completed(
        self,
        batch_id: str,
        trained_adapters: List[Any],
        failed_conversations: List[str]
    ) -> None:
        """Mark a batch as completed and update tracking."""
        if batch_id in self.active_batches:
            batch = self.active_batches[batch_id]
            batch.status = "completed"
            batch.completed_at = datetime.utcnow()
            batch.trained_adapters = trained_adapters
            batch.failed_conversations = failed_conversations
            
            # Move to history
            self.batch_history.append(batch)
            del self.active_batches[batch_id]
            
            logger.info(f"Batch {batch_id} completed: {len(trained_adapters)} adapters trained, {len(failed_conversations)} failed")
    
    async def mark_batch_failed(self, batch_id: str, error_message: str) -> None:
        """Mark a batch as failed."""
        if batch_id in self.active_batches:
            batch = self.active_batches[batch_id]
            batch.status = "failed"
            batch.completed_at = datetime.utcnow()
            batch.failed_conversations = batch.conversation_ids
            
            # Move to history
            self.batch_history.append(batch)
            del self.active_batches[batch_id]
            
            logger.error(f"Batch {batch_id} failed: {error_message}")


# Factory function for creating training scheduler
def create_training_scheduler(settings: Optional[Settings] = None) -> TrainingScheduler:
    """
    Create a training scheduler instance.
    
    Args:
        settings: Optional settings, uses global settings if None
        
    Returns:
        TrainingScheduler instance
    """
    return TrainingScheduler(settings)