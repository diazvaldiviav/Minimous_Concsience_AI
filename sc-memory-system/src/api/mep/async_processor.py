"""
Week 3 Async Processing Enhancement for MEP (Memory Exchange Protocol).

This module implements production-ready async processing pipeline with:
- Multi-stage processing (staging → validating → training → consolidated)
- Concurrent processing with configurable worker pools
- Retry logic with exponential backoff
- Status persistence and recovery
- Progress tracking with detailed stage information
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ...core.config import Settings, get_settings
from ...core.exceptions import (
    MEPError,
    MEPQueueError,
    ConsolidationError,
    ServiceUnavailableError
)
from ...core.models import (
    AsyncProposalStatus,
    ProposalStage,
    MEPProposal,
    ResourceUsage
)
from ..mep.schemas import MEPProposalRequest


logger = logging.getLogger(__name__)


class ProcessingStageEnum(str, Enum):
    """Processing stages for MEP proposals."""
    STAGING = "staging"
    VALIDATING = "validating"
    TRAINING = "training"
    CONSOLIDATED = "consolidated"
    FAILED = "failed"


class ProcessingStatusEnum(str, Enum):
    """Processing status for stages."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AsyncProposalProcessor:
    """Advanced async processor for MEP proposals with staging pipeline."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the async processor.

        Args:
            settings: Application settings, uses global settings if None
        """
        self.settings = settings or get_settings()
        self.processing_queue: Dict[str, AsyncProposalStatus] = {}
        self.worker_pools: Dict[str, asyncio.Queue] = {
            stage.value: asyncio.Queue(maxsize=self.settings.async_processing.max_queue_size)
            for stage in ProcessingStageEnum
            if stage != ProcessingStageEnum.FAILED
        }
        self.workers: List[asyncio.Task] = []
        self.running = False
        self._lock = asyncio.Lock()
        
        # Resource monitoring
        self.resource_usage_history: List[ResourceUsage] = []
        
        logger.info(f"AsyncProposalProcessor initialized with {self.settings.async_processing.max_concurrent_workers} workers")

    async def start(self) -> None:
        """Start the async processing system."""
        if self.running:
            logger.warning("AsyncProposalProcessor already running")
            return

        self.running = True
        
        # Load persisted state if enabled
        if self.settings.async_processing.persistence_enabled:
            await self._load_persisted_state()

        # Start worker pools for each stage
        for stage in ProcessingStageEnum:
            if stage == ProcessingStageEnum.FAILED:
                continue
                
            # Start multiple workers per stage
            workers_per_stage = max(1, self.settings.async_processing.max_concurrent_workers // len(self.worker_pools))
            
            for worker_id in range(workers_per_stage):
                worker_task = asyncio.create_task(
                    self._stage_worker(stage, f"{stage.value}_worker_{worker_id}")
                )
                self.workers.append(worker_task)

        # Start resource monitoring task
        monitor_task = asyncio.create_task(self._resource_monitor())
        self.workers.append(monitor_task)

        # Start persistence task
        if self.settings.async_processing.persistence_enabled:
            persistence_task = asyncio.create_task(self._persistence_worker())
            self.workers.append(persistence_task)

        logger.info(f"AsyncProposalProcessor started with {len(self.workers)} workers")

    async def stop(self) -> None:
        """Stop the async processing system."""
        if not self.running:
            return

        self.running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to complete
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        # Save final state if persistence enabled
        if self.settings.async_processing.persistence_enabled:
            await self._save_persisted_state()

        logger.info("AsyncProposalProcessor stopped")

    async def submit_proposal(self, proposal: MEPProposalRequest) -> str:
        """
        Submit a MEP proposal for async processing.

        Args:
            proposal: The MEP proposal to process

        Returns:
            str: Unique proposal ID for tracking

        Raises:
            MEPQueueError: If queue is full or service unavailable
        """
        if not self.running:
            raise ServiceUnavailableError("AsyncProposalProcessor is not running")

        proposal_id = str(uuid4())
        
        # Create async proposal status
        async_status = AsyncProposalStatus(
            proposal_id=proposal_id,
            overall_status=ProcessingStatusEnum.PENDING.value,
            current_stage=ProcessingStageEnum.STAGING.value,
            stages=[
                ProposalStage(
                    stage_name=ProcessingStageEnum.STAGING.value,
                    status=ProcessingStatusEnum.PENDING.value
                ),
                ProposalStage(
                    stage_name=ProcessingStageEnum.VALIDATING.value,
                    status=ProcessingStatusEnum.PENDING.value
                ),
                ProposalStage(
                    stage_name=ProcessingStageEnum.TRAINING.value,
                    status=ProcessingStatusEnum.PENDING.value
                ),
                ProposalStage(
                    stage_name=ProcessingStageEnum.CONSOLIDATED.value,
                    status=ProcessingStatusEnum.PENDING.value
                )
            ],
            submitted_at=datetime.utcnow(),
            processing_metadata={
                "proposal": proposal.model_dump(exclude={'token_usage': {'utilization_ratio', 'capacity_ratio'}, 'message_span': {'span_length'}}),
                "submission_timestamp": time.time()
            }
        )

        async with self._lock:
            # Check queue capacity
            if len(self.processing_queue) >= self.settings.async_processing.max_queue_size:
                raise MEPQueueError(
                    "Processing queue is full",
                    queue_size=len(self.processing_queue),
                    max_size=self.settings.async_processing.max_queue_size
                )

            # Add to processing queue
            self.processing_queue[proposal_id] = async_status

        # Add to staging queue for processing
        try:
            await self.worker_pools[ProcessingStageEnum.STAGING.value].put(proposal_id)
        except asyncio.QueueFull:
            # Remove from processing queue if staging queue is full
            async with self._lock:
                del self.processing_queue[proposal_id]
            raise MEPQueueError(
                f"Staging queue is full",
                queue_size=self.worker_pools[ProcessingStageEnum.STAGING.value].qsize(),
                max_size=self.settings.async_processing.max_queue_size
            )

        logger.info(f"Submitted proposal {proposal_id} for async processing")
        return proposal_id

    async def get_proposal_status(self, proposal_id: str) -> Optional[AsyncProposalStatus]:
        """
        Get the current status of a proposal.

        Args:
            proposal_id: The proposal ID to check

        Returns:
            AsyncProposalStatus if found, None otherwise
        """
        async with self._lock:
            return self.processing_queue.get(proposal_id)

    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status and statistics.

        Returns:
            Dictionary with queue status information
        """
        async with self._lock:
            total_proposals = len(self.processing_queue)
            
            # Count proposals by stage
            stage_counts = {stage.value: 0 for stage in ProcessingStageEnum}
            for status in self.processing_queue.values():
                stage_counts[status.current_stage] += 1

            # Queue sizes
            queue_sizes = {
                stage: queue.qsize() 
                for stage, queue in self.worker_pools.items()
            }

            # Recent resource usage
            recent_resource_usage = None
            if self.resource_usage_history:
                recent_resource_usage = self.resource_usage_history[-1].model_dump()

            return {
                "total_proposals": total_proposals,
                "stage_counts": stage_counts,
                "queue_sizes": queue_sizes,
                "active_workers": len([w for w in self.workers if not w.done()]),
                "total_workers": len(self.workers),
                "processing_enabled": self.running,
                "recent_resource_usage": recent_resource_usage,
                "uptime_seconds": time.time() - self.processing_queue.get("_start_time", time.time())
            }

    async def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status information for a specific proposal.

        Args:
            proposal_id: The ID of the proposal to get status for

        Returns:
            Dictionary with proposal status and logs, or None if not found
        """
        async with self._lock:
            if proposal_id not in self.processing_queue:
                return None

            status = self.processing_queue[proposal_id]

            # Build comprehensive status response
            response = {
                "proposal_id": proposal_id,
                "status": getattr(status, 'overall_status', 'unknown'),
                "stage": getattr(status, 'current_stage', 'unknown'),
                "progress": getattr(status, 'progress', 0.0),
                "created_at": getattr(status, 'created_at', None).isoformat() if getattr(status, 'created_at', None) else None,
                "updated_at": getattr(status, 'updated_at', None).isoformat() if getattr(status, 'updated_at', None) else None,
                "error": getattr(status, 'error_message', None),
                "logs": ""
            }

            # Add stage-specific details
            stage_details = getattr(status, 'stage_details', None)
            if stage_details:
                response["stage_details"] = stage_details

            # Add training logs if available
            logs_list = []
            stage_outputs = getattr(status, 'stage_outputs', None)
            if stage_outputs:
                for stage_name, output in stage_outputs.items():
                    if isinstance(output, dict) and 'logs' in output:
                        logs_list.append(f"[{stage_name.upper()}] {output['logs']}")
                    elif isinstance(output, str):
                        logs_list.append(f"[{stage_name.upper()}] {output}")

            # Add any error details to logs
            error_message = getattr(status, 'error_message', None)
            if error_message:
                logs_list.append(f"[ERROR] {error_message}")

            response["logs"] = "\n".join(logs_list) if logs_list else "No logs available"

            # Add training metrics if in training stage
            current_stage = getattr(status, 'current_stage', None)
            if (current_stage == ProcessingStageEnum.TRAINING.value and
                stage_details and
                "training_metrics" in stage_details):
                response["training_metrics"] = stage_details["training_metrics"]

            # Add adapter path if completed
            overall_status = getattr(status, 'overall_status', None)
            if (overall_status == ProcessingStatusEnum.COMPLETED.value and
                stage_outputs and
                ProcessingStageEnum.TRAINING.value in stage_outputs):
                training_output = stage_outputs[ProcessingStageEnum.TRAINING.value]
                if isinstance(training_output, dict) and "adapter_path" in training_output:
                    response["adapter_path"] = training_output["adapter_path"]

            return response

    async def _stage_worker(self, stage: ProcessingStageEnum, worker_id: str) -> None:
        """
        Worker coroutine for processing a specific stage.

        Args:
            stage: The processing stage this worker handles
            worker_id: Unique identifier for this worker
        """
        logger.info(f"Starting stage worker {worker_id} for {stage.value}")
        
        while self.running:
            try:
                # Get next proposal from stage queue
                proposal_id = await asyncio.wait_for(
                    self.worker_pools[stage.value].get(),
                    timeout=self.settings.async_processing.queue_check_interval
                )

                # Process the proposal
                await self._process_proposal_stage(proposal_id, stage, worker_id)

            except asyncio.TimeoutError:
                # No proposals to process, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause before continuing

        logger.info(f"Stage worker {worker_id} for {stage.value} stopped")

    async def _process_proposal_stage(
        self, 
        proposal_id: str, 
        stage: ProcessingStageEnum, 
        worker_id: str
    ) -> None:
        """
        Process a single proposal at a specific stage.

        Args:
            proposal_id: ID of the proposal to process
            stage: Current processing stage
            worker_id: ID of the worker processing this proposal
        """
        async with self._lock:
            if proposal_id not in self.processing_queue:
                logger.warning(f"Proposal {proposal_id} not found in queue")
                return

            status = self.processing_queue[proposal_id]

        logger.info(f"Worker {worker_id} processing proposal {proposal_id} at stage {stage.value}")

        try:
            # Update stage status to in_progress
            await self._update_stage_status(
                proposal_id, 
                stage.value, 
                ProcessingStatusEnum.IN_PROGRESS,
                worker_id=worker_id
            )

            # Process based on stage
            success = False
            if stage == ProcessingStageEnum.STAGING:
                success = await self._process_staging(proposal_id, status)
            elif stage == ProcessingStageEnum.VALIDATING:
                success = await self._process_validating(proposal_id, status)
            elif stage == ProcessingStageEnum.TRAINING:
                success = await self._process_training(proposal_id, status)
            elif stage == ProcessingStageEnum.CONSOLIDATED:
                success = await self._process_consolidation(proposal_id, status)

            if success:
                # Mark stage as completed
                await self._update_stage_status(
                    proposal_id,
                    stage.value,
                    ProcessingStatusEnum.COMPLETED
                )

                # Move to next stage if not final stage
                next_stage = self._get_next_stage(stage)
                if next_stage:
                    await self.worker_pools[next_stage.value].put(proposal_id)
                    async with self._lock:
                        self.processing_queue[proposal_id].current_stage = next_stage.value
                else:
                    # Final stage completed
                    async with self._lock:
                        self.processing_queue[proposal_id].overall_status = ProcessingStatusEnum.COMPLETED.value

            else:
                # Stage failed
                await self._handle_stage_failure(proposal_id, stage, worker_id)

        except Exception as e:
            logger.error(f"Error processing proposal {proposal_id} at stage {stage.value}: {e}", exc_info=True)
            await self._handle_stage_failure(proposal_id, stage, worker_id, str(e))

    async def _process_staging(self, proposal_id: str, status: AsyncProposalStatus) -> bool:
        """
        Process the staging phase - validate proposal and prepare for processing.

        Args:
            proposal_id: Proposal ID
            status: Current proposal status

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract proposal from metadata
            proposal_dict = status.processing_metadata.get("proposal")
            if not proposal_dict:
                raise ValueError("No proposal data found in metadata")

            # Validate proposal structure
            proposal = MEPProposalRequest.model_validate(proposal_dict)

            # Basic validation checks
            if not proposal.key_facts:
                logger.warning(f"Proposal {proposal_id} has no key facts")

            if proposal.token_usage.used_tokens <= 0:
                raise ValueError("Invalid token usage information")

            # Update processing metadata with validation results
            async with self._lock:
                self.processing_queue[proposal_id].processing_metadata.update({
                    "staging_completed_at": datetime.utcnow().isoformat(),
                    "validation_checks_passed": True,
                    "facts_count": len(proposal.key_facts),
                    "message_span_turns": proposal.message_span.to_turn - proposal.message_span.from_turn + 1
                })

            # Simulate some processing time
            await asyncio.sleep(0.1)

            logger.info(f"Staging completed for proposal {proposal_id}")
            return True

        except Exception as e:
            logger.error(f"Staging failed for proposal {proposal_id}: {e}")
            return False

    async def _process_validating(self, proposal_id: str, status: AsyncProposalStatus) -> bool:
        """
        Process the validation phase - validate facts using truth model.

        Args:
            proposal_id: Proposal ID  
            status: Current proposal status

        Returns:
            True if successful, False otherwise
        """
        try:
            # Import truth model
            from ...memory.truth_model import create_truth_model

            # Extract proposal from metadata
            proposal_dict = status.processing_metadata.get("proposal")
            proposal = MEPProposalRequest.model_validate(proposal_dict)

            # Initialize truth model
            truth_model = create_truth_model()

            # Validate key facts
            conversation_context = status.processing_metadata.get("summary_text", "")
            validated_facts = await truth_model.validate_key_facts(
                proposal.key_facts,
                conversation_context
            )

            # Update processing metadata with validation results
            async with self._lock:
                self.processing_queue[proposal_id].processing_metadata.update({
                    "validating_completed_at": datetime.utcnow().isoformat(),
                    "validated_facts": [fact.model_dump() for fact in validated_facts],
                    "validated_facts_count": len(validated_facts),
                    "validation_success_rate": len(validated_facts) / len(proposal.key_facts) if proposal.key_facts else 0
                })

            logger.info(f"Validation completed for proposal {proposal_id}: {len(validated_facts)} facts validated")
            return True

        except Exception as e:
            logger.error(f"Validation failed for proposal {proposal_id}: {e}")
            return False

    async def _process_training(self, proposal_id: str, status: AsyncProposalStatus) -> bool:
        """
        Process the training phase - convert to training data and train LoRA adapter.

        Args:
            proposal_id: Proposal ID
            status: Current proposal status

        Returns:
            True if successful, False otherwise
        """
        try:
            # Import required components
            from ...memory.conversation_processor import create_conversation_processor
            from ...memory.lora_trainer import create_lora_trainer

            # Extract data from metadata
            proposal_dict = status.processing_metadata.get("proposal")
            proposal = MEPProposalRequest.model_validate(proposal_dict)
            validated_facts = status.processing_metadata.get("validated_facts", [])

            # Convert validated facts back to ValidatedFact objects
            from ...core.models import ValidatedFact
            validated_fact_objects = [
                ValidatedFact.model_validate(fact) for fact in validated_facts
            ]

            # Process conversation to training data
            conversation_processor = create_conversation_processor()
            training_examples = await conversation_processor.process_conversation_to_training_data(
                proposal, validated_fact_objects
            )

            if not training_examples:
                logger.warning(f"No training examples generated for proposal {proposal_id}")
                return False

            # Train LoRA adapter
            lora_trainer = create_lora_trainer()
            
            # Save training data to temporary file
            training_data_path = Path(f"./data/training/temp_{proposal_id}.jsonl")
            training_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(training_data_path, 'w') as f:
                for example in training_examples:
                    json.dump(example.model_dump(), f)
                    f.write('\n')

            # Train the adapter
            adapter = await lora_trainer.train_conversation_adapter(
                training_data_path, 
                proposal.external_chat_id
            )

            # Clean up temporary file
            training_data_path.unlink(exist_ok=True)

            # Update processing metadata with training results
            async with self._lock:
                self.processing_queue[proposal_id].processing_metadata.update({
                    "training_completed_at": datetime.utcnow().isoformat(),
                    "adapter_id": adapter.adapter_id,
                    "training_examples_count": len(training_examples),
                    "adapter_path": str(adapter.adapter_path),
                    "training_time_seconds": adapter.training_time_seconds
                })

            logger.info(f"Training completed for proposal {proposal_id}: adapter {adapter.adapter_id} created")
            return True

        except Exception as e:
            logger.error(f"Training failed for proposal {proposal_id}: {e}")
            return False

    async def _process_consolidation(self, proposal_id: str, status: AsyncProposalStatus) -> bool:
        """
        Process the consolidation phase - finalize and mark as consolidated.

        Args:
            proposal_id: Proposal ID
            status: Current proposal status

        Returns:
            True if successful, False otherwise
        """
        try:
            # Final consolidation steps
            async with self._lock:
                self.processing_queue[proposal_id].processing_metadata.update({
                    "consolidation_completed_at": datetime.utcnow().isoformat(),
                    "final_status": "consolidated"
                })

                # Update overall status
                self.processing_queue[proposal_id].overall_status = ProcessingStatusEnum.COMPLETED.value
                self.processing_queue[proposal_id].current_stage = ProcessingStageEnum.CONSOLIDATED.value

            logger.info(f"Consolidation completed for proposal {proposal_id}")
            return True

        except Exception as e:
            logger.error(f"Consolidation failed for proposal {proposal_id}: {e}")
            return False

    async def _update_stage_status(
        self, 
        proposal_id: str, 
        stage_name: str, 
        status: ProcessingStatusEnum,
        worker_id: Optional[str] = None,
        error_message: Optional[str] = None,
        progress_percent: float = 0.0
    ) -> None:
        """Update the status of a specific stage for a proposal."""
        async with self._lock:
            if proposal_id not in self.processing_queue:
                return

            proposal_status = self.processing_queue[proposal_id]
            
            # Find and update the stage
            for stage in proposal_status.stages:
                if stage.stage_name == stage_name:
                    stage.status = status.value
                    stage.progress_percent = progress_percent
                    
                    if status == ProcessingStatusEnum.IN_PROGRESS:
                        stage.started_at = datetime.utcnow()
                        proposal_status.worker_id = worker_id
                        if not proposal_status.started_processing_at:
                            proposal_status.started_processing_at = datetime.utcnow()
                    elif status in [ProcessingStatusEnum.COMPLETED, ProcessingStatusEnum.FAILED]:
                        stage.completed_at = datetime.utcnow()
                        stage.progress_percent = 100.0
                        
                    if error_message:
                        stage.error_message = error_message
                    
                    break

    async def _handle_stage_failure(
        self, 
        proposal_id: str, 
        stage: ProcessingStageEnum, 
        worker_id: str,
        error_message: Optional[str] = None
    ) -> None:
        """Handle failure of a stage, including retry logic."""
        async with self._lock:
            if proposal_id not in self.processing_queue:
                return

            status = self.processing_queue[proposal_id]

        # Check if we should retry
        if status.retry_count < self.settings.async_processing.retry_attempts:
            # Increment retry count
            async with self._lock:
                self.processing_queue[proposal_id].retry_count += 1

            # Calculate retry delay (exponential backoff)
            delay = self.settings.async_processing.retry_delay_base ** status.retry_count
            
            logger.info(f"Retrying proposal {proposal_id} at stage {stage.value} after {delay}s (attempt {status.retry_count})")
            
            # Schedule retry
            asyncio.create_task(self._schedule_retry(proposal_id, stage, delay))
        else:
            # Max retries exceeded, mark as failed
            await self._update_stage_status(
                proposal_id,
                stage.value,
                ProcessingStatusEnum.FAILED,
                error_message=error_message or f"Max retries ({self.settings.async_processing.retry_attempts}) exceeded"
            )

            async with self._lock:
                self.processing_queue[proposal_id].overall_status = ProcessingStatusEnum.FAILED.value
                self.processing_queue[proposal_id].current_stage = ProcessingStageEnum.FAILED.value

            logger.error(f"Proposal {proposal_id} failed at stage {stage.value} after {status.retry_count} retries")

    async def _schedule_retry(self, proposal_id: str, stage: ProcessingStageEnum, delay: float) -> None:
        """Schedule a retry for a failed stage."""
        await asyncio.sleep(delay)
        
        # Reset stage status and re-queue
        await self._update_stage_status(
            proposal_id,
            stage.value,
            ProcessingStatusEnum.PENDING
        )
        
        await self.worker_pools[stage.value].put(proposal_id)

    def _get_next_stage(self, current_stage: ProcessingStageEnum) -> Optional[ProcessingStageEnum]:
        """Get the next processing stage."""
        stage_order = [
            ProcessingStageEnum.STAGING,
            ProcessingStageEnum.VALIDATING,
            ProcessingStageEnum.TRAINING,
            ProcessingStageEnum.CONSOLIDATED
        ]
        
        try:
            current_index = stage_order.index(current_stage)
            if current_index < len(stage_order) - 1:
                return stage_order[current_index + 1]
        except ValueError:
            pass
            
        return None

    async def _resource_monitor(self) -> None:
        """Monitor system resource usage."""
        import psutil
        
        while self.running:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                # GPU monitoring (if available)
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

                # Disk I/O (basic measurement)
                disk_io = psutil.disk_io_counters()
                disk_io_read = disk_io.read_bytes / (1024 * 1024)  # MB
                disk_io_write = disk_io.write_bytes / (1024 * 1024)  # MB

                # Active workers count
                active_workers = len([w for w in self.workers if not w.done()])

                resource_usage = ResourceUsage(
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    gpu_percent=gpu_percent,
                    gpu_memory_percent=gpu_memory_percent,
                    disk_io_read=disk_io_read,
                    disk_io_write=disk_io_write,
                    active_workers=active_workers
                )

                # Store resource usage (keep last 100 measurements)
                self.resource_usage_history.append(resource_usage)
                if len(self.resource_usage_history) > 100:
                    self.resource_usage_history.pop(0)

                await asyncio.sleep(self.settings.training_scheduler.resource_monitoring_interval)

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(10)  # Longer wait on error

    async def _persistence_worker(self) -> None:
        """Periodically save processing state to disk."""
        while self.running:
            try:
                await self._save_persisted_state()
                await asyncio.sleep(30)  # Save every 30 seconds
            except Exception as e:
                logger.error(f"Error in persistence worker: {e}")
                await asyncio.sleep(60)  # Longer wait on error

    async def _save_persisted_state(self) -> None:
        """Save current processing state to disk."""
        try:
            persistence_file = self.settings.async_processing.persistence_file
            persistence_file.parent.mkdir(parents=True, exist_ok=True)

            async with self._lock:
                # Convert processing queue to serializable format
                state_data = {
                    "processing_queue": {
                        proposal_id: status.model_dump(mode='json')
                        for proposal_id, status in self.processing_queue.items()
                    },
                    "saved_at": datetime.utcnow().isoformat(),
                    "queue_sizes": {
                        stage: queue.qsize()
                        for stage, queue in self.worker_pools.items()
                    }
                }

            with open(persistence_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            logger.debug(f"Saved processing state to {persistence_file}")

        except Exception as e:
            logger.error(f"Failed to save processing state: {e}")

    async def _load_persisted_state(self) -> None:
        """Load processing state from disk."""
        try:
            persistence_file = self.settings.async_processing.persistence_file
            
            if not persistence_file.exists():
                logger.info("No persisted state found, starting fresh")
                return

            with open(persistence_file, 'r') as f:
                state_data = json.load(f)

            # Restore processing queue
            async with self._lock:
                for proposal_id, status_data in state_data.get("processing_queue", {}).items():
                    try:
                        status = AsyncProposalStatus.model_validate(status_data)
                        self.processing_queue[proposal_id] = status
                        
                        # Re-queue incomplete proposals
                        if status.overall_status != ProcessingStatusEnum.COMPLETED.value:
                            current_stage = status.current_stage
                            if current_stage in self.worker_pools:
                                await self.worker_pools[current_stage].put(proposal_id)
                    
                    except Exception as e:
                        logger.error(f"Failed to restore proposal {proposal_id}: {e}")

            logger.info(f"Loaded persisted state: {len(self.processing_queue)} proposals restored")

        except Exception as e:
            logger.error(f"Failed to load persisted state: {e}")


# Global processor instance
_global_processor: Optional[AsyncProposalProcessor] = None


async def get_async_processor() -> AsyncProposalProcessor:
    """Get the global async processor instance."""
    global _global_processor
    
    if _global_processor is None:
        _global_processor = AsyncProposalProcessor()
        await _global_processor.start()
    
    return _global_processor


async def shutdown_async_processor() -> None:
    """Shutdown the global async processor."""
    global _global_processor
    
    if _global_processor is not None:
        await _global_processor.stop()
        _global_processor = None