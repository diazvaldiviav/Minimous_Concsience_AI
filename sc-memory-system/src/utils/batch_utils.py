"""
Batch processing utilities for Week 3 enhancements.

This module provides utilities for batch processing, resource optimization,
and efficient handling of multiple conversations in training scenarios.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Callable, Awaitable
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BatchGroup:
    """Represents a group of items to be processed together."""
    
    def __init__(self, 
                 batch_id: str,
                 items: List[Any],
                 priority: int = 1,
                 estimated_processing_time: float = 0.0):
        """
        Initialize a batch group.

        Args:
            batch_id: Unique identifier for this batch
            items: List of items to process
            priority: Priority level (higher = more important)
            estimated_processing_time: Estimated time to process this batch
        """
        self.batch_id = batch_id
        self.items = items
        self.priority = priority
        self.estimated_processing_time = estimated_processing_time
        self.created_at = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"BatchGroup(id={self.batch_id}, items={len(self.items)}, priority={self.priority})"


class BatchOptimizer:
    """Optimizes batching strategies based on resource constraints and similarities."""
    
    def __init__(self, 
                 max_batch_size: int = 10,
                 min_batch_size: int = 1,
                 similarity_threshold: float = 0.8):
        """
        Initialize batch optimizer.

        Args:
            max_batch_size: Maximum items per batch
            min_batch_size: Minimum items per batch
            similarity_threshold: Threshold for grouping similar items
        """
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.similarity_threshold = similarity_threshold

    def optimize_batches(self, 
                        items: List[Any],
                        similarity_func: Optional[Callable[[Any, Any], float]] = None,
                        priority_func: Optional[Callable[[Any], int]] = None) -> List[BatchGroup]:
        """
        Optimize items into batches based on similarity and priority.

        Args:
            items: Items to batch
            similarity_func: Function to compute similarity between items
            priority_func: Function to determine item priority

        Returns:
            List of optimized batch groups
        """
        if not items:
            return []

        # Group items by similarity if similarity function provided
        if similarity_func:
            groups = self._group_by_similarity(items, similarity_func)
        else:
            groups = [items]

        # Create batch groups
        batches = []
        batch_counter = 0

        for group in groups:
            # Sort by priority if priority function provided
            if priority_func:
                group.sort(key=priority_func, reverse=True)

            # Split large groups into multiple batches
            for i in range(0, len(group), self.max_batch_size):
                batch_items = group[i:i + self.max_batch_size]
                
                if len(batch_items) >= self.min_batch_size or i + self.max_batch_size >= len(group):
                    # Calculate batch priority as average of item priorities
                    if priority_func:
                        avg_priority = sum(priority_func(item) for item in batch_items) / len(batch_items)
                    else:
                        avg_priority = 1

                    batch = BatchGroup(
                        batch_id=f"batch_{batch_counter:04d}",
                        items=batch_items,
                        priority=int(avg_priority),
                        estimated_processing_time=len(batch_items) * 30.0  # Rough estimate
                    )
                    batches.append(batch)
                    batch_counter += 1

        # Sort batches by priority
        batches.sort(key=lambda b: b.priority, reverse=True)
        return batches

    def _group_by_similarity(self, 
                           items: List[Any],
                           similarity_func: Callable[[Any, Any], float]) -> List[List[Any]]:
        """Group items by similarity using a greedy clustering approach."""
        if not items:
            return []

        groups = []
        ungrouped = items.copy()

        while ungrouped:
            # Start new group with first ungrouped item
            current_group = [ungrouped.pop(0)]
            
            # Find similar items to add to this group
            i = 0
            while i < len(ungrouped):
                item = ungrouped[i]
                
                # Check similarity with items in current group
                similar = any(
                    similarity_func(item, group_item) >= self.similarity_threshold
                    for group_item in current_group
                )
                
                if similar and len(current_group) < self.max_batch_size:
                    current_group.append(ungrouped.pop(i))
                else:
                    i += 1

            groups.append(current_group)

        return groups


class ResourceAwareBatchProcessor:
    """Batch processor that adapts to resource availability."""
    
    def __init__(self,
                 max_cpu_usage: float = 0.8,
                 max_memory_usage: float = 0.8,
                 max_gpu_usage: float = 0.8,
                 monitoring_interval: float = 5.0):
        """
        Initialize resource-aware batch processor.

        Args:
            max_cpu_usage: Maximum CPU usage threshold (0.0-1.0)
            max_memory_usage: Maximum memory usage threshold (0.0-1.0)  
            max_gpu_usage: Maximum GPU usage threshold (0.0-1.0)
            monitoring_interval: How often to check resources (seconds)
        """
        self.max_cpu_usage = max_cpu_usage
        self.max_memory_usage = max_memory_usage
        self.max_gpu_usage = max_gpu_usage
        self.monitoring_interval = monitoring_interval
        self.running = False
        self.current_concurrency = 1
        self.max_concurrency = 10

    async def process_batches(self,
                            batches: List[BatchGroup],
                            batch_processor: Callable[[BatchGroup], Awaitable[Any]]) -> List[Any]:
        """
        Process batches with resource awareness.

        Args:
            batches: List of batch groups to process
            batch_processor: Async function to process each batch

        Returns:
            List of processing results
        """
        self.running = True
        results = []
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(self._monitor_resources())
        
        try:
            # Process batches with dynamic concurrency
            semaphore = asyncio.Semaphore(self.current_concurrency)
            
            async def process_with_semaphore(batch: BatchGroup) -> Any:
                async with semaphore:
                    return await batch_processor(batch)

            # Create tasks for all batches
            tasks = [
                asyncio.create_task(process_with_semaphore(batch))
                for batch in batches
            ]

            # Process with adaptive concurrency
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                
                # Update semaphore if concurrency changed
                if semaphore._value != self.current_concurrency:
                    # Create new semaphore with updated concurrency
                    # (This is a simplified approach; in practice you might want 
                    # more sophisticated concurrency control)
                    pass

        finally:
            self.running = False
            monitor_task.cancel()
            
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        return results

    async def _monitor_resources(self) -> None:
        """Monitor system resources and adjust concurrency."""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available, resource monitoring disabled")
            return

        while self.running:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                # GPU monitoring (if available)
                gpu_percent = 0.0
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_percent = gpus[0].load * 100
                except ImportError:
                    pass

                # Determine if resources are under pressure
                resource_pressure = max(
                    cpu_percent / 100.0 / self.max_cpu_usage,
                    memory_percent / 100.0 / self.max_memory_usage,
                    gpu_percent / 100.0 / self.max_gpu_usage if gpu_percent > 0 else 0
                )

                # Adjust concurrency based on resource pressure
                if resource_pressure > 1.2:  # High pressure
                    self.current_concurrency = max(1, self.current_concurrency - 1)
                elif resource_pressure < 0.7:  # Low pressure
                    self.current_concurrency = min(self.max_concurrency, self.current_concurrency + 1)

                logger.debug(f"Resource pressure: {resource_pressure:.2f}, concurrency: {self.current_concurrency}")
                
                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)


class PriorityBatchQueue:
    """Priority queue for batch processing with time-based scheduling."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize priority batch queue.

        Args:
            max_size: Maximum number of batches in queue
        """
        self.max_size = max_size
        self.queues: Dict[int, List[BatchGroup]] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def put(self, batch: BatchGroup) -> None:
        """Add a batch to the queue."""
        async with self.lock:
            if self.size() >= self.max_size:
                raise asyncio.QueueFull("Batch queue is full")
            
            self.queues[batch.priority].append(batch)

    async def get(self) -> BatchGroup:
        """Get the highest priority batch from the queue."""
        async with self.lock:
            if self.empty():
                raise asyncio.QueueEmpty("Batch queue is empty")

            # Get highest priority level with batches
            max_priority = max(self.queues.keys())
            batch = self.queues[max_priority].pop(0)
            
            # Clean up empty priority levels
            if not self.queues[max_priority]:
                del self.queues[max_priority]

            return batch

    async def get_nowait(self) -> BatchGroup:
        """Get a batch without waiting."""
        if self.empty():
            raise asyncio.QueueEmpty("Batch queue is empty")
        return await self.get()

    def empty(self) -> bool:
        """Check if queue is empty."""
        return not any(self.queues.values())

    def size(self) -> int:
        """Get total number of batches in queue."""
        return sum(len(batches) for batches in self.queues.values())

    def priority_counts(self) -> Dict[int, int]:
        """Get count of batches at each priority level."""
        return {priority: len(batches) for priority, batches in self.queues.items()}


class BatchScheduler:
    """Scheduler for batch processing with timing and resource constraints."""
    
    def __init__(self,
                 batch_timeout: float = 300.0,
                 max_concurrent_batches: int = 5,
                 resource_check_interval: float = 10.0):
        """
        Initialize batch scheduler.

        Args:
            batch_timeout: Maximum time to wait for a batch to complete
            max_concurrent_batches: Maximum concurrent batch processing
            resource_check_interval: How often to check resource availability
        """
        self.batch_timeout = batch_timeout
        self.max_concurrent_batches = max_concurrent_batches
        self.resource_check_interval = resource_check_interval
        self.queue = PriorityBatchQueue()
        self.running = False
        self.workers: List[asyncio.Task] = []

    async def start(self, 
                   batch_processor: Callable[[BatchGroup], Awaitable[Any]]) -> None:
        """
        Start the batch scheduler.

        Args:
            batch_processor: Function to process each batch
        """
        if self.running:
            return

        self.running = True

        # Start worker tasks
        for i in range(self.max_concurrent_batches):
            worker = asyncio.create_task(
                self._worker(f"batch_worker_{i}", batch_processor)
            )
            self.workers.append(worker)

        logger.info(f"BatchScheduler started with {len(self.workers)} workers")

    async def stop(self) -> None:
        """Stop the batch scheduler."""
        if not self.running:
            return

        self.running = False

        # Cancel and wait for workers
        for worker in self.workers:
            worker.cancel()

        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers.clear()
        logger.info("BatchScheduler stopped")

    async def schedule_batch(self, batch: BatchGroup) -> None:
        """Schedule a batch for processing."""
        await self.queue.put(batch)
        logger.info(f"Scheduled batch {batch.batch_id} with priority {batch.priority}")

    async def _worker(self, 
                     worker_id: str,
                     batch_processor: Callable[[BatchGroup], Awaitable[Any]]) -> None:
        """Worker coroutine for processing batches."""
        logger.info(f"Batch worker {worker_id} started")

        while self.running:
            try:
                # Wait for next batch
                try:
                    batch = await asyncio.wait_for(
                        self._wait_for_batch(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the batch with timeout
                logger.info(f"Worker {worker_id} processing batch {batch.batch_id}")
                
                start_time = time.time()
                try:
                    result = await asyncio.wait_for(
                        batch_processor(batch),
                        timeout=self.batch_timeout
                    )
                    
                    processing_time = time.time() - start_time
                    logger.info(
                        f"Worker {worker_id} completed batch {batch.batch_id} "
                        f"in {processing_time:.2f}s"
                    )

                except asyncio.TimeoutError:
                    logger.error(
                        f"Worker {worker_id} timed out processing batch {batch.batch_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Worker {worker_id} failed to process batch {batch.batch_id}: {e}",
                        exc_info=True
                    )

            except Exception as e:
                logger.error(f"Worker {worker_id} encountered unexpected error: {e}")
                await asyncio.sleep(1)

        logger.info(f"Batch worker {worker_id} stopped")

    async def _wait_for_batch(self) -> BatchGroup:
        """Wait for a batch to become available."""
        while self.running:
            try:
                return await self.queue.get()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.1)
        
        raise asyncio.CancelledError("Scheduler stopping")

    async def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "running": self.running,
            "queue_size": self.queue.size(),
            "priority_counts": self.queue.priority_counts(),
            "active_workers": len([w for w in self.workers if not w.done()]),
            "total_workers": len(self.workers),
            "max_concurrent_batches": self.max_concurrent_batches
        }


def estimate_batch_processing_time(items: List[Any], 
                                 base_time_per_item: float = 30.0,
                                 overhead_per_batch: float = 5.0) -> float:
    """
    Estimate processing time for a batch.

    Args:
        items: List of items in the batch
        base_time_per_item: Base processing time per item in seconds
        overhead_per_batch: Additional overhead per batch in seconds

    Returns:
        Estimated processing time in seconds
    """
    return len(items) * base_time_per_item + overhead_per_batch


def calculate_optimal_batch_size(total_items: int,
                                max_batch_size: int = 20,
                                min_batch_size: int = 1,
                                target_batches: Optional[int] = None) -> int:
    """
    Calculate optimal batch size for given constraints.

    Args:
        total_items: Total number of items to process
        max_batch_size: Maximum allowed batch size
        min_batch_size: Minimum allowed batch size
        target_batches: Target number of batches (optional)

    Returns:
        Optimal batch size
    """
    if total_items == 0:
        return min_batch_size

    if target_batches:
        # Calculate batch size to achieve target number of batches
        calculated_size = max(min_batch_size, total_items // target_batches)
        return min(calculated_size, max_batch_size)
    else:
        # Use maximum batch size unless items don't divide evenly
        if total_items <= max_batch_size:
            return total_items
        
        # Find batch size that minimizes remainder
        best_size = max_batch_size
        min_remainder = total_items % max_batch_size
        
        for size in range(max_batch_size, min_batch_size - 1, -1):
            remainder = total_items % size
            if remainder == 0:
                return size
            elif remainder < min_remainder:
                min_remainder = remainder
                best_size = size
        
        return best_size


async def process_items_in_batches(
    items: List[T],
    batch_processor: Callable[[List[T]], Awaitable[Any]],
    batch_size: int = 10,
    max_concurrency: int = 5
) -> List[Any]:
    """
    Process items in batches with concurrency control.

    Args:
        items: Items to process
        batch_processor: Function to process each batch
        batch_size: Size of each batch
        max_concurrency: Maximum concurrent batch processing

    Returns:
        List of results from batch processing
    """
    if not items:
        return []

    # Split items into batches
    batches = [
        items[i:i + batch_size] 
        for i in range(0, len(items), batch_size)
    ]

    # Process batches with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_batch_with_semaphore(batch: List[T]) -> Any:
        async with semaphore:
            return await batch_processor(batch)

    # Process all batches concurrently
    tasks = [
        asyncio.create_task(process_batch_with_semaphore(batch))
        for batch in batches
    ]

    return await asyncio.gather(*tasks)