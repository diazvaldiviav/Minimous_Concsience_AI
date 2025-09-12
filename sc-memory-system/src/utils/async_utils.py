"""
Async utilities for Week 3 enhancements.

This module provides helper functions and utilities for async processing,
resource monitoring, and concurrent task management.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def run_with_timeout(
    coro: Awaitable[T], 
    timeout_seconds: float,
    timeout_message: Optional[str] = None
) -> T:
    """
    Run a coroutine with a timeout.

    Args:
        coro: The coroutine to run
        timeout_seconds: Timeout in seconds
        timeout_message: Optional custom timeout message

    Returns:
        The result of the coroutine

    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        message = timeout_message or f"Operation timed out after {timeout_seconds} seconds"
        logger.error(message)
        raise asyncio.TimeoutError(message)


async def run_with_retry(
    coro_func: Callable[..., Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> T:
    """
    Run a coroutine function with exponential backoff retry.

    Args:
        coro_func: The coroutine function to run
        max_retries: Maximum number of retries
        base_delay: Base delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
        *args: Arguments to pass to coro_func
        **kwargs: Keyword arguments to pass to coro_func

    Returns:
        The result of the coroutine function

    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            
            if attempt == max_retries:
                # Final attempt failed
                logger.error(f"All {max_retries} retry attempts failed. Last error: {e}")
                raise
            
            # Calculate delay with exponential backoff
            delay = base_delay * (backoff_multiplier ** attempt)
            
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay:.2f} seconds..."
            )
            
            await asyncio.sleep(delay)
    
    # This should never be reached, but just in case
    raise last_exception or Exception("Unexpected retry loop exit")


async def gather_with_concurrency(
    coroutines: List[Awaitable[T]], 
    max_concurrency: int = 10
) -> List[T]:
    """
    Run multiple coroutines with limited concurrency.

    Args:
        coroutines: List of coroutines to run
        max_concurrency: Maximum number of concurrent coroutines

    Returns:
        List of results in the same order as input coroutines
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def run_with_semaphore(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro
    
    # Wrap all coroutines with semaphore
    limited_coroutines = [run_with_semaphore(coro) for coro in coroutines]
    
    return await asyncio.gather(*limited_coroutines)


async def gather_with_error_handling(
    coroutines: List[Awaitable[T]], 
    return_exceptions: bool = True,
    max_concurrency: Optional[int] = None
) -> List[Any]:
    """
    Run multiple coroutines with proper error handling and optional concurrency limit.

    Args:
        coroutines: List of coroutines to run
        return_exceptions: If True, exceptions are returned as results
        max_concurrency: Optional maximum number of concurrent coroutines

    Returns:
        List of results or exceptions
    """
    if max_concurrency:
        return await gather_with_concurrency_and_error_handling(
            coroutines, max_concurrency, return_exceptions
        )
    else:
        return await asyncio.gather(*coroutines, return_exceptions=return_exceptions)


async def gather_with_concurrency_and_error_handling(
    coroutines: List[Awaitable[T]],
    max_concurrency: int,
    return_exceptions: bool = True
) -> List[Any]:
    """
    Run coroutines with both concurrency limit and error handling.

    Args:
        coroutines: List of coroutines to run
        max_concurrency: Maximum number of concurrent coroutines
        return_exceptions: If True, exceptions are returned as results

    Returns:
        List of results or exceptions
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def run_with_semaphore_and_error_handling(coro: Awaitable[T]) -> Any:
        async with semaphore:
            try:
                return await coro
            except Exception as e:
                if return_exceptions:
                    return e
                raise
    
    # Wrap all coroutines
    wrapped_coroutines = [
        run_with_semaphore_and_error_handling(coro) 
        for coro in coroutines
    ]
    
    return await asyncio.gather(*wrapped_coroutines, return_exceptions=False)


class AsyncContextManager:
    """A generic async context manager for resource management."""
    
    def __init__(self, 
                 setup_func: Callable[[], Awaitable[Any]],
                 cleanup_func: Callable[[Any], Awaitable[None]]):
        """
        Initialize async context manager.

        Args:
            setup_func: Async function to set up the resource
            cleanup_func: Async function to clean up the resource
        """
        self.setup_func = setup_func
        self.cleanup_func = cleanup_func
        self.resource = None

    async def __aenter__(self):
        self.resource = await self.setup_func()
        return self.resource

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.resource is not None:
            await self.cleanup_func(self.resource)
        return False


class AsyncBatchProcessor:
    """Process items in async batches with configurable batch size and timing."""
    
    def __init__(self, 
                 batch_size: int = 10,
                 batch_timeout: float = 5.0,
                 max_concurrency: int = 5):
        """
        Initialize batch processor.

        Args:
            batch_size: Maximum items per batch
            batch_timeout: Maximum time to wait for batch to fill
            max_concurrency: Maximum concurrent batch processing
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_concurrency = max_concurrency
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None

    async def add_item(self, item: Any) -> None:
        """Add an item to the batch queue."""
        await self.queue.put(item)

    async def start_processing(self, 
                              batch_handler: Callable[[List[Any]], Awaitable[None]]) -> None:
        """
        Start batch processing.

        Args:
            batch_handler: Async function to process each batch
        """
        if self.running:
            return

        self.running = True
        self.worker_task = asyncio.create_task(
            self._batch_worker(batch_handler)
        )

    async def stop_processing(self) -> None:
        """Stop batch processing and process remaining items."""
        self.running = False
        
        if self.worker_task:
            await self.worker_task

    async def _batch_worker(self, 
                           batch_handler: Callable[[List[Any]], Awaitable[None]]) -> None:
        """Worker that processes items in batches."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        while self.running or not self.queue.empty():
            batch = []
            batch_start_time = time.time()
            
            # Collect items for batch
            while (len(batch) < self.batch_size and 
                   (time.time() - batch_start_time) < self.batch_timeout):
                
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=min(0.1, self.batch_timeout - (time.time() - batch_start_time))
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # Process batch if not empty
            if batch:
                async with semaphore:
                    try:
                        await batch_handler(batch)
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}", exc_info=True)

            # Short pause to prevent busy waiting
            await asyncio.sleep(0.01)


def async_lru_cache(maxsize: int = 128, ttl: float = 3600):
    """
    LRU cache decorator for async functions with TTL support.

    Args:
        maxsize: Maximum cache size
        ttl: Time to live in seconds
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        cache: Dict[str, Dict[str, Any]] = {}
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Create cache key
            key = str(hash((args, tuple(sorted(kwargs.items())))))
            
            # Check cache
            if key in cache:
                cached_item = cache[key]
                if time.time() - cached_item['timestamp'] < ttl:
                    return cached_item['result']
                else:
                    # Remove expired item
                    del cache[key]
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            
            # Manage cache size
            if len(cache) >= maxsize:
                # Remove oldest item
                oldest_key = min(cache.keys(), key=lambda k: cache[k]['timestamp'])
                del cache[oldest_key]
            
            # Cache the result
            cache[key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            return result
        
        return wrapper
    return decorator


async def create_task_with_error_logging(
    coro: Awaitable[T], 
    task_name: str = "unnamed_task"
) -> asyncio.Task[T]:
    """
    Create a task with automatic error logging.

    Args:
        coro: The coroutine to run
        task_name: Name for logging purposes

    Returns:
        The created task
    """
    async def wrapped_coro() -> T:
        try:
            return await coro
        except Exception as e:
            logger.error(f"Task '{task_name}' failed with error: {e}", exc_info=True)
            raise

    task = asyncio.create_task(wrapped_coro())
    task.set_name(task_name)
    return task


class AsyncRateLimiter:
    """Rate limiter for async operations."""
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire rate limit permission."""
        async with self.lock:
            now = time.time()
            
            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < self.time_window]
            
            # If we're at the limit, wait
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    # Recursive call to try again
                    return await self.acquire()
            
            # Record this call
            self.calls.append(now)


async def wait_for_condition(
    condition_func: Callable[[], Awaitable[bool]],
    timeout: float = 30.0,
    check_interval: float = 0.5,
    timeout_message: Optional[str] = None
) -> None:
    """
    Wait for an async condition to become true.

    Args:
        condition_func: Async function that returns True when condition is met
        timeout: Maximum time to wait in seconds
        check_interval: How often to check the condition
        timeout_message: Custom timeout message

    Raises:
        asyncio.TimeoutError: If condition is not met within timeout
    """
    start_time = time.time()
    
    while True:
        if await condition_func():
            return
        
        if time.time() - start_time > timeout:
            message = timeout_message or f"Condition not met within {timeout} seconds"
            raise asyncio.TimeoutError(message)
        
        await asyncio.sleep(check_interval)