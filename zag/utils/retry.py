"""
Retry utilities for handling transient failures

Provides context manager and decorator for automatic retry with exponential backoff
"""

import time
import asyncio
from typing import Optional, Callable, TypeVar, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryContext:
    """
    Context manager for retry logic with exponential backoff
    
    Example:
        with retry(max_attempts=5, backoff_factor=2.0):
            result = api_call()
        
        # Async version
        async with aretry(max_attempts=5):
            result = await async_api_call()
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 1.5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: tuple = (Exception,),
        on_retry: Optional[Callable[[Exception, int], None]] = None
    ):
        """
        Initialize retry context
        
        Args:
            max_attempts: Maximum number of attempts (default: 3)
            backoff_factor: Multiplier for delay between retries (default: 1.5)
            initial_delay: Initial delay in seconds (default: 1.0)
            max_delay: Maximum delay in seconds (default: 60.0)
            exceptions: Tuple of exceptions to catch (default: all Exception)
            on_retry: Optional callback function(exception, attempt_num) called on each retry
        """
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.on_retry = on_retry
        self.attempt = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False
        
        if not issubclass(exc_type, self.exceptions):
            return False
        
        self.attempt += 1
        
        if self.attempt >= self.max_attempts:
            logger.error(f"Max retry attempts ({self.max_attempts}) reached, giving up")
            return False
        
        # Calculate delay with exponential backoff
        delay = min(
            self.initial_delay * (self.backoff_factor ** (self.attempt - 1)),
            self.max_delay
        )
        
        logger.warning(
            f"Attempt {self.attempt}/{self.max_attempts} failed: {exc_val}. "
            f"Retrying in {delay:.1f}s..."
        )
        
        if self.on_retry:
            self.on_retry(exc_val, self.attempt)
        
        time.sleep(delay)
        return True


class AsyncRetryContext:
    """Async version of RetryContext"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 1.5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: tuple = (Exception,),
        on_retry: Optional[Callable[[Exception, int], None]] = None
    ):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.on_retry = on_retry
        self.attempt = 0
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False
        
        if not issubclass(exc_type, self.exceptions):
            return False
        
        self.attempt += 1
        
        if self.attempt >= self.max_attempts:
            logger.error(f"Max retry attempts ({self.max_attempts}) reached, giving up")
            return False
        
        # Calculate delay with exponential backoff
        delay = min(
            self.initial_delay * (self.backoff_factor ** (self.attempt - 1)),
            self.max_delay
        )
        
        logger.warning(
            f"Attempt {self.attempt}/{self.max_attempts} failed: {exc_val}. "
            f"Retrying in {delay:.1f}s..."
        )
        
        if self.on_retry:
            if asyncio.iscoroutinefunction(self.on_retry):
                await self.on_retry(exc_val, self.attempt)
            else:
                self.on_retry(exc_val, self.attempt)
        
        await asyncio.sleep(delay)
        return True


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 1.5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Context manager for retry with exponential backoff
    
    Example:
        with retry(max_attempts=5, backoff_factor=2.0):
            result = risky_operation()
    """
    return RetryContext(
        max_attempts=max_attempts,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exceptions=exceptions,
        on_retry=on_retry
    )


def aretry(
    max_attempts: int = 3,
    backoff_factor: float = 1.5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Async context manager for retry with exponential backoff
    
    Example:
        async with aretry(max_attempts=5):
            result = await async_risky_operation()
    """
    return AsyncRetryContext(
        max_attempts=max_attempts,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exceptions=exceptions,
        on_retry=on_retry
    )


def retry_decorator(
    max_attempts: int = 3,
    backoff_factor: float = 1.5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for automatic retry with exponential backoff
    
    Example:
        @retry_decorator(max_attempts=5, backoff_factor=2.0)
        def risky_function():
            return api_call()
        
        @retry_decorator(max_attempts=3)
        async def async_risky_function():
            return await async_api_call()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                last_exception = None
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            delay = min(
                                initial_delay * (backoff_factor ** attempt),
                                max_delay
                            )
                            logger.warning(
                                f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                                f"Retrying in {delay:.1f}s..."
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                f"{func.__name__} failed after {max_attempts} attempts"
                            )
                
                raise last_exception
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                last_exception = None
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            delay = min(
                                initial_delay * (backoff_factor ** attempt),
                                max_delay
                            )
                            logger.warning(
                                f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                                f"Retrying in {delay:.1f}s..."
                            )
                            time.sleep(delay)
                        else:
                            logger.error(
                                f"{func.__name__} failed after {max_attempts} attempts"
                            )
                
                raise last_exception
            
            return sync_wrapper
    
    return decorator
