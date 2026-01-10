"""
Progress display utilities

Provides decorators for adding progress bars to methods.
Decorators handle normalization, progress display, and error handling.
"""

from functools import wraps
from typing import Union, Callable, Any
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn,
    TimeElapsedColumn,
)


def with_spinner_progress(description: str = "Processing {count} items"):
    """
    Decorator: Spinner + count + elapsed time
    
    Suitable for batch operations where intermediate progress cannot be reported.
    
    Display format: "Adding 100 units to index... ⠋  0:00:02"
    
    Args:
        description: Operation description (supports {count} placeholder)
    
    Features:
        - Automatically normalizes single item to list
        - Skips empty lists
        - Shows spinner animation + elapsed time
        - Displays item count in description
    
    Example:
        >>> @with_spinner_progress("Adding {count} units to index")
        >>> def add(self, units_list: list[BaseUnit]) -> None:
        ...     self.store.add(units_list)
        
        >>> @with_spinner_progress("Indexing {count} documents")
        >>> async def aadd(self, units_list: list[BaseUnit]) -> None:
        ...     await self.store.aadd(units_list)
    
    Note:
        - The decorated method receives a list (already normalized)
        - Return value is preserved
        - Works with both sync and async methods
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(self, units_or_list: Union[Any, list], *args, **kwargs):
            # Normalize to list
            if not isinstance(units_or_list, list):
                items = [units_or_list]
            else:
                items = units_or_list
            
            if not items:
                return None
            
            # Format description with count
            desc = description.format(count=len(items))
            
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold blue]{desc}..."),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(desc)
                result = await func(self, items, *args, **kwargs)
                progress.update(task, completed=100)
                return result
        
        @wraps(func)
        def sync_wrapper(self, units_or_list: Union[Any, list], *args, **kwargs):
            # Normalize to list
            if not isinstance(units_or_list, list):
                items = [units_or_list]
            else:
                items = units_or_list
            
            if not items:
                return None
            
            # Format description with count
            desc = description.format(count=len(items))
            
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold blue]{desc}..."),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(desc)
                result = func(self, items, *args, **kwargs)
                progress.update(task, completed=100)
                return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
