"""
Base postprocessor abstract class
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..schemas.base import BaseUnit


class BasePostprocessor(ABC):
    """
    Abstract base class for all postprocessors
    
    All postprocessors (including rerankers, filters, augmentors, etc.) 
    implement this unified interface. This allows flexible composition 
    and nesting of postprocessors.
    
    Design Philosophy:
        - Single responsibility: each postprocessor focuses on one task
        - Composability: postprocessors can be combined and nested
        - Uniformity: all postprocessors use the same interface
    
    Examples:
        >>> class MyFilter(BasePostprocessor):
        ...     def process(self, query: str, units: list[BaseUnit]) -> list[BaseUnit]:
        ...         return [u for u in units if u.score > 0.7]
    """
    
    @abstractmethod
    def process(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Post-process retrieved units
        
        Args:
            query: Original query text
            units: Units to process
            
        Returns:
            Processed units (may change count, order, or content)
        """
        pass
    
    async def aprocess(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Async version of process (optional implementation)
        
        Args:
            query: Original query text
            units: Units to process
            
        Returns:
            Processed units
        """
        # Default implementation: call sync version
        return self.process(query, units)
