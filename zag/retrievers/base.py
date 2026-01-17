"""
Base retriever abstract class
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..schemas import BaseUnit


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers
    
    All retrievers (both basic and composite) implement this unified interface.
    This allows flexible composition and nesting of retrievers.
    
    Design Philosophy:
        - Single responsibility: each retriever focuses on one retrieval strategy
        - Composability: retrievers can be combined and nested
        - Uniformity: all retrievers use the same interface
    """
    
    @abstractmethod
    def retrieve(
        self, 
        query: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[BaseUnit]:
        """
        Retrieve relevant units for the given query
        
        Args:
            query: Query text
            top_k: Maximum number of units to return
            filters: Optional metadata filters
            
        Returns:
            List of retrieved units, sorted by relevance
        """
        pass
    
    async def aretrieve(
        self, 
        query: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[BaseUnit]:
        """
        Async version of retrieve (optional implementation)
        
        Args:
            query: Query text
            top_k: Maximum number of units to return
            filters: Optional metadata filters
            
        Returns:
            List of retrieved units, sorted by relevance
        """
        # Default implementation: call sync version
        return self.retrieve(query, top_k, filters)
