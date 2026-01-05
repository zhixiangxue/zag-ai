"""
Provider base class for reranker implementations (internal use only)
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """
    Base provider class for reranker implementations
    
    All provider implementations must inherit from this class
    and implement the required methods.
    """
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        documents: list[str],
        top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """
        Rerank documents for a given query
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = all)
            
        Returns:
            List of (document, score) tuples, sorted by score (descending)
        """
        pass
