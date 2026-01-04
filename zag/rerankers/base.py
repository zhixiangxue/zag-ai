"""
Base reranker class
"""

from abc import ABC, abstractmethod

from ..retrievers.base import RetrievalResult


class BaseReranker(ABC):
    """
    Base class for all rerankers
    
    Rerankers reorder retrieval results for better relevance
    """
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results
        
        Args:
            query: Original search query
            results: Initial retrieval results
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked results
        """
        pass
