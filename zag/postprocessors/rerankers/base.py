"""
Base reranker class
"""

from abc import abstractmethod
from typing import Optional

from ..base import BasePostprocessor
from ...schemas.base import BaseUnit


class BaseReranker(BasePostprocessor):
    """
    Base class for all rerankers (special type of postprocessor)
    
    Rerankers use more accurate models to recompute relevance scores
    and reorder results.
    
    Characteristics:
        - Changes result ordering (by recomputing scores)
        - Usually uses complex models (Cross-encoder, LLM, etc.)
        - Computationally intensive but significantly improves quality
    
    Note:
        Reranker inherits from BasePostprocessor, implementing the process()
        interface while also providing the rerank() method for semantic clarity.
    
    Examples:
        >>> class MyReranker(BaseReranker):
        ...     def rerank(self, query: str, units: list[BaseUnit], top_k=None):
        ...         # Compute new scores
        ...         for unit in units:
        ...             unit.score = compute_score(query, unit)
        ...         return sorted(units, key=lambda x: x.score, reverse=True)[:top_k]
    """
    
    def process(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Implement BasePostprocessor interface
        
        Internally calls rerank() method
        """
        return self.rerank(query, units)
    
    @abstractmethod
    def rerank(
        self, 
        query: str,
        units: list[BaseUnit],
        top_k: Optional[int] = None
    ) -> list[BaseUnit]:
        """
        Rerank units by recomputing relevance scores
        
        Args:
            query: Original query text
            units: Units to rerank
            top_k: Maximum number of results to return (None = all)
            
        Returns:
            Reranked units, sorted by new relevance scores
        """
        pass
