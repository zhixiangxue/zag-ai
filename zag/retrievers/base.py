"""
Base retriever class
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from zag.schemas.base import BaseUnit


class RetrievalResult:
    """Result from retrieval"""
    
    def __init__(
        self,
        unit: BaseUnit,
        score: float,
        metadata: Optional[dict[str, Any]] = None
    ):
        self.unit = unit
        self.score = score
        self.metadata = metadata or {}


class BaseRetriever(ABC):
    """
    Base class for all retrievers
    
    Retrievers search and return relevant units
    """
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant units for a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieval results with scores
        """
        pass
