"""
Base vector store class
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """Result from vector search"""
    
    unit_id: str
    score: float
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    """
    Base class for vector storage
    
    VectorStore stores embeddings and performs similarity search
    """
    
    @abstractmethod
    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: Optional[list[dict[str, Any]]] = None
    ) -> None:
        """
        Add vectors to store
        
        Args:
            ids: List of unit IDs
            vectors: List of embedding vectors
            metadatas: Optional list of metadata dicts
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_dict: Optional[dict[str, Any]] = None
    ) -> list[VectorSearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results with unit IDs and scores
        """
        pass
    
    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        Delete vectors by IDs
        
        Args:
            ids: List of unit IDs to delete
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all vectors from store
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Get the embedding dimension
        
        Returns:
            Dimension of stored vectors
        """
        pass
