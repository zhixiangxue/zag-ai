"""
Vector retriever - basic layer retriever for vector search
"""

from typing import Any, Optional

from ..base import BaseRetriever
from ...schemas import BaseUnit
from ...schemas import RetrievalSource
from ...storages.vector.base import BaseVectorStore


class VectorRetriever(BaseRetriever):
    """
    Vector retriever (Basic Layer)
    
    Encapsulates vector search logic:
    1. Uses VectorStore to perform similarity search
    2. Returns units with similarity scores attached
    
    Responsibilities:
        - Wraps VectorStore search functionality
        - Provides retriever interface for composition
    
    Note:
        Unlike the design document which separates VectorStore and DocStore,
        our VectorStore already manages complete Unit objects and handles
        embeddings automatically, so this retriever is a simpler wrapper.
    
    Examples:
        >>> from zag.embedders import Embedder
        >>> from zag.storages.vector.chroma import ChromaVectorStore
        >>> from zag.retrievers import VectorRetriever
        >>> 
        >>> # Initialize components
        >>> embedder = Embedder("bailian/text-embedding-v3", api_key="sk-xxx")
        >>> vector_store = ChromaVectorStore(embedder=embedder)
        >>> 
        >>> # Create retriever
        >>> retriever = VectorRetriever(vector_store=vector_store)
        >>> 
        >>> # Retrieve
        >>> units = retriever.retrieve("What is RAG?", top_k=5)
        >>> for unit in units:
        >>>     print(f"Score: {unit.score}, Content: {unit.content[:100]}")
    """
    
    def __init__(
        self, 
        vector_store: BaseVectorStore,
        top_k: int = 10,
    ):
        """
        Initialize vector retriever
        
        Args:
            vector_store: Vector store instance
            top_k: Default number of results to return
        """
        self.vector_store = vector_store
        self.default_top_k = top_k
    
    def retrieve(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None
    ) -> list[BaseUnit]:
        """
        Execute vector search
        
        Args:
            query: Query text
            top_k: Number of results to return (overrides default if provided)
            filters: Optional metadata filters
            
        Returns:
            List of retrieved units, sorted by similarity score
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # VectorStore.search() already returns complete Units with scores
        units = self.vector_store.search(
            query=query,
            top_k=k,
            filter=filters,
        )
        
        # Set retrieval source
        for unit in units:
            unit.source = RetrievalSource.VECTOR
        
        return units
    
    async def aretrieve(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None
    ) -> list[BaseUnit]:
        """
        Async vector search
        
        Args:
            query: Query text
            top_k: Number of results to return (overrides default if provided)
            filters: Optional metadata filters
            
        Returns:
            List of retrieved units, sorted by similarity score
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # Use VectorStore's async search
        units = await self.vector_store.asearch(
            query=query,
            top_k=k,
            filter=filters,
        )
        
        # Set retrieval source
        for unit in units:
            unit.source = RetrievalSource.VECTOR
        
        return units
