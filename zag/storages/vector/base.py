"""
Base vector store class
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class BaseVectorStore(ABC):
    """
    Base class for vector storage
    
    VectorStore manages embedder(s) and provides vector storage/retrieval capabilities.
    Supports both single multimodal embedder and separate embedders for different content types.
    
    Design Philosophy:
        - VectorStore naturally knows its embedding model(s)
        - Users don't need to manually manage embedding process
        - Ensures consistency between storage and retrieval embeddings
        - Supports both multimodal and specialized embedders
    
    Responsibilities:
        - Manage embedder(s) (ensure same model for add/search)
        - Route different unit types to appropriate embedders
        - Automatically convert units to vectors and store
        - Automatically convert query to vector and search
    """
    
    def __init__(
        self,
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ):
        """
        Initialize vector store with embedder(s)
        
        Two usage patterns:
        
        Pattern 1 (Recommended): Single multimodal embedder
            Use when your embedder supports multiple content types (text, image, etc.)
            Example: OpenAI CLIP, Alibaba Tongyi Multimodal
            
        Pattern 2: Separate embedders for different types
            Use when you want specialized embedders for each content type
            Example: Bailian for text + CLIP for images
        
        Args:
            embedder: Single embedder for all content types (multimodal)
                     If provided, this will be used for all unit types
            text_embedder: Embedder specifically for text and table units
                          Only used when 'embedder' is not provided
            image_embedder: Embedder specifically for image units
                           Only used when 'embedder' is not provided
            **kwargs: Implementation-specific parameters
        
        Raises:
            ValueError: If neither 'embedder' nor 'text_embedder' is provided
        
        Examples:
            # Pattern 1: Multimodal embedder (handles text and images)
            >>> multimodal_emb = Embedder('openai/clip-vit')
            >>> store = ChromaVectorStore(embedder=multimodal_emb)
            
            # Pattern 2: Separate embedders for text and images
            >>> text_emb = Embedder('bailian/text-embedding-v3')
            >>> image_emb = Embedder('openai/clip-vit')
            >>> store = ChromaVectorStore(
            ...     text_embedder=text_emb,
            ...     image_embedder=image_emb
            ... )
            
            # Pattern 3: Text-only (most common case)
            >>> text_emb = Embedder('bailian/text-embedding-v3')
            >>> store = ChromaVectorStore(embedder=text_emb)
            >>> # or
            >>> store = ChromaVectorStore(text_embedder=text_emb)
        """
        if embedder:
            # Pattern 1: Single multimodal embedder
            # Use the same embedder for all content types
            self.embedder = embedder
            self.text_embedder = embedder
            self.image_embedder = embedder
            self._is_multimodal = True
        else:
            # Pattern 2: Separate embedders
            # Require at least text_embedder for text and table units
            if not text_embedder:
                raise ValueError(
                    "Must provide either 'embedder' (for multimodal) "
                    "or at least 'text_embedder' for text/table units"
                )
            self.embedder = None
            self.text_embedder = text_embedder
            self.image_embedder = image_embedder
            self._is_multimodal = False
    
    def _get_embedder_for_unit(self, unit: 'BaseUnit') -> 'BaseEmbedder':
        """
        Get appropriate embedder for a given unit (routing logic)
        
        Routing rules:
        - Multimodal mode: All unit types use the same embedder
        - Separate mode:
            * TextUnit → text_embedder
            * TableUnit → text_embedder (tables are converted to text at unit level)
            * ImageUnit → image_embedder
        
        Args:
            unit: The unit to get embedder for
        
        Returns:
            The appropriate embedder for this unit type
        
        Raises:
            ValueError: If image_embedder is required but not provided
        """
        from zag.schemas.base import UnitType
        
        if self._is_multimodal:
            # Multimodal embedder handles all types
            return self.embedder
        
        # Separate embedders: route by type
        if unit.unit_type == UnitType.IMAGE:
            if not self.image_embedder:
                raise ValueError(
                    f"ImageUnit requires 'image_embedder', but it was not provided. "
                    f"Please provide either a multimodal 'embedder' or 'image_embedder'."
                )
            return self.image_embedder
        else:
            # TextUnit and TableUnit both use text_embedder
            # (TableUnit.content should already be text representation)
            return self.text_embedder
    
    @abstractmethod
    def add(self, units: Union['BaseUnit', list['BaseUnit']]) -> None:
        """
        Add unit(s) to vector store
        
        Supports both single unit and batch operations:
        - Single: store.add(unit)
        - Batch: store.add([unit1, unit2, ...])
        
        Args:
            units: Single unit or list of units to store
            
        Internal Process:
            1. Normalize input to list format
            2. Use self.embedder to generate vectors for units
            3. Store vectors and metadata to vector database
            
        Note:
            - Users don't need to pre-embed units
            - VectorStore automatically calls embedder
            - Ensures unified embedding model
            - Batch operations are more efficient when adding multiple units
        """
        pass
    
    @abstractmethod
    async def aadd(self, units: Union['BaseUnit', list['BaseUnit']]) -> None:
        """
        Async version of add - must be implemented by subclass
        
        Add unit(s) to vector store asynchronously.
        
        Args:
            units: Single unit or list of units to store
            
        Implementation Notes:
            - If backend supports async natively (e.g., Qdrant), implement true async I/O
            - If backend is sync-only (e.g., ChromaDB), can use executor wrapper or raise NotImplementedError
            - Make it clear in subclass docstring whether true async is supported
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: Union[str, 'BaseUnit'],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None
    ) -> list['BaseUnit']:
        """
        Search for similar units
        
        Args:
            query: Query content (can be text or Unit)
            top_k: Number of results to return
            filter: Optional metadata filters
            
        Returns:
            List of matching units, sorted by similarity
            
        Internal Process:
            1. Use self.embedder to convert query to vector
            2. Search similar vectors in vector database
            3. Return corresponding units
            
        Note:
            - Users don't need to pre-embed query
            - Automatically uses same embedder as storage
        """
        pass
    
    @abstractmethod
    async def asearch(
        self,
        query: Union[str, 'BaseUnit'],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None
    ) -> list['BaseUnit']:
        """
        Async version of search - must be implemented by subclass
        
        Search for similar units asynchronously.
        
        Args:
            query: Query content (can be text or Unit)
            top_k: Number of results to return
            filter: Optional metadata filters
            
        Returns:
            List of matching units, sorted by similarity
            
        Implementation Notes:
            - If backend supports async natively, implement true async I/O
            - If backend is sync-only, can use executor wrapper or raise NotImplementedError
        """
        pass
    
    @abstractmethod
    def delete(self, unit_ids: list[str]) -> None:
        """
        Delete units by IDs
        
        Args:
            unit_ids: List of unit IDs to delete
        """
        pass
    
    @abstractmethod
    async def adelete(self, unit_ids: list[str]) -> None:
        """
        Async version of delete - must be implemented by subclass
        
        Delete units by IDs asynchronously.
        
        Args:
            unit_ids: List of unit IDs to delete
        """
        pass
    
    @abstractmethod
    def get(self, unit_ids: list[str]) -> list['BaseUnit']:
        """
        Get units by IDs
        
        Args:
            unit_ids: List of unit IDs
            
        Returns:
            List of corresponding units
        """
        pass
    
    @abstractmethod
    async def aget(self, unit_ids: list[str]) -> list['BaseUnit']:
        """
        Async version of get - must be implemented by subclass
        
        Get units by IDs asynchronously.
        
        Args:
            unit_ids: List of unit IDs
            
        Returns:
            List of corresponding units
        """
        pass
    
    @abstractmethod
    def update(self, units: Union['BaseUnit', list['BaseUnit']]) -> None:
        """
        Update existing unit(s)
        
        Supports both single unit and batch operations.
        
        Args:
            units: Single unit or list of units to update
            
        Internal Process:
            1. Re-generate vectors using self.embedder
            2. Update records in vector database
        """
        pass
    
    @abstractmethod
    async def aupdate(self, units: Union['BaseUnit', list['BaseUnit']]) -> None:
        """
        Async version of update - must be implemented by subclass
        
        Update existing unit(s) asynchronously.
        
        Args:
            units: Single unit or list of units to update
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all vectors from store
        """
        pass
    
    @abstractmethod
    async def aclear(self) -> None:
        """
        Async version of clear - must be implemented by subclass
        
        Clear all vectors from store asynchronously.
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Get vector dimension
        
        Returns:
            Vector dimension (determined by embedder)
        """
        pass
    
    def get_embedder(self) -> 'BaseEmbedder':
        """
        Get current embedder
        
        Returns:
            Current embedder instance
        """
        return self.embedder
