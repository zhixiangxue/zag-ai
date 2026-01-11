"""
ChromaDB vector store implementation
"""

import asyncio
from typing import Any, Optional, Union

from .base import BaseVectorStore
from ...schemas.base import BaseUnit, UnitType
from ...schemas.unit import TextUnit


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store with multiple deployment modes
    
    Supports three deployment modes:
    1. **In-memory**: For testing and prototyping
    2. **Local persistent**: For development and single-node deployment
    3. **Server**: For production and multi-client access
    
    Usage:
        # In-memory mode (for testing)
        store = ChromaVectorStore.in_memory(
            collection_name="test",
            embedder=embedder
        )
        
        # Local persistent mode (for development)
        store = ChromaVectorStore.local(
            path="./chroma_db",
            collection_name="docs",
            embedder=embedder
        )
        
        # Server mode (for production)
        store = ChromaVectorStore.server(
            host="localhost",
            port=8000,
            collection_name="docs",
            embedder=embedder
        )
    
    **Async Support:**
    - Async methods (aadd, asearch, etc.) use thread pool executor wrapper
    - This is **not true async I/O**, just running sync code in a thread
    - For true async support with server mode, see ChromaDB documentation
    """
    
    def __init__(
        self,
        client: 'chromadb.ClientAPI',
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ):
        """
        Private constructor - use factory methods instead
        
        Factory methods:
        - ChromaVectorStore.in_memory() - for testing
        - ChromaVectorStore.local() - for development
        - ChromaVectorStore.server() - for production
        
        Args:
            client: ChromaDB client instance
            collection_name: Name of the Chroma collection
            embedder: Single embedder for all content types (multimodal)
            text_embedder: Embedder specifically for text/table units
            image_embedder: Embedder specifically for image units
            **kwargs: Additional parameters
        
        Note:
            See BaseVectorStore.__init__ for detailed embedder usage patterns
        """
        super().__init__(
            embedder=embedder,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            **kwargs
        )
        
        self.client = client
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    @classmethod
    def in_memory(
        cls,
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ) -> 'ChromaVectorStore':
        """
        Create in-memory Chroma store (for testing/prototyping)
        
        Data is lost when process exits.
        Perfect for unit tests and quick experiments.
        
        Args:
            collection_name: Name of the collection
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            **kwargs: Additional parameters
        
        Returns:
            ChromaVectorStore instance
        
        Example:
            >>> embedder = Embedder('bailian/text-embedding-v3')
            >>> store = ChromaVectorStore.in_memory(
            ...     collection_name="test",
            ...     embedder=embedder
            ... )
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )
        
        client = chromadb.Client()
        return cls(
            client=client,
            collection_name=collection_name,
            embedder=embedder,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            **kwargs
        )
    
    @classmethod
    def local(
        cls,
        path: str,
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ) -> 'ChromaVectorStore':
        """
        Create local persistent Chroma store (for development)
        
        Data persists to local directory.
        Suitable for development and single-node deployment.
        
        Args:
            path: Local directory to store data
            collection_name: Name of the collection
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            **kwargs: Additional parameters
        
        Returns:
            ChromaVectorStore instance
        
        Example:
            >>> embedder = Embedder('bailian/text-embedding-v3')
            >>> store = ChromaVectorStore.local(
            ...     path="./chroma_db",
            ...     collection_name="docs",
            ...     embedder=embedder
            ... )
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )
        
        client = chromadb.PersistentClient(path=path)
        return cls(
            client=client,
            collection_name=collection_name,
            embedder=embedder,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            **kwargs
        )
    
    @classmethod
    def server(
        cls,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ) -> 'ChromaVectorStore':
        """
        Connect to Chroma server (for production)
        
        Connects to a running Chroma server.
        Suitable for production deployments with multiple clients.
        
        Args:
            host: Chroma server host (default: localhost)
            port: Chroma server port (default: 8000)
            collection_name: Name of the collection
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            **kwargs: Additional parameters
        
        Returns:
            ChromaVectorStore instance
        
        Example:
            >>> # Start Chroma server first:
            >>> # chroma run --host localhost --port 8000
            >>> 
            >>> embedder = Embedder('bailian/text-embedding-v3')
            >>> store = ChromaVectorStore.server(
            ...     host="localhost",
            ...     port=8000,
            ...     collection_name="docs",
            ...     embedder=embedder
            ... )
        
        Note:
            Start Chroma server with:
            chroma run --host localhost --port 8000
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )
        
        client = chromadb.HttpClient(host=host, port=port)
        return cls(
            client=client,
            collection_name=collection_name,
            embedder=embedder,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            **kwargs
        )
    
    def add(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Add unit(s) to Chroma vector store
        
        Supports both single unit and batch operations:
        - Single: store.add(unit)
        - Batch: store.add([unit1, unit2, ...])
        
        Supports mixed unit types (text, table, image) in a single call.
        The method automatically:
        1. Normalizes input to list format
        2. Groups units by type
        3. Routes each group to appropriate embedder
        4. Batches embedding calls for efficiency
        5. Stores all units together
        
        Args:
            units: Single unit or list of units to store (can be mixed types)
        """
        # Normalize input to list format
        if isinstance(units, BaseUnit):
            units = [units]
        
        if not units:
            return
        
        # Group units by type for efficient batch embedding
        # text/table units share the same embedder, image units use separate one
        text_like_units = []  # TextUnit and TableUnit
        image_units = []      # ImageUnit
        
        for unit in units:
            if unit.unit_type == UnitType.IMAGE:
                image_units.append(unit)
            else:
                # text and table both use text_embedder
                text_like_units.append(unit)
        
        # Prepare collections for final storage
        all_ids = []
        all_embeddings = []
        all_metadatas = []
        all_documents = []
        
        # Process text-like units (text + table)
        if text_like_units:
            # Extract content for embedding
            contents = []
            for unit in text_like_units:
                if isinstance(unit, TextUnit):
                    contents.append(unit.content)
                else:
                    # For non-text units (e.g., TableUnit), convert to string
                    # TableUnit should already have text representation in content
                    contents.append(str(unit.content))
            
            # Get appropriate embedder and embed
            embedder = self._get_embedder_for_unit(text_like_units[0])
            embeddings = embedder.embed_batch(contents)
            
            # Collect data
            for unit, content, embedding in zip(text_like_units, contents, embeddings):
                all_ids.append(unit.unit_id)
                all_embeddings.append(embedding)
                all_documents.append(content)
                all_metadatas.append(self._extract_metadata(unit))
        
        # Process image units
        if image_units:
            # Extract image content
            image_contents = [unit.content for unit in image_units]
            
            # Get image embedder and embed
            embedder = self._get_embedder_for_unit(image_units[0])
            embeddings = embedder.embed_batch(image_contents)
            
            # Collect data
            for unit, content, embedding in zip(image_units, image_contents, embeddings):
                all_ids.append(unit.unit_id)
                all_embeddings.append(embedding)
                # For images, store a placeholder or empty string as document
                # (Chroma requires documents field)
                all_documents.append(f"[Image: {unit.unit_id}]")
                all_metadatas.append(self._extract_metadata(unit))
        
        # Add all units to Chroma in one batch
        self.collection.add(
            ids=all_ids,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            documents=all_documents
        )
    
    async def aadd(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of add (uses thread pool executor)
        
        **Important:** This is NOT true async I/O. ChromaDB's PersistentClient
        (used in embedded mode) does not support native async operations.
        This method runs the synchronous add() in a thread pool executor.
        
        For true async I/O, you need:
        1. Run ChromaDB as a server: `chroma run`
        2. Use chromadb.AsyncHttpClient instead of PersistentClient
        3. See: https://cookbook.chromadb.dev/core/clients/#http-client
        
        Args:
            units: Single unit or list of units to store
            
        Note:
            While this provides async interface compatibility, it doesn't
            eliminate blocking I/O. It only moves the blocking operation
            to a thread pool, which may help with concurrency but won't
            improve I/O performance like true async would.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.add, units)
    
    def _extract_metadata(self, unit: BaseUnit) -> dict:
        """
        Extract metadata from unit for storage
        
        Args:
            unit: Unit to extract metadata from
        
        Returns:
            Dictionary of metadata (Chroma compatible)
        """
        metadata = {
            "unit_type": unit.unit_type.value,  # Store enum value as string
            "source_doc_id": unit.source_doc_id or "",
        }
        
        # Add context_path if available
        if unit.metadata and unit.metadata.context_path:
            metadata["context_path"] = unit.metadata.context_path
        
        # Add custom metadata (only Chroma-compatible types)
        if unit.metadata and unit.metadata.custom:
            for key, value in unit.metadata.custom.items():
                # Chroma metadata values must be str, int, float, or bool
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"custom_{key}"] = value
        
        return metadata
    
    def search(
        self,
        query: Union[str, BaseUnit],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None
    ) -> list[BaseUnit]:
        """
        Search for similar units in Chroma
        
        Automatically routes query to appropriate embedder based on query type:
        - String query → text_embedder
        - TextUnit/TableUnit → text_embedder
        - ImageUnit → image_embedder
        
        Args:
            query: Query content (can be text string or Unit)
            top_k: Number of results to return
            filter: Optional metadata filters (Chroma where clause)
            
        Returns:
            List of matching units, sorted by similarity
        """
        # Determine query type and extract content
        if isinstance(query, str):
            # String query → use text_embedder
            query_content = query
            embedder = self.text_embedder
        elif isinstance(query, BaseUnit):
            # Unit query → route to appropriate embedder
            embedder = self._get_embedder_for_unit(query)
            
            if isinstance(query, TextUnit):
                query_content = query.content
            elif query.unit_type == UnitType.IMAGE:
                query_content = query.content  # bytes for image
            else:
                query_content = str(query.content)
        else:
            raise TypeError(
                f"Query must be str or BaseUnit, got {type(query).__name__}"
            )
        
        # Embed the query using appropriate embedder
        query_embedding = embedder.embed(query_content)
        
        # Search in Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        
        # Convert results to Units
        units = []
        if results['ids'] and results['ids'][0]:
            for i, unit_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else None
                
                # Reconstruct unit based on type
                # Note: In production, you might want to store full unit data
                # or retrieve from a separate document store
                unit = TextUnit(
                    unit_id=unit_id,
                    content=document,
                    unit_type=metadata.get('unit_type', 'text')
                )
                
                # Restore metadata
                if 'context_path' in metadata:
                    unit.metadata.context_path = metadata['context_path']
                
                if 'source_doc_id' in metadata and metadata['source_doc_id']:
                    unit.source_doc_id = metadata['source_doc_id']
                
                # Attach similarity score
                # Chroma uses cosine distance by default (lower is more similar)
                # Convert to similarity score: similarity = 1 - distance
                if distance is not None:
                    unit.score = 1.0 - distance
                
                units.append(unit)
        
        return units
    
    async def asearch(
        self,
        query: Union[str, BaseUnit],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None
    ) -> list[BaseUnit]:
        """
        Async version of search (uses thread pool executor)
        
        **Important:** This is NOT true async I/O. See aadd() docstring for details.
        
        Args:
            query: Query content (can be text string or Unit)
            top_k: Number of results to return
            filter: Optional metadata filters
            
        Returns:
            List of matching units, sorted by similarity
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.search, query, top_k, filter)
    
    def delete(self, unit_ids: list[str]) -> None:
        """
        Delete units by IDs from Chroma
        
        Args:
            unit_ids: List of unit IDs to delete
        """
        if not unit_ids:
            return
        
        self.collection.delete(ids=unit_ids)
    
    async def adelete(self, unit_ids: list[str]) -> None:
        """
        Async version of delete (uses thread pool executor)
        
        **Important:** This is NOT true async I/O. See aadd() docstring for details.
        
        Args:
            unit_ids: List of unit IDs to delete
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.delete, unit_ids)
    
    def get(self, unit_ids: list[str]) -> list[BaseUnit]:
        """
        Get units by IDs from Chroma
        
        Args:
            unit_ids: List of unit IDs
            
        Returns:
            List of corresponding units
        """
        if not unit_ids:
            return []
        
        results = self.collection.get(ids=unit_ids)
        
        units = []
        if results['ids']:
            for i, unit_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                document = results['documents'][i]
                
                # Reconstruct unit
                unit = TextUnit(
                    unit_id=unit_id,
                    content=document,
                    unit_type=metadata.get('unit_type', 'text')
                )
                
                # Restore metadata
                if 'context_path' in metadata:
                    unit.metadata.context_path = metadata['context_path']
                
                if 'source_doc_id' in metadata and metadata['source_doc_id']:
                    unit.source_doc_id = metadata['source_doc_id']
                
                units.append(unit)
        
        return units
    
    async def aget(self, unit_ids: list[str]) -> list[BaseUnit]:
        """
        Async version of get (uses thread pool executor)
        
        **Important:** This is NOT true async I/O. See aadd() docstring for details.
        
        Args:
            unit_ids: List of unit IDs
            
        Returns:
            List of corresponding units
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get, unit_ids)
    
    def update(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Update existing unit(s) in Chroma
        
        Supports both single unit and batch operations.
        
        Args:
            units: Single unit or list of units to update
        """
        # Normalize input to list format
        if isinstance(units, BaseUnit):
            units = [units]
        
        if not units:
            return
        
        # Chroma doesn't have a native update, so we delete and re-add
        unit_ids = [unit.unit_id for unit in units]
        self.delete(unit_ids)
        self.add(units)
    
    async def aupdate(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of update (uses thread pool executor)
        
        **Important:** This is NOT true async I/O. See aadd() docstring for details.
        
        Args:
            units: Single unit or list of units to update
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.update, units)
    
    def clear(self) -> None:
        """
        Clear all vectors from Chroma collection
        """
        # Delete and recreate collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    async def aclear(self) -> None:
        """
        Async version of clear (uses thread pool executor)
        
        **Important:** This is NOT true async I/O. See aadd() docstring for details.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.clear)
    
    @property
    def dimension(self) -> int:
        """
        Get vector dimension
        
        Returns the dimension of the primary embedder (text_embedder).
        Note: In multimodal mode, all embedders should have the same dimension.
        
        Returns:
            Vector dimension
        """
        # Use text_embedder as the primary embedder
        # (In multimodal mode, text_embedder == image_embedder)
        return self.text_embedder.dimension
    
    def count(self) -> int:
        """
        Get total number of units in collection
        
        Returns:
            Number of units stored
        """
        return self.collection.count()
