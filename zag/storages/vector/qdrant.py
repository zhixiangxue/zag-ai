"""
Qdrant vector store implementation
"""

import asyncio
from typing import Any, Optional, Union

from .base import BaseVectorStore
from ...schemas import BaseUnit, UnitType
from ...schemas.unit import TextUnit


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store with multiple deployment modes
    
    Supports four deployment modes:
    1. **In-memory**: For testing and prototyping (data lost on exit)
    2. **Local persistent**: For development and single-node deployment
    3. **Server**: For production self-hosted deployment
    4. **Cloud**: For Qdrant Cloud managed service
    
    Usage:
        # In-memory mode (for testing)
        store = QdrantVectorStore.in_memory(
            collection_name="test",
            embedder=embedder
        )
        
        # Local persistent mode (for development)
        store = QdrantVectorStore.local(
            path="./qdrant_data",
            collection_name="docs",
            embedder=embedder
        )
        
        # Server mode (for production self-hosted)
        store = QdrantVectorStore.server(
            host="localhost",
            port=6333,
            collection_name="docs",
            embedder=embedder
        )
        
        # Cloud mode (for Qdrant Cloud)
        store = QdrantVectorStore.cloud(
            url="https://xxx.qdrant.io",
            api_key="your_key",
            collection_name="docs",
            embedder=embedder
        )
    
    **Async Support:**
    - Async methods (aadd, asearch, etc.) use thread pool executor wrapper
    - This is **not true async I/O**, just running sync code in a thread
    - For true async support, use Qdrant's AsyncQdrantClient
    """
    
    def __init__(
        self,
        client: 'QdrantClient',
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ):
        """
        Private constructor - use factory methods instead
        
        Factory methods:
        - QdrantVectorStore.in_memory() - for testing
        - QdrantVectorStore.local() - for development
        - QdrantVectorStore.server() - for production
        - QdrantVectorStore.cloud() - for managed service
        
        Args:
            client: Qdrant client instance
            collection_name: Name of the Qdrant collection
            embedder: Single embedder for all content types (multimodal)
            text_embedder: Embedder specifically for text/table units
            image_embedder: Embedder specifically for image units
            **kwargs: Additional parameters
        
        Note:
            See BaseVectorStore.__init__ for detailed embedder usage patterns
        """
        # Initialize client and collection name first
        self.client = client
        self.collection_name = collection_name
        
        # Call parent constructor to initialize embedders
        super().__init__(
            embedder=embedder,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            **kwargs
        )
        
        # Create collection after embedders are initialized
        # (need embedder.dimension for vector size)
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        from qdrant_client.models import Distance, VectorParams
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            # Create collection with vector configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                )
            )
    
    @classmethod
    def in_memory(
        cls,
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ) -> 'QdrantVectorStore':
        """
        Create in-memory Qdrant store (for testing/prototyping)
        
        Data is lost when process exits.
        Perfect for unit tests and quick experiments.
        
        Args:
            collection_name: Name of the collection
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            **kwargs: Additional parameters
        
        Returns:
            QdrantVectorStore instance
        
        Example:
            >>> embedder = Embedder('bailian/text-embedding-v3')
            >>> store = QdrantVectorStore.in_memory(
            ...     collection_name="test",
            ...     embedder=embedder
            ... )
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install it with: pip install qdrant-client"
            )
        
        client = QdrantClient(location=":memory:")
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
    ) -> 'QdrantVectorStore':
        """
        Create local persistent Qdrant store (for development)
        
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
            QdrantVectorStore instance
        
        Example:
            >>> embedder = Embedder('bailian/text-embedding-v3')
            >>> store = QdrantVectorStore.local(
            ...     path="./qdrant_data",
            ...     collection_name="docs",
            ...     embedder=embedder
            ... )
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install it with: pip install qdrant-client"
            )
        
        client = QdrantClient(path=path)
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
        port: int = 6333,
        grpc_port: Optional[int] = None,
        prefer_grpc: bool = True,
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ) -> 'QdrantVectorStore':
        """
        Connect to Qdrant server (for production)
        
        Connects to a running Qdrant server.
        Suitable for production deployments with multiple clients.
        
        Args:
            host: Qdrant server host (default: localhost)
            port: Qdrant HTTP port (default: 6333)
            grpc_port: Qdrant gRPC port (optional, default: port+1)
            prefer_grpc: Use gRPC if available (default: True, much faster)
            collection_name: Name of the collection
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            **kwargs: Additional parameters
        
        Returns:
            QdrantVectorStore instance
        
        Example:
            >>> # Start Qdrant server first:
            >>> # docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
            >>> 
            >>> embedder = Embedder('bailian/text-embedding-v3')
            >>> 
            >>> # Use gRPC (faster, recommended)
            >>> store = QdrantVectorStore.server(
            ...     host="localhost",
            ...     port=6333,
            ...     grpc_port=6334,
            ...     collection_name="docs",
            ...     embedder=embedder
            ... )
            >>> 
            >>> # Or use HTTP only
            >>> store = QdrantVectorStore.server(
            ...     host="localhost",
            ...     port=6333,
            ...     prefer_grpc=False,
            ...     collection_name="docs",
            ...     embedder=embedder
            ... )
        
        Note:
            gRPC is significantly faster than HTTP for high-frequency operations.
            If grpc_port is not specified, it defaults to port+1.
            
            Start Qdrant server with:
            docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
            
            Or with docker-compose. See: https://qdrant.tech/documentation/quick-start/
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install it with: pip install qdrant-client"
            )
        
        # Determine gRPC port if not specified
        if grpc_port is None and prefer_grpc:
            grpc_port = port + 1
        
        # Create client with gRPC preference
        if prefer_grpc and grpc_port:
            client = QdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=True
            )
        else:
            client = QdrantClient(host=host, port=port)
        
        return cls(
            client=client,
            collection_name=collection_name,
            embedder=embedder,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            **kwargs
        )
    
    @classmethod
    def cloud(
        cls,
        url: str,
        api_key: str,
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ) -> 'QdrantVectorStore':
        """
        Connect to Qdrant Cloud (for managed service)
        
        Connects to Qdrant Cloud managed service.
        Suitable for production without managing infrastructure.
        
        Args:
            url: Qdrant Cloud cluster URL
            api_key: API key for authentication
            collection_name: Name of the collection
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            **kwargs: Additional parameters
        
        Returns:
            QdrantVectorStore instance
        
        Example:
            >>> embedder = Embedder('bailian/text-embedding-v3')
            >>> store = QdrantVectorStore.cloud(
            ...     url="https://xxx-yyy-zzz.qdrant.io",
            ...     api_key="your_api_key_here",
            ...     collection_name="docs",
            ...     embedder=embedder
            ... )
        
        Note:
            Sign up for Qdrant Cloud: https://cloud.qdrant.io/
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install it with: pip install qdrant-client"
            )
        
        client = QdrantClient(url=url, api_key=api_key)
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
        Add unit(s) to Qdrant vector store
        
        Supports both single unit and batch operations.
        Supports mixed unit types (text, table, image) in a single call.
        
        Args:
            units: Single unit or list of units to store (can be mixed types)
        """
        # Normalize input to list format
        if isinstance(units, BaseUnit):
            units = [units]
        
        if not units:
            return
        
        # Group units by type for separate processing
        text_units = []
        table_units = []
        image_units = []
        
        for unit in units:
            if unit.unit_type == UnitType.TEXT:
                text_units.append(unit)
            elif unit.unit_type == UnitType.TABLE:
                table_units.append(unit)
            elif unit.unit_type == UnitType.IMAGE:
                image_units.append(unit)
        
        # Process each type separately and collect points
        points = []
        
        if text_units:
            points.extend(self._process_text_units(text_units))
        
        if table_units:
            points.extend(self._process_table_units(table_units))
        
        if image_units:
            points.extend(self._process_image_units(image_units))
        
        # Upsert all points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
    
    def _process_text_units(self, text_units: list[TextUnit]) -> list:
        """
        Process TextUnits: embed and build points
        
        Args:
            text_units: List of TextUnits
            
        Returns:
            List of PointStruct objects
        """
        from qdrant_client.models import PointStruct
        
        # Batch embedding
        embedding_contents = [
            unit.embedding_content if unit.embedding_content else unit.content
            for unit in text_units
        ]
        embedder = self._get_embedder_for_unit(text_units[0])
        embeddings = embedder.embed_batch(embedding_contents)
        
        points = []
        for unit, embedding in zip(text_units, embeddings):
            payload = {
                "unit_id": unit.unit_id,
                "content": unit.content,
                "embedding_content": unit.embedding_content or unit.content,
                "unit_type": unit.unit_type.value,
                "prev_unit_id": unit.prev_unit_id,
                "next_unit_id": unit.next_unit_id,
                "doc_id": unit.doc_id,
                "relations": unit.relations
            }
            
            if unit.metadata:
                payload["metadata"] = unit.metadata.model_dump()
            
            points.append(PointStruct(
                id=self._unit_id_to_qdrant_id(unit.unit_id),
                vector=embedding,
                payload=payload
            ))
        
        return points
    
    def _process_table_units(self, table_units: list) -> list:
        """
        Process TableUnits: embed and build points with table-specific fields
        
        Args:
            table_units: List of TableUnits
            
        Returns:
            List of PointStruct objects
        """
        from qdrant_client.models import PointStruct
        
        # Batch embedding
        embedding_contents = [
            unit.embedding_content if unit.embedding_content else unit.content
            for unit in table_units
        ]
        embedder = self._get_embedder_for_unit(table_units[0])
        embeddings = embedder.embed_batch(embedding_contents)
        
        points = []
        for unit, embedding in zip(table_units, embeddings):
            payload = {
                "unit_id": unit.unit_id,
                "content": unit.content,  # Markdown table
                "embedding_content": unit.embedding_content or unit.content,
                "unit_type": unit.unit_type.value,
                "prev_unit_id": unit.prev_unit_id,
                "next_unit_id": unit.next_unit_id,
                "doc_id": unit.doc_id,
                "relations": unit.relations,
                # TableUnit-specific: caption (may be None)
                "caption": unit.caption
            }
            
            # Serialize DataFrame if exists
            if hasattr(unit, 'df') and unit.df is not None:
                try:
                    payload["df_data"] = unit.df.to_dict(orient='records')
                    payload["df_columns"] = list(unit.df.columns)
                except Exception:
                    # If serialization fails, skip (content will be used as fallback)
                    pass
            
            if unit.metadata:
                payload["metadata"] = unit.metadata.model_dump()
            
            points.append(PointStruct(
                id=self._unit_id_to_qdrant_id(unit.unit_id),
                vector=embedding,
                payload=payload
            ))
        
        return points
    
    def _process_image_units(self, image_units: list) -> list:
        """
        Process ImageUnits: embed and build points
        
        Args:
            image_units: List of ImageUnits
            
        Returns:
            List of PointStruct objects
        """
        from qdrant_client.models import PointStruct
        
        # Batch embedding (use content directly for images)
        image_contents = [unit.content for unit in image_units]
        embedder = self._get_embedder_for_unit(image_units[0])
        embeddings = embedder.embed_batch(image_contents)
        
        points = []
        for unit, embedding in zip(image_units, embeddings):
            payload = {
                "unit_id": unit.unit_id,
                "content": unit.content,  # Image bytes
                "unit_type": unit.unit_type.value,
                "prev_unit_id": unit.prev_unit_id,
                "next_unit_id": unit.next_unit_id,
                "doc_id": unit.doc_id,
                "relations": unit.relations
            }
            
            if unit.metadata:
                payload["metadata"] = unit.metadata.model_dump()
            
            points.append(PointStruct(
                id=self._unit_id_to_qdrant_id(unit.unit_id),
                vector=embedding,
                payload=payload
            ))
        
        return points
    
    async def aadd(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of add (uses executor wrapper)
        
        Args:
            units: Single unit or list of units to store
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.add, units)
    
    def search(
        self,
        query: Union[str, BaseUnit],
        top_k: int = 10,
        filter: Optional[Union[dict[str, Any], Any]] = None
    ) -> list[BaseUnit]:
        """
        Search for similar units
        
        Args:
            query: Query content (can be text or Unit)
            top_k: Number of results to return
            filter: Optional metadata filters
                   - dict: MongoDB-style filter (auto-converted)
                   - Filter: Qdrant native Filter object (direct use)
            
        Returns:
            List of matching units with scores, sorted by similarity
        """
        # Get query embedding
        if isinstance(query, str):
            # Text query
            embedder = self.text_embedder
            query_vector = embedder.embed(query)
        elif isinstance(query, BaseUnit):
            # Unit query
            embedder = self._get_embedder_for_unit(query)
            if isinstance(query, TextUnit):
                query_vector = embedder.embed(query.content)
            else:
                query_vector = embedder.embed(str(query.content))
        else:
            raise ValueError(f"Unsupported query type: {type(query)}")
        
        # Convert filter to Qdrant format
        qdrant_filter = None
        if filter:
            if isinstance(filter, dict):
                # MongoDB-style dict: convert to Qdrant Filter
                from ...utils.filter_converter import QdrantFilterConverter
                converter = QdrantFilterConverter()
                qdrant_filter = converter.convert(filter)
            else:
                # Native Qdrant Filter object: use directly
                qdrant_filter = filter
        
        # Search in Qdrant using query_points (new API)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,  # Dense vector for nearest search
            limit=top_k,
            query_filter=qdrant_filter  # Qdrant Filter object
        )
        
        # Convert results to units
        units = []
        for scored_point in response.points:
            unit = self._point_to_unit(scored_point)
            unit.score = scored_point.score
            units.append(unit)
        
        return units
    
    async def asearch(
        self,
        query: Union[str, BaseUnit],
        top_k: int = 10,
        filter: Optional[Union[dict[str, Any], Any]] = None
    ) -> list[BaseUnit]:
        """
        Async version of search (uses executor wrapper)
        
        Args:
            query: Query content
            top_k: Number of results to return
            filter: Optional metadata filters
                   - dict: MongoDB-style filter (auto-converted)
                   - Filter: Qdrant native Filter object (direct use)
            
        Returns:
            List of matching units
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, top_k, filter)
    
    def delete(self, unit_ids: list[str]) -> None:
        """
        Delete units by IDs
        
        Args:
            unit_ids: List of unit IDs to delete
        """
        # Convert string IDs to Qdrant integer IDs
        qdrant_ids = [self._unit_id_to_qdrant_id(uid) for uid in unit_ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_ids
        )
    
    async def adelete(self, unit_ids: list[str]) -> None:
        """
        Async version of delete (uses executor wrapper)
        
        Args:
            unit_ids: List of unit IDs to delete
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.delete, unit_ids)
    
    def get(self, unit_ids: list[str]) -> list[BaseUnit]:
        """
        Get units by IDs
        
        Args:
            unit_ids: List of unit IDs
            
        Returns:
            List of corresponding units
        """
        # Convert string IDs to Qdrant integer IDs
        qdrant_ids = [self._unit_id_to_qdrant_id(uid) for uid in unit_ids]
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=qdrant_ids,
            with_payload=True,
            with_vectors=False
        )
        
        return [self._point_to_unit(point) for point in results]
    
    async def aget(self, unit_ids: list[str]) -> list[BaseUnit]:
        """
        Async version of get (uses executor wrapper)
        
        Args:
            unit_ids: List of unit IDs
            
        Returns:
            List of corresponding units
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, unit_ids)
    
    def update(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Update existing unit(s)
        
        Supports both single unit and batch operations.
        
        Args:
            units: Single unit or list of units to update
        """
        # Update is same as add in Qdrant (upsert)
        self.add(units)
    
    async def aupdate(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of update (uses executor wrapper)
        
        Args:
            units: Single unit or list of units to update
        """
        await self.aadd(units)
    
    def clear(self) -> None:
        """Clear all vectors from store"""
        # Delete and recreate collection
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
    
    async def aclear(self) -> None:
        """
        Async version of clear (uses executor wrapper)
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.clear)
    
    def count(self) -> int:
        """
        Get total number of units in store
        
        Returns:
            Total count of units
        """
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.points_count
    
    @property
    def dimension(self) -> int:
        """
        Get vector dimension
        
        Returns:
            Vector dimension (determined by embedder)
        """
        return self.text_embedder.dimension
    
    def _unit_id_to_qdrant_id(self, unit_id: str) -> int:
        """
        Convert unit ID (string) to Qdrant point ID (integer)
        
        Qdrant requires either UUID or integer as point ID.
        We use hash to convert string IDs to integers.
        
        Args:
            unit_id: String unit ID
            
        Returns:
            Integer point ID for Qdrant
        """
        # Use hash to convert string to integer
        # Using abs() to ensure positive value
        return abs(hash(unit_id)) % (2**63)  # Keep within 64-bit signed int range
    
    def _qdrant_id_to_unit_id(self, qdrant_id: int, stored_unit_id: str) -> str:
        """
        Get original unit ID from Qdrant point
        
        Since we hash unit_id for Qdrant, we store the original
        unit_id in the payload for retrieval.
        
        Args:
            qdrant_id: Qdrant point ID (integer)
            stored_unit_id: Original unit ID from payload
            
        Returns:
            Original unit ID string
        """
        return stored_unit_id
    

    def _point_to_unit(self, point) -> BaseUnit:
        """
        Convert Qdrant point to Unit
        
        Args:
            point: Qdrant point object
            
        Returns:
            Corresponding Unit instance
        """
        payload = point.payload
        unit_type = payload.get("unit_type", "text")
        
        # Get original unit_id from payload
        unit_id = payload.get("unit_id", str(point.id))
        
        # Extract content
        content = payload.get("content", "")
        
        # Reconstruct metadata from nested object
        from ...schemas import UnitMetadata
        metadata_dict = payload.get("metadata", {})
        if metadata_dict:
            metadata = UnitMetadata(**metadata_dict)
        else:
            metadata = UnitMetadata()
        
        # Restore chain relationships
        prev_unit_id = payload.get("prev_unit_id")
        next_unit_id = payload.get("next_unit_id")
        doc_id = payload.get("doc_id")
        
        # Restore semantic relationships
        relations = payload.get("relations", {})
        
        # Create unit based on type
        if unit_type == "text":
            unit = TextUnit(
                unit_id=unit_id,
                content=content,
                metadata=metadata
            )
        elif unit_type == "table":
            # Import TableUnit dynamically
            from ...schemas.unit import TableUnit
            
            # Restore DataFrame if exists
            df = None
            if "df_data" in payload and "df_columns" in payload:
                try:
                    import pandas as pd
                    df = pd.DataFrame(payload["df_data"])
                except Exception:
                    # If restoration fails, df remains None (fallback to content)
                    pass
            
            unit = TableUnit(
                unit_id=unit_id,
                content=content,
                embedding_content=payload.get("embedding_content"),
                caption=payload.get("caption"),
                df=df,
                metadata=metadata
            )
        elif unit_type == "image":
            # Import ImageUnit dynamically
            from ...schemas.unit import ImageUnit
            unit = ImageUnit(
                unit_id=unit_id,
                content=content,
                metadata=metadata
            )
        else:
            # Fallback to TextUnit
            unit = TextUnit(
                unit_id=unit_id,
                content=content,
                metadata=metadata
            )
        
        # Restore relationships
        unit.prev_unit_id = prev_unit_id
        unit.next_unit_id = next_unit_id
        unit.doc_id = doc_id
        unit.relations = relations
        
        return unit
