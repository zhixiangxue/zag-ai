"""
Qdrant vector store implementation
"""

import asyncio
from typing import Any, Optional, Union, Dict

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
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self.client.delete_collection(collection_name=self.collection_name)
    
    async def adelete_collection(self) -> None:
        """Async version of delete_collection"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.delete_collection)
    
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
        timeout: int = 300,
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
            timeout: Request timeout in seconds (default: 300, i.e. 5 minutes)
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
            ...     timeout=300,
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
        
        # Create client with gRPC preference and timeout
        if prefer_grpc and grpc_port:
            client = QdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=True,
                timeout=timeout
            )
        else:
            client = QdrantClient(
                host=host,
                port=port,
                timeout=timeout
            )
        
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
            List of PointStruct objects (may be fewer than input if embedding fails)
        """
        from qdrant_client.models import PointStruct
        from ...utils import logger
        
        # Batch embedding with error handling
        embedding_contents = [
            unit.embedding_content if unit.embedding_content else unit.content
            for unit in text_units
        ]
        embedder = self._get_embedder_for_unit(text_units[0])
        
        try:
            embeddings = embedder.embed_batch(embedding_contents)
        except Exception as e:
            logger.error(f"Failed to embed {len(text_units)} text units: {e}")
            logger.warning(f"Skipping all {len(text_units)} text units due to embedding failure")
            return []  # Skip all units in this batch
        
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
                payload["metadata"] = unit.metadata.to_json_safe()
            
            # Store views (serialize ContentView list)
            if unit.views:
                payload["views"] = [view.model_dump() for view in unit.views]
            
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
            List of PointStruct objects (may be fewer than input if embedding fails)
        """
        from qdrant_client.models import PointStruct
        from ...utils import logger
        
        # Batch embedding with error handling
        embedding_contents = [
            unit.embedding_content if unit.embedding_content else unit.content
            for unit in table_units
        ]
        embedder = self._get_embedder_for_unit(table_units[0])
        
        try:
            embeddings = embedder.embed_batch(embedding_contents)
        except Exception as e:
            logger.error(f"Failed to embed {len(table_units)} table units: {e}")
            logger.warning(f"Skipping all {len(table_units)} table units due to embedding failure")
            return []  # Skip all units in this batch
        
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
            
            # Serialize DataFrame if exists (use json_data property for safety)
            if hasattr(unit, 'json_data') and unit.json_data is not None:
                payload["df_data"] = unit.json_data  # Already safe: tolist() done in property
            
            if unit.metadata:
                payload["metadata"] = unit.metadata.to_json_safe()
            
            # Store views
            if unit.views:
                payload["views"] = [view.model_dump() for view in unit.views]
            
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
                payload["metadata"] = unit.metadata.to_json_safe()
            
            # Store views
            if unit.views:
                payload["views"] = [view.model_dump() for view in unit.views]
            
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
    
    def remove(self, filters: Dict[str, Any]) -> None:
        """
        Remove units by metadata filters (same structure as query filters)

        Args:
            filters: Filter conditions using dot notation, e.g.
                    {"doc_id": "xxx"}
                    {"doc_id": "xxx", "metadata.custom.mode": "lod"}
                    Complex filters: {"$and": [...], "$or": [...]}

        Raises:
            ValueError: If filters is empty (safety check)
        """
        if not filters:
            raise ValueError("filters cannot be empty")

        # Reuse QdrantFilterConverter (same as query)
        from ...utils.filter_converter import QdrantFilterConverter
        converter = QdrantFilterConverter()
        qdrant_filter = converter.convert(filters)

        # Delete all points matching the filter
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_filter
        )

    async def aremove(self, filters: Dict[str, Any]) -> None:
        """
        Async version of remove (uses executor wrapper)

        Args:
            filters: Filter conditions using dot notation
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.remove, filters)
    
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
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """
        Set a value in a nested dict using dot notation path.
        
        Example:
            data = {}
            _set_nested_value(data, "metadata.custom.lender", "Plaza")
            # data is now {"metadata": {"custom": {"lender": "Plaza"}}}
        """
        keys = path.split(".")
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def patch(
        self,
        unit_ids: Union[str, list[str]],
        fields: Dict[str, Any],
    ) -> None:
        """
        Partially update specific fields for given unit IDs without touching vectors.
        
        Uses Qdrant's native set_payload with 'key' parameter for nested field updates.
        This is the safest approach: no need to fetch/merge/overwrite, Qdrant handles
        the merge internally at the specified nested path.
        
        Args:
            unit_ids: Single unit ID or list of unit IDs to update
            fields: Dict of fields to update. Supports dot notation for nested fields.
                   Example: {"metadata.custom.lender": "Plaza", "status": "processed"}
        
        Example:
            # Update nested field metadata.custom.lender
            store.patch(
                unit_ids=["unit-1", "unit-2"],
                fields={"metadata.custom.lender": "Plaza"}
            )
            # Qdrant will set payload.metadata.custom.lender = "Plaza"
            # without touching any other fields
        """
        # Normalize to list
        if isinstance(unit_ids, str):
            unit_ids = [unit_ids]
        
        if not unit_ids or not fields:
            return
        
        # Convert string IDs to Qdrant integer IDs
        qdrant_ids = [self._unit_id_to_qdrant_id(uid) for uid in unit_ids]
        
        # Use set_payload with 'key' parameter for each field
        for path, value in fields.items():
            parts = path.rsplit(".", 1)
            
            if len(parts) == 2:
                # Nested path: e.g., "metadata.custom.lender" -> key="metadata.custom", payload={"lender": value}
                parent_key, field_name = parts
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={field_name: value},
                    key=parent_key,
                    points=qdrant_ids,
                )
            else:
                # Top-level field: e.g., "status" -> key=None, payload={"status": value}
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={path: value},
                    points=qdrant_ids,
                )
    
    async def apatch(
        self,
        unit_ids: Union[str, list[str]],
        fields: Dict[str, Any],
    ) -> None:
        """
        Async version of patch (uses executor wrapper)
        
        Args:
            unit_ids: Single unit ID or list of unit IDs to update
            fields: Dict of fields to update
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.patch, unit_ids, fields)
    
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

    def fetch(self, filters: Optional[Dict[str, Any]] = None) -> list[BaseUnit]:
        """
        Fetch all units matching the filters (no vector search)
        
        Uses Qdrant scroll API to retrieve units purely by metadata filters
        without performing any vector similarity search.
        
        Args:
            filters: Optional metadata filters using dot notation, e.g.
                    {"doc_id": "xxx"}
                    {"doc_id": "xxx", "metadata.custom.mode": "lod"}
                    If None, returns all units (use with caution on large collections)
        
        Returns:
            List of matching units (unsorted)
        """
        # Convert filters to Qdrant format if provided
        qdrant_filter = None
        if filters:
            from ...utils.filter_converter import QdrantFilterConverter
            converter = QdrantFilterConverter()
            qdrant_filter = converter.convert(filters)
        
        # Use scroll to fetch all matching points
        all_units = []
        offset = None
        batch_size = 1000
        
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                offset=offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert points to units
            for point in results:
                unit = self._point_to_unit(point)
                all_units.append(unit)
            
            # Check if there are more results
            if not next_offset:
                break
            offset = next_offset
        
        return all_units

    async def afetch(self, filters: Optional[Dict[str, Any]] = None) -> list[BaseUnit]:
        """
        Async version of fetch (uses executor wrapper)
        
        Args:
            filters: Optional metadata filters using dot notation
            
        Returns:
            List of matching units (unsorted)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fetch, filters)
    
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
        We use stable hash (hashlib.sha256) to convert string IDs to integers.
        
        Args:
            unit_id: String unit ID
            
        Returns:
            Integer point ID for Qdrant
        """
        # Use hashlib for stable hash across processes
        import hashlib
        hash_bytes = hashlib.sha256(unit_id.encode('utf-8')).digest()
        # Take first 8 bytes and convert to int
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big', signed=False)
        # Keep within 63-bit range (signed int64)
        return hash_int % (2**63)
    
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
        
        # Restore views
        from ...schemas import ContentView, LODLevel
        views = None
        if "views" in payload and payload["views"]:
            views = [ContentView(**view_data) for view_data in payload["views"]]
        
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
        
        # Restore views
        unit.views = views
        
        return unit
