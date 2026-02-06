"""
Milvus vector store implementation
"""

import asyncio
from typing import Any, Optional, Union

from .base import BaseVectorStore
from ...schemas import BaseUnit, UnitType, UnitMetadata
from ...schemas.unit import TextUnit


class MilvusVectorStore(BaseVectorStore):
    """
    Milvus vector store - high-performance cloud-native vector database
    
    Milvus is a high-performance, cloud-native vector database built for scale.
    Written in Go and C++, it features hardware acceleration and distributed architecture.
    
    Supported deployment modes:
    1. **Local (Milvus Lite)**: Embedded mode for development and small-scale applications
    2. **Server (Standalone)**: Single-node server for production
    3. **Server (Cluster)**: Distributed cluster for large-scale production
    4. **Cloud (Zilliz)**: Fully managed service on Zilliz Cloud
    
    Usage:
        # Local mode (Milvus Lite - embedded)
        store = MilvusVectorStore.local(
            path="./milvus.db",
            collection_name="docs",
            embedder=embedder
        )
        
        # Server mode (Standalone/Cluster)
        store = MilvusVectorStore.server(
            host="localhost",
            port=19530,
            collection_name="docs",
            embedder=embedder
        )
        
        # Cloud mode (Zilliz Cloud)
        store = MilvusVectorStore.cloud(
            url="https://xxx.api.gcp-us-west1.zillizcloud.com:443",
            api_key="your_api_key",
            collection_name="docs",
            embedder=embedder
        )
    
    **Async Support:**
    - Async methods (aadd, asearch, etc.) use thread pool executor wrapper
    - This is **not true async I/O**, just running sync code in a thread
    - For true async support, consider using aiomilvus (if available)
    
    **Key Features:**
    - High-performance vector search at scale
    - Distributed architecture (separates compute and storage)
    - Hardware acceleration (CPU/GPU)
    - Multi-tenancy support
    - Advanced metadata filtering
    - Hybrid search (dense + sparse vectors)
    - Full-text search with BM25
    - RBAC and data security
    
    References:
        - Official Site: https://milvus.io/
        - Python SDK: https://github.com/milvus-io/pymilvus
        - Docs: https://milvus.io/docs
    """
    
    def __init__(
        self,
        client: 'MilvusClient',
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ):
        """
        Private constructor - use factory methods instead
        
        Factory methods:
        - MilvusVectorStore.local() - for Milvus Lite (embedded)
        - MilvusVectorStore.server() - for Standalone/Cluster
        - MilvusVectorStore.cloud() - for Zilliz Cloud
        
        Args:
            client: MilvusClient instance
            collection_name: Name of the Milvus collection
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
        self._dimension_cache = None
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure collection exists, create if not"""
        # Check if collection exists
        if not self.client.has_collection(collection_name=self.collection_name):
            # Create collection with vector field
            # Will create on first add when we know the dimension
            pass
    
    def _create_collection_if_needed(self, dimension: int):
        """Create collection if it doesn't exist"""
        if self.client.has_collection(collection_name=self.collection_name):
            return
        
        from pymilvus import DataType
        
        # Create collection with schema
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,  # Allow dynamic metadata fields
        )
        
        # Add fields
        schema.add_field(field_name="unit_id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="unit_type", datatype=DataType.VARCHAR, max_length=50)
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            metric_type="COSINE",  # Use cosine similarity
        )
        
        # Create index for better search performance
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",  # Auto-select best index type
            metric_type="COSINE"
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
        
        # Load collection into memory (required for search)
        self.client.load_collection(collection_name=self.collection_name)
    
    @classmethod
    def local(
        cls,
        path: str,
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ) -> 'MilvusVectorStore':
        """
        Create local Milvus Lite store (embedded mode)
        
        Milvus Lite is a lightweight version running embedded in your application.
        Perfect for development, testing, and small-scale applications.
        Data persists to local file.
        
        Args:
            path: Local file path for data storage (e.g., "./milvus.db")
            collection_name: Name of the collection
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            **kwargs: Additional parameters
        
        Returns:
            MilvusVectorStore instance
        
        Example:
            >>> embedder = Embedder('ollama/jina-embeddings-v2-base-en')
            >>> store = MilvusVectorStore.local(
            ...     path="./milvus_data.db",
            ...     collection_name="products",
            ...     embedder=embedder
            ... )
        
        Note:
            **IMPORTANT**: Milvus Lite currently only supports Linux and macOS!
            On Windows, use server() mode instead:
            
            **Recommended: WSL2 (much easier than Docker Desktop!)**
            ```bash
            # In WSL2 (Ubuntu)
            curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
            bash standalone_embed.sh start
            ```
            
            Then connect from Windows:
            ```python
            store = MilvusVectorStore.server(host="localhost", port=19530)
            ```
            
            **Alternative: Docker Desktop (if you prefer)**
            ```bash
            docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
            ```
            
            Suitable for < 1M vectors. For larger scale, use server() or cloud()
        """
        import platform
        import sys
        
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError(
                "pymilvus is required for MilvusVectorStore. "
                "Install it with: pip install pymilvus"
            )
        
        # Check platform compatibility
        if platform.system() == "Windows":
            raise RuntimeError(
                "\n"
                "❌ Milvus Lite does not support Windows!\n"
                "\n"
                "Please use one of these alternatives:\n"
                "\n"
                "💡 Recommended: WSL2 (easiest way!)\n"
                "   # In WSL2 (Ubuntu):\n"
                "   curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh\n"
                "   bash standalone_embed.sh start\n"
                "   \n"
                "   # Then in Windows Python:\n"
                "   store = MilvusVectorStore.server(host='localhost', port=19530)\n"
                "\n"
                "2️⃣  Docker Desktop (if you prefer):\n"
                "   docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest\n"
                "   store = MilvusVectorStore.server(host='localhost', port=19530)\n"
                "\n"
                "3️⃣  Zilliz Cloud (free tier available):\n"
                "   https://zilliz.com/\n"
                "\n"
                "See: https://milvus.io/docs/install_standalone-windows.md"
            )
        
        # Create MilvusClient with local file
        try:
            client = MilvusClient(uri=path)
        except Exception as e:
            if "milvus_lite" in str(e).lower() or "milvus-lite" in str(e).lower():
                raise RuntimeError(
                    f"Failed to start Milvus Lite: {e}\n\n"
                    "Make sure milvus-lite is installed:\n"
                    "  pip install pymilvus[milvus_lite]\n\n"
                    "Note: Milvus Lite only supports Linux and macOS."
                ) from e
            raise
        
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
        port: int = 19530,
        collection_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs
    ) -> 'MilvusVectorStore':
        """
        Create Milvus server store (Standalone or Cluster)
        
        Connects to Milvus Standalone (single-node) or Distributed (cluster).
        Suitable for production deployments with high performance requirements.
        
        Args:
            host: Milvus server host (default: "localhost")
            port: Milvus server port (default: 19530)
            collection_name: Name of the collection
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            username: Username for authentication (optional)
            password: Password for authentication (optional)
            **kwargs: Additional parameters
        
        Returns:
            MilvusVectorStore instance
        
        Example:
            >>> # Without authentication
            >>> store = MilvusVectorStore.server(
            ...     host="localhost",
            ...     port=19530,
            ...     collection_name="docs",
            ...     embedder=embedder
            ... )
            >>> 
            >>> # With authentication
            >>> store = MilvusVectorStore.server(
            ...     host="localhost",
            ...     port=19530,
            ...     collection_name="docs",
            ...     embedder=embedder,
            ...     username="root",
            ...     password="Milvus"
            ... )
        
        Note:
            - Port 19530 is the default gRPC port
            - Supports both Standalone and Distributed deployments
        """
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError(
                "pymilvus is required for MilvusVectorStore. "
                "Install it with: pip install pymilvus"
            )
        
        # Build URI
        uri = f"http://{host}:{port}"
        
        # Build token if username/password provided
        token = None
        if username and password:
            token = f"{username}:{password}"
        
        # Create MilvusClient
        client = MilvusClient(uri=uri, token=token)
        
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
    ) -> 'MilvusVectorStore':
        """
        Create Zilliz Cloud store (managed service)
        
        Connects to Zilliz Cloud, the fully managed Milvus service.
        Supports Serverless, Dedicated, and BYOC deployment options.
        
        Args:
            url: Zilliz Cloud endpoint URL (e.g., "https://xxx.api.gcp-us-west1.zillizcloud.com:443")
            api_key: Zilliz Cloud API key
            collection_name: Name of the collection
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            **kwargs: Additional parameters
        
        Returns:
            MilvusVectorStore instance
        
        Example:
            >>> store = MilvusVectorStore.cloud(
            ...     url="https://xxx.api.gcp-us-west1.zillizcloud.com:443",
            ...     api_key="your_zilliz_api_key",
            ...     collection_name="docs",
            ...     embedder=embedder
            ... )
        
        Note:
            Get your Zilliz Cloud credentials at https://zilliz.com/
        """
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError(
                "pymilvus is required for MilvusVectorStore. "
                "Install it with: pip install pymilvus"
            )
        
        # Create MilvusClient for Zilliz Cloud
        client = MilvusClient(uri=url, token=api_key)
        
        return cls(
            client=client,
            collection_name=collection_name,
            embedder=embedder,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            **kwargs
        )
    
    def _extract_metadata(self, unit: BaseUnit) -> dict:
        """
        Extract metadata from unit for storage
        
        Preserves original data types for proper filtering:
        - Strings remain strings
        - Numbers (int/float) remain numbers
        - Booleans remain booleans
        - Other types are converted to strings
        """
        metadata = {}
        
        if unit.metadata:
            if unit.metadata.context_path:
                metadata['context_path'] = unit.metadata.context_path
            
            # Store custom fields with type preservation (flatten with custom_ prefix)
            if unit.metadata.custom:
                for key, value in unit.metadata.custom.items():
                    # Preserve native types for filtering
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[f'custom_{key}'] = value
                    else:
                        # Convert complex types to string
                        metadata[f'custom_{key}'] = str(value)
        
        return metadata
    
    def add(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Add unit(s) to Milvus collection
        
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
        
        # Group units by type for efficient batch embedding
        text_like_units = []  # TextUnit and TableUnit
        image_units = []      # ImageUnit
        
        for unit in units:
            if unit.unit_type == UnitType.IMAGE:
                image_units.append(unit)
            else:
                text_like_units.append(unit)
        
        # Prepare records for insertion
        records = []
        
        # Process text-like units
        if text_like_units:
            contents = []
            for unit in text_like_units:
                if isinstance(unit, TextUnit):
                    contents.append(unit.content)
                else:
                    contents.append(str(unit.content))
            
            embedder = self._get_embedder_for_unit(text_like_units[0])
            embeddings = embedder.embed_batch(contents)
            
            # Create collection if needed (now we know dimension)
            if not self.client.has_collection(collection_name=self.collection_name):
                self._create_collection_if_needed(len(embeddings[0]))
            
            for unit, content, embedding in zip(text_like_units, contents, embeddings):
                record = {
                    'unit_id': unit.unit_id,
                    'vector': embedding,
                    'content': content,
                    'unit_type': unit.unit_type.value,
                    **self._extract_metadata(unit)
                }
                
                # Store views
                if unit.views:
                    record["views"] = [view.model_dump() for view in unit.views]
                
                records.append(record)
        
        # Process image units
        if image_units:
            image_contents = [unit.content for unit in image_units]
            embedder = self._get_embedder_for_unit(image_units[0])
            embeddings = embedder.embed_batch(image_contents)
            
            # Create collection if needed
            if not self.client.has_collection(collection_name=self.collection_name):
                self._create_collection_if_needed(len(embeddings[0]))
            
            for unit, content, embedding in zip(image_units, image_contents, embeddings):
                record = {
                    'unit_id': unit.unit_id,
                    'vector': embedding,
                    'content': f"[Image: {unit.unit_id}]",
                    'unit_type': unit.unit_type.value,
                    **self._extract_metadata(unit)
                }
                
                # Store views
                if unit.views:
                    record["views"] = [view.model_dump() for view in unit.views]
                
                records.append(record)
        
        # Insert records
        if records:
            self.client.insert(
                collection_name=self.collection_name,
                data=records
            )
            
            # Load collection into memory for search (Milvus requires this)
            self.client.load_collection(collection_name=self.collection_name)
    
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
        filter: Optional[Union[str, dict[str, Any]]] = None
    ) -> list[BaseUnit]:
        """
        Search for similar units
        
        Args:
            query: Query content (can be text or Unit)
            top_k: Number of results to return
            filter: Optional metadata filters
                   - str: Milvus boolean expression (recommended for full capabilities)
                   - dict: Simple key-value filters (only supports equality with AND logic)
            
        Returns:
            List of matching units with scores, sorted by similarity
            
        Filter Expression Guide:
            Milvus supports rich boolean expressions. Use string filters for full power:
            
            Basic operators:
                - Comparison: ==, !=, >, <, >=, <=
                - Logical: and, or, not
                - Membership: in [...]
            
            Advanced features:
                - JSON fields: product["model"] == "ABC"
                - Arrays: history_temperatures[10] > 23
                - Text match: TEXT_MATCH(description, "keyword")
                - Array ops: ARRAY_CONTAINS(tags, "sale")
            
            Reference: https://milvus.io/docs/zh/boolean.md
            
        Examples:
            >>> # String expression - Simple equality
            >>> results = store.search(
            ...     "running shoes",
            ...     filter="custom_brand == 'Nike'"
            ... )
            >>> 
            >>> # String expression - Range query
            >>> results = store.search(
            ...     "running shoes",
            ...     filter="custom_price >= 1000 and custom_price <= 2000"
            ... )
            >>> 
            >>> # String expression - IN operator (recommended for multiple values)
            >>> results = store.search(
            ...     "running shoes",
            ...     filter="custom_brand in ['Nike', 'Adidas', 'Puma']"
            ... )
            >>> 
            >>> # String expression - Complex conditions
            >>> results = store.search(
            ...     "running shoes",
            ...     filter="custom_category == 'Apparel' and custom_brand in ['Nike', 'Adidas'] and custom_price < 2000"
            ... )
            >>> 
            >>> # Dict filter - Only for simple equality (NOT recommended for complex cases)
            >>> results = store.search(
            ...     "running shoes",
            ...     filter={"custom_brand": "Nike", "custom_season": "Summer"}  # equivalent to: custom_brand == 'Nike' and custom_season == 'Summer'
            ... )
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
        
        # Build filter expression
        # Milvus supports rich boolean expressions, we pass them through directly
        # Reference: https://milvus.io/docs/zh/boolean.md
        filter_expr = None
        if filter:
            if isinstance(filter, str):
                # Direct Milvus expression (recommended)
                # Examples:
                #   - Simple: "custom_brand == 'Nike'"
                #   - Range: "custom_price >= 1000 and custom_price <= 2000"
                #   - IN operator: "custom_brand in ['Nike', 'Adidas', 'Puma']"
                #   - Complex: "custom_category == 'Apparel' and custom_brand in ['Nike', 'Adidas'] and custom_price < 2000"
                filter_expr = filter
            elif isinstance(filter, dict):
                # For simple dict filters, convert to basic equality expressions
                # Note: This only supports simple equality checks with AND logic
                # For advanced queries (IN, OR, ranges), use string expressions instead
                expressions = []
                for key, value in filter.items():
                    if isinstance(value, str):
                        expressions.append(f'{key} == "{value}"')
                    elif isinstance(value, bool):
                        expressions.append(f'{key} == {str(value).lower()}')
                    elif isinstance(value, (int, float)):
                        expressions.append(f'{key} == {value}')
                    elif value is None:
                        expressions.append(f'{key} == null')
                
                if expressions:
                    filter_expr = " and ".join(expressions)
        
        # Perform search
        search_params = {
            "collection_name": self.collection_name,
            "data": [query_vector],
            "limit": top_k,
            "output_fields": ["unit_id", "content", "unit_type", "context_path"]
        }
        
        if filter_expr:
            search_params["filter"] = filter_expr
        
        results = self.client.search(**search_params)
        
        # Convert results to units
        units = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit['entity']
                unit_type_str = entity.get('unit_type', 'text')
                unit_type = UnitType(unit_type_str)
                
                if unit_type == UnitType.TEXT:
                    metadata = UnitMetadata(
                        context_path=entity.get('context_path')
                    )
                    
                    unit = TextUnit(
                        unit_id=entity['unit_id'],
                        content=entity['content'],
                        metadata=metadata
                    )
                    
                    # Restore views
                    if 'views' in entity and entity['views']:
                        from ...schemas import ContentView
                        try:
                            unit.views = [ContentView(**view_data) for view_data in entity['views']]
                        except Exception:
                            # If restoration fails, views remain None
                            pass
                    
                    # Add score
                    unit.score = float(hit.get('distance', 0.0))
                    
                    units.append(unit)
        
        return units
    
    async def asearch(
        self,
        query: Union[str, BaseUnit],
        top_k: int = 10,
        filter: Optional[Union[str, dict[str, Any]]] = None
    ) -> list[BaseUnit]:
        """
        Async version of search (uses executor wrapper)
        
        Args:
            query: Query content (can be text or Unit)
            top_k: Number of results to return
            filter: Optional metadata filters
                   - str: Milvus expression (recommended)
                   - dict: Simple key-value filters
            
        Returns:
            List of matching units with scores
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, top_k, filter)
    
    def delete(self, unit_ids: list[str]) -> None:
        """
        Delete units by IDs
        
        Args:
            unit_ids: List of unit IDs to delete
        """
        if not unit_ids:
            return
        
        # Build filter expression for deletion
        ids_str = ", ".join([f'"{id}"' for id in unit_ids])
        filter_expr = f"unit_id in [{ids_str}]"
        
        self.client.delete(
            collection_name=self.collection_name,
            filter=filter_expr
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
        if not unit_ids:
            return []
        
        # Query by IDs
        ids_str = ", ".join([f'"{id}"' for id in unit_ids])
        filter_expr = f"unit_id in [{ids_str}]"
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["unit_id", "content", "unit_type", "context_path"]
        )
        
        # Convert results to units
        units = []
        for entity in results:
            unit_type_str = entity.get('unit_type', 'text')
            unit_type = UnitType(unit_type_str)
            
            if unit_type == UnitType.TEXT:
                metadata = UnitMetadata(
                    context_path=entity.get('context_path')
                )
                
                unit = TextUnit(
                    unit_id=entity['unit_id'],
                    content=entity['content'],
                    metadata=metadata
                )
                
                # Restore views
                if 'views' in entity and entity['views']:
                    from ...schemas import ContentView
                    try:
                        unit.views = [ContentView(**view_data) for view_data in entity['views']]
                    except Exception:
                        # If restoration fails, views remain None
                        pass
                
                units.append(unit)
        
        return units
    
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
        
        Milvus doesn't support direct update. We delete and re-insert.
        
        Args:
            units: Single unit or list of units to update
        """
        # Normalize to list
        if isinstance(units, BaseUnit):
            units = [units]
        
        if not units:
            return
        
        # Delete old records
        unit_ids = [unit.unit_id for unit in units]
        self.delete(unit_ids)
        
        # Re-insert with updated data
        self.add(units)
    
    async def aupdate(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of update (uses executor wrapper)
        
        Args:
            units: Single unit or list of units to update
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.update, units)
    
    def clear(self) -> None:
        """
        Clear all vectors from collection
        """
        # Drop and recreate collection
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
    
    async def aclear(self) -> None:
        """
        Async version of clear (uses executor wrapper)
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.clear)
    
    def count(self) -> int:
        """
        Get total number of vectors in collection
        
        Returns:
            Number of vectors stored
        """
        if not self.client.has_collection(collection_name=self.collection_name):
            return 0
        
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return int(stats.get('row_count', 0))
    
    @property
    def dimension(self) -> int:
        """
        Get vector dimension
        
        Returns:
            Vector dimension (determined by embedder)
        """
        if self._dimension_cache is None:
            self._dimension_cache = self.text_embedder.dimension
        return self._dimension_cache
