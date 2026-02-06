"""
LanceDB vector store implementation
"""

import asyncio
from typing import Any, Optional, Union
from pathlib import Path

from .base import BaseVectorStore
from ...schemas import BaseUnit, UnitType
from ...schemas.unit import TextUnit


class LanceDBVectorStore(BaseVectorStore):
    """
    LanceDB vector store - embedded vector database
    
    LanceDB is a lightweight, embedded vector database built on Lance format.
    It's designed to be fast, serverless, and easy to use (similar to SQLite for vectors).
    
    Supported deployment modes:
    1. **Local persistent**: For development and production (primary use case)
    2. **Cloud**: For LanceDB Cloud managed service
    
    Not supported:
    - **In-memory**: LanceDB is always persistent
    - **Separate server**: Use local mode (embedded) or cloud mode
    
    Usage:
        # Local mode (most common)
        store = LanceDBVectorStore.local(
            path="./lancedb_data",
            collection_name="docs",
            embedder=embedder
        )
        
        # Cloud mode (managed service)
        store = LanceDBVectorStore.cloud(
            url="db://my_database",
            api_key="ldb_...",
            collection_name="docs",
            embedder=embedder
        )
    
    **Async Support:**
    - Async methods (aadd, asearch, etc.) use thread pool executor wrapper
    - This is **not true async I/O**, just running sync code in a thread
    - LanceDB is fundamentally synchronous (embedded database)
    
    **Key Features:**
    - Zero-copy reads with Apache Arrow
    - Fast vector search with IVF-PQ index
    - Columnar storage format (Lance)
    - Embedded (no server needed)
    - Multi-modal support
    
    References:
        - Quickstart: https://docs.lancedb.com/quickstart
        - Python SDK: https://lancedb.github.io/lancedb/python/python/
    """
    
    def __init__(
        self,
        connection: 'lancedb.DBConnection',
        table_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ):
        """
        Private constructor - use factory methods instead
        
        Factory methods:
        - LanceDBVectorStore.local() - for local persistent storage
        - LanceDBVectorStore.cloud() - for LanceDB Cloud
        
        Args:
            connection: LanceDB connection instance
            table_name: Name of the LanceDB table
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
        
        self.connection = connection
        self.table_name = table_name
        self._table = None
        self._dimension_cache = None
        
        # Initialize table if it doesn't exist
        self._ensure_table()
    
    def _ensure_table(self):
        """Ensure table exists, create if not"""
        try:
            self._table = self.connection.open_table(self.table_name)
        except Exception:
            # Table doesn't exist, will create on first add
            self._table = None
    
    @classmethod
    def local(
        cls,
        path: str,
        table_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ) -> 'LanceDBVectorStore':
        """
        Create local persistent LanceDB store (recommended)
        
        Data persists to local directory in Lance columnar format.
        Suitable for both development and production (embedded database).
        
        Args:
            path: Local directory to store data
            table_name: Name of the table
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            **kwargs: Additional parameters
        
        Returns:
            LanceDBVectorStore instance
        
        Example:
            >>> embedder = Embedder('ollama/jina-embeddings-v2-base-en')
            >>> store = LanceDBVectorStore.local(
            ...     path="./lancedb_data",
            ...     table_name="products",
            ...     embedder=embedder
            ... )
        
        Note:
            LanceDB is always persistent (no in-memory mode).
            This is the recommended deployment mode for most use cases.
        """
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "lancedb is required for LanceDBVectorStore. "
                "Install it with: pip install lancedb"
            )
        
        # Connect to local database
        connection = lancedb.connect(path)
        
        return cls(
            connection=connection,
            table_name=table_name,
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
        table_name: str = "default",
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ) -> 'LanceDBVectorStore':
        """
        Create LanceDB Cloud store (managed service)
        
        Connects to LanceDB Cloud managed service.
        Requires LanceDB Cloud account and API key.
        
        Args:
            url: LanceDB Cloud database URL (format: "db://database_name")
            api_key: LanceDB Cloud API key (format: "ldb_...")
            table_name: Name of the table
            embedder: Single embedder for all content types
            text_embedder: Embedder for text/table units
            image_embedder: Embedder for image units
            **kwargs: Additional parameters
        
        Returns:
            LanceDBVectorStore instance
        
        Example:
            >>> embedder = Embedder('ollama/jina-embeddings-v2-base-en')
            >>> store = LanceDBVectorStore.cloud(
            ...     url="db://my_database",
            ...     api_key="ldb_...",
            ...     table_name="products",
            ...     embedder=embedder
            ... )
        
        Note:
            Sign up for LanceDB Cloud: https://cloud.lancedb.com/
        """
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "lancedb is required for LanceDBVectorStore. "
                "Install it with: pip install lancedb"
            )
        
        # Connect to LanceDB Cloud
        connection = lancedb.connect(url, api_key=api_key)
        
        return cls(
            connection=connection,
            table_name=table_name,
            embedder=embedder,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            **kwargs
        )
    
    def _extract_metadata(self, unit: BaseUnit) -> dict:
        """Extract metadata from unit for storage
        
        Preserves original data types for proper SQL filtering:
        - Strings remain strings
        - Numbers (int/float) remain numbers
        - Booleans remain booleans
        - Other types are converted to strings
        """
        metadata = {}
        
        if unit.metadata:
            if unit.metadata.context_path:
                metadata['context_path'] = unit.metadata.context_path
            
            # Add page_numbers if available
            if unit.metadata.page_numbers:
                # Store as comma-separated string (SQL doesn't support lists well)
                metadata['page_numbers'] = ",".join(map(str, unit.metadata.page_numbers))
            
            # Store custom fields directly (preserve original structure)
            if unit.metadata.custom:
                for key, value in unit.metadata.custom.items():
                    # Preserve native types for SQL filtering
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[key] = value
                    else:
                        # Convert complex types to string
                        metadata[key] = str(value)
        
        return metadata
    
    def add(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Add unit(s) to LanceDB table
        
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
        
        # Create table if it doesn't exist, otherwise add records
        if self._table is None:
            self._table = self.connection.create_table(self.table_name, data=records)
        else:
            self._table.add(records)
    
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
                   - str: SQL WHERE clause (e.g., "price > 1000 AND brand = 'Nike'")
                   - dict: MongoDB-style filter (auto-converted to SQL WHERE)
            
        Returns:
            List of matching units with scores, sorted by similarity
            
        Examples:
            >>> # MongoDB-style filter (recommended)
            >>> results = store.search(
            ...     "running shoes",
            ...     filter={"metadata.custom.brand": {"$in": ["Nike", "Adidas"]}, 
            ...             "metadata.custom.price": {"$lt": 2000}}
            ... )
            >>> 
            >>> # SQL WHERE clause (advanced)
            >>> results = store.search(
            ...     "running shoes",
            ...     filter="(brand IN ('Nike', 'Adidas')) AND (price < 2000)"
            ... )
        """
        if self._table is None:
            return []
        
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
        
        # Build search query
        search_query = self._table.search(query_vector).limit(top_k)
        
        # Apply filter if provided
        if filter:
            if isinstance(filter, str):
                # Direct SQL WHERE clause - most powerful!
                search_query = search_query.where(filter)
            elif isinstance(filter, dict):
                # Convert MongoDB-style filter to SQL WHERE clause
                from ...utils.filter_converter import LanceDBFilterConverter
                converter = LanceDBFilterConverter()
                where_clause = converter.convert(filter)
                if where_clause:
                    search_query = search_query.where(where_clause)
        
        # Execute search
        results = search_query.to_pandas()
        
        # Convert results to units
        units = []
        for _, row in results.iterrows():
            unit_type_str = row.get('unit_type', 'text')
            unit_type = UnitType(unit_type_str)
            
            if unit_type == UnitType.TEXT:
                from ...schemas import UnitMetadata
                
                metadata = UnitMetadata(
                    context_path=row.get('context_path')
                )
                
                unit = TextUnit(
                    unit_id=row['unit_id'],
                    content=row['content'],
                    metadata=metadata
                )
                
                # Restore views
                if 'views' in row and row['views']:
                    from ...schemas import ContentView
                    try:
                        unit.views = [ContentView(**view_data) for view_data in row['views']]
                    except Exception:
                        # If restoration fails, views remain None
                        pass
                
                # Add score if available
                if '_distance' in row:
                    unit.score = float(1.0 / (1.0 + row['_distance']))  # Convert distance to similarity
                
                units.append(unit)
        
        return units
    
    async def asearch(
        self,
        query: Union[str, BaseUnit],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None
    ) -> list[BaseUnit]:
        """
        Async version of search (uses executor wrapper)
        
        Args:
            query: Query content (can be text or Unit)
            top_k: Number of results to return
            filter: Optional metadata filters (MongoDB-style dict)
            
        Returns:
            List of matching units, sorted by similarity
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, top_k, filter)
    
    def delete(self, unit_ids: list[str]) -> None:
        """
        Delete units by IDs
        
        Args:
            unit_ids: List of unit IDs to delete
        """
        if self._table is None:
            return
        
        # Build delete condition
        id_list = "', '".join(unit_ids)
        self._table.delete(f"unit_id IN ('{id_list}')")
    
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
        if self._table is None:
            return []
        
        # Query by unit IDs
        id_list = "', '".join(unit_ids)
        results = self._table.search().where(f"unit_id IN ('{id_list}')").to_pandas()
        
        # Convert results to units
        units = []
        for _, row in results.iterrows():
            unit_type_str = row.get('unit_type', 'text')
            unit_type = UnitType(unit_type_str)
            
            if unit_type == UnitType.TEXT:
                from ...schemas import UnitMetadata
                
                metadata = UnitMetadata(
                    context_path=row.get('context_path')
                )
                
                unit = TextUnit(
                    unit_id=row['unit_id'],
                    content=row['content'],
                    metadata=metadata
                )
                
                # Restore views
                if 'views' in row and row['views']:
                    from ...schemas import ContentView
                    try:
                        unit.views = [ContentView(**view_data) for view_data in row['views']]
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
        
        Supports both single unit and batch operations.
        LanceDB uses upsert semantics (insert or replace).
        
        Args:
            units: Single unit or list of units to update
        """
        # In LanceDB, we delete old records and add new ones
        if isinstance(units, BaseUnit):
            units = [units]
        
        if not units:
            return
        
        # Delete existing records
        unit_ids = [unit.unit_id for unit in units]
        self.delete(unit_ids)
        
        # Add new records
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
        Clear all vectors from store (drop and recreate table)
        """
        if self._table is not None:
            self.connection.drop_table(self.table_name)
            self._table = None
    
    async def aclear(self) -> None:
        """
        Async version of clear (uses executor wrapper)
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.clear)
    
    def count(self) -> int:
        """
        Get total number of vectors in store
        
        Returns:
            Number of vectors stored
        """
        if self._table is None:
            return 0
        
        return self._table.count_rows()
    
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
