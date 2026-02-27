"""
Fulltext indexer implementation

FullTextIndexer uses Meilisearch for full-text search capabilities.
"""

from typing import Union, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

import meilisearch
from meilisearch.errors import MeilisearchApiError

from ..indexers.base import BaseIndexer
from ..schemas import BaseUnit, TextUnit, TableUnit
from ..utils.progress import with_spinner_progress


class FullTextIndexer(BaseIndexer):
    """
    Full-text search indexer powered by Meilisearch
    
    This indexer provides full-text search capabilities by directly
    integrating with Meilisearch service via official Python SDK.
    
    Features:
        - Full-text search with typo tolerance
        - Filterable and sortable attributes
        - Customizable searchable attributes
        - Unified sync/async interfaces
        - Support for single/batch operations
    
    Design:
        - Direct integration with Meilisearch (no intermediate store layer)
        - Uses official meilisearch Python SDK
        - Converts BaseUnit to Meilisearch documents
        - All unit.metadata.custom fields are flattened into document
    
    ⚠️  IMPORTANT - Field Configuration:
        All metadata.custom fields are automatically flattened into the Meilisearch document.
        Use configure_settings() to control which fields are:
        - searchable (full-text search)
        - filterable (filter-only, not searchable)
        - sortable (can be used in sorting)
        
        If you don't call configure_settings(), ALL fields are searchable by default!
    
    Usage:
        >>> from zag.indexers import FullTextIndexer
        >>> 
        >>> # Create indexer
        >>> indexer = FullTextIndexer(
        ...     url="http://127.0.0.1:7700",
        ...     index_name="documents",
        ...     primary_key="unit_id"
        ... )
        >>> 
        >>> # Configure search settings (IMPORTANT!)
        >>> indexer.configure_settings(
        ...     searchable_attributes=["content", "title"],  # Only these for full-text search
        ...     filterable_attributes=["source", "type", "price"]  # Can filter but not search
        ... )
        >>> 
        >>> # Add units
        >>> indexer.add(units)
        >>> 
        >>> # Upsert units
        >>> indexer.upsert(updated_units)
    """
    
    def __init__(
        self,
        url: str = "http://127.0.0.1:7700",
        index_name: str = "documents",
        primary_key: str = "unit_id",
        api_key: Optional[str] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        auto_create_index: bool = True
    ):
        """
        Initialize full-text indexer
        
        Args:
            url: Meilisearch server URL
            index_name: Name of the Meilisearch index
            primary_key: Primary key field name (default: "unit_id")
            api_key: Optional API key for authentication
            executor: Optional thread pool for async operations
                     If None, creates a default executor with 4 workers
            auto_create_index: Whether to automatically create index if not exists
        
        Note:
            Meilisearch SDK methods are synchronous. Async methods use
            executor to run sync operations in thread pool.
        """
        self.url = url
        self.index_name = index_name
        self.primary_key = primary_key
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
        
        # Initialize Meilisearch client
        self.client = meilisearch.Client(url, api_key)
        
        # Get or create index
        if auto_create_index:
            try:
                self.index = self.client.get_index(index_name)
            except MeilisearchApiError:
                # Index doesn't exist, create it
                task = self.client.create_index(index_name, {"primaryKey": primary_key})
                self.client.wait_for_task(task.task_uid)
                self.index = self.client.get_index(index_name)
        else:
            self.index = self.client.get_index(index_name)
    
    def _unit_to_document(self, unit: BaseUnit) -> Dict[str, Any]:
        """
        Convert BaseUnit to Meilisearch document
        
        IMPORTANT:
            - Preserve nested metadata structure so that field paths (e.g.
              metadata.custom.mode) are consistent with Qdrant payloads.
            - Exclude dynamic fields like embedding/score/source which are
              not needed for fulltext index.
        
        Args:
            unit: BaseUnit instance
        
        Returns:
            Dictionary representation for Meilisearch (JSON-serializable)
        """
        # Start from JSON dump of the unit, excluding dynamic fields
        doc: Dict[str, Any] = unit.model_dump(
            mode="json",
            exclude={"embedding", "score", "source"}
        )
        
        # Normalize content/embedding_content:
        # - Always remove embedding_content from the doc
        # - If content is empty, promote embedding_content to content first
        embedding_content = doc.pop("embedding_content", None) or ""
        if not (doc.get("content") or "") and embedding_content:
            doc["content"] = embedding_content
        
        # Ensure primary key field is present and correct
        doc[self.primary_key] = unit.unit_id
        
        # For TableUnit, expose JSON-safe table data under df_data
        if isinstance(unit, TableUnit):
            json_data = unit.json_data
            if json_data is not None:
                doc["df_data"] = json_data
        
        return doc
    
    def _normalize_units(
        self, 
        units: Union[BaseUnit, list[BaseUnit]]
    ) -> list[BaseUnit]:
        """Normalize input to list format"""
        if isinstance(units, BaseUnit):
            return [units]
        return units
    
    def _normalize_ids(
        self, 
        unit_ids: Union[str, list[str]]
    ) -> list[str]:
        """Normalize input to list format"""
        if isinstance(unit_ids, str):
            return [unit_ids]
        return unit_ids
    
    def configure_settings(
        self,
        searchable_attributes: Optional[list[str]] = None,
        filterable_attributes: Optional[list[str]] = None,
        sortable_attributes: Optional[list[str]] = None,
        displayed_attributes: Optional[list[str]] = None,
        ranking_rules: Optional[list[str]] = None
    ) -> None:
        """
        Configure Meilisearch index settings
        
        Args:
            searchable_attributes: Attributes to search in
            filterable_attributes: Attributes that can be used in filters
            sortable_attributes: Attributes that can be used for sorting
            displayed_attributes: Attributes to return in search results
            ranking_rules: Custom ranking rules
        
        IMPORTANT:
            ⚠️  When you specify searchable_attributes, ONLY those fields will be searchable!
            ⚠️  All fields from unit.metadata.custom are flattened into the document.
            ⚠️  If not specified, Meilisearch defaults to ALL fields being searchable.
            
            Common pattern:
            - searchable_attributes: ["content", "title", ...]  # Fields for full-text search
            - filterable_attributes: ["price", "city", ...]     # Fields for filtering only
            - sortable_attributes: ["timestamp", "price", ...]  # Fields for sorting
        
        Example:
            >>> indexer.configure_settings(
            ...     searchable_attributes=["content", "title"],  # Only these are searchable
            ...     filterable_attributes=["source", "type", "timestamp"],  # Can filter but not search
            ...     sortable_attributes=["timestamp"]
            ... )
        """
        settings = {}
        
        if searchable_attributes is not None:
            settings["searchableAttributes"] = searchable_attributes
        if filterable_attributes is not None:
            settings["filterableAttributes"] = filterable_attributes
        if sortable_attributes is not None:
            settings["sortableAttributes"] = sortable_attributes
        if displayed_attributes is not None:
            settings["displayedAttributes"] = displayed_attributes
        if ranking_rules is not None:
            settings["rankingRules"] = ranking_rules
        
        if settings:
            task = self.index.update_settings(settings)
            self.client.wait_for_task(task.task_uid)
    
    @with_spinner_progress("Adding {count} units to fulltext index")
    def add(self, units_list: list[BaseUnit]) -> None:
        """
        Add units to index
        
        Args:
            units_list: List of units to add (normalized by decorator)
        
        Note:
            In Meilisearch, add_documents has upsert semantics by default.
            If a document with the same primary key exists, it will be updated.
        
        Example:
            >>> indexer.add(new_unit)
            >>> indexer.add([unit1, unit2])
        """
        documents = [self._unit_to_document(u) for u in units_list]
        task = self.index.add_documents(documents)
        self.client.wait_for_task(task.task_uid)
    
    @with_spinner_progress("Adding {count} units to fulltext index")
    async def aadd(self, units_list: list[BaseUnit]) -> None:
        """
        Async version of add
        
        Args:
            units_list: List of units to add (normalized by decorator)
        """
        loop = asyncio.get_event_loop()
        # Build documents list
        documents = [self._unit_to_document(u) for u in units_list]
        # Execute add_documents in thread pool
        def _sync_add():
            task = self.index.add_documents(documents)
            self.client.wait_for_task(task.task_uid)
        await loop.run_in_executor(self._executor, _sync_add)
    
    def update(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Update existing units in the index
        
        Args:
            units: Single unit or list of units to update
        
        Implementation:
            Uses update_documents which only updates existing documents.
            Non-existing documents will be ignored.
        
        Example:
            >>> modified_unit.content = "Updated content"
            >>> indexer.update(modified_unit)
        """
        units_list = self._normalize_units(units)
        
        if not units_list:
            return
        
        documents = [self._unit_to_document(u) for u in units_list]
        task = self.index.update_documents(documents)
        self.client.wait_for_task(task.task_uid)
    
    async def aupdate(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of update
        
        Args:
            units: Single unit or list of units to update
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self.update, units)
    
    def upsert(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Update or insert units (upsert)
        
        Args:
            units: Single unit or list of units to upsert
        
        Implementation:
            Meilisearch's add_documents has upsert semantics by default,
            so this delegates to add()
        
        Example:
            >>> indexer.upsert(unit)  # Add if new, update if exists
        """
        # In Meilisearch, add_documents = upsert by default
        self.add(units)
    
    async def aupsert(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of upsert
        
        Args:
            units: Single unit or list of units to upsert
        """
        await self.aadd(units)
    
    def delete(self, unit_ids: Union[str, list[str]]) -> None:
        """
        Delete units from index
        
        Args:
            unit_ids: Single unit ID or list of unit IDs to delete
        
        Example:
            >>> indexer.delete("unit_123")
            >>> indexer.delete(["unit_1", "unit_2"])
        """
        ids_list = self._normalize_ids(unit_ids)
        
        if not ids_list:
            return
        
        task = self.index.delete_documents(ids_list)
        self.client.wait_for_task(task.task_uid)
    
    async def adelete(self, unit_ids: Union[str, list[str]]) -> None:
        """
        Async version of delete
        
        Args:
            unit_ids: Single unit ID or list of unit IDs to delete
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self.delete, unit_ids)
    
    def clear(self) -> None:
        """
        Clear all units from the index
        
        Example:
            >>> indexer.clear()  # Remove all documents
        """
        task = self.index.delete_all_documents()
        self.client.wait_for_task(task.task_uid)
    
    async def aclear(self) -> None:
        """Async version of clear"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self.clear)
    
    def count(self) -> int:
        """
        Get total number of units in the index
        
        Returns:
            Number of units currently indexed
        
        Example:
            >>> count = indexer.count()
            >>> print(f"Index contains {count} units")
        """
        stats = self.index.get_stats()
        return stats.number_of_documents
    
    def exists(self, unit_id: str) -> bool:
        """
        Check if a unit exists in the index
        
        Args:
            unit_id: Unit ID to check
        
        Returns:
            True if unit exists, False otherwise
        
        Example:
            >>> if indexer.exists("unit_123"):
            ...     print("Unit found")
        """
        try:
            doc = self.index.get_document(unit_id)
            return doc is not None
        except MeilisearchApiError:
            return False
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"FullTextIndexer("
            f"url='{self.url}', "
            f"index='{self.index_name}'"
            f")"
        )
