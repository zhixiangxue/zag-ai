"""
Vector indexer implementation

VectorIndexer wraps a VectorStore and provides high-level index management.
"""

from typing import Union, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from zag.indexers.base import BaseIndexer
from zag.schemas.base import BaseUnit
from zag.storages.vector.base import BaseVectorStore


class VectorIndexer(BaseIndexer):
    """
    Vector indexer for managing vector-based indices
    
    This indexer wraps a VectorStore and provides:
    - High-level index management (build, add, update, delete)
    - Unified sync/async interfaces
    - Support for single/batch operations
    
    Design:
        - Delegates storage operations to VectorStore
        - Adds index management logic (build vs add semantics)
        - Provides async wrappers for sync-only backends
    
    Usage:
        >>> from zag import Embedder
        >>> from zag.storages.vector import ChromaVectorStore
        >>> from zag.indexers import VectorIndexer
        >>> 
        >>> # Create vector store
        >>> embedder = Embedder('bailian/text-embedding-v3', api_key='...')
        >>> store = ChromaVectorStore(embedder=embedder, collection_name='docs')
        >>> 
        >>> # Create indexer
        >>> indexer = VectorIndexer(vector_store=store)
        >>> 
        >>> # Build index
        >>> indexer.build(units)  # Full rebuild
        >>> 
        >>> # Add more units
        >>> indexer.add(new_units)  # Incremental
    """
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize vector indexer
        
        Args:
            vector_store: VectorStore backend for storage operations
            executor: Optional thread pool for async operations on sync backends
                     If None, creates a default executor with 4 workers
        
        Note:
            The executor is only used when the backend doesn't support true async.
            For true async backends (e.g., Qdrant), the executor is not needed.
        """
        self.vector_store = vector_store
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
    
    def _normalize_units(
        self, 
        units: Union[BaseUnit, list[BaseUnit]]
    ) -> list[BaseUnit]:
        """
        Normalize input to list format
        
        Args:
            units: Single unit or list of units
        
        Returns:
            List of units
        """
        if isinstance(units, BaseUnit):
            return [units]
        return units
    
    def _normalize_ids(
        self, 
        unit_ids: Union[str, list[str]]
    ) -> list[str]:
        """
        Normalize input to list format
        
        Args:
            unit_ids: Single ID or list of IDs
        
        Returns:
            List of IDs
        """
        if isinstance(unit_ids, str):
            return [unit_ids]
        return unit_ids
    
    def build(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Build index from units (full rebuild)
        
        Clears the existing index and builds from scratch.
        
        Args:
            units: Single unit or list of units to index
        
        Process:
            1. Clear existing index
            2. Add all units to empty index
        
        Example:
            >>> indexer.build(all_units)  # Rebuild entire index
        """
        units_list = self._normalize_units(units)
        
        # Clear existing index
        self.vector_store.clear()
        
        # Add all units
        if units_list:
            self.vector_store.add(units_list)
    
    async def abuild(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of build
        
        Args:
            units: Single unit or list of units to index
        
        Implementation:
            Try to use async methods, fallback to sync with executor
        """
        units_list = self._normalize_units(units)
        
        try:
            # Try true async methods
            await self.vector_store.aclear()
            if units_list:
                await self.vector_store.aadd(units_list)
        except (AttributeError, NotImplementedError):
            # Fallback to sync with executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self.build, units)
    
    def add(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Add units to existing index (incremental)
        
        Args:
            units: Single unit or list of units to add
        
        Example:
            >>> indexer.add(new_unit)  # Add single
            >>> indexer.add([unit1, unit2])  # Add batch
        """
        units_list = self._normalize_units(units)
        
        if units_list:
            self.vector_store.add(units_list)
    
    async def aadd(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of add
        
        Args:
            units: Single unit or list of units to add
        """
        units_list = self._normalize_units(units)
        
        if not units_list:
            return
        
        try:
            # Try true async method
            await self.vector_store.aadd(units_list)
        except (AttributeError, NotImplementedError):
            # Fallback to sync with executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self.vector_store.add, units_list)
    
    def update(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Update existing units in the index
        
        Args:
            units: Single unit or list of units to update
        
        Implementation:
            Implemented as delete + add for vector stores
            Most vector stores don't distinguish between update and upsert,
            so we explicitly delete first to ensure update-only semantics
        
        Note:
            If you want to handle both insert and update, use upsert() instead
        """
        units_list = self._normalize_units(units)
        
        if not units_list:
            return
        
        # Delete + Add to ensure update semantics
        unit_ids = [unit.unit_id for unit in units_list]
        self.vector_store.delete(unit_ids)
        self.vector_store.add(units_list)
    
    async def aupdate(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of update
        
        Args:
            units: Single unit or list of units to update
        """
        units_list = self._normalize_units(units)
        
        if not units_list:
            return
        
        unit_ids = [unit.unit_id for unit in units_list]
        
        try:
            # Try true async methods
            await self.vector_store.adelete(unit_ids)
            await self.vector_store.aadd(units_list)
        except (AttributeError, NotImplementedError):
            # Fallback to sync with executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self.update, units)
    
    def upsert(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Update or insert units (upsert)
        
        Args:
            units: Single unit or list of units to upsert
        
        Implementation:
            Most vector databases (Chroma, Pinecone, etc.) have upsert semantics
            in their add() method, so we delegate directly to add()
        """
        units_list = self._normalize_units(units)
        
        if not units_list:
            return
        
        # Most vector stores implement upsert semantics in add()
        # Same ID will update, new ID will insert
        self.vector_store.add(units_list)
    
    async def aupsert(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of upsert
        
        Args:
            units: Single unit or list of units to upsert
        """
        units_list = self._normalize_units(units)
        
        if not units_list:
            return
        
        try:
            # Try true async method
            await self.vector_store.aadd(units_list)
        except (AttributeError, NotImplementedError):
            # Fallback to sync with executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self.vector_store.add, units_list)
    
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
        
        if ids_list:
            self.vector_store.delete(ids_list)
    
    async def adelete(self, unit_ids: Union[str, list[str]]) -> None:
        """
        Async version of delete
        
        Args:
            unit_ids: Single unit ID or list of unit IDs to delete
        """
        ids_list = self._normalize_ids(unit_ids)
        
        if not ids_list:
            return
        
        try:
            # Try true async method
            await self.vector_store.adelete(ids_list)
        except (AttributeError, NotImplementedError):
            # Fallback to sync with executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self.vector_store.delete, ids_list)
    
    def clear(self) -> None:
        """
        Clear all units from the index
        
        Example:
            >>> indexer.clear()  # Remove all
        """
        self.vector_store.clear()
    
    async def aclear(self) -> None:
        """
        Async version of clear
        """
        try:
            # Try true async method
            await self.vector_store.aclear()
        except (AttributeError, NotImplementedError):
            # Fallback to sync with executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self.vector_store.clear)
    
    def count(self) -> int:
        """
        Get total number of units in the index
        
        Returns:
            Number of units currently indexed
        
        Example:
            >>> count = indexer.count()
            >>> print(f"Index contains {count} units")
        """
        if hasattr(self.vector_store, 'count'):
            return self.vector_store.count()
        else:
            raise NotImplementedError(
                f"{self.vector_store.__class__.__name__} does not implement count()"
            )
    
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
        if hasattr(self.vector_store, 'exists'):
            return self.vector_store.exists(unit_id)
        else:
            # Fallback: try to get the unit
            try:
                result = self.vector_store.get([unit_id])
                return len(result) > 0
            except:
                return False
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"VectorIndexer("
            f"vector_store={self.vector_store.__class__.__name__}"
            f")"
        )
