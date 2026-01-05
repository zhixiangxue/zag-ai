"""
Base indexer class
"""

from abc import ABC, abstractmethod
from typing import Union, Optional

from zag.schemas.base import BaseUnit


class BaseIndexer(ABC):
    """
    Base class for all indexers
    
    Indexer is a high-level abstraction that manages indices for efficient retrieval.
    It orchestrates one or more storage backends (vector store, document store, etc.)
    and provides unified indexing operations.
    
    Design Philosophy:
        - Indexer = Index Manager (high-level orchestration)
        - VectorStore = Storage Backend (low-level operations)
        - Indexer coordinates multiple stores if needed
    
    Key Differences from VectorStore:
        - Indexer: build(), add(), delete() (index management)
        - VectorStore: add(), search(), delete() (storage operations)
    
    Responsibilities:
        - Build and manage indices (may involve multiple storages)
        - Provide high-level indexing operations
        - Support index persistence and loading
        - Unified sync/async interfaces
    
    Note:
        All methods support both single unit and batch operations:
        - Single: indexer.add(unit)
        - Batch: indexer.add([unit1, unit2, ...])
    """
    
    @abstractmethod
    def build(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Build index from units (full rebuild)
        
        This method clears the existing index and builds from scratch.
        Use this when you want to completely replace the index content.
        
        Args:
            units: Single unit or list of units to index
        
        Note:
            - This is a destructive operation (clears existing index)
            - For incremental updates, use add() instead
            - Batch operations are more efficient
        
        Example:
            >>> indexer.build(units)  # Rebuild entire index
        """
        pass
    
    @abstractmethod
    async def abuild(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of build - build index from units
        
        Args:
            units: Single unit or list of units to index
        
        Implementation Notes:
            - True async if backend supports it (e.g., Qdrant)
            - Use executor wrapper for sync backends (e.g., ChromaDB)
            - Document async support level in subclass
        """
        pass
    
    @abstractmethod
    def add(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Add units to existing index (incremental)
        
        This method adds units to the existing index without clearing it.
        Use this for incremental updates to an existing index.
        
        Args:
            units: Single unit or list of units to add
        
        Note:
            - Non-destructive operation
            - Existing index content is preserved
            - Duplicate unit_ids may cause issues (implementation-specific)
        
        Example:
            >>> indexer.add(new_unit)  # Add single unit
            >>> indexer.add([unit1, unit2])  # Add multiple units
        """
        pass
    
    @abstractmethod
    async def aadd(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of add - add units to existing index
        
        Args:
            units: Single unit or list of units to add
        """
        pass
    
    @abstractmethod
    def update(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Update existing units in the index
        
        Updates units that already exist in the index (based on unit_id).
        Behavior for non-existing units depends on implementation.
        
        Args:
            units: Single unit or list of units to update
        
        Note:
            - Designed for updating existing units only
            - May raise error or ignore non-existing units (implementation-specific)
            - Use upsert() if you want to handle both insert and update
        
        Example:
            >>> modified_unit.content = "Updated content"
            >>> indexer.update(modified_unit)
        """
        pass
    
    @abstractmethod
    async def aupdate(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of update - update existing units
        
        Args:
            units: Single unit or list of units to update
        """
        pass
    
    @abstractmethod
    def upsert(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Update or insert units (upsert)
        
        This method updates units if they exist (based on unit_id),
        or inserts them if they don't exist.
        
        Args:
            units: Single unit or list of units to upsert
        
        Note:
            - Most flexible operation for incremental updates
            - Handles both new and existing units
            - Recommended when you don't know if unit exists
        
        Example:
            >>> indexer.upsert(unit)  # Add if new, update if exists
            >>> indexer.upsert([unit1, unit2])  # Batch upsert
        """
        pass
    
    @abstractmethod
    async def aupsert(self, units: Union[BaseUnit, list[BaseUnit]]) -> None:
        """
        Async version of upsert - update or insert units
        
        Args:
            units: Single unit or list of units to upsert
        """
        pass
    
    @abstractmethod
    def delete(self, unit_ids: Union[str, list[str]]) -> None:
        """
        Delete units from index
        
        Removes units from the index based on their unit_ids.
        
        Args:
            unit_ids: Single unit ID or list of unit IDs to delete
        
        Note:
            - Silent if unit_id doesn't exist (no error)
            - Cascading deletion if multi-store (vector + doc store)
        
        Example:
            >>> indexer.delete("unit_123")  # Delete single unit
            >>> indexer.delete(["unit_1", "unit_2"])  # Delete multiple
        """
        pass
    
    @abstractmethod
    async def adelete(self, unit_ids: Union[str, list[str]]) -> None:
        """
        Async version of delete - delete units from index
        
        Args:
            unit_ids: Single unit ID or list of unit IDs to delete
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all units from the index
        
        Removes all indexed units, effectively resetting the index.
        
        Warning:
            This is a destructive operation that cannot be undone.
        
        Example:
            >>> indexer.clear()  # Remove all indexed data
        """
        pass
    
    @abstractmethod
    async def aclear(self) -> None:
        """
        Async version of clear - clear all units from index
        """
        pass
    
    def count(self) -> int:
        """
        Get total number of units in the index
        
        Returns:
            Number of units currently indexed
        
        Note:
            This is an optional method. Subclasses may override it.
            Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement count()"
        )
    
    def exists(self, unit_id: str) -> bool:
        """
        Check if a unit exists in the index
        
        Args:
            unit_id: Unit ID to check
        
        Returns:
            True if unit exists, False otherwise
        
        Note:
            This is an optional method. Subclasses may override it.
            Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement exists()"
        )
