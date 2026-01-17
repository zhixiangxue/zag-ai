"""
Base unit store class
"""

from abc import ABC, abstractmethod
from typing import Optional

from ...schemas import BaseUnit


class BaseUnitStore(ABC):
    """
    Base class for unit storage
    
    UnitStore maintains the mapping: unit_id -> Unit object
    Similar to LlamaIndex's NodeStore
    
    This is the primary storage for Units after splitting.
    VectorStore only stores embeddings, while UnitStore stores complete Unit objects.
    """
    
    @abstractmethod
    def add(self, units: list[BaseUnit]) -> None:
        """
        Add units to store
        
        Args:
            units: List of units to add
        """
        pass
    
    @abstractmethod
    def get(self, unit_id: str) -> Optional[BaseUnit]:
        """
        Get unit by ID
        
        Args:
            unit_id: ID of the unit to retrieve
            
        Returns:
            Unit object if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_batch(self, unit_ids: list[str]) -> list[BaseUnit]:
        """
        Get multiple units by IDs
        
        Args:
            unit_ids: List of unit IDs
            
        Returns:
            List of units (missing IDs are skipped)
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
    def list_all(self) -> list[str]:
        """
        List all unit IDs in store
        
        Returns:
            List of all unit IDs
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all units from store
        """
        pass
