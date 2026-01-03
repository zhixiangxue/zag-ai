"""
Base indexer class
"""

from abc import ABC, abstractmethod
from typing import Any

from zag.schemas.base import BaseUnit


class BaseIndexer(ABC):
    """
    Base class for all indexers
    
    Indexers build and manage indices for efficient retrieval
    """
    
    @abstractmethod
    def build(self, units: list[BaseUnit]) -> None:
        """
        Build index from units
        
        Args:
            units: List of units to index
        """
        pass
    
    @abstractmethod
    def add(self, units: list[BaseUnit]) -> None:
        """
        Add units to existing index
        
        Args:
            units: List of units to add
        """
        pass
    
    @abstractmethod
    def delete(self, unit_ids: list[str]) -> None:
        """
        Delete units from index
        
        Args:
            unit_ids: List of unit IDs to delete
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save index to disk
        
        Args:
            path: Path to save the index
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load index from disk
        
        Args:
            path: Path to load the index from
        """
        pass
