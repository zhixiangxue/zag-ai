"""
Base extractor class for extracting metadata from units
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Sequence


class BaseExtractor(ABC):
    """
    Base class for all extractors.
    
    Extractors process units to extract additional metadata.
    All extractors must implement the aextract() method.
    """
    
    @abstractmethod
    async def aextract(self, units: Sequence) -> List[Dict]:
        """
        Extract metadata from units (async).
        
        Args:
            units: Sequence of units to process
            
        Returns:
            List of metadata dictionaries, one per unit
        """
        pass
    
    def extract(self, units: Sequence) -> List[Dict]:
        """
        Extract metadata from units (sync).
        
        Args:
            units: Sequence of units to process
            
        Returns:
            List of metadata dictionaries
        """
        import asyncio
        return asyncio.run(self.aextract(units))
    
    def __call__(self, units: Sequence) -> List:
        """
        Process units: extract metadata and update units.
        
        Args:
            units: Sequence of units to process
            
        Returns:
            Updated units with enriched metadata
        """
        metadata_list = self.extract(units)
        
        # Update unit metadata
        for unit, metadata in zip(units, metadata_list):
            # metadata.custom is a dict that can be updated
            unit.metadata.custom.update(metadata)
        
        return units
