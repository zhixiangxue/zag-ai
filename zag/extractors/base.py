"""
Base extractor class and default extractors
"""

from abc import ABC, abstractmethod

from ..schemas.base import BaseUnit


class BaseExtractor(ABC):
    """
    Base class for all extractors
    Extractors process units to extract or enrich metadata
    """
    
    @abstractmethod
    def process(self, unit: BaseUnit) -> BaseUnit:
        """
        Process a unit and return the processed unit
        
        Args:
            unit: The unit to process
            
        Returns:
            The processed unit (can be the same object or a new one)
        """
        pass


class IdentityExtractor(BaseExtractor):
    """
    Default extractor that does nothing
    Useful as a no-op default for optional extraction
    """
    
    def process(self, unit: BaseUnit) -> BaseUnit:
        """Return the unit unchanged"""
        return unit
