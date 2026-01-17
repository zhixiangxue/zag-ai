"""
Base reader class for all readers
"""

from abc import ABC, abstractmethod

from ..schemas import BaseDocument


class BaseReader(ABC):
    """
    Base class for all readers
    Simple interface definition - readers parse files and return Document objects
    
    Note:
        For optional utility functions (source validation, file type detection),
        see zag.utils.source.SourceUtils
    """
    
    @abstractmethod
    def read(self, source: str) -> BaseDocument:
        """
        Read and parse a file
        
        Args:
            source: File path (relative/absolute) or URL
            
        Returns:
            A Document object containing structured parsed results
        """
        pass
