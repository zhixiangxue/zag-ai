"""
PDF document schema
"""

from typing import Any
from pydantic import Field

from .base import PageableDocument, UnitCollection


class PDF(PageableDocument):
    """
    PDF document with page structure
    """
    
    content: Any = None  # Raw PDF data or structured content
    
    def split(self, splitter: 'BaseSplitter') -> UnitCollection:
        """
        Split PDF document into units
        
        Args:
            splitter: The splitter to use
            
        Returns:
            UnitCollection of split units
        """
        return splitter.split(self)
