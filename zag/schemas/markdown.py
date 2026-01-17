"""
Markdown document schema
"""

from .document import BaseDocument
from .unit import UnitCollection


class Markdown(BaseDocument):
    """
    Markdown document
    Simple text-based document without page structure
    """
    
    content: str = ""  # Markdown content
    
    def split(self, splitter: 'BaseSplitter') -> UnitCollection:
        """
        Split markdown document into units
        
        Args:
            splitter: The splitter to use
            
        Returns:
            UnitCollection of split units
        """
        return splitter.split(self)
