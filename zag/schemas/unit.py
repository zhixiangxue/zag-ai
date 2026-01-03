"""
Concrete unit types: TextUnit, TableUnit, ImageUnit
"""

from typing import Any, Optional
from zag.schemas.base import BaseUnit


class TextUnit(BaseUnit):
    """Text unit for representing text chunks"""
    
    content: str = ""
    unit_type: str = "text"


class TableUnit(BaseUnit):
    """Table unit for representing tables"""
    
    content: Any = None  # Can be DataFrame, List[List], dict, etc.
    unit_type: str = "table"
    caption: Optional[str] = None


class ImageUnit(BaseUnit):
    """Image unit for representing images"""
    
    content: bytes = b""
    unit_type: str = "image"
    format: Optional[str] = None  # "png", "jpg", "webp", etc.
    caption: Optional[str] = None
