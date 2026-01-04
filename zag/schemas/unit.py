"""
Concrete unit types: TextUnit, TableUnit, ImageUnit
"""

from typing import Any, Optional
from zag.schemas.base import BaseUnit, UnitType


class TextUnit(BaseUnit):
    """Text unit for representing text chunks"""
    
    content: str = ""
    unit_type: UnitType = UnitType.TEXT


class TableUnit(BaseUnit):
    """Table unit for representing tables"""
    
    content: Any = None  # Can be DataFrame, List[List], dict, etc.
    unit_type: UnitType = UnitType.TABLE
    caption: Optional[str] = None


class ImageUnit(BaseUnit):
    """Image unit for representing images"""
    
    content: bytes = b""
    unit_type: UnitType = UnitType.IMAGE
    format: Optional[str] = None  # "png", "jpg", "webp", etc.
    caption: Optional[str] = None
