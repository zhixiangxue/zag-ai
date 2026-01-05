"""
Concrete unit types: TextUnit, TableUnit, ImageUnit
"""

from typing import Any, Optional, Dict

from .base import BaseUnit, UnitType


class TextUnit(BaseUnit):
    """Text unit for representing text chunks"""
    
    content: str = ""
    unit_type: UnitType = UnitType.TEXT


class TableUnit(BaseUnit):
    """Table unit for representing tables
    
    Attributes:
        content: Original table representation (Markdown, HTML, or plain text)
        json_data: Structured table data as dict (parsed by docling/minerU)
                   Format: {"headers": [...], "rows": [[...], [...]]}
        caption: Optional table caption or title
    """
    
    content: str = ""  # Markdown/HTML/text format
    json_data: Optional[Dict] = None  # Structured table data
    unit_type: UnitType = UnitType.TABLE
    caption: Optional[str] = None


class ImageUnit(BaseUnit):
    """Image unit for representing images"""
    
    content: bytes = b""
    unit_type: UnitType = UnitType.IMAGE
    format: Optional[str] = None  # "png", "jpg", "webp", etc.
    caption: Optional[str] = None
