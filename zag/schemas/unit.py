"""
Concrete unit types: TextUnit, TableUnit, ImageUnit
"""

from typing import Any, Optional, Dict

from .base import BaseUnit, UnitType


class TextUnit(BaseUnit):
    """
    Text unit for representing text chunks
    
    Attributes:
        content: Original content (full text with Markdown tables, for LLM)
        embedding_content: Processed content for embedding (tables replaced with summaries)
        unit_type: Unit type (always TEXT)
    
    Design:
        - content: Used by LLM (preserves full information including Markdown tables)
        - embedding_content: Used for vector embedding/search (cleaner signal)
        
        When embedding_content is None, use content for embedding.
    """
    
    content: str = ""
    embedding_content: Optional[str] = None
    unit_type: UnitType = UnitType.TEXT
    
    def __add__(self, other: 'TextUnit') -> 'TextUnit':
        """
        Merge two TextUnits using + operator
        
        Creates a new TextUnit with:
        - Combined content (separated by double newline)
        - Context path from the first unit
        - New unit_id
        - Metadata tracks merged units
        
        Example:
            >>> unit1 = TextUnit(content="Part 1")
            >>> unit2 = TextUnit(content="Part 2")
            >>> merged = unit1 + unit2
            >>> merged.content
            "Part 1\n\nPart 2"
        
        Args:
            other: Another TextUnit to merge with
        
        Returns:
            New merged TextUnit
        """
        if not isinstance(other, TextUnit):
            raise TypeError(f"Cannot add TextUnit with {type(other)}")
        
        # Merge content
        merged_content = f"{self.content}\n\n{other.content}"
        
        # Merge embedding_content if exists
        merged_embedding_content = None
        if self.embedding_content or other.embedding_content:
            embed_self = self.embedding_content if self.embedding_content else self.content
            embed_other = other.embedding_content if other.embedding_content else other.content
            merged_embedding_content = f"{embed_self}\n\n{embed_other}"
        
        # Create new unit with merged content
        from .base import UnitMetadata
        from uuid import uuid4
        
        # Build merged metadata
        merged_custom = {}
        
        # Inherit custom metadata from first unit
        if self.metadata and self.metadata.custom:
            merged_custom.update(self.metadata.custom)
        
        # Track merged units
        merged_from = merged_custom.get("merged_from", [])
        if isinstance(merged_from, list):
            merged_from.extend([self.unit_id, other.unit_id])
        else:
            merged_from = [self.unit_id, other.unit_id]
        
        merged_custom["merged_from"] = merged_from
        merged_custom["merged_count"] = len(merged_from)
        
        # Create merged unit
        merged = TextUnit(
            unit_id=str(uuid4()),
            content=merged_content,
            embedding_content=merged_embedding_content,
            metadata=UnitMetadata(
                context_path=self.metadata.context_path if self.metadata else None,
                custom=merged_custom
            )
        )
        
        # Inherit source_doc_id from first unit
        if hasattr(self, 'source_doc_id') and self.source_doc_id:
            merged.source_doc_id = self.source_doc_id
        
        return merged


class TableUnit(BaseUnit):
    """
    Table unit for representing tables
    
    Attributes:
        content: Original table representation (Markdown, HTML, or plain text)
        embedding_content: Processed content for embedding (e.g., natural language summary)
        json_data: Structured table data as dict (parsed by docling/minerU)
                   Format: {"headers": [...], "rows": [[...], [...]]}
        caption: Optional table caption or title
    
    Design:
        - content: Original table format (for LLM and display)
        - embedding_content: Table summary or structured description (for vector search)
        - json_data: Structured data (for programmatic access)
        
        When embedding_content is None, use content for embedding.
    """
    
    content: str = ""  # Markdown/HTML/text format
    embedding_content: Optional[str] = None  # Summary or description for embedding
    json_data: Optional[Dict] = None  # Structured table data
    unit_type: UnitType = UnitType.TABLE
    caption: Optional[str] = None


class ImageUnit(BaseUnit):
    """Image unit for representing images"""
    
    content: bytes = b""
    unit_type: UnitType = UnitType.IMAGE
    format: Optional[str] = None  # "png", "jpg", "webp", etc.
    caption: Optional[str] = None
