"""
Concrete unit types: TextUnit, TableUnit, ImageUnit
"""

from typing import Any, Optional, Dict
from uuid import uuid4

from .base import BaseUnit, UnitType, UnitMetadata


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
        
        # Merge page numbers
        # If both units have page_numbers, merge them (union + sort)
        # This handles cross-page merged units
        merged_page_numbers = None
        if self.metadata and other.metadata:
            pages_self = self.metadata.page_numbers or []
            pages_other = other.metadata.page_numbers or []
            if pages_self or pages_other:
                # Merge and deduplicate
                all_pages = sorted(set(pages_self + pages_other))
                merged_page_numbers = all_pages if all_pages else None
        
        # Create merged unit
        merged = TextUnit(
            unit_id=str(uuid4()),
            content=merged_content,
            embedding_content=merged_embedding_content,
            metadata=UnitMetadata(
                context_path=self.metadata.context_path if self.metadata else None,
                page_numbers=merged_page_numbers,
            )
        )
        
        # Inherit doc_id from first unit
        if hasattr(self, 'doc_id') and self.doc_id:
            merged.doc_id = self.doc_id
        
        return merged


class TableUnit(BaseUnit):
    """
    Table unit for representing tables
    
    Attributes:
        content: Original table representation (Markdown, HTML, or plain text)
        embedding_content: Processed content for embedding (e.g., natural language summary)
        df: Table data as pandas DataFrame (for flexible data operations)
        caption: Optional table caption or title
    
    Design:
        - content: Original table format (for LLM and display)
        - embedding_content: Table summary or structured description (for vector search)
        - df: Structured data as DataFrame (for analysis, filtering, transformation)
        
        When embedding_content is None, use content for embedding.
        
    Why pandas DataFrame:
        - Flexible data operations (filter, group, aggregate)
        - Easy serialization (to_json, to_dict, to_csv)
        - Rich ecosystem integration
        - Natural representation for tabular data
    """
    
    content: str = ""  # Markdown/HTML/text format
    embedding_content: Optional[str] = None  # Summary or description for embedding
    df: Optional[Any] = None  # pandas DataFrame (use Any to avoid pandas dependency)
    unit_type: UnitType = UnitType.TABLE
    caption: Optional[str] = None
    
    def __add__(self, other: 'TableUnit') -> 'TableUnit':
        """
        Merge two TableUnits using + operator
        
        Creates a new TableUnit with:
        - Combined content (separated by double newline)
        - Context path from the first unit
        - New unit_id
        - Metadata tracks merged units
        - bbox set to None (merged tables have no valid bbox)
        
        Note: json_data merging is complex and not implemented.
              Use content (Markdown) for merged representation.
        
        Example:
            >>> table1 = TableUnit(content="| A | B |\n|---|---|\n| 1 | 2 |")
            >>> table2 = TableUnit(content="| C | D |\n|---|---|\n| 3 | 4 |")
            >>> merged = table1 + table2
        
        Args:
            other: Another TableUnit to merge with
        
        Returns:
            New merged TableUnit
        """
        if not isinstance(other, TableUnit):
            raise TypeError(f"Cannot add TableUnit with {type(other)}")
        
        # Merge content
        merged_content = f"{self.content}\n\n{other.content}"
        
        # Merge embedding_content if exists
        merged_embedding_content = None
        if self.embedding_content or other.embedding_content:
            embed_self = self.embedding_content if self.embedding_content else self.content
            embed_other = other.embedding_content if other.embedding_content else other.content
            merged_embedding_content = f"{embed_self}\n\n{embed_other}"
        
        # Merge captions
        merged_caption = None
        if self.caption and other.caption:
            merged_caption = f"{self.caption}; {other.caption}"
        elif self.caption:
            merged_caption = self.caption
        elif other.caption:
            merged_caption = other.caption
        
        # Merge page numbers
        merged_page_numbers = None
        if self.metadata and other.metadata:
            pages_self = self.metadata.page_numbers or []
            pages_other = other.metadata.page_numbers or []
            if pages_self or pages_other:
                all_pages = sorted(set(pages_self + pages_other))
                merged_page_numbers = all_pages if all_pages else None
        
        # Create merged unit
        merged = TableUnit(
            unit_id=str(uuid4()),
            content=merged_content,
            embedding_content=merged_embedding_content,
            caption=merged_caption,
            df=None,  # TODO: Implement DataFrame merging logic (concat, append, etc.)
            metadata=UnitMetadata(
                context_path=self.metadata.context_path if self.metadata else None,
                page_numbers=merged_page_numbers,
                bbox=None,  # Merged table has no valid bbox
                header=None  # Tables are never headers
            )
        )
        
        # Inherit doc_id from first unit
        if hasattr(self, 'doc_id') and self.doc_id:
            merged.doc_id = self.doc_id
        
        return merged


class ImageUnit(BaseUnit):
    """Image unit for representing images"""
    
    content: bytes = b""
    unit_type: UnitType = UnitType.IMAGE
    format: Optional[str] = None  # "png", "jpg", "webp", etc.
    caption: Optional[str] = None
