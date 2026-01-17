"""
Document-related classes

This module contains base document classes and page structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field

from .metadata import DocumentMetadata
from .unit import BaseUnit, UnitCollection


class BaseDocument(BaseModel, ABC):
    """
    Base class for all document types
    Acts as a container for structured parsed results
    
    Note:
        doc_id must be provided by Reader (typically based on file hash for idempotency)
        metadata is now a structured DocumentMetadata object
    """
    
    doc_id: str  # Required: Reader should provide hash-based ID
    metadata: DocumentMetadata
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }
    
    @abstractmethod
    def split(self, splitter: 'BaseSplitter') -> 'UnitCollection':
        """
        Split document into units using the given splitter
        
        Args:
            splitter: The splitter to use for splitting
            
        Returns:
            UnitCollection containing the split units
        """
        pass


class Page(BaseModel):
    """
    Generic page structure for documents with page-level data
    
    Design:
        - content: Human-readable page content (Markdown text with tables)
        - units: Machine-processable structured units (TextUnit, TableUnit, etc.)
    
    Both fields serve different purposes:
        - content: For display, LLM context, human reading
        - units: For RAG processing, vector indexing, structured operations
    """
    
    page_number: int
    content: str = ""  # Full page text in Markdown format
    units: list['BaseUnit'] = Field(default_factory=list)  # Structured units
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True,
    }


class PageableDocument(BaseDocument):
    """
    Base class for documents with page structure (PDF, DOCX, PPTX, etc.)
    """
    
    pages: list[Page] = Field(default_factory=list)
    
    def get_page(self, page_num: int) -> Optional[Page]:
        """
        Get page by page number
        
        Args:
            page_num: Page number to retrieve
            
        Returns:
            Page object if found, None otherwise
        """
        for page in self.pages:
            if page.page_number == page_num:
                return page
        return None
    
    def get_page_count(self) -> int:
        """
        Get total number of pages
        
        Returns:
            Total page count
        """
        return len(self.pages)
