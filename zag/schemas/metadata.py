"""
Metadata classes for documents and units

This module contains structured metadata definitions used throughout the schema system.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentMetadata(BaseModel):
    """
    Structured metadata for documents
    Type-safe alternative to plain dict
    """
    
    # Source information
    source: str
    source_type: str  # "local" or "url"
    file_type: str  # "pdf", "markdown", etc.
    
    # File information
    file_name: Optional[str] = None
    file_size: Optional[int] = None  # in bytes
    file_extension: Optional[str] = None
    md5: str  # File hash (xxhash), REQUIRED for integrity check
    
    # Content information
    content_length: int = 0  # length in characters
    mime_type: Optional[str] = None
    
    # Processing information
    created_at: datetime = Field(default_factory=datetime.now)
    reader_name: Optional[str] = None
    
    # Custom fields (for extensibility)
    custom: dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True,
    }


class UnitMetadata(BaseModel):
    """
    Universal metadata for units
    
    Core position fields:
        - context_path: Hierarchical context (e.g., "Chapter 1 > Section 1.1")
        - page_numbers: Pages where unit appears (e.g., [1], [1,2], None)
        - bbox: Bounding box coordinates {"l": x1, "t": y1, "r": x2, "b": y2}
    
    Document tracking:
        - document: Complete document metadata dict (for filtering/tracking)
    
    Content extraction:
        - keywords: Extracted keywords from content (set by KeywordExtractor)
    
    Design principles:
        - Position fields are framework-level (not business-specific)
        - document field stores source document info (file_name, md5, etc.)
        - keywords field stores extracted keywords (set by Extractor)
        - custom dict is for business metadata only (framework should not write to it)
        - bbox becomes None after unit merging/splitting
        - page_numbers inferred automatically by PDF.split() using fuzzy matching
    """
    
    context_path: Optional[str] = None
    """
    Hierarchical path providing full context
    
    This field represents the heading hierarchy where this unit appears.
    
    Examples:
        - Markdown: "Introduction > Background > History"
        - PDF with structure: "Chapter 1 > Section 1.2 > Subsection"
        - PDF without structure: None (cannot infer hierarchy)
        - Word: "Chapter1 > Section1.2"
        - Excel: "Sheet1 > TableA"
    
    Set to None when hierarchy cannot be determined.
    """
    
    page_numbers: Optional[list[int]] = None
    """
    Page numbers where this unit appears (supports cross-page units)
    
    Examples:
        - Single page: [1]
        - Cross-page: [1, 2] (unit spans pages 1 and 2)
        - Multiple pages: [2, 3, 5] (unit appears on non-consecutive pages)
        - Unknown: None (cannot determine, e.g., after heavy processing)
    
    Note:
        - Automatically inferred by PDF.split() using fuzzy position matching
        - May be None for non-pageable documents (Excel, Markdown)
        - May be None if matching confidence is too low
        - Check metadata.custom['page_match_confidence'] for match quality
    """
    
    document: Optional[dict[str, Any]] = None
    """
    Source document metadata (for filtering/tracking/auditing)
    
    This is a dict representation of DocumentMetadata containing:
        - source: File path or URL
        - file_name: Original file name
        - md5: File hash for integrity
        - file_type: pdf, markdown, docx, etc.
        - reader_name: Which reader processed this
        - And other document-level information
    
    Examples:
        - Filter by file: units.filter(lambda u: u.metadata.document['file_name'] == 'doc.pdf')
        - Track source: u.metadata.document['md5']
        - Audit processing: u.metadata.document['reader_name']
    
    This field is automatically populated by Document.split()
    """
    
    keywords: Optional[list[str]] = None
    """
    Extracted keywords from unit content
    
    This field is populated by KeywordExtractor or StructuredExtractor.
    
    Examples:
        - unit.metadata.keywords = ['mortgage', 'fixed rate', '30-year']
        - Filter by keyword: units.filter(lambda u: 'mortgage' in (u.metadata.keywords or []))
    
    Note:
        - Set by Extractors, not by framework core
        - None means extraction not performed yet
        - Empty list [] means no keywords found
    """
    
    custom: dict[str, Any] = Field(default_factory=dict)
    """
    Business-specific custom metadata fields
    
    IMPORTANT: This is for APPLICATION/BUSINESS use only.
    The zag framework itself should NOT write to this field.
    
    Examples:
        - Business: {"lender_name": "GlobalTrust", "product_type": "mortgage"}
        - Business: {"department": "HR", "confidential": True}
        - Business: {"category": "legal", "contract_id": "CNT-2024-001"}
    """
    
    model_config = {
        "arbitrary_types_allowed": True,
    }
