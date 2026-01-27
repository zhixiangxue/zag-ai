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
    
    def model_dump_deep(self, **kwargs) -> dict[str, Any]:
        """
        Convert to dict with deep copy (safe for nested mutable objects)
        
        Use this when passing metadata to UnitMetadata or storing separately
        to avoid shared references to nested mutable objects (like 'custom' dict).
        
        Example:
            # Safe: deep copy prevents shared references
            unit_metadata = UnitMetadata(
                document=pdf_doc.metadata.model_dump_deep()
            )
            
            # Unsafe: shallow copy may cause shared references
            unit_metadata = UnitMetadata(
                document=pdf_doc.metadata.model_dump()  # Don't use this!
            )
        
        Args:
            **kwargs: Additional arguments passed to model_dump()
        
        Returns:
            Deep-copied dict representation
        """
        import copy
        return copy.deepcopy(self.model_dump(**kwargs))


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
    
    def model_dump_deep(self, **kwargs) -> dict[str, Any]:
        """
        Convert to dict with deep copy (safe for nested mutable objects)
        
        Use this when storing or passing metadata separately to avoid
        shared references to nested mutable objects (like 'custom' dict,
        'keywords' list, or 'document' dict).
        
        Example:
            # Safe: deep copy prevents shared references
            metadata_copy = unit.metadata.model_dump_deep()
            
            # Unsafe: shallow copy may cause shared references
            metadata_copy = unit.metadata.model_dump()  # Don't use this!
        
        Args:
            **kwargs: Additional arguments passed to model_dump()
        
        Returns:
            Deep-copied dict representation
        """
        import copy
        return copy.deepcopy(self.model_dump(**kwargs))
    
    def to_json_safe(self, **kwargs) -> dict[str, Any]:
        """
        Convert to JSON-safe dict (handles NaN, inf, numpy types, etc.)
        
        This method ensures all values are JSON-serializable by:
        - Converting numpy types to Python types
        - Converting NaN/inf to None
        - Converting all dict keys to strings
        - Recursively processing nested structures
        
        Use this when storing to vector databases (Qdrant, Milvus, etc.)
        or any system requiring strict JSON compatibility.
        
        Example:
            # For vector store payload
            payload["metadata"] = unit.metadata.to_json_safe()
            
            # For JSON serialization
            json.dumps(unit.metadata.to_json_safe())
        
        Args:
            **kwargs: Additional arguments passed to model_dump()
        
        Returns:
            JSON-compatible dict (safe for Qdrant, Milvus, JSON, etc.)
        """
        import math
        try:
            import numpy as np
            has_numpy = True
        except ImportError:
            has_numpy = False
        
        def convert_value(val):
            """Recursively convert values to JSON-serializable types"""
            if val is None:
                return None
            elif isinstance(val, (str, bool)):
                return val
            elif isinstance(val, (int, float)):
                # Handle NaN and inf
                if math.isnan(val) or math.isinf(val):
                    return None  # Convert NaN/inf to None for safety
                return val
            elif has_numpy and isinstance(val, (np.integer, np.floating)):
                # Convert numpy types to Python types
                val_python = val.item()
                # Handle NaN and inf from numpy
                if math.isnan(val_python) or math.isinf(val_python):
                    return None
                return val_python
            elif isinstance(val, (list, tuple)):
                return [convert_value(v) for v in val]
            elif isinstance(val, dict):
                return {str(k): convert_value(v) for k, v in val.items()}
            else:
                # Fallback: convert to string
                return str(val)
        
        try:
            metadata_dict = self.model_dump(**kwargs)
            return convert_value(metadata_dict)
        except Exception:
            # Ultimate fallback: return empty dict
            return {}
