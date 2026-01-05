"""
Base classes and utilities for all schema types.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Callable
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
import uuid


class RelationType(str, Enum):
    """Predefined relationship types between units"""
    
    # Reference relationships
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"
    
    # Hierarchical relationships
    PARENT = "parent"
    CHILDREN = "children"
    
    # Semantic relationships
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    
    # Structural relationships
    FOOTNOTE = "footnote"
    CAPTION_OF = "caption_of"
    VISUAL_CONTEXT = "visual_context"


class UnitType(str, Enum):
    """Unit content types"""
    
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    BASE = "base"  # Base/unknown type


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
    Minimalist design inspired by LlamaIndex
    
    Focuses on providing context path for LLM understanding
    """
    
    context_path: Optional[str] = None
    """
    Hierarchical path providing full context
    
    This single field encodes:
    - Context hierarchy (by splitting "/")
    - Current section/page title (last segment)
    - Parent context (all but last segment)
    - Depth level (count "/")
    
    Examples:
        - Markdown: "Introduction/Background/History"
        - PDF: "Page3/Section2"
        - Word: "Chapter1/Section1.2"
        - Excel: "Sheet1/TableA"
    """
    
    custom: dict[str, Any] = Field(default_factory=dict)
    """
    Optional custom fields for document-type specific metadata
    
    Examples:
        - Markdown: {"is_code_block": True, "language": "python"}
        - PDF: {"page_number": 3, "bbox": [x, y, w, h]}
        - Table: {"row_count": 10, "col_count": 5}
    """
    
    model_config = {
        "arbitrary_types_allowed": True,
    }


class UnitRegistry:
    """Global unit registry for runtime ID-to-object resolution"""
    
    _units: dict[str, 'BaseUnit'] = {}
    
    @classmethod
    def register(cls, unit: 'BaseUnit') -> None:
        """Register a unit to the store"""
        cls._units[unit.unit_id] = unit
    
    @classmethod
    def get(cls, unit_id: str) -> Optional['BaseUnit']:
        """Get unit by ID"""
        return cls._units.get(unit_id)
    
    @classmethod
    def get_many(cls, unit_ids: list[str]) -> list['BaseUnit']:
        """Get multiple units by IDs"""
        return [cls._units[uid] for uid in unit_ids if uid in cls._units]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all units (useful for testing)"""
        cls._units.clear()
    
    @classmethod
    def count(cls) -> int:
        """Get total number of registered units"""
        return len(cls._units)


class BaseUnit(BaseModel):
    """Base class for all units (text, table, image, etc.)"""
    
    unit_id: str
    content: Any
    unit_type: UnitType = UnitType.BASE
    metadata: UnitMetadata = Field(default_factory=UnitMetadata)
    
    # Embedding vector (optional, for caching)
    embedding: Optional[list[float]] = None
    
    # Retrieval score (set by retrievers)
    score: Optional[float] = None
    
    # Chain relationships (managed by Splitter)
    prev_unit_id: Optional[str] = None
    next_unit_id: Optional[str] = None
    source_doc_id: Optional[str] = None
    
    # Semantic relationships (stored as ID lists)
    relations: dict[str, list[str]] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }
    
    @model_validator(mode='after')
    def register_to_store(self):
        """Auto-register to UnitRegistry after initialization"""
        UnitRegistry.register(self)
        return self
    
    # ============ Relationship Methods (Object-based) ============
    
    def add_reference(self, target: 'BaseUnit', bidirectional: bool = True) -> 'BaseUnit':
        """
        Add reference relationship: this unit references target
        
        Args:
            target: The unit being referenced
            bidirectional: If True, automatically add reverse relationship
        """
        self._add_relation(RelationType.REFERENCES, target.unit_id)
        if bidirectional:
            target._add_relation(RelationType.REFERENCED_BY, self.unit_id)
        return self
    
    def add_referenced_by(self, source: 'BaseUnit', bidirectional: bool = True) -> 'BaseUnit':
        """
        Add referenced-by relationship: this unit is referenced by source
        
        Args:
            source: The unit that references this unit
            bidirectional: If True, automatically add reverse relationship
        """
        self._add_relation(RelationType.REFERENCED_BY, source.unit_id)
        if bidirectional:
            source._add_relation(RelationType.REFERENCES, self.unit_id)
        return self
    
    def set_parent(self, parent: 'BaseUnit', bidirectional: bool = True) -> 'BaseUnit':
        """
        Set parent unit (e.g., parent section)
        
        Args:
            parent: The parent unit
            bidirectional: If True, automatically add this unit to parent's children
        """
        self.relations[RelationType.PARENT] = [parent.unit_id]
        if bidirectional:
            parent._add_relation(RelationType.CHILDREN, self.unit_id)
        return self
    
    def add_child(self, child: 'BaseUnit', bidirectional: bool = True) -> 'BaseUnit':
        """
        Add child unit
        
        Args:
            child: The child unit
            bidirectional: If True, automatically set this unit as child's parent
        """
        self._add_relation(RelationType.CHILDREN, child.unit_id)
        if bidirectional:
            child.relations[RelationType.PARENT] = [self.unit_id]
        return self
    
    def add_related(self, related: 'BaseUnit', bidirectional: bool = True) -> 'BaseUnit':
        """
        Add related content
        
        Args:
            related: The related unit
            bidirectional: If True, automatically add reverse relationship
        """
        self._add_relation(RelationType.RELATED_TO, related.unit_id)
        if bidirectional:
            related._add_relation(RelationType.RELATED_TO, self.unit_id)
        return self
    
    def set_caption_of(self, element: 'BaseUnit', bidirectional: bool = True) -> 'BaseUnit':
        """
        Mark this unit as caption/description of another element
        
        Args:
            element: The element this unit describes
            bidirectional: If True, automatically add visual context relationship
        """
        self.relations[RelationType.CAPTION_OF] = [element.unit_id]
        if bidirectional:
            element._add_relation(RelationType.VISUAL_CONTEXT, self.unit_id)
        return self
    
    def add_visual_context(self, visual: 'BaseUnit', bidirectional: bool = True) -> 'BaseUnit':
        """
        Add visual context (e.g., related chart/image)
        
        Args:
            visual: The visual element
            bidirectional: If True, automatically add caption relationship
        """
        self._add_relation(RelationType.VISUAL_CONTEXT, visual.unit_id)
        if bidirectional:
            visual._add_relation(RelationType.CAPTION_OF, self.unit_id)
        return self
    
    # ============ Query Methods (return objects) ============
    
    def get_references(self) -> list['BaseUnit']:
        """Get all referenced units as objects"""
        return UnitRegistry.get_many(self.get_reference_ids())
    
    def get_referenced_by(self) -> list['BaseUnit']:
        """Get all units that reference this unit as objects"""
        return UnitRegistry.get_many(self.get_referenced_by_ids())
    
    def get_parent(self) -> Optional['BaseUnit']:
        """Get parent unit as object"""
        parent_id = self.get_parent_id()
        return UnitRegistry.get(parent_id) if parent_id else None
    
    def get_children(self) -> list['BaseUnit']:
        """Get all child units as objects"""
        return UnitRegistry.get_many(self.get_children_ids())
    
    def get_related(self) -> list['BaseUnit']:
        """Get all related units as objects"""
        return UnitRegistry.get_many(self.relations.get(RelationType.RELATED_TO, []))
    
    def get_prev(self) -> Optional['BaseUnit']:
        """Get previous unit in chain"""
        return UnitRegistry.get(self.prev_unit_id) if self.prev_unit_id else None
    
    def get_next(self) -> Optional['BaseUnit']:
        """Get next unit in chain"""
        return UnitRegistry.get(self.next_unit_id) if self.next_unit_id else None
    
    # ============ Query Methods (return IDs) ============
    
    def get_reference_ids(self) -> list[str]:
        """Get all referenced unit IDs"""
        return self.relations.get(RelationType.REFERENCES, [])
    
    def get_referenced_by_ids(self) -> list[str]:
        """Get all unit IDs that reference this unit"""
        return self.relations.get(RelationType.REFERENCED_BY, [])
    
    def get_parent_id(self) -> Optional[str]:
        """Get parent unit ID"""
        parents = self.relations.get(RelationType.PARENT, [])
        return parents[0] if parents else None
    
    def get_children_ids(self) -> list[str]:
        """Get all child unit IDs"""
        return self.relations.get(RelationType.CHILDREN, [])
    
    # ============ Utility Methods ============
    
    def has_references(self) -> bool:
        """Check if this unit has any references"""
        return len(self.get_reference_ids()) > 0
    
    def has_parent(self) -> bool:
        """Check if this unit has a parent"""
        return self.get_parent_id() is not None
    
    def _add_relation(self, rel_type: str, target_id: str) -> None:
        """Internal method: add a relation"""
        if rel_type not in self.relations:
            self.relations[rel_type] = []
        if target_id not in self.relations[rel_type]:
            self.relations[rel_type].append(target_id)


class UnitCollection(list):
    """
    Collection of units with chainable methods
    Supports filtering, extraction, and other operations
    """
    
    def __init__(self, units: list[BaseUnit]):
        super().__init__(units)
    
    def extract(self, extractor: 'BaseExtractor') -> 'UnitCollection':
        """
        Apply extractor to all units
        
        Args:
            extractor: The extractor to apply
            
        Returns:
            New UnitCollection with processed units
        """
        return UnitCollection([extractor.process(unit) for unit in self])
    
    def filter_by_type(self, unit_type: str) -> 'UnitCollection':
        """
        Filter units by type
        
        Args:
            unit_type: The unit type to filter ("text", "table", "image")
            
        Returns:
            New UnitCollection with filtered units
        """
        return UnitCollection([u for u in self if u.unit_type == unit_type])
    
    def filter(self, predicate: Callable[[BaseUnit], bool]) -> 'UnitCollection':
        """
        Filter units by custom predicate
        
        Args:
            predicate: A function that takes a unit and returns bool
            
        Returns:
            New UnitCollection with filtered units
        """
        return UnitCollection([u for u in self if predicate(u)])
    
    def get_by_id(self, unit_id: str) -> Optional[BaseUnit]:
        """
        Get unit by ID from this collection
        
        Args:
            unit_id: The unit ID to search for
            
        Returns:
            The unit if found, None otherwise
        """
        for unit in self:
            if unit.unit_id == unit_id:
                return unit
        return None
    
    def get_text_units(self) -> 'UnitCollection':
        """Shortcut to get all text units"""
        return self.filter_by_type("text")
    
    def get_table_units(self) -> 'UnitCollection':
        """Shortcut to get all table units"""
        return self.filter_by_type("table")
    
    def get_image_units(self) -> 'UnitCollection':
        """Shortcut to get all image units"""
        return self.filter_by_type("image")


class BaseDocument(BaseModel, ABC):
    """
    Base class for all document types
    Acts as a container for structured parsed results
    
    Note:
        doc_id is automatically generated if not provided
        metadata is now a structured DocumentMetadata object
    """
    
    doc_id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}")
    metadata: DocumentMetadata
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }
    
    @abstractmethod
    def split(self, splitter: 'BaseSplitter') -> UnitCollection:
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
    """
    
    page_number: int
    content: Any = None
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
