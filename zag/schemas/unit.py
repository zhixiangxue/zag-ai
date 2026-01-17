"""
Unit-related classes

This module contains all unit-related classes including:
- BaseUnit: Base class for all units
- UnitRegistry: Global unit registry for ID-to-object resolution  
- UnitCollection: Collection of units with chainable operations
- TextUnit, TableUnit, ImageUnit: Concrete unit implementations
"""

from typing import Any, Optional, Callable
from uuid import uuid4
from pydantic import BaseModel, Field, model_validator

from .types import UnitType, RelationType, RetrievalSource
from .metadata import UnitMetadata


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
    
    # Retrieval metadata (set by retrievers)
    score: Optional[float] = None
    source: Optional['RetrievalSource'] = None
    
    # Chain relationships (managed by Splitter)
    prev_unit_id: Optional[str] = None
    next_unit_id: Optional[str] = None
    doc_id: Optional[str] = None
    
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
