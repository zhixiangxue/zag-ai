"""
Schemas module - Core data structures
"""

# Import from new modular structure
from .types import UnitType, RelationType, RetrievalSource, ProcessingMode
from .metadata import DocumentMetadata, UnitMetadata
from .document import BaseDocument, Page, PageableDocument
from .unit import BaseUnit, UnitRegistry, UnitCollection, TextUnit, TableUnit, ImageUnit, LODLevel, ContentView
from .tree import TreeNode, DocTree, TreeRetrievalResult

__all__ = [
    # Enums
    "UnitType",
    "RelationType",
    "RetrievalSource",
    "ProcessingMode",
    # Metadata
    "DocumentMetadata",
    "UnitMetadata",
    # Document classes
    "BaseDocument",
    "Page",
    "PageableDocument",
    # Unit classes
    "BaseUnit",
    "UnitRegistry",
    "UnitCollection",
    # Concrete unit types
    "TextUnit",
    "TableUnit",
    "ImageUnit",
    # LOD and views
    "LODLevel",
    "ContentView",
    # Tree schemas
    "TreeNode",
    "DocTree",
    "TreeRetrievalResult",
]