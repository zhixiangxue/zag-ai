"""
Type definitions for schemas module

This module contains all enum types used across the schema system.
"""

from enum import Enum


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


class RetrievalSource(str, Enum):
    """Retrieval source types - indicates which storage system the unit came from"""
    
    VECTOR = "vector"        # From vector database (e.g., ChromaDB)
    FULLTEXT = "fulltext"    # From fulltext search engine (e.g., Meilisearch)
