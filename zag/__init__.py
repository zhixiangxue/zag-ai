"""
Zag AI - RAG Framework
"""

__version__ = "0.1.0"

# Exceptions
from zag.exceptions import (
    ZagError,
    ConfigurationError,
    ValidationError,
    ProcessingError,
    ResourceError,
)

# Core schemas
from zag.schemas.base import (
    BaseUnit, 
    UnitRegistry, 
    RelationType,
    DocumentMetadata,
    UnitMetadata,
    UnitCollection, 
    BaseDocument,
    Page,
    PageableDocument,
)
from zag.schemas.unit import TextUnit, TableUnit, ImageUnit
from zag.schemas.pdf import PDF
from zag.schemas.markdown import Markdown

# Readers
from zag.readers.base import BaseReader

# Splitters
from zag.splitters.base import BaseSplitter

# Extractors
from zag.extractors.base import BaseExtractor
from zag.extractors import TableExtractor, StructuredExtractor, KeywordExtractor

# Embedders
from zag.embedders import Embedder

# Retrievers
from zag.retrievers import (
    BaseRetriever,
    VectorRetriever,
    QueryFusionRetriever,
    FusionMode,
)

__all__ = [
    # Version
    "__version__",
    # Exceptions
    "ZagError",
    "ConfigurationError",
    "ValidationError",
    "ProcessingError",
    "ResourceError",
    # Schemas
    "BaseUnit",
    "UnitRegistry",
    "RelationType",
    "DocumentMetadata",
    "UnitMetadata",
    "TextUnit",
    "TableUnit",
    "ImageUnit",
    "BaseDocument",
    "Page",
    "PageableDocument",
    "UnitCollection",
    "PDF",
    "Markdown",
    # Readers
    "BaseReader",
    # Splitters
    "BaseSplitter",
    # Extractors
    "BaseExtractor",
    "TableExtractor",
    "StructuredExtractor",
    "KeywordExtractor",
    # Embedders
    "Embedder",
    # Retrievers
    "BaseRetriever",
    "VectorRetriever",
    "QueryFusionRetriever",
    "FusionMode",
]