"""
Zag AI - RAG Framework
"""

import logging
from rich.console import Console
from rich.logging import RichHandler

__version__ = "0.1.0"

# Configure rich logging for better UX with progress bars
_console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(
        console=_console,
        rich_tracebacks=True,
        show_time=True,
        show_path=False
    )]
)

# Reduce verbosity for HTTP requests to avoid interfering with progress bars
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Exceptions
from .exceptions import (
    ZagError,
    ConfigurationError,
    ValidationError,
    ProcessingError,
    ResourceError,
)

# Core schemas
from .schemas.base import (
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
from .schemas.unit import TextUnit, TableUnit, ImageUnit
from .schemas.pdf import PDF
from .schemas.markdown import Markdown

# Readers
from .readers.base import BaseReader

# Splitters
from .splitters.base import BaseSplitter

# Extractors
from .extractors.base import BaseExtractor
from .extractors import TableExtractor, StructuredExtractor, KeywordExtractor

# Embedders
from .embedders import Embedder

# Retrievers
from .retrievers import (
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