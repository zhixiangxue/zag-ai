"""
Postprocessors module for optimizing retrieval results

Architecture:
    - base.py: BasePostprocessor abstract class
    - rerankers/: Reranking postprocessors (recompute relevance scores)
    - filters/: Filtering postprocessors (remove unwanted results)
    - augmentors/: Augmentation postprocessors (add context)
    - compressors/: Compression postprocessors (reduce content)
    - composite/: Composite postprocessors (orchestrate multiple processors)
"""

from .base import BasePostprocessor

# Rerankers
from .rerankers import (
    BaseReranker,
    Reranker,
)

# Filters
from .filters import (
    SimilarityFilter,
    Deduplicator,
)

# Augmentors
from .augmentors import (
    ContextAugmentor,
)

# Compressors
from .compressors import (
    TokenCompressor,
)

# Composite
from .composite import (
    ChainPostprocessor,
    ConditionalPostprocessor,
)

__all__ = [
    # Base
    "BasePostprocessor",
    # Rerankers
    "BaseReranker",
    "Reranker",
    # Filters
    "SimilarityFilter",
    "Deduplicator",
    # Augmentors
    "ContextAugmentor",
    # Compressors
    "TokenCompressor",
    # Composite
    "ChainPostprocessor",
    "ConditionalPostprocessor",
]
