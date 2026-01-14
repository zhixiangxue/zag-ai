"""
Postprocessors module for optimizing retrieval results

Architecture:
    - base.py: BasePostprocessor abstract class and PostprocessorPipeline
    - rerankers/: Reranking postprocessors (recompute relevance scores)
    - filters/: Filtering postprocessors (remove unwanted results)
    - augmentors/: Augmentation postprocessors (add context)
    - compressors/: Compression postprocessors (reduce content)
    - selectors/: Selection postprocessors (extract relevant passages)

Usage:
    >>> # Chain postprocessors using | operator
    >>> pipeline = Reranker(...) | SimilarityFilter(0.7) | TokenCompressor(4000)
    >>> result = pipeline.process(query, units)
"""

from .base import BasePostprocessor, PostprocessorPipeline

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

# Selectors
from .selectors import (
    LLMSelector,
)

__all__ = [
    # Base
    "BasePostprocessor",
    "PostprocessorPipeline",
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
    # Selectors
    "LLMSelector",
]
