"""
Retriever module for semantic search and retrieval

Architecture:
    - base.py: BaseRetriever abstract class
    - basic/: Basic layer retrievers (encapsulate storage calls)
    - composite/: Composite layer retrievers (combine multiple retrievers)
    - tree/: Tree-based retrievers operating on DocTree/TreeNode
"""

from .base import BaseRetriever

# Basic layer retrievers
from .basic import VectorRetriever, FullTextRetriever

# Composite layer retrievers
from .composite import QueryFusionRetriever, FusionMode

# Tree-based retrievers
from .tree import SimpleRetriever, MCTSRetriever, TreeRetrievalResult

__all__ = [
    # Base
    "BaseRetriever",
    # Basic layer
    "VectorRetriever",
    "FullTextRetriever",
    # Composite layer
    "QueryFusionRetriever",
    "FusionMode",
    # Tree layer
    "SimpleRetriever",
    "MCTSRetriever",
    "TreeRetrievalResult",
]
