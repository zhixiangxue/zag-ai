"""
Composite layer retrievers

Composite retrievers combine multiple retrievers and implement retrieval strategies.
They depend only on BaseRetriever interface, not directly on storage layer.
"""

from .fusion_retriever import QueryFusionRetriever, FusionMode
from .rewrite_retriever import QueryRewriteRetriever

__all__ = [
    "QueryFusionRetriever",
    "FusionMode",
    "QueryRewriteRetriever",
]
