"""
Query rewrite retriever - decorator that applies a query transformation before retrieval.
"""

from typing import Any, Callable, Optional

from ..base import BaseRetriever
from ...schemas import BaseUnit


class QueryRewriteRetriever(BaseRetriever):
    """
    Query rewrite retriever (Composite Layer - Decorator)

    Wraps any BaseRetriever and applies a query transformation function
    before delegating to the inner retriever. The inner retriever is
    completely unaware of the rewrite.

    Typical use cases:
        - BM25 keyword extraction before full-text search
        - Query expansion / translation before vector search
        - Any scenario where different retrievers need different query forms

    Design:
        - Pure decorator: adds one responsibility, delegates everything else
        - Zero coupling to inner retriever type or storage backend
        - Sync and async paths both supported

    Examples:
        >>> from zag.retrievers import FullTextRetriever, QueryRewriteRetriever
        >>>
        >>> def to_keywords(query: str) -> str:
        ...     # e.g., call an LLM to extract BM25 keywords
        ...     return extract_keywords(query)
        >>>
        >>> retriever = QueryRewriteRetriever(
        ...     retriever=FullTextRetriever(url="http://localhost:7700", index_name="docs"),
        ...     rewrite_fn=to_keywords,
        ... )
        >>> units = retriever.retrieve("What are the income requirements for DSCR loans?")
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        rewrite_fn: Callable[[str], str],
    ):
        """
        Initialize query rewrite retriever.

        Args:
            retriever: Inner retriever to delegate search to.
            rewrite_fn: Callable that transforms the query string before retrieval.
                        Receives the original query and returns the rewritten query.
        """
        self.retriever = retriever
        self.rewrite_fn = rewrite_fn

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[BaseUnit]:
        """Apply rewrite then delegate to inner retriever."""
        rewritten = self.rewrite_fn(query)
        return self.retriever.retrieve(rewritten, top_k=top_k, filters=filters)

    async def aretrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[BaseUnit]:
        """Async version: apply rewrite then delegate to inner retriever."""
        rewritten = self.rewrite_fn(query)
        return await self.retriever.aretrieve(rewritten, top_k=top_k, filters=filters)

    def __repr__(self) -> str:
        return f"QueryRewriteRetriever(retriever={self.retriever!r})"
