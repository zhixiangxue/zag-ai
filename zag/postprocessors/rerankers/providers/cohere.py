"""
Cohere reranker provider

Uses Cohere's Rerank API for high-quality reranking.
Official documentation: https://docs.cohere.com/reference/rerank

Supported models:
- rerank-english-v3.0
- rerank-multilingual-v3.0
- rerank-english-v2.0
- rerank-multilingual-v2.0

Requirements:
    pip install cohere
"""

from typing import Optional
from .base import BaseProvider


class CohereProvider(BaseProvider):
    """
    Cohere Rerank API provider
    
    Uses Cohere's Rerank API for document reranking.
    Requires a Cohere API key.
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_chunks_per_doc: Optional[int] = None
    ):
        """
        Initialize Cohere reranker provider
        
        Args:
            model: Model name (e.g., "rerank-english-v3.0", "rerank-multilingual-v3.0")
            api_key: Cohere API key (if not provided, uses COHERE_API_KEY env var)
            base_url: Custom API base URL (optional, for enterprise deployments)
            max_chunks_per_doc: Maximum chunks per document for long texts (optional)
        """
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_chunks_per_doc = max_chunks_per_doc
        self._client = None
        self._async_client = None  # lazy-init AsyncClient for arerank()

    def _load_client(self):
        """Lazy load the Cohere sync client."""
        if self._client is None:
            try:
                import cohere
            except ImportError:
                raise ImportError(
                    "cohere is required for this provider. "
                    "Install it with: pip install cohere"
                )

            kwargs = {}
            if self.api_key:
                kwargs['api_key'] = self.api_key
            if self.base_url:
                kwargs['base_url'] = self.base_url

            self._client = cohere.Client(**kwargs)

    def _load_async_client(self):
        """Lazy load the Cohere async client."""
        if self._async_client is None:
            try:
                import cohere
            except ImportError:
                raise ImportError(
                    "cohere is required for this provider. "
                    "Install it with: pip install cohere"
                )

            kwargs = {}
            if self.api_key:
                kwargs['api_key'] = self.api_key
            if self.base_url:
                kwargs['base_url'] = self.base_url

            self._async_client = cohere.AsyncClient(**kwargs)
    
    def rerank(
        self, 
        query: str, 
        documents: list[str],
        top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """
        Rerank documents using Cohere Rerank API
        
        Args:
            query: The search query
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            List of (document, score) tuples sorted by relevance score
        """
        if not documents:
            return []
        
        # Load client
        self._load_client()
        
        # Prepare rerank request parameters
        kwargs = {
            'query': query,
            'documents': documents,
            'model': self.model_name,
        }
        
        # Add optional parameters
        if top_k is not None:
            kwargs['top_n'] = top_k
        if self.max_chunks_per_doc is not None:
            kwargs['max_chunks_per_doc'] = self.max_chunks_per_doc
        
        # Call Cohere Rerank API
        try:
            response = self._client.rerank(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Cohere rerank API error: {e}")
        
        # Extract results
        results = []
        for result in response.results:
            doc_text = documents[result.index]
            score = result.relevance_score
            results.append((doc_text, score))
        
        # Results are already sorted by relevance from Cohere
        return results

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """Async version of rerank using cohere.AsyncClient.

        Zero threads blocked — pure async HTTP to Cohere API.
        """
        if not documents:
            return []

        self._load_async_client()

        kwargs = {
            'query': query,
            'documents': documents,
            'model': self.model_name,
        }
        if top_k is not None:
            kwargs['top_n'] = top_k
        if self.max_chunks_per_doc is not None:
            kwargs['max_chunks_per_doc'] = self.max_chunks_per_doc

        try:
            response = await self._async_client.rerank(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Cohere async rerank API error: {e}")

        results = []
        for result in response.results:
            doc_text = documents[result.index]
            score = result.relevance_score
            results.append((doc_text, score))

        return results
