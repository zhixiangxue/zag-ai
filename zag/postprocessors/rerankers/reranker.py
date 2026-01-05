"""
Unified reranker implementation
"""

from typing import Any, Optional

from .base import BaseReranker
from .uri import parse_reranker_uri
from .providers import create_provider, get_available_providers
from .exceptions import RerankerError
from ...schemas.base import BaseUnit


class Reranker(BaseReranker):
    """
    Unified reranker class supporting multiple providers
    
    This is the main reranker implementation that supports different
    providers (local models, remote APIs, etc.) through a unified interface.
    
    Characteristics:
        - Supports multiple providers (local, cohere, jina, etc.)
        - URI-based or parameter-based configuration
        - Can be used standalone or in postprocessor pipelines
    
    Examples:
        >>> from zag.postprocessors import Reranker
        >>> 
        >>> # URI format - simple
        >>> reranker = Reranker("local/cross-encoder/ms-marco-MiniLM-L-12-v2")
        >>> reranked = reranker.rerank(query, units, top_k=10)
        >>> 
        >>> # Parameter format
        >>> reranker = Reranker(
        ...     provider="local",
        ...     model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        ...     device="cuda"
        ... )
        >>> 
        >>> # In a pipeline
        >>> from zag.postprocessors import ChainPostprocessor
        >>> pipeline = ChainPostprocessor([
        ...     SimilarityFilter(threshold=0.7),
        ...     Reranker("local/cross-encoder/ms-marco-MiniLM-L-12-v2"),
        ... ])
    """
    
    def __init__(
        self,
        uri: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        **config: Any
    ):
        """
        Initialize the reranker
        
        Args:
            uri: Reranker URI string (e.g., "local/cross-encoder-ms-marco")
            provider: Provider name (required if uri is not provided)
            model: Model name (required if uri is not provided)
            **config: Additional configuration (device, batch_size, api_key, etc.)
            
        Examples:
            >>> # URI format
            >>> reranker = Reranker("local/cross-encoder/ms-marco-MiniLM-L-12-v2")
            >>> 
            >>> # Parameter format
            >>> reranker = Reranker(
            ...     provider="local",
            ...     model="cross-encoder/ms-marco-MiniLM-L-12-v2",
            ...     device="cuda",
            ...     batch_size=64
            ... )
        """
        if uri:
            parsed = parse_reranker_uri(uri)
            provider = parsed['provider']
            model = parsed['model']
            
            uri_params = parsed.get('params', {})
            if parsed.get('base_url'):
                uri_params['base_url'] = parsed['base_url']
            
            config = {**uri_params, **config}
        
        if not provider or not model:
            available = get_available_providers()
            raise ValueError(
                "Must provide either 'uri' or both 'provider' and 'model'. "
                f"Available providers: {', '.join(available)}"
            )
        
        config['model'] = model
        
        self._provider = create_provider(provider, config)
        self._provider_name = provider
        self._model = model
    
    def rerank(
        self, 
        query: str,
        units: list[BaseUnit],
        top_k: Optional[int] = None
    ) -> list[BaseUnit]:
        """
        Rerank units by recomputing relevance scores
        
        Args:
            query: Original query text
            units: Units to rerank
            top_k: Maximum number of results to return (None = all)
            
        Returns:
            Reranked units, sorted by new relevance scores (descending)
            
        Example:
            >>> reranker = Reranker("local/cross-encoder/ms-marco")
            >>> units = [unit1, unit2, unit3]
            >>> reranked = reranker.rerank("machine learning", units, top_k=2)
        """
        if not units:
            return []
        
        # Extract document texts from units
        documents = [str(unit.content) for unit in units]
        
        # Call provider to get scores
        results = self._provider.rerank(query, documents, top_k=top_k)
        
        # Create a mapping from document text to score
        doc_to_score = {doc: score for doc, score in results}
        
        # Update units with new scores
        reranked_units = []
        for unit in units:
            doc_text = str(unit.content)
            if doc_text in doc_to_score:
                # Create a copy to avoid modifying original
                unit_copy = unit.model_copy()
                unit_copy.score = doc_to_score[doc_text]
                reranked_units.append(unit_copy)
        
        # Sort by score (descending)
        reranked_units.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k results
        if top_k is not None:
            return reranked_units[:top_k]
        return reranked_units
    
    @property
    def provider(self) -> str:
        """Get the provider name"""
        return self._provider_name
    
    @property
    def model(self) -> str:
        """Get the model name"""
        return self._model
    
    def __repr__(self) -> str:
        return f"Reranker(provider='{self._provider_name}', model='{self._model}')"
