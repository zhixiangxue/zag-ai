"""
Unified reranker implementation
"""

from typing import Any, Optional

from .base import BaseReranker
from .uri import parse_reranker_uri
from .providers import create_provider, get_available_providers
from .exceptions import RerankerError
from ...schemas import BaseUnit


class Reranker(BaseReranker):
    """
    Unified reranker class supporting multiple providers
    
    This is the main reranker implementation that supports different
    providers (local models, remote APIs, etc.) through a unified interface.
    
    Characteristics:
        - Supports multiple providers (sentence_transformers, cohere, jina, etc.)
        - URI-based or parameter-based configuration
        - Can be used standalone or in postprocessor pipelines
    
    Examples:
        >>> from zag.postprocessors import Reranker
        >>> 
        >>> # URI format - simple
        >>> reranker = Reranker("sentence_transformers/cross-encoder/ms-marco-MiniLM-L-12-v2")
        >>> reranked = reranker.rerank(query, units, top_k=10)
        >>> 
        >>> # Parameter format
        >>> reranker = Reranker(
        ...     provider="sentence_transformers",
        ...     model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        ...     device="cuda"
        ... )
        >>> 
        >>> # In a pipeline
        >>> from zag.postprocessors import ChainPostprocessor
        >>> pipeline = ChainPostprocessor([
        ...     SimilarityFilter(threshold=0.7),
        ...     Reranker("sentence_transformers/cross-encoder/ms-marco-MiniLM-L-12-v2"),
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
            uri: Reranker URI string (e.g., "sentence_transformers/cross-encoder-ms-marco")
            provider: Provider name (required if uri is not provided)
            model: Model name (required if uri is not provided)
            **config: Additional configuration (device, batch_size, api_key, etc.)
            
        Examples:
            >>> # URI format
            >>> reranker = Reranker("sentence_transformers/cross-encoder/ms-marco-MiniLM-L-12-v2")
            >>> 
            >>> # Parameter format
            >>> reranker = Reranker(
            ...     provider="sentence_transformers",
            ...     model="cross-encoder/ms-marco-MiniLM-L-12-v2",
            ...     device="cuda",
            ...     batch_size=64
            ... )
        """
        if uri:
            parsed = parse_reranker_uri(uri)
            provider = parsed['provider']
            model = parsed['model']
            
            # Merge URI params with config
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
        
        # Create provider with explicit parameters
        self._provider = create_provider(provider, model, **config)
        self._provider_name = provider
        self._model = model

    def _build_document(self, unit: BaseUnit) -> str:
        """Build reranker document text from a unit.

        Prefixes core structural signals (lender, program/guideline, context
        path, processing mode) before the raw content, so cross-encoder
        rerankers can better capture query intent like "Non-QM foreclosure
        seasoning" instead of only seeing flat text.
        """
        parts: list[str] = []
        metadata = getattr(unit, "metadata", None)

        if metadata is not None:
            # Custom business fields: lender, guideline/program, mode, etc.
            custom = getattr(metadata, "custom", None)
            if isinstance(custom, dict):
                lender = custom.get("lender")
                guideline = custom.get("guideline")
                mode = custom.get("mode")
                if lender:
                    parts.append(f"[Lender: {lender}]")
                if guideline:
                    parts.append(f"[Program: {guideline}]")
                if mode:
                    parts.append(f"[Mode: {mode}]")

            # Hierarchical path in the document (e.g. Credit > Derogatory > Foreclosure)
            context_path = getattr(metadata, "context_path", None)
            if context_path:
                path_str = str(context_path)
                if path_str:
                    parts.append(f"[Path: {path_str}]")

            # Extracted keywords (if available)
            keywords = getattr(metadata, "keywords", None)
            if keywords:
                # Limit number of keywords to keep prompt compact
                kw_preview = ", ".join(keywords[:5])
                parts.append(f"[Keywords: {kw_preview}]")

            # Source document info (file_name is a strong signal for program/guide)
            document_meta = getattr(metadata, "document", None)
            if isinstance(document_meta, dict):
                file_name = document_meta.get("file_name")
                if file_name:
                    parts.append(f"[File: {file_name}]")

        prefix = " ".join(parts)
        content = str(getattr(unit, "content", "") or "")

        if prefix:
            doc = prefix + "\n" + content
        else:
            doc = content

        # Truncate to keep reranker input reasonably bounded
        max_len = 2000
        if len(doc) > max_len:
            return doc[:max_len]
        return doc
    
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
            >>> reranker = Reranker("sentence_transformers/cross-encoder/ms-marco")
            >>> units = [unit1, unit2, unit3]
            >>> reranked = reranker.rerank("machine learning", units, top_k=2)
        """
        if not units:
            return []
        
        # Extract document texts from units (with structural prefix)
        documents = [self._build_document(unit) for unit in units]
        
        # Call provider to get scores
        results = self._provider.rerank(query, documents, top_k=top_k)
        
        # Create a mapping from document text to score
        doc_to_score = {doc: score for doc, score in results}
        
        # Update units with new scores
        reranked_units = []
        for unit in units:
            doc_text = self._build_document(unit)
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
