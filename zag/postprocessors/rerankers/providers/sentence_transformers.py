"""
Sentence-Transformers cross-encoder reranker provider

Uses sentence-transformers library for local cross-encoder reranking.
Official documentation: https://www.sbert.net/docs/pretrained_cross-encoders.html

Supported models:
- ms-marco-MiniLM-L-12-v2
- ms-marco-TinyBERT-L-2-v2
- cross-encoder/ms-marco-electra-base
- Any cross-encoder model from HuggingFace
"""

from typing import Optional
from .base import BaseProvider


class SentenceTransformersProvider(BaseProvider):
    """
    Sentence-Transformers cross-encoder reranker
    
    Uses CrossEncoder models from sentence-transformers library.
    Runs locally without external API dependencies.
    """
    
    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize sentence-transformers cross-encoder provider
        
        Args:
            model: Model name or path (e.g., "cross-encoder/ms-marco-MiniLM-L-12-v2")
            device: Computing device ("cuda" or "cpu", None for auto)
            batch_size: Batch size for inference (default: 32)
        """
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self._model = None
    
    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for this provider. "
                    "Install it with: pip install sentence-transformers"
                )
            
            self._model = CrossEncoder(self.model_name, device=self.device)
    
    def rerank(
        self, 
        query: str, 
        documents: list[str],
        top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """
        Rerank documents using local cross-encoder
        
        Args:
            query: The search query
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            List of (document, score) tuples sorted by score
        """
        if not documents:
            return []
        
        # Load model
        self._load_model()
        
        # Build query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Compute scores
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        
        # Combine documents with scores
        results = list(zip(documents, scores))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        if top_k is not None:
            return results[:top_k]
        return results
