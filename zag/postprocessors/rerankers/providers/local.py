"""
Local cross-encoder reranker provider
"""

from typing import Any
from .base import BaseProvider


class LocalProvider(BaseProvider):
    """
    Local cross-encoder reranker using sentence-transformers
    
    Supports running cross-encoder models locally.
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize local cross-encoder provider
        
        Args:
            config: Configuration dict with keys:
                - model: Model name or path (required)
                - device: Computing device ("cuda" or "cpu", None for auto)
                - batch_size: Batch size for inference
        """
        self.model_name = config['model']
        self.device = config.get('device')
        self.batch_size = config.get('batch_size', 32)
        self._model = None
    
    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local reranker. "
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
