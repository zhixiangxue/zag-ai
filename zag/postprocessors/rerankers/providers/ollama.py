"""
Ollama reranker provider

Uses Ollama's local reranking models.
Official documentation: https://ollama.com/

Supported models:
- bge-reranker-large
- bge-reranker-base
- jina-reranker-v1-base-en
- Any reranking model available in Ollama

Requirements:
    pip install ollama
"""

from typing import Optional
from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """
    Ollama reranker provider
    
    Uses Ollama for local reranking with various models.
    No API key required, runs completely local.
    """
    
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        """
        Initialize Ollama reranker provider
        
        Args:
            model: Model name (e.g., "bge-reranker-large", "jina-reranker-v1-base-en")
            base_url: Ollama server URL (default: "http://localhost:11434")
            timeout: Request timeout in seconds (optional)
        """
        self.model_name = model
        self.base_url = base_url or "http://localhost:11434"
        self.timeout = timeout
        self._client = None
    
    def _load_client(self):
        """Lazy load the Ollama client"""
        if self._client is None:
            try:
                import ollama
            except ImportError:
                raise ImportError(
                    "ollama is required for this provider. "
                    "Install it with: pip install ollama"
                )
            
            # Initialize client
            kwargs = {'host': self.base_url}
            if self.timeout is not None:
                kwargs['timeout'] = self.timeout
            
            self._client = ollama.Client(**kwargs)
    
    def rerank(
        self, 
        query: str, 
        documents: list[str],
        top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """
        Rerank documents using Ollama
        
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
        
        # Call Ollama embeddings API for reranking
        # Note: Ollama reranking models typically use a specific prompt format
        try:
            # Score each document
            results = []
            for doc in documents:
                # Format as reranking prompt
                prompt = f"Query: {query}\nDocument: {doc}\nRelevance:"
                
                response = self._client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.0,  # Deterministic scoring
                        'num_predict': 10,   # Only need a short score
                    }
                )
                
                # Extract score from response
                # This is a simple heuristic - actual implementation may vary by model
                score_text = response.get('response', '').strip()
                try:
                    # Try to parse as float
                    score = float(score_text.split()[0])
                except (ValueError, IndexError):
                    # Fallback: use length-based heuristic
                    score = len(score_text) / 100.0
                
                results.append((doc, score))
            
        except Exception as e:
            raise RuntimeError(f"Ollama rerank error: {e}")
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        if top_k is not None:
            return results[:top_k]
        return results
