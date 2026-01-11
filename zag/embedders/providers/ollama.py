"""
Ollama Embedder Provider

Ollama is a local LLM platform for running models locally.
Official documentation: https://ollama.ai/

Supported embedding models:
- nomic-embed-text
- mxbai-embed-large
- all-minilm
- etc. (any embedding model available in Ollama)
"""

from typing import Optional
from pydantic import BaseModel, field_validator

from . import register_provider
from .base import BaseProvider

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class OllamaProviderConfig(BaseModel):
    """Configuration for Ollama embedder provider"""
    
    model: str
    base_url: Optional[str] = "http://localhost:11434"
    timeout: int = 60
    max_retries: int = 3
    
    class Config:
        extra = "allow"  # Allow additional parameters
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model cannot be empty")
        return v
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Ollama"""
        return v or "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """
    Ollama embedder implementation using official Ollama Python client
    
    This provider uses the official ollama package to communicate with
    the local Ollama server.
    """
    
    def __init__(self, config: OllamaProviderConfig):
        if not HAS_OLLAMA:
            raise ImportError(
                "Ollama SDK is required for Ollama provider. "
                "Install it with: pip install ollama"
            )
        
        self.config = config
        self._dimension_cache = None
        self._client = ollama.Client(
            host=config.base_url,
            timeout=config.timeout
        )
    
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text using Ollama API
        
        Args:
            text: Input text to embed
            
        Returns:
            Vector representation of the text
        """
        response = self._client.embed(
            model=self.config.model,
            input=text
        )
        return response["embeddings"][0]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts using Ollama API
        
        The official Ollama client supports batch embedding.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of vector representations
        """
        response = self._client.embed(
            model=self.config.model,
            input=texts
        )
        return response["embeddings"]
    
    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension
        
        Makes a test call to determine the dimension if not cached.
        
        Returns:
            Dimension of the embedding vectors
        """
        if self._dimension_cache is None:
            # Make a test call to get dimension
            sample_vec = self.embed_text("dimension test")
            self._dimension_cache = len(sample_vec)
        return self._dimension_cache


# Register the Ollama provider factory
@register_provider("ollama")
def create_ollama_provider(**config) -> BaseProvider:
    """
    Factory function for creating Ollama provider instances
    
    Args:
        **config: Configuration parameters (model, base_url, etc.)
        
    Returns:
        OllamaProvider instance
        
    Example:
        >>> provider = create_ollama_provider(
        ...     model="nomic-embed-text",
        ...     base_url="http://localhost:11434"
        ... )
    """
    validated_config = OllamaProviderConfig(**config)
    return OllamaProvider(validated_config)
