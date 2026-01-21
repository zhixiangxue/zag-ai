"""
Bailian (DashScope) Embedder Provider

Bailian is Alibaba Cloud's LLM platform, using OpenAI-compatible API.
Official documentation: https://help.aliyun.com/zh/model-studio/

Supported embedding models:
- text-embedding-v3
- text-embedding-v2
- text-embedding-v1
"""

from typing import Optional
from pydantic import BaseModel, field_validator

from . import register_provider
from .base import BaseProvider

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class BailianProviderConfig(BaseModel):
    """Configuration for Bailian embedder provider"""
    
    api_key: str
    model: str
    base_url: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout: int = 30
    max_retries: int = 3
    dimensions: Optional[int] = None  # Optional dimension parameter
    
    class Config:
        extra = "allow"  # Allow additional parameters
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("API key cannot be empty")
        return v
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model cannot be empty")
        return v
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Bailian (DashScope compatible mode)"""
        return v or "https://dashscope.aliyuncs.com/compatible-mode/v1"


class BailianProvider(BaseProvider):
    """
    Bailian embedder implementation using OpenAI-compatible API
    
    This provider uses the OpenAI Python SDK to communicate with
    Bailian's DashScope API in compatibility mode.
    """
    
    def __init__(self, config: BailianProviderConfig):
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI SDK is required for Bailian provider. "
                "Install it with: pip install openai"
            )
        
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        self._dimension_cache = config.dimensions
    
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text using Bailian API
        
        Args:
            text: Input text to embed
            
        Returns:
            Vector representation of the text
        """
        response = self._client.embeddings.create(
            model=self.config.model,
            input=text,
            dimensions=self.config.dimensions
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in batch using Bailian API
        
        Bailian API has a batch size limit of 10. This method automatically
        splits large batches into chunks and processes them sequentially.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of vector representations
        """
        # Bailian API batch size limit
        BATCH_SIZE = 10
        
        if len(texts) <= BATCH_SIZE:
            # Single batch
            response = self._client.embeddings.create(
                model=self.config.model,
                input=texts,
                dimensions=self.config.dimensions
            )
            return [item.embedding for item in response.data]
        
        # Multiple batches - split and process
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self.config.model,
                input=batch,
                dimensions=self.config.dimensions
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension
        
        If dimensions parameter is set in config, use it.
        Otherwise, make a test call to determine the dimension.
        
        Returns:
            Dimension of the embedding vectors
        """
        if self._dimension_cache is None:
            # Make a test call to get dimension
            sample_vec = self.embed_text("dimension test")
            self._dimension_cache = len(sample_vec)
        return self._dimension_cache


# Register the Bailian provider factory
@register_provider("bailian")
def create_bailian_provider(**config) -> BaseProvider:
    """
    Factory function for creating Bailian provider instances
    
    Args:
        **config: Configuration parameters (api_key, model, base_url, etc.)
        
    Returns:
        BailianProvider instance
        
    Example:
        >>> provider = create_bailian_provider(
        ...     api_key="sk-xxx",
        ...     model="text-embedding-v3",
        ...     dimensions=1024
        ... )
    """
    validated_config = BailianProviderConfig(**config)
    return BailianProvider(validated_config)
