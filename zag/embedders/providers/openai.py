"""
OpenAI Embedder Provider

OpenAI provides state-of-the-art embedding models via API.
Official documentation: https://platform.openai.com/docs/guides/embeddings

Supported models:
- text-embedding-3-small (1536 dimensions, cost-effective)
- text-embedding-3-large (3072 dimensions, best quality)
- text-embedding-ada-002 (1536 dimensions, legacy)

Features:
- Batch embedding support (up to 2048 inputs per request)
- Dimension reduction support (text-embedding-3 models)
- Automatic retry with exponential backoff
"""

import os
import logging
from typing import Optional
from pydantic import BaseModel, field_validator, model_validator

from . import register_provider
from .base import BaseProvider

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

logger = logging.getLogger(__name__)


class OpenAIProviderConfig(BaseModel):
    """Configuration for OpenAI embedder provider"""
    
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    dimensions: Optional[int] = None  # Only for text-embedding-3 models
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
    
    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key(cls, v):
        """Get API key from config or environment variable"""
        if v:
            return v
        # Try to get from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set it via config or OPENAI_API_KEY environment variable."
            )
        return api_key
    
    @model_validator(mode='after')
    def validate_dimensions(self):
        """Validate dimensions parameter for text-embedding-3 models"""
        if self.dimensions is not None:
            if not self.model.startswith('text-embedding-3'):
                raise ValueError(
                    f"Dimension reduction is only supported for text-embedding-3 models. "
                    f"Got model: {self.model}"
                )
            if self.dimensions <= 0:
                raise ValueError(f"Dimensions must be positive, got: {self.dimensions}")
        return self


class OpenAIProvider(BaseProvider):
    """
    OpenAI embedder implementation using official OpenAI Python client
    
    This provider uses the official openai package to communicate with
    OpenAI's embedding API.
    """
    
    def __init__(self, config: OpenAIProviderConfig):
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI SDK is required for OpenAI provider. "
                "Install it with: pip install openai"
            )
        
        self.config = config
        self._dimension_cache = None
        self._tokenizer = None
        
        # Initialize tiktoken encoder if available
        if HAS_TIKTOKEN:
            try:
                # Use cl100k_base encoding for text-embedding-3 models
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}. Token counting will use estimation.")
        
        # Initialize OpenAI client
        client_kwargs = {
            "api_key": config.api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }
        
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        
        if config.organization:
            client_kwargs["organization"] = config.organization
        
        self._client = OpenAI(**client_kwargs)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or estimation"""
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        # Fallback: rough estimation (1 token â‰?4 characters)
        return len(text) // 4
    
    def _create_token_aware_batches(self, texts: list[str], max_tokens: int = 250000, max_inputs: int = 2048) -> list[list[str]]:
        """
        Split texts into batches based on both token count and input count
        
        Args:
            texts: List of texts to batch
            max_tokens: Maximum tokens per batch (default: 250,000 for safety)
            max_inputs: Maximum number of inputs per batch (default: 2,048)
        
        Returns:
            List of batches, where each batch is a list of texts
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            token_count = self._count_tokens(text)
            
            # Check if adding this text would exceed limits
            would_exceed_tokens = current_tokens + token_count > max_tokens
            would_exceed_inputs = len(current_batch) >= max_inputs
            
            if current_batch and (would_exceed_tokens or would_exceed_inputs):
                # Start a new batch
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = token_count
            else:
                # Add to current batch
                current_batch.append(text)
                current_tokens += token_count
        
        # Add the last batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text using OpenAI API
        
        Args:
            text: Input text to embed
            
        Returns:
            Vector representation of the text
            
        Raises:
            APIError: If API request fails
            RateLimitError: If rate limit is exceeded
        """
        # Prepare request parameters
        request_params = {
            "model": self.config.model,
            "input": text,
        }
        
        if self.config.dimensions is not None:
            request_params["dimensions"] = self.config.dimensions
        
        try:
            response = self._client.embeddings.create(**request_params)
            return response.data[0].embedding
        except (APIError, RateLimitError, APIConnectionError) as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts using OpenAI API with token-aware batching
        
        Automatically splits large batches to respect both:
        - Input count limit: 2,048 inputs per request
        - Token limit: 250,000 tokens per request (with safety buffer)
        
        Args:
            texts: List of input texts
            
        Returns:
            List of vector representations
            
        Raises:
            APIError: If API request fails
            RateLimitError: If rate limit is exceeded
        """
        if not texts:
            return []
        
        # Create token-aware batches
        batches = self._create_token_aware_batches(texts, max_tokens=250000, max_inputs=2048)
        
        if len(batches) > 1:
            logger.info(f"Split {len(texts)} texts into {len(batches)} batches to respect token limits")
        
        all_embeddings = []
        
        for batch_idx, batch in enumerate(batches, 1):
            # Prepare request parameters
            request_params = {
                "model": self.config.model,
                "input": batch,
            }
            
            if self.config.dimensions is not None:
                request_params["dimensions"] = self.config.dimensions
            
            try:
                logger.debug(f"Processing batch {batch_idx}/{len(batches)} with {len(batch)} texts")
                response = self._client.embeddings.create(**request_params)
                # Extract embeddings in correct order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except (APIError, RateLimitError, APIConnectionError) as e:
                logger.error(f"Batch {batch_idx}/{len(batches)} failed: {e}")
                raise RuntimeError(f"OpenAI API error: {e}")
        
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension
        
        For text-embedding-3 models with custom dimensions, returns the configured value.
        Otherwise, makes a test call to determine the dimension.
        
        Returns:
            Dimension of the embedding vectors
        """
        if self._dimension_cache is None:
            # If dimensions is explicitly set, use it
            if self.config.dimensions is not None:
                self._dimension_cache = self.config.dimensions
            else:
                # Make a test call to get dimension
                sample_vec = self.embed_text("dimension test")
                self._dimension_cache = len(sample_vec)
        return self._dimension_cache


# Register the OpenAI provider factory
@register_provider("openai")
def create_openai_provider(**config) -> BaseProvider:
    """
    Factory function for creating OpenAI provider instances
    
    Args:
        **config: Configuration parameters
                 - model: Model name (default: "text-embedding-3-small")
                 - api_key: OpenAI API key (or set OPENAI_API_KEY env var)
                 - base_url: Custom API endpoint (optional, for Azure OpenAI)
                 - organization: OpenAI organization ID (optional)
                 - dimensions: Output dimension (optional, text-embedding-3 only)
                 - timeout: Request timeout in seconds (default: 60)
                 - max_retries: Maximum retry attempts (default: 3)
        
    Returns:
        OpenAIProvider instance
        
    Examples:
        >>> # Basic usage (API key from environment)
        >>> provider = create_openai_provider(
        ...     model="text-embedding-3-small"
        ... )
        >>> 
        >>> # With explicit API key
        >>> provider = create_openai_provider(
        ...     model="text-embedding-3-large",
        ...     api_key="sk-..."
        ... )
        >>> 
        >>> # With dimension reduction
        >>> provider = create_openai_provider(
        ...     model="text-embedding-3-small",
        ...     dimensions=512  # Reduce from 1536 to 512
        ... )
        >>> 
        >>> # Azure OpenAI
        >>> provider = create_openai_provider(
        ...     model="text-embedding-3-small",
        ...     api_key="your-azure-key",
        ...     base_url="https://your-resource.openai.azure.com/"
        ... )
    """
    validated_config = OpenAIProviderConfig(**config)
    return OpenAIProvider(validated_config)
