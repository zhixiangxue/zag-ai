"""
Unified embedder entry point
"""

import asyncio
from typing import Any

from cachetools import LRUCache

from .uri import parse_embedder_uri
from .providers import create_provider, get_available_providers
from .exceptions import EmbedderError


class Embedder:
    """
    Unified embedder class - the main entry point for all embedding operations
    
    This class provides a simple, consistent interface for text embedding
    across different providers. It supports both URI-based configuration
    (similar to database connection strings) and traditional parameter-based
    initialization.
    
    Examples:
        # URI format - simple
        >>> embedder = Embedder("bailian/text-embedding-v3", api_key="sk-xxx")
        >>> vec = embedder.embed("hello world")
        >>> print(f"Dimension: {embedder.dimension}")
        
        # URI format - full (with custom base_url and parameters)
        >>> embedder = Embedder(
        ...     "bailian@https://dashscope.aliyuncs.com/compatible-mode/v1:text-embedding-v3?dimensions=512",
        ...     api_key="sk-xxx"
        ... )
        
        # Parameter format (alternative)
        >>> embedder = Embedder(
        ...     provider="bailian",
        ...     model="text-embedding-v3",
        ...     api_key="sk-xxx",
        ...     dimensions=1024
        ... )
        
        # Batch embedding
        >>> texts = ["hello", "world", "test"]
        >>> vectors = embedder.embed_batch(texts)
        >>> print(f"Embedded {len(vectors)} texts")
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
        Initialize the embedder
        
        Args:
            uri: Embedder URI string in one of two formats:
                 - Simple: "provider/model" (e.g., "bailian/text-embedding-v3")
                 - Full: "provider@base_url:model?params" 
                         (e.g., "bailian@~:text-embedding-v3?dimensions=1024")
            provider: Provider name (required if uri is not provided)
            model: Model name (required if uri is not provided)
            **config: Additional configuration parameters including:
                     - api_key: API key for the provider (usually required)
                     - base_url: Custom base URL (optional, provider-specific default used if not set)
                     - timeout: Request timeout in seconds (default: 30)
                     - dimensions: Embedding dimension (optional, model-specific)
                     - And other provider-specific parameters
        
        Raises:
            ValueError: If neither uri nor (provider, model) is provided
            EmbedderError: If provider is not found or configuration is invalid
        """
        if uri:
            # Parse URI to extract provider, model, base_url, and parameters
            parsed = parse_embedder_uri(uri)
            provider = parsed['provider']
            model = parsed['model']
            
            # Extract parameters from URI
            uri_params = parsed.get('params', {})
            
            # Handle base_url specially
            if parsed.get('base_url'):
                uri_params['base_url'] = parsed['base_url']
            
            # Merge configurations (config parameter takes precedence over URI params)
            config = {**uri_params, **config}
        
        # Validate that we have both provider and model
        if not provider or not model:
            available = get_available_providers()
            raise ValueError(
                "Must provide either 'uri' or both 'provider' and 'model'. "
                f"Available providers: {', '.join(available)}"
            )
        
        # Add model to configuration
        config['model'] = model
        
        # Create the underlying provider instance
        self._provider = create_provider(provider, config)
        self._provider_name = provider
        self._model = model
        # LRU cache for embedding vectors.
        # Key: raw text string. Value: embedding vector (list[float]).
        # Deterministic embeddings make caching safe; maxsize=1024 limits memory
        # to ~12 MB for 3072-dim float32 vectors.
        # cachetools.LRUCache is not thread-safe, but GIL-protected dict operations
        # are atomic enough for our concurrent-async use case (idempotent on miss).
        self._vector_cache: LRUCache = LRUCache(maxsize=1024)
        # Limits concurrent async embed calls to prevent burst 429 from OpenAI.
        # Lazy-init in aembed() so it is always created inside the event loop.
        self._embed_semaphore: asyncio.Semaphore | None = None
    
    def embed(self, text: str) -> list[float]:
        """
        Embed a single text into a vector
        
        Args:
            text: Input text to embed
            
        Returns:
            Vector representation as a list of floats
            
        Example:
            >>> embedder = Embedder("bailian/text-embedding-v3", api_key="sk-xxx")
            >>> vec = embedder.embed("hello world")
            >>> print(len(vec))  # e.g., 1536
        """
        cached = self._vector_cache.get(text)
        if cached is not None:
            return cached
        vector = self._provider.embed_text(text)
        self._vector_cache[text] = vector
        return vector
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in batch

        Batch embedding is typically more efficient than calling embed()
        multiple times, as it reduces API calls and network overhead.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of vector representations, one for each input text
            
        Example:
            >>> embedder = Embedder("bailian/text-embedding-v3", api_key="sk-xxx")
            >>> texts = ["hello", "world", "test"]
            >>> vectors = embedder.embed_batch(texts)
            >>> print(len(vectors))  # 3
        """
        return self._provider.embed_batch(texts)
    
    async def aembed(self, text: str) -> list[float]:
        """
        Async version of embed.

        Uses provider's native async method if available (e.g. OpenAI AsyncOpenAI),
        otherwise falls back to running the sync embed in a thread pool executor.
        This ensures zero threads are blocked for providers that support true async.

        Args:
            text: Input text to embed

        Returns:
            Vector representation as a list of floats
        """
        cached = self._vector_cache.get(text)
        if cached is not None:
            return cached
        if self._embed_semaphore is None:
            # Limit concurrent async embedding calls to stay under OpenAI rate limit.
            # Formula: RPM_limit / (60 / avg_latency_s) / num_workers
            # Example (10000 RPM, ~1.2s/call, 4 workers): 10000/50/4 = 50
            # Adjust if RPM tier or worker count changes.
            self._embed_semaphore = asyncio.Semaphore(50)
        async with self._embed_semaphore:
            if hasattr(self._provider, 'aembed_text'):
                vector = await self._provider.aembed_text(text)
            else:
                # Fallback: run sync embed in thread pool (non-async providers)
                loop = asyncio.get_running_loop()
                vector = await loop.run_in_executor(None, self.embed, text)
        self._vector_cache[text] = vector
        return vector
    
    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension
        
        Returns:
            Dimension of the embedding vectors produced by this embedder
            
        Example:
            >>> embedder = Embedder("bailian/text-embedding-v3", api_key="sk-xxx")
            >>> print(embedder.dimension)  # e.g., 1536
        """
        return self._provider.dimension
    
    @property
    def provider(self) -> str:
        """
        Get the provider name
        
        Returns:
            Name of the provider (e.g., "bailian", "openai")
        """
        return self._provider_name
    
    @property
    def model(self) -> str:
        """
        Get the model name
        
        Returns:
            Name of the model (e.g., "text-embedding-v3")
        """
        return self._model
    
    def __repr__(self) -> str:
        """
        String representation of the embedder
        
        Returns:
            Human-readable string describing the embedder
        """
        return f"Embedder(provider='{self._provider_name}', model='{self._model}', dimension={self.dimension})"
