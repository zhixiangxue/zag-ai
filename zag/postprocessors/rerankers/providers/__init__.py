"""
Reranker provider registry and factory
"""

from typing import Any
from .base import BaseProvider
from .sentence_transformers import SentenceTransformersProvider
from .cohere import CohereProvider
from .ollama import OllamaProvider
from ..exceptions import ProviderError


# Provider registry
_PROVIDERS = {
    "sentence_transformers": SentenceTransformersProvider,
    "cohere": CohereProvider,
    "ollama": OllamaProvider,
}


def create_provider(provider_name: str, model: str, **kwargs: Any) -> BaseProvider:
    """
    Create a reranker provider instance
    
    Args:
        provider_name: Name of the provider ("sentence_transformers")
        model: Model name or path
        **kwargs: Additional provider-specific configuration
        
    Returns:
        Provider instance
        
    Raises:
        ProviderError: If provider is not found
    """
    if provider_name not in _PROVIDERS:
        available = ', '.join(_PROVIDERS.keys())
        raise ProviderError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {available}"
        )
    
    provider_class = _PROVIDERS[provider_name]
    return provider_class(model=model, **kwargs)


def get_available_providers() -> list[str]:
    """
    Get list of available provider names
    
    Returns:
        List of provider names
    """
    return list(_PROVIDERS.keys())


__all__ = [
    "BaseProvider",
    "SentenceTransformersProvider",
    "CohereProvider",
    "OllamaProvider",
    "create_provider",
    "get_available_providers",
]
