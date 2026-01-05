"""
Reranker provider registry and factory
"""

from typing import Any
from .base import BaseProvider
from .local import LocalProvider
from ..exceptions import ProviderError


# Provider registry
_PROVIDERS = {
    "local": LocalProvider,
}


def create_provider(provider_name: str, config: dict[str, Any]) -> BaseProvider:
    """
    Create a reranker provider instance
    
    Args:
        provider_name: Name of the provider (e.g., "local", "cohere")
        config: Configuration dictionary
        
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
    return provider_class(config)


def get_available_providers() -> list[str]:
    """
    Get list of available provider names
    
    Returns:
        List of provider names
    """
    return list(_PROVIDERS.keys())


__all__ = [
    "BaseProvider",
    "LocalProvider",
    "create_provider",
    "get_available_providers",
]
