"""
Provider registration and factory
"""

from typing import Dict, Callable, Any

from .base import BaseProvider

# Global provider registry: provider_name -> factory_function
_PROVIDERS: Dict[str, Callable] = {}


def register_provider(name: str):
    """
    Decorator for registering provider factory functions
    
    Args:
        name: Provider name (e.g., "bailian", "openai")
        
    Example:
        @register_provider("bailian")
        def create_bailian_provider(**config) -> BaseProvider:
            validated_config = BailianProviderConfig(**config)
            return BailianProvider(validated_config)
    """
    def decorator(factory_func: Callable) -> Callable:
        if name.lower() in _PROVIDERS:
            raise ValueError(f"Provider '{name}' is already registered")
        _PROVIDERS[name.lower()] = factory_func
        return factory_func
    return decorator


def create_provider(provider_name: str, config: Dict[str, Any]) -> BaseProvider:
    """
    Create a provider instance using the factory pattern
    
    Args:
        provider_name: Name of the provider (e.g., "bailian", "openai")
        config: Configuration dictionary for the provider
        
    Returns:
        Provider instance implementing BaseProvider
        
    Raises:
        ValueError: If the provider is not registered
    """
    provider_name = provider_name.lower()
    
    if provider_name not in _PROVIDERS:
        available = ', '.join(sorted(_PROVIDERS.keys()))
        raise ValueError(
            f"Provider '{provider_name}' not found. "
            f"Available providers: {available or 'none'}"
        )
    
    factory = _PROVIDERS[provider_name]
    return factory(**config)


def get_available_providers() -> list[str]:
    """
    Get list of all registered providers
    
    Returns:
        Sorted list of provider names
    """
    return sorted(_PROVIDERS.keys())


# Import all provider implementations to trigger registration
# Add new providers here when implemented
try:
    from . import bailian
except ImportError:
    pass  # Provider not yet implemented or dependencies missing


__all__ = [
    'BaseProvider',
    'register_provider', 
    'create_provider', 
    'get_available_providers'
]
