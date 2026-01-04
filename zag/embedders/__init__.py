"""Embedders module - unified text embedding interface"""

from .embedder import Embedder
from .providers import get_available_providers
from .exceptions import EmbedderError, URIError, ProviderError, ConfigError

__all__ = [
    'Embedder',
    'get_available_providers',
    'EmbedderError',
    'URIError',
    'ProviderError',
    'ConfigError',
]
