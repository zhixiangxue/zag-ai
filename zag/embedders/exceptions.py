"""
Embedder exceptions
"""

from ..exceptions import ZagError


class EmbedderError(ZagError):
    """Base exception for embedder errors"""
    pass


class URIError(EmbedderError):
    """URI parsing or validation error"""
    pass


class ProviderError(EmbedderError):
    """Provider-specific error"""
    pass


class ConfigError(EmbedderError):
    """Configuration validation error"""
    pass
