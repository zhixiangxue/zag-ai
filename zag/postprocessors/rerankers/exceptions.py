"""
Reranker exceptions
"""


class RerankerError(Exception):
    """Base exception for reranker errors"""
    pass


class URIError(RerankerError):
    """Raised when URI parsing fails"""
    pass


class ProviderError(RerankerError):
    """Raised when provider operations fail"""
    pass


class ConfigurationError(RerankerError):
    """Raised when configuration is invalid"""
    pass
