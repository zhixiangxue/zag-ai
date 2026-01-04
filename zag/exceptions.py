"""
Zag framework exceptions

Base exception hierarchy for all zag modules
"""


class ZagError(Exception):
    """
    Base exception for all zag framework errors
    
    All module-specific exceptions should inherit from this class
    to maintain a consistent exception hierarchy across the framework.
    """
    pass


class ConfigurationError(ZagError):
    """Configuration or initialization error"""
    pass


class ValidationError(ZagError):
    """Data validation error"""
    pass


class ProcessingError(ZagError):
    """Data processing error"""
    pass


class ResourceError(ZagError):
    """Resource access or management error"""
    pass
