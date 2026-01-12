from .base import BaseReader

# Lazy imports for optional readers
# These will only be imported when actually used
__all__ = [
    "BaseReader",
    "DoclingReader",
    "MarkItDownReader",
    "MinerUReader",
]

def __getattr__(name: str):
    """Lazy import for optional readers"""
    if name == "DoclingReader":
        from .docling import DoclingReader
        return DoclingReader
    elif name == "MarkItDownReader":
        from .markitdown import MarkItDownReader
        return MarkItDownReader
    elif name == "MinerUReader":
        from .mineru import MinerUReader
        return MinerUReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
