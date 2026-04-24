from .base import BaseReader

# Lazy imports for optional readers
# These will only be imported when actually used
__all__ = [
    "BaseReader",
    "CamelotReader",
    "ClaudeVisionReader",
    "DoclingReader",
    "LightOnOCRReader",
    "MarkItDownReader",
    "MinerUReader",
    "PDFPlumberReader",
    "PyMuPDF4LLMReader",
    "TabulaReader",
    "MarkdownTreeReader",
]


def __getattr__(name: str):
    """Lazy import for optional readers"""
    if name == "ClaudeVisionReader":
        from .claude_vision import ClaudeVisionReader

        return ClaudeVisionReader
    elif name == "CamelotReader":
        from .camelot import CamelotReader

        return CamelotReader
    elif name == "DoclingReader":
        from .docling import DoclingReader

        return DoclingReader
    elif name == "LightOnOCRReader":
        from .lightonocr import LightOnOCRReader

        return LightOnOCRReader
    elif name == "MarkItDownReader":
        from .markitdown import MarkItDownReader

        return MarkItDownReader
    elif name == "MinerUReader":
        from .mineru import MinerUReader

        return MinerUReader
    elif name == "PDFPlumberReader":
        from .pdfplumber import PDFPlumberReader

        return PDFPlumberReader
    elif name == "TabulaReader":
        from .tabula import TabulaReader

        return TabulaReader
    elif name == "PyMuPDF4LLMReader":
        from .pymupdf4llm import PyMuPDF4LLMReader

        return PyMuPDF4LLMReader
    elif name == "MarkdownTreeReader":
        from .markdown_tree import MarkdownTreeReader

        return MarkdownTreeReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
