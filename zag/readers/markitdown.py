"""
MarkItDown reader for converting various document formats
"""

from pathlib import Path
from typing import Union

from markitdown import MarkItDown

from .base import BaseReader
from ..schemas import BaseDocument, DocumentMetadata
from ..schemas.plain import PlainText
from ..utils.source import SourceUtils, FileType, SourceInfo
from ..utils.hash import calculate_file_hash, calculate_string_hash


class MarkItDownReader(BaseReader):
    """
    Reader using Microsoft's MarkItDown library
    Supports: PDF, DOCX, PPTX, XLSX, HTML, XML, CSV, JSON, ZIP, TXT, MD, etc.
    
    Automatically returns PlainText for every format, because MarkItDown
    discards page boundaries regardless of input type.
    Use MinerU (ClassicProcessor) for paginated PDF structure, or
    DoclingReader for paginated DOCX structure.
    
    Usage:
        reader = MarkItDownReader()
        doc = reader.read("path/to/file.pdf")   # Returns PlainText
        doc = reader.read("file.md")             # Returns PlainText
        units = doc.split(splitter)
    """
    
    @staticmethod
    def _get_supported_types() -> set[FileType]:
        """Get supported file types"""
        return {
            FileType.PDF,
            FileType.MARKDOWN,
            FileType.WORD,
            FileType.POWERPOINT,
            FileType.EXCEL,
            FileType.HTML,
            FileType.XML,
            FileType.JSON,
            FileType.CSV,
            FileType.TEXT,
            FileType.ZIP,
        }
    
    def __init__(self):
        """Initialize MarkItDown reader"""
        self._reader = MarkItDown()
    
    def read(self, source: Union[str, Path]) -> BaseDocument:
        """
        Read and convert a file to appropriate Document type
        
        Args:
            source: File path (str or Path object, relative/absolute) or URL
            
        Returns:
            PlainText document (single synthetic page)
            
        Raises:
            ValueError: If source is invalid or file format is not supported
        """
        # Validate and get complete info in ONE call
        info = SourceUtils.validate(source, check_accessibility=True, timeout=5)
        
        # Check if valid
        if not info.is_valid:
            raise ValueError(info.error_message)
        
        # Check if file type is supported
        if info.file_type not in self._get_supported_types():
            raise ValueError(
                f"Unsupported file type: {info.file_type.value}. "
                f"Supported: {', '.join(t.value for t in self._get_supported_types())}"
            )
        
        # Convert file using MarkItDown
        result = self._reader.convert(source)
        
        # Create appropriate document type
        return self._create_document(info, result.text_content)
    
    def _create_document(self, info: SourceInfo, content: str) -> BaseDocument:
        """
        Create a PlainText document for any input format.

        MarkItDown always returns a single flat string regardless of the source
        format — it discards page boundaries for PDF, DOCX, PPTX, etc.
        Returning PlainText (single synthetic page) honestly reflects what we
        actually have.  Use MinerU (via ClassicProcessor) for paginated PDF, or
        DoclingReader for paginated DOCX.
        """
        metadata = self._build_metadata(info, content)
        return PlainText.from_text(
            content=content,
            doc_id=metadata.md5,
            metadata=metadata,
        )
    
    def _build_metadata(self, info: SourceInfo, content: str) -> DocumentMetadata:
        """
        Build metadata object from SourceInfo
        
        Args:
            info: SourceInfo from validation
            content: Document content
            
        Returns:
            DocumentMetadata object
        """
        # Get file size and calculate hash
        file_size = None
        file_hash = None
        
        if info.source_type.value == "local":
            try:
                file_path = Path(info.source)
                file_size = file_path.stat().st_size
                # Calculate file hash using xxhash
                file_hash = calculate_file_hash(file_path)
            except Exception as e:
                # If hash calculation fails, raise error (md5 is required)
                raise RuntimeError(
                    f"Failed to calculate file hash for {info.source}: {e}"
                )
        else:
            # For URL sources, use URL string hash as document ID
            file_hash = calculate_string_hash(info.source)
        
        return DocumentMetadata(
            source=info.source,
            source_type=info.source_type.value,
            file_type=info.file_type.value if info.file_type else "unknown",
            file_name=Path(info.source).name if info.source_type.value == "local" else None,
            file_size=file_size,
            file_extension=info.file_extension if info.file_extension else None,
            md5=file_hash,  # Required field
            content_length=len(content),
            mime_type=info.mime_type if info.mime_type else None,
            reader_name="MarkItDownReader"
        )
