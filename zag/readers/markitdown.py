"""
MarkItDown reader for converting various document formats
"""

from pathlib import Path

from markitdown import MarkItDown

from .base import BaseReader
from ..schemas import BaseDocument, DocumentMetadata
from ..schemas.pdf import PDF
from ..schemas.markdown import Markdown
from ..utils.source import SourceUtils, FileType, SourceInfo
from ..utils.hash import calculate_file_hash, calculate_string_hash


class MarkItDownReader(BaseReader):
    """
    Reader using Microsoft's MarkItDown library
    Supports: PDF, DOCX, PPTX, XLSX, HTML, XML, CSV, JSON, ZIP, TXT, MD, etc.
    
    Automatically returns appropriate document type:
    - PDF files → PDF document
    - Other formats → Markdown document
    
    Usage:
        reader = MarkItDownReader()
        doc = reader.read("path/to/file.pdf")  # Returns PDF
        doc = reader.read("file.md")  # Returns Markdown
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
    
    def read(self, source: str) -> BaseDocument:
        """
        Read and convert a file to appropriate Document type
        
        Args:
            source: File path (relative/absolute) or URL
            
        Returns:
            PDF or Markdown document based on file type
            
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
        Create appropriate document based on file type
        
        Args:
            info: SourceInfo from validation
            content: Converted content
            
        Returns:
            PDF or Markdown document
        """
        metadata = self._build_metadata(info, content)
        
        if info.file_type == FileType.PDF:
            return PDF(
                doc_id=metadata.md5,  # Use file hash as document ID
                content=content,
                metadata=metadata
            )
        else:
            # All other types converted to Markdown
            return Markdown(
                doc_id=metadata.md5,  # Use file hash as document ID
                content=content,
                metadata=metadata
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
