"""
Source validation and file type detection utilities
Optional utilities for reader implementations
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from mimetypes import guess_extension
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class SourceType(str, Enum):
    """Source type enumeration"""
    LOCAL = "local"
    URL = "url"
    UNKNOWN = "unknown"


@dataclass
class SourceInfo:
    """
    Complete information about a source
    Returned by SourceUtils.validate() to avoid multiple network requests
    """
    # Basic info
    source: str
    source_type: SourceType
    is_valid: bool
    error_message: str = ""
    
    # File type info
    file_type: Optional['FileType'] = None
    file_extension: str = ""
    mime_type: str = ""
    
    # URL-specific info
    status_code: Optional[int] = None
    
    def __str__(self) -> str:
        if self.is_valid:
            return f"SourceInfo(valid, type={self.source_type.value}, file_type={self.file_type.value if self.file_type else 'unknown'})"
        else:
            return f"SourceInfo(invalid, error={self.error_message})"


class FileType(str, Enum):
    """
    File type enumeration
    Similar to MIME types but simplified for document processing
    """
    
    # Document types
    PDF = "pdf"
    MARKDOWN = "markdown"
    WORD = "word"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    
    # Web formats
    HTML = "html"
    XML = "xml"
    JSON = "json"
    
    # Plain text
    TEXT = "text"
    CSV = "csv"
    
    # Archive
    ZIP = "zip"
    
    # Unknown
    UNKNOWN = "unknown"


class SourceUtils:
    """
    Utility class for source validation and file type detection
    Can be optionally used by any reader implementation
    
    Main API: validate() - returns complete SourceInfo in one call
    """
    
    # Extension to FileType mapping
    EXT_TO_TYPE = {
        ".pdf": FileType.PDF,
        ".md": FileType.MARKDOWN,
        ".markdown": FileType.MARKDOWN,
        ".docx": FileType.WORD,
        ".doc": FileType.WORD,
        ".xlsx": FileType.EXCEL,
        ".xls": FileType.EXCEL,
        ".pptx": FileType.POWERPOINT,
        ".ppt": FileType.POWERPOINT,
        ".html": FileType.HTML,
        ".htm": FileType.HTML,
        ".xml": FileType.XML,
        ".json": FileType.JSON,
        ".txt": FileType.TEXT,
        "": FileType.TEXT,
        ".csv": FileType.CSV,
        ".zip": FileType.ZIP,
    }
    
    # MIME type to FileType mapping
    MIME_TO_TYPE = {
        "application/pdf": FileType.PDF,
        "text/markdown": FileType.MARKDOWN,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.WORD,
        "application/msword": FileType.WORD,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileType.EXCEL,
        "application/vnd.ms-excel": FileType.EXCEL,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": FileType.POWERPOINT,
        "application/vnd.ms-powerpoint": FileType.POWERPOINT,
        "text/html": FileType.HTML,
        "application/xml": FileType.XML,
        "text/xml": FileType.XML,
        "application/json": FileType.JSON,
        "text/plain": FileType.TEXT,
        "text/csv": FileType.CSV,
        "application/zip": FileType.ZIP,
    }
    
    @staticmethod
    def validate(source: str, check_accessibility: bool = True, timeout: int = 5) -> SourceInfo:
        """
        Validate source and return complete information in one call
        
        This is the MAIN API - avoids multiple network requests
        
        Args:
            source: File path or URL
            check_accessibility: Whether to check URL accessibility (requires network request)
            timeout: Timeout in seconds for network requests
            
        Returns:
            SourceInfo object containing all information
            
        Example:
            >>> info = SourceUtils.validate("https://example.com/file.pdf")
            >>> if info.is_valid:
            ...     print(f"File type: {info.file_type}")
            ...     print(f"Is URL: {info.source_type == SourceType.URL}")
            >>> else:
            ...     print(f"Error: {info.error_message}")
        """
        # Detect source type
        source_type = SourceUtils._detect_source_type(source)
        
        if source_type == SourceType.URL:
            return SourceUtils._validate_url(source, check_accessibility, timeout)
        elif source_type == SourceType.LOCAL:
            return SourceUtils._validate_local(source)
        else:
            return SourceInfo(
                source=source,
                source_type=SourceType.UNKNOWN,
                is_valid=False,
                error_message="Invalid source format"
            )
    
    @staticmethod
    def _detect_source_type(source: str) -> SourceType:
        """Detect if source is URL or local path"""
        try:
            result = urlparse(source)
            if result.scheme and result.netloc and result.scheme in ('http', 'https', 'ftp'):
                return SourceType.URL
            elif result.scheme == 'file':
                return SourceType.LOCAL
            else:
                # No scheme, treat as local path
                return SourceType.LOCAL
        except Exception:
            return SourceType.UNKNOWN
    
    @staticmethod
    def _validate_local(source: str) -> SourceInfo:
        """Validate local file path"""
        path = Path(source)
        
        # Check existence
        if not path.exists():
            return SourceInfo(
                source=source,
                source_type=SourceType.LOCAL,
                is_valid=False,
                error_message=f"File not found: {source}"
            )
        
        if not path.is_file():
            return SourceInfo(
                source=source,
                source_type=SourceType.LOCAL,
                is_valid=False,
                error_message=f"Path is not a file: {source}"
            )
        
        # Get file extension and type
        file_ext = path.suffix.lower()
        file_type = SourceUtils.EXT_TO_TYPE.get(file_ext, FileType.UNKNOWN)
        
        return SourceInfo(
            source=source,
            source_type=SourceType.LOCAL,
            is_valid=True,
            file_type=file_type,
            file_extension=file_ext
        )
    
    @staticmethod
    def _validate_url(source: str, check_accessibility: bool, timeout: int) -> SourceInfo:
        """
        Validate URL and get file type from Content-Type header
        
        Only makes ONE network request (HEAD) to get both accessibility and file type
        """
        if not HAS_REQUESTS:
            # If requests not installed, can't check accessibility
            return SourceInfo(
                source=source,
                source_type=SourceType.URL,
                is_valid=not check_accessibility,
                error_message="requests library not installed" if check_accessibility else "",
                file_type=SourceUtils._guess_type_from_url_path(source)
            )
        
        if not check_accessibility:
            # Don't check, just guess from URL path
            return SourceInfo(
                source=source,
                source_type=SourceType.URL,
                is_valid=True,
                file_type=SourceUtils._guess_type_from_url_path(source),
                file_extension=SourceUtils._get_extension_from_url(source)
            )
        
        # Make ONE HEAD request to check both accessibility and get Content-Type
        try:
            response = requests.head(source, timeout=timeout, allow_redirects=True)
            
            # Check status
            if response.status_code != 200:
                return SourceInfo(
                    source=source,
                    source_type=SourceType.URL,
                    is_valid=False,
                    error_message=f"URL returned status code: {response.status_code}",
                    status_code=response.status_code
                )
            
            # Get Content-Type
            content_type = response.headers.get('Content-Type', '').split(';')[0].strip()
            
            # Try to determine file type
            file_type = SourceUtils.MIME_TO_TYPE.get(content_type, FileType.UNKNOWN)
            
            # If still unknown, try from URL path
            if file_type == FileType.UNKNOWN:
                file_type = SourceUtils._guess_type_from_url_path(source)
            
            # Get extension
            file_ext = SourceUtils._get_extension_from_url(source)
            if not file_ext and content_type:
                # Guess extension from MIME type
                file_ext = guess_extension(content_type) or ""
            
            return SourceInfo(
                source=source,
                source_type=SourceType.URL,
                is_valid=True,
                file_type=file_type,
                file_extension=file_ext,
                mime_type=content_type,
                status_code=response.status_code
            )
            
        except requests.Timeout:
            return SourceInfo(
                source=source,
                source_type=SourceType.URL,
                is_valid=False,
                error_message=f"URL timeout after {timeout}s"
            )
        except requests.RequestException as e:
            return SourceInfo(
                source=source,
                source_type=SourceType.URL,
                is_valid=False,
                error_message=f"URL error: {str(e)}"
            )
    
    @staticmethod
    def _get_extension_from_url(url: str) -> str:
        """Extract file extension from URL path"""
        try:
            path = urlparse(url).path
            return Path(path).suffix.lower()
        except Exception:
            return ""
    
    @staticmethod
    def _guess_type_from_url_path(url: str) -> FileType:
        """Guess file type from URL path extension"""
        ext = SourceUtils._get_extension_from_url(url)
        return SourceUtils.EXT_TO_TYPE.get(ext, FileType.UNKNOWN)
