"""PDF reader using pdfplumber for complex table extraction."""

from pathlib import Path
from typing import Dict, List, Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError(
        "pdfplumber is required. Install with: pip install pdfplumber"
    )

from zag.schemas import BaseDocument, DocumentMetadata, Page
from zag.schemas.pdf import PDF
from .base import BaseReader
from ..utils.source import SourceUtils, FileType
from ..utils.hash import calculate_file_hash


class PDFPlumberReader(BaseReader):
    """
    Reader for PDF files using pdfplumber.
    
    Excellent for:
    - Native PDF files with text (not scanned images)
    - Complex tables with borders
    - Tables with merged cells
    
    Features:
    - Extracts tables as structured data (DataFrame-compatible)
    - Preserves table formatting
    - Handles multi-page documents
    - Lightweight, no GPU required
    
    Example:
        >>> reader = PDFPlumberReader()
        >>> doc = reader.read("document.pdf")
        >>> print(doc.content)
    """
    
    def __init__(
        self,
        table_settings: Optional[Dict] = None,
        extract_images: bool = False
    ):
        """
        Initialize PDFPlumber reader.
        
        Args:
            table_settings: Custom settings for table extraction
                See: https://github.com/jsvine/pdfplumber#table-extraction-settings
                Example: {
                    "vertical_strategy": "lines",  # or "text"
                    "horizontal_strategy": "lines",
                    "explicit_vertical_lines": [],
                    "explicit_horizontal_lines": [],
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                    "min_words_vertical": 3,
                    "min_words_horizontal": 1,
                    "intersection_tolerance": 3,
                }
            extract_images: Whether to extract images (experimental)
        """
        self.table_settings = table_settings or {}
        self.extract_images = extract_images
    
    @property
    def supported_formats(self) -> List[str]:
        """Return supported file formats."""
        return [".pdf"]
    
    def read(self, file_path: str, **kwargs) -> BaseDocument:
        """
        Read PDF file and extract content with tables.
        
        Args:
            file_path: Path to PDF file
            **kwargs: Additional arguments (currently unused)
        
        Returns:
            PDF document with extracted content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported: {self.supported_formats}"
            )
        
        pages = []
        full_text = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text() or ""
                
                # Extract tables
                tables = page.extract_tables(table_settings=self.table_settings)
                
                # Format tables as Markdown
                table_content = []
                if tables:
                    for i, table in enumerate(tables, start=1):
                        if table and len(table) > 0:
                            md_table = self._table_to_markdown(table)
                            table_content.append(f"\n**Table {i}:**\n{md_table}\n")
                
                # Combine text and tables
                page_content = text
                if table_content:
                    page_content += "\n\n" + "\n".join(table_content)
                
                # Create Page object
                pages.append(Page(
                    page_number=page_num,
                    content=page_content,
                    units=[],  # Empty - no unit construction
                    metadata={}
                ))
                
                full_text.append(page_content)
            
            # Create metadata
            markdown_content = "\n\n".join(full_text)
            metadata = self._extract_metadata(file_path, markdown_content, pdf)
        
        # Create PDF document with doc_id based on file hash
        # This ensures idempotency: same file -> same doc_id
        pdf_doc = PDF(
            doc_id=metadata.md5,  # Use file hash as doc_id
            content=markdown_content,
            metadata=metadata,
            pages=pages
        )
        
        return pdf_doc
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """
        Convert table data to Markdown format.
        
        Args:
            table: List of rows, where each row is a list of cells
        
        Returns:
            Markdown-formatted table string
        """
        if not table or len(table) == 0:
            return ""
        
        # Clean cells (handle None values)
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        # Get max column count
        max_cols = max(len(row) for row in cleaned_table)
        
        # Pad rows to same length
        for row in cleaned_table:
            while len(row) < max_cols:
                row.append("")
        
        # Build Markdown table
        lines = []
        
        # Header (first row)
        if cleaned_table:
            header = cleaned_table[0]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            
            # Data rows
            for row in cleaned_table[1:]:
                lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def _extract_metadata(self, file_path: Path, content: str, pdf) -> DocumentMetadata:
        """
        Extract metadata from PDF.
        
        Args:
            file_path: Path to PDF file
            content: Document content
            pdf: pdfplumber PDF object
        
        Returns:
            DocumentMetadata object
        """
        # Calculate file hash using xxhash
        try:
            file_size = file_path.stat().st_size
            file_hash = calculate_file_hash(file_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to calculate file hash for {file_path}: {e}"
            )
        
        # Get PDF info
        info = pdf.metadata or {}
        
        # Custom metadata
        custom = {
            "doc_name": file_path.name,
            "total_pages": len(pdf.pages),
        }
        
        # Add title if available
        title = info.get("Title") or info.get("title")
        if title:
            custom["title"] = title
        
        # Add author if available
        author = info.get("Author") or info.get("author")
        if author:
            custom["author"] = author
        
        # Add PDF version if available
        if hasattr(pdf, "pdf_version") and pdf.pdf_version:
            custom["pdf_version"] = pdf.pdf_version
        
        # Add all other PDF metadata
        for key, value in info.items():
            if key not in ["Author", "Title", "CreationDate"] and value:
                custom[f"pdf_{key.lower()}"] = str(value)
        
        return DocumentMetadata(
            source=str(file_path),
            source_type="local",
            file_type="pdf",
            file_name=file_path.name,
            file_size=file_size,
            file_extension=".pdf",
            md5=file_hash,  # Required field
            content_length=len(content),
            mime_type="application/pdf",
            reader_name="PDFPlumberReader",
            custom=custom
        )
