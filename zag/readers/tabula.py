"""PDF reader using Tabula for table extraction."""

from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import tabula
except ImportError:
    raise ImportError(
        "tabula-py is required. Install with: pip install tabula-py"
    )

import pandas as pd

from zag.schemas import BaseDocument, DocumentMetadata, Page
from zag.schemas.pdf import PDF
from .base import BaseReader
from ..utils.source import SourceUtils, FileType
from ..utils.hash import calculate_file_hash


class TabulaReader(BaseReader):
    """
    Reader for PDF files using Tabula for table extraction.
    
    Excellent for:
    - PDF files with structured tables
    - Tables with clear row/column boundaries
    - Multi-page documents with consistent table formats
    
    Features:
    - Direct DataFrame output (no intermediate parsing)
    - Better accuracy than pdfplumber/Camelot for many table types
    - Handles merged cells and complex layouts
    - Supports multiple table extraction strategies
    
    Note: Requires Java Runtime Environment (JRE) to be installed.
    
    Example:
        >>> reader = TabulaReader()
        >>> doc = reader.read("document.pdf")
        >>> print(doc.content)
        
        >>> # With custom settings
        >>> reader = TabulaReader(
        ...     lattice=True,  # For tables with borders
        ...     multiple_tables=True
        ... )
    """
    
    def __init__(
        self,
        lattice: bool = False,
        stream: bool = True,
        multiple_tables: bool = True,
        pages: str = 'all',
        guess: bool = True,
        **kwargs
    ):
        """
        Initialize Tabula reader.
        
        Args:
            lattice: Force PDF to be extracted using lattice-mode extraction
                (if there are ruling lines separating each cell, as in a PDF of an Excel spreadsheet)
            stream: Force PDF to be extracted using stream-mode extraction
                (if there are no ruling lines separating each cell)
            multiple_tables: Extract multiple tables per page (default: True)
            pages: Pages to extract. 'all' or page numbers like '1,2,3' or '1-3'
            guess: Guess the portion of the page to analyze per page (default: True)
            **kwargs: Additional tabula-py parameters
                - area: Portion of the page to analyze [top,left,bottom,right]
                - columns: X coordinates of column boundaries
                - relative_area: Use relative area coordinates (default: False)
                - encoding: Encoding type for pandas (default: 'utf-8')
        """
        self.lattice = lattice
        self.stream = stream
        self.multiple_tables = multiple_tables
        self.pages = pages
        self.guess = guess
        self.kwargs = kwargs
    
    @property
    def supported_formats(self) -> List[str]:
        """Return supported file formats."""
        return [".pdf"]
    
    def read(self, file_path: str, **kwargs) -> BaseDocument:
        """
        Read PDF file and extract tables using Tabula.
        
        Args:
            file_path: Path to PDF file
            **kwargs: Additional arguments (override init settings)
        
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
        
        # Merge init settings with runtime kwargs
        extract_kwargs = {
            'lattice': self.lattice,
            'stream': self.stream,
            'multiple_tables': self.multiple_tables,
            'pages': self.pages,
            'guess': self.guess,
            **self.kwargs,
            **kwargs
        }
        
        # Extract tables from PDF
        try:
            dfs = tabula.read_pdf(str(file_path), **extract_kwargs)
        except Exception as e:
            raise RuntimeError(f"Tabula extraction failed: {e}")
        
        # If no tables found, try alternate strategy
        if not dfs or len(dfs) == 0:
            # Try switching lattice/stream
            alternate_kwargs = extract_kwargs.copy()
            alternate_kwargs['lattice'] = not self.lattice
            alternate_kwargs['stream'] = not self.stream
            
            try:
                dfs = tabula.read_pdf(str(file_path), **alternate_kwargs)
            except:
                pass  # Still no tables, continue with empty result
        
        # Build pages with table content
        pages = self._build_pages(dfs, file_path)
        
        # Create full content
        full_content = "\n\n".join(page.content for page in pages)
        
        # Extract metadata
        metadata = self._extract_metadata(file_path, full_content, len(pages))
        
        # Create PDF document
        pdf_doc = PDF(
            doc_id=metadata.md5,
            content=full_content,
            metadata=metadata,
            pages=pages
        )
        
        return pdf_doc
    
    def _build_pages(self, dfs: List[pd.DataFrame], file_path: Path) -> List[Page]:
        """
        Build Page objects from extracted DataFrames.
        
        Args:
            dfs: List of DataFrames extracted by Tabula
            file_path: Path to PDF file
        
        Returns:
            List of Page objects
        """
        if not dfs:
            # No tables found, return single empty page
            return [Page(
                page_number=1,
                content="*[No tables detected]*",
                units=[],
                metadata={"tables_count": 0}
            )]
        
        # Group tables by page (Tabula doesn't provide page info directly in DataFrame)
        # For now, treat each table as from a sequential page
        pages = []
        
        for page_num, df in enumerate(dfs, start=1):
            # Convert DataFrame to Markdown
            if df is not None and not df.empty:
                md_table = self._dataframe_to_markdown(df)
                content = f"**Table {page_num}:**\n\n{md_table}"
            else:
                content = f"*[Empty table {page_num}]*"
            
            # Create Page object
            pages.append(Page(
                page_number=page_num,
                content=content,
                units=[],
                metadata={
                    "table_index": page_num - 1,
                    "rows": len(df) if df is not None else 0,
                    "columns": len(df.columns) if df is not None else 0
                }
            ))
        
        return pages
    
    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame to Markdown table format.
        
        Args:
            df: pandas DataFrame
        
        Returns:
            Markdown-formatted table string
        """
        if df is None or df.empty:
            return "*[Empty table]*"
        
        # Clean DataFrame
        df = df.fillna("")  # Replace NaN with empty string
        
        # Convert to string and strip whitespace
        df = df.astype(str).apply(lambda x: x.str.strip())
        
        # Build Markdown table
        lines = []
        
        # Header row (use first row as header if no column names)
        if all(isinstance(col, int) for col in df.columns):
            # Use first row as header
            header = df.iloc[0].tolist()
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            
            # Data rows (skip first row)
            for idx in range(1, len(df)):
                row = df.iloc[idx].tolist()
                lines.append("| " + " | ".join(row) + " |")
        else:
            # Use column names as header
            header = [str(col) for col in df.columns]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            
            # Data rows
            for idx in range(len(df)):
                row = df.iloc[idx].tolist()
                lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def _extract_metadata(self, file_path: Path, content: str, num_pages: int) -> DocumentMetadata:
        """
        Extract metadata from PDF.
        
        Args:
            file_path: Path to PDF file
            content: Document content
            num_pages: Number of pages/tables
        
        Returns:
            DocumentMetadata object
        """
        # Calculate file hash
        try:
            file_size = file_path.stat().st_size
            file_hash = calculate_file_hash(file_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to calculate file hash for {file_path}: {e}"
            )
        
        # Custom metadata
        custom = {
            "doc_name": file_path.name,
            "total_tables": num_pages,
            "extraction_method": "tabula",
            "lattice_mode": self.lattice,
            "stream_mode": self.stream,
        }
        
        return DocumentMetadata(
            source=str(file_path),
            source_type="local",
            file_type="pdf",
            file_name=file_path.name,
            file_size=file_size,
            file_extension=".pdf",
            md5=file_hash,
            content_length=len(content),
            mime_type="application/pdf",
            reader_name="TabulaReader",
            custom=custom
        )
