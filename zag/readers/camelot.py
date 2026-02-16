"""PDF reader using Camelot for advanced table extraction."""

from pathlib import Path
from typing import List, Literal, Optional

try:
    import camelot
except ImportError:
    raise ImportError(
        "camelot-py is required. Install with: pip install 'camelot-py[base]'"
    )

from zag.schemas import BaseDocument, DocumentMetadata, Page
from zag.schemas.pdf import PDF
from zag.schemas.unit import TableUnit
from zag.schemas.metadata import UnitMetadata
from .base import BaseReader
from ..utils.hash import calculate_file_hash
from uuid import uuid4


class CamelotReader(BaseReader):
    """
    Reader for PDF files using Camelot for advanced table extraction.
    
    Camelot is specialized for extracting tables from PDFs with high accuracy.
    It supports two extraction modes:
    
    1. **Lattice mode** (default): Uses lines/borders to detect tables
       - Best for: Tables with clear borders and grid lines
       - Pros: High accuracy for bordered tables, handles merged cells well
       - Cons: Cannot detect borderless tables
    
    2. **Stream mode**: Uses text positions to detect tables
       - Best for: Tables without borders, whitespace-separated data
       - Pros: Can detect borderless tables
       - Cons: Less accurate than Lattice for bordered tables
    
    Features:
    - Extracts tables as structured data (DataFrame-compatible)
    - Handles complex tables with merged cells
    - Supports multi-page documents
    - Provides accuracy scores for each table
    - Configurable extraction parameters
    
    Example:
        >>> # Use default Lattice mode for bordered tables
        >>> reader = CamelotReader()
        >>> doc = reader.read("document.pdf")
        
        >>> # Use Stream mode for borderless tables
        >>> reader = CamelotReader(flavor="stream")
        >>> doc = reader.read("document.pdf")
        
        >>> # Custom Lattice settings
        >>> reader = CamelotReader(
        ...     flavor="lattice",
        ...     line_scale=15,  # Line detection sensitivity
        ...     copy_text=['v']  # Copy text direction
        ... )
        
        >>> # Custom Stream settings for complex tables
        >>> reader = CamelotReader(
        ...     flavor="stream",
        ...     row_tol=5,     # Moderate: combine text into rows
        ...     column_tol=2,  # Moderate: combine text into columns
        ...     edge_tol=75    # Moderate: extend text edges
        ... )
        
        >>> # Flatten complex tables (Lattice mode only)
        >>> reader = CamelotReader(
        ...     flavor="lattice",
        ...     copy_text=['h', 'v'],  # Copy merged cell text in both directions
        ...     split_text=True        # Split spanning text
        ... )
    """
    
    def __init__(
        self,
        flavor: Literal["lattice", "stream"] = "lattice",
        pages: str = "all",
        **kwargs
    ):
        """
        Initialize Camelot reader.
        
        Args:
            flavor: Extraction mode
                - "lattice": Line-based detection (default, best for bordered tables)
                - "stream": Text position-based detection (best for borderless tables)
            pages: Pages to extract tables from
                - "all": All pages (default)
                - "1": Only first page
                - "1,2,3": Specific pages
                - "1-5": Page range
            **kwargs: Additional Camelot parameters
                Common (both modes):
                    - split_text: Split text spanning multiple cells (default: False)
                        * WARNING: May break words apart, use with caution
                    - strip_text: Remove characters from text (default: '')
                    
                For Lattice mode (bordered tables):
                    - line_scale: Line detection sensitivity (default: 15)
                    - copy_text: Copy merged cell text to spanning cells
                        * ['h']: Copy horizontally
                        * ['v']: Copy vertically  
                        * ['h', 'v']: Copy in both directions (flatten)
                    - shift_text: Text flow direction in merged cells (default: ['l', 't'])
                        * 'l': left, 'r': right, 't': top, 'b': bottom
                    - process_background: Process background lines (default: False)
                    
                For Stream mode (borderless tables):
                    - row_tol: Vertical text combination tolerance (default: 2)
                        * Increase MODERATELY (e.g., 5) to combine text into rows
                        * Too high will merge different rows incorrectly
                    - column_tol: Horizontal text combination tolerance (default: 0)
                        * Increase MODERATELY (e.g., 2) to combine text into columns
                        * Too high will merge different columns incorrectly
                    - edge_tol: Text edge extension tolerance (default: 50)
                        * Increase MODERATELY (e.g., 75) for better handling
                    - columns: Manually specify column x-coordinates (e.g., ['100,200,300'])
                        * Use when automatic column detection fails
                    
                Tips for complex tables:
                    - Lattice: copy_text=['h', 'v'], split_text=True
                    - Stream: row_tol=5, column_tol=2, edge_tol=75 (avoid split_text)
                    - If columns merge: Use lower tolerance or specify columns parameter
        """
        self.flavor = flavor
        self.pages = pages
        self.kwargs = kwargs
    
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
        
        # Extract tables using Camelot
        tables = camelot.read_pdf(
            str(file_path),
            flavor=self.flavor,
            pages=self.pages,
            **self.kwargs
        )
        
        # Build TableUnits from extracted tables
        table_units = self._build_table_units(tables, file_path)
        
        # Build full content from all tables
        full_text = "\n\n".join(unit.content for unit in table_units)
        
        # Get total pages
        num_pages = self._get_page_count(tables, file_path)
        
        # Create metadata
        metadata = self._extract_metadata(file_path, full_text, num_pages)
        
        # Create PDF document with doc_id based on file hash
        pdf_doc = PDF(
            doc_id=metadata.md5,
            content=full_text,
            metadata=metadata,
            pages=[],  # Camelot focuses on tables, not page structure
            units=table_units
        )
        
        return pdf_doc
    
    def _get_page_count(self, tables, file_path: Path) -> int:
        """
        Get total page count from PDF.
        
        Args:
            tables: Camelot TableList object
            file_path: Path to PDF file
        
        Returns:
            Number of pages in PDF
        """
        if tables:
            # Get max page number from tables
            return max(table.page for table in tables)
        
        # Try to get page count from PDF
        try:
            import pypdf
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                return len(pdf_reader.pages)
        except Exception:
            return 1
    
    def _build_table_units(self, tables, file_path: Path) -> List[TableUnit]:
        """
        Build TableUnits from extracted tables.
        
        Args:
            tables: Camelot TableList object
            file_path: Path to PDF file
        
        Returns:
            List of TableUnit objects
        """
        if not tables:
            return []
        
        # Build document metadata for units
        # This will be injected into each unit's metadata.document
        file_size = file_path.stat().st_size
        file_hash = calculate_file_hash(file_path)
        
        doc_metadata = DocumentMetadata(
            source=str(file_path),
            source_type="local",
            file_type="pdf",
            file_name=file_path.name,
            file_size=file_size,
            file_extension=".pdf",
            md5=file_hash,
            content_length=0,  # Will be updated later
            mime_type="application/pdf",
            reader_name="CamelotReader",
            custom={
                "extraction_flavor": self.flavor,
            }
        )
        
        # Convert each table to TableUnit
        table_units = []
        for idx, table in enumerate(tables, start=1):
            # Get table metadata
            accuracy = table.accuracy if hasattr(table, 'accuracy') else None
            whitespace = table.whitespace if hasattr(table, 'whitespace') else None
            page_num = table.page
            
            # Build markdown content (just the table, no metadata)
            content = self._table_to_markdown(table.df)
            
            # Create TableUnit
            table_unit = TableUnit(
                unit_id=str(uuid4()),
                content=content,
                df=table.df,
                metadata=UnitMetadata(
                    page_numbers=[page_num],
                    document=doc_metadata.model_dump(exclude={'custom'}),
                    custom={}
                )
            )
            
            table_units.append(table_unit)
        
        return table_units
    
    def _table_to_markdown(self, df) -> str:
        """
        Convert DataFrame to Markdown format.
        
        Args:
            df: pandas DataFrame
        
        Returns:
            Markdown-formatted table string
        """
        if df is None or df.empty:
            return "*[Empty table]*"
        
        # Convert DataFrame to markdown
        lines = []
        
        # Header row (use first row as header)
        header = df.iloc[0].tolist()
        header = [str(cell).strip() if cell else "" for cell in header]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Data rows
        for idx in range(1, len(df)):
            row = df.iloc[idx].tolist()
            row = [str(cell).strip() if cell else "" for cell in row]
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def _extract_metadata(self, file_path: Path, content: str, num_pages: int) -> DocumentMetadata:
        """
        Extract metadata from PDF.
        
        Args:
            file_path: Path to PDF file
            content: Document content
            num_pages: Number of pages
        
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
        
        # Try to get PDF metadata
        custom = {
            "doc_name": file_path.name,
            "total_pages": num_pages,
            "extraction_flavor": self.flavor,
        }
        
        # Try to extract PDF metadata using pypdf
        try:
            import pypdf
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                info = pdf_reader.metadata or {}
                
                # Add title if available
                if info.get('/Title'):
                    custom["title"] = info.get('/Title')
                
                # Add author if available
                if info.get('/Author'):
                    custom["author"] = info.get('/Author')
        except Exception:
            # If pypdf fails, just skip metadata extraction
            pass
        
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
            reader_name="CamelotReader",
            custom=custom
        )
