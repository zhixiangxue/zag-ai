"""
MinerU reader for high-accuracy PDF parsing
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Optional, Literal, Union, Tuple

from .base import BaseReader
from ..schemas import BaseDocument, DocumentMetadata, Page
from ..schemas.pdf import PDF
from ..utils.source import SourceUtils, FileType, SourceInfo
from ..utils.hash import calculate_file_hash, calculate_string_hash


class MinerUReader(BaseReader):
    """
    Reader using MinerU library for high-accuracy PDF parsing
    
    MinerU Features:
    - High accuracy: 82+ (pipeline) / 90+ (hybrid/vlm)
    - Remove headers, footers, footnotes for semantic coherence
    - Preserve document structure (headings, paragraphs, lists)
    - Extract images, tables with LaTeX/HTML format
    - Auto-detect scanned PDFs and enable OCR
    - Support 109 languages for OCR
    - Multiple backend options
    
    Backends:
    - pipeline: Traditional CV + OCR, good compatibility, pure CPU support
    - hybrid-auto-engine: Mixed VLM + Pipeline, high accuracy, requires GPU (10GB+ VRAM)
    - vlm-auto-engine: Pure VLM, highest accuracy, requires GPU (8GB+ VRAM)
    - hybrid-http-client: Remote VLM API, requires server_url
    - vlm-http-client: Remote VLM API, requires server_url
    
    Returns PDF document with:
    - content: Full markdown representation
    - pages: List of Page objects with structured data
    - metadata: Document and parsing information
    
    Usage:
        # Basic usage - hybrid backend (recommended)
        reader = MinerUReader()
        doc = reader.read("path/to/file.pdf")
        
        # Use pipeline backend for CPU-only environment
        reader = MinerUReader(backend="pipeline")
        doc = reader.read("path/to/file.pdf")
        
        # Use VLM for highest accuracy
        reader = MinerUReader(backend="vlm-auto-engine")
        doc = reader.read("path/to/file.pdf")
        
        # Specify language for better OCR
        reader = MinerUReader(lang="en")  # English
        reader = MinerUReader(lang="ch")  # Chinese
        
        # Parse specific page range
        doc = reader.read("path/to/file.pdf", page_range=(1, 100))
        doc = reader.read("path/to/file.pdf", page_range=(101, 200))
        
        # Use remote VLM service
        reader = MinerUReader(
            backend="hybrid-http-client",
            server_url="http://127.0.0.1:30000"
        )
        doc = reader.read("path/to/file.pdf")
        
        # Disable formula or table parsing
        reader = MinerUReader(formula_enable=False, table_enable=False)
        doc = reader.read("path/to/file.pdf")
    
    Note:
        MinerU only supports PDF files. For other formats, use DoclingReader or MarkItDownReader.
    """
    
    @staticmethod
    def _get_supported_types() -> set[FileType]:
        """
        Get supported file types
        
        Note: MinerU only supports PDF format
        """
        return {FileType.PDF}
    
    def __init__(
        self,
        backend: Literal[
            "pipeline",
            "hybrid-auto-engine", 
            "vlm-auto-engine",
            "hybrid-http-client",
            "vlm-http-client"
        ] = "hybrid-auto-engine",
        parse_method: Literal["auto", "txt", "ocr"] = "auto",
        lang: str = "ch",
        formula_enable: bool = True,
        table_enable: bool = True,
        server_url: Optional[str] = None,
    ):
        """
        Initialize MinerU reader
        
        Args:
            backend: Parsing backend
                - "pipeline": Traditional CV + OCR (CPU support)
                - "hybrid-auto-engine": Mixed VLM + Pipeline (GPU required, default)
                - "vlm-auto-engine": Pure VLM (GPU required)
                - "hybrid-http-client": Remote VLM API (requires server_url)
                - "vlm-http-client": Remote VLM API (requires server_url)
            parse_method: Parsing method
                - "auto": Auto-detect (default)
                - "txt": Text extraction
                - "ocr": Force OCR for scanned PDFs
            lang: OCR language code, default "ch" (Chinese)
                Supported: ch, en, korean, japan, chinese_cht, ta, te, ka, th, el,
                          latin, arabic, east_slavic, cyrillic, devanagari, etc.
            formula_enable: Enable formula recognition
            table_enable: Enable table recognition
            server_url: Server URL for http-client backends
        """
        self.backend = backend
        self.parse_method = parse_method
        self.lang = lang
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.server_url = server_url
        
        # Validate http-client backend requires server_url
        if backend in ["hybrid-http-client", "vlm-http-client"] and not server_url:
            raise ValueError(f"Backend '{backend}' requires server_url parameter")
    
    def read(self, source: Union[str, Path], page_range: Optional[Tuple[int, int]] = None) -> BaseDocument:
        """
        Read and parse a PDF file using MinerU
        
        Args:
            source: File path (str or Path object, relative/absolute) or URL
            page_range: Optional page range as (start_page, end_page).
                       Page numbers are 1-based and inclusive.
                       Example: (1, 100) means read pages 1 through 100.
                       None means read all pages (default).
            
        Returns:
            PDF document with markdown content and structured pages
            
        Raises:
            ValueError: If source is invalid, file format is not supported,
                       or page_range is invalid (negative, end < start)
            ImportError: If mineru is not installed
            
        Note:
            Page range validation:
            - Negative values or end < start → ValueError (invalid input)
            - Range exceeds actual pages → Auto-adjusted to actual range (no error)
            - Example: PDF has 50 pages, request (1, 100) → reads pages 1-50
            
            When page_range is specified, the doc_id is still based on the
            original file (not the page range). Use PDF.__add__ to merge
            multiple page ranges and then set the correct doc_id.
        """
        # Check if mineru is installed
        try:
            from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
            from mineru.data.data_reader_writer import FileBasedDataWriter
            from mineru.utils.engine_utils import get_vlm_engine
            from mineru.utils.enum_class import MakeMode
            from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
            from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
            from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
            from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
            from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze
        except ImportError as e:
            raise ImportError(
                f"mineru is not installed or missing dependencies: {e}. "
                "Install it with: pip install mineru[all]"
            )
        
        # Convert Path object to string if needed (for compatibility with SourceUtils)
        source = str(source)
        
        # Validate and get complete info
        info = SourceUtils.validate(source, check_accessibility=True, timeout=5)
        
        # Check if valid
        if not info.is_valid:
            raise ValueError(info.error_message)
        
        # Check if file type is supported
        if info.file_type not in self._get_supported_types():
            raise ValueError(
                f"Unsupported file type: {info.file_type.value}. "
                "MinerUReader only supports PDF files."
            )
        
        # Get local file path (download if URL)
        local_path = self._get_local_path(info)
        
        # Get total page count for validation
        actual_pages = self._get_pdf_page_count(local_path)
        
        # Convert page_range to MinerU's start_page_id/end_page_id (0-based)
        # Note: pypdfium2's end_page_id is INCLUSIVE, so we need end_page - 1
        start_page_id = 0
        end_page_id = None
        if page_range is not None:
            start_page, end_page = page_range
            
            # Validate: negative values
            if start_page < 1 or end_page < 1:
                raise ValueError(
                    f"Invalid page_range: page numbers must be >= 1. "
                    f"Got: start={start_page}, end={end_page}"
                )
            
            # Validate: end < start
            if end_page < start_page:
                raise ValueError(
                    f"Invalid page_range: end ({end_page}) must be >= start ({start_page})"
                )
            
            # Auto-adjust if exceeds actual pages
            if actual_pages is not None:
                if start_page > actual_pages:
                    raise ValueError(
                        f"Invalid page_range: start ({start_page}) exceeds "
                        f"total pages ({actual_pages})"
                    )
                if end_page > actual_pages:
                    end_page = actual_pages
            
            start_page_id = start_page - 1  # Convert 1-based to 0-based
            end_page_id = end_page - 1      # Convert 1-based to 0-based (inclusive)
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_output_dir:
            # Read PDF bytes
            pdf_bytes = read_fn(local_path)
            pdf_file_name = Path(local_path).stem
            
            # Handle page range
            if start_page_id > 0 or end_page_id is not None:
                pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                    pdf_bytes, start_page_id, end_page_id
                )
            
            # Parse based on backend
            if self.backend == "pipeline":
                # Pipeline backend
                parse_method = self.parse_method
                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                    [pdf_bytes], [self.lang], 
                    parse_method=parse_method,
                    formula_enable=self.formula_enable,
                    table_enable=self.table_enable
                )
                
                # Prepare environment
                local_image_dir, local_md_dir = prepare_env(temp_output_dir, pdf_file_name, parse_method)
                image_writer = FileBasedDataWriter(local_image_dir)
                md_writer = FileBasedDataWriter(local_md_dir)
                
                # Convert to middle JSON
                model_list = infer_results[0]
                images_list = all_image_lists[0]
                pdf_doc = all_pdf_docs[0]
                _lang = lang_list[0]
                _ocr_enable = ocr_enabled_list[0]
                middle_json = pipeline_result_to_middle_json(
                    model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, self.formula_enable
                )
                
            elif self.backend.startswith("vlm-"):
                # VLM backend
                backend = self.backend[4:]  # Remove 'vlm-' prefix
                if backend == "auto-engine":
                    backend = get_vlm_engine(inference_engine='auto', is_async=False)
                
                parse_method = "vlm"
                local_image_dir, local_md_dir = prepare_env(temp_output_dir, pdf_file_name, parse_method)
                image_writer = FileBasedDataWriter(local_image_dir)
                md_writer = FileBasedDataWriter(local_md_dir)
                
                middle_json, infer_result = vlm_doc_analyze(
                    pdf_bytes,
                    image_writer=image_writer,
                    backend=backend,
                    server_url=self.server_url
                )
                
            elif self.backend.startswith("hybrid-"):
                # Hybrid backend
                backend = self.backend[7:]  # Remove 'hybrid-' prefix
                if backend == "auto-engine":
                    backend = get_vlm_engine(inference_engine='auto', is_async=False)
                
                parse_method = f"hybrid_{self.parse_method}"
                local_image_dir, local_md_dir = prepare_env(temp_output_dir, pdf_file_name, parse_method)
                image_writer = FileBasedDataWriter(local_image_dir)
                md_writer = FileBasedDataWriter(local_md_dir)
                
                middle_json, infer_result, _vlm_ocr_enable = hybrid_doc_analyze(
                    pdf_bytes,
                    image_writer=image_writer,
                    backend=backend,
                    parse_method=parse_method,
                    language=self.lang,
                    inline_formula_enable=self.formula_enable,
                    server_url=self.server_url,
                )
            
            # Generate markdown content
            pdf_info = middle_json["pdf_info"]
            image_dir = Path(local_image_dir).name
            
            if self.backend == "pipeline":
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
            else:
                content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
            
            # Build content and pages together from content_list
            # This ensures consistency and accurate page boundaries
            md_content, pages = self._build_content_and_pages_from_content_list(
                content_list, start_page_id
            )
            
            # Extract custom metadata
            custom_metadata = self._extract_custom_metadata(middle_json)
        
        # Build metadata
        metadata = self._build_metadata(info, md_content, custom_metadata)
        
        # Create PDF document
        return PDF(
            doc_id=metadata.md5,  # Use file hash as document ID
            content=md_content,
            metadata=metadata,
            pages=pages
        )
    
    def _get_local_path(self, info: SourceInfo) -> str:
        """
        Get local file path, download if URL
        
        Args:
            info: SourceInfo from validation
            
        Returns:
            Local file path
        """
        if info.source_type.value == "local":
            return info.source
        else:
            # TODO: Implement URL download to temp file
            # For now, assume URL is directly accessible by MinerU
            return info.source
    
    @staticmethod
    def _get_pdf_page_count(source: str) -> Optional[int]:
        """
        Get total page count of a PDF file
        
        Args:
            source: Path to PDF file
            
        Returns:
            Total page count, or None if cannot determine
        """
        try:
            # Use pypdf for lightweight page count
            from pypdf import PdfReader
            reader = PdfReader(source)
            return len(reader.pages)
        except Exception:
            # Fallback: try pypdfium2 (already used by MinerU)
            try:
                import pypdfium2
                pdf = pypdfium2.PdfDocument(source)
                count = len(pdf)
                pdf.close()
                return count
            except Exception:
                return None
    

    def _build_content_and_pages_from_content_list(
        self, 
        content_list: list[dict], 
        start_page_id: int = 0
    ) -> tuple[str, list[Page]]:
        """
        Build pdf.content and pages together from content_list.
        
        This ensures:
        1. pdf.content and page.content are consistent (built from same source)
        2. Page boundaries in pdf.content are known (for accurate span mapping)
        
        Args:
            content_list: Content list from MinerU output (preserves original order)
            start_page_id: Start page ID (0-based) from original PDF
            
        Returns:
            Tuple of (full_content, pages) where each page has metadata.span set
        """
        # Group by page number, preserving order within each page
        page_items = {}
        
        for item in content_list:
            page_idx = item.get("page_idx", 0)
            page_num = page_idx + 1 + start_page_id
            
            if page_num not in page_items:
                page_items[page_num] = []
            
            page_items[page_num].append(item)
        
        # Build content and pages together
        all_parts = []
        pages = []
        
        for page_num in sorted(page_items.keys()):
            items = page_items[page_num]
            
            # Record where this page starts in the full content
            page_start = len("\n\n".join(all_parts)) if all_parts else 0
            
            # Build page content preserving original order
            page_parts = []
            text_count = 0
            table_count = 0
            image_count = 0
            
            for item in items:
                item_type = item.get("type", "text")
                
                if item_type == "text":
                    text = item.get("text", "")
                    if text.strip():
                        page_parts.append(text)
                        text_count += 1
                        
                elif item_type == "table":
                    # Prefer html field, fall back to table_body
                    html = item.get("html") or item.get("table_body")
                    if html:
                        page_parts.append(html)
                        table_count += 1
                        
                elif item_type in ["image", "figure"]:
                    caption = item.get("caption")
                    if caption:
                        page_parts.append(f"[Image: {caption}]")
                    image_count += 1
            
            page_content = "\n\n".join(page_parts)
            
            # Add to full content
            if page_parts:
                all_parts.extend(page_parts)
            
            # Calculate page end position
            full_content_so_far = "\n\n".join(all_parts)
            page_end = len(full_content_so_far)
            
            # Create Page with span info
            page = Page(
                page_number=page_num,
                content=page_content,
                units=[],
                metadata={
                    "text_count": text_count,
                    "table_count": table_count,
                    "image_count": image_count,
                    "span": (page_start, page_end)  # Position in pdf.content
                }
            )
            pages.append(page)
        
        # Build final full content
        full_content = "\n\n".join(all_parts)
        
        return full_content, pages

    def _extract_custom_metadata(self, middle_json: dict) -> dict[str, Any]:
        """
        Extract custom metadata from MinerU middle JSON
        
        Args:
            middle_json: Middle JSON from MinerU output
            
        Returns:
            Dict of custom metadata
        """
        pdf_info = middle_json.get("pdf_info", [])
        
        return {
            "page_count": len(pdf_info),
            "backend": self.backend,
            "parse_method": self.parse_method,
            "lang": self.lang,
            "formula_enabled": self.formula_enable,
            "table_enabled": self.table_enable,
        }
    
    def _build_metadata(
        self, 
        info: SourceInfo, 
        content: str, 
        custom: dict[str, Any]
    ) -> DocumentMetadata:
        """
        Build metadata object
        
        Args:
            info: SourceInfo from validation
            content: Document content
            custom: Custom metadata from MinerU
            
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
            source=str(info.source),  # Ensure source is string, not Path object
            source_type=info.source_type.value,
            file_type=info.file_type.value if info.file_type else "unknown",
            file_name=Path(info.source).name if info.source_type.value == "local" else None,
            file_size=file_size,
            file_extension=info.file_extension if info.file_extension else None,
            md5=file_hash,  # Required field
            content_length=len(content),
            mime_type=info.mime_type if info.mime_type else None,
            reader_name="MinerUReader",
            custom=custom
        )
