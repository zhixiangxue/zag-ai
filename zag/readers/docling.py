"""
Docling reader for advanced PDF understanding
"""

from pathlib import Path
from typing import Any, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions
from docling.pipeline.vlm_pipeline import VlmPipeline

from .base import BaseReader
from ..schemas.base import BaseDocument, DocumentMetadata, Page
from ..schemas.pdf import PDF
from ..utils.source import SourceUtils, FileType, SourceInfo
from ..utils.hash import calculate_file_hash


class DoclingReader(BaseReader):
    """
    Reader using Docling library for advanced PDF understanding
    
    Features:
    - Deep layout analysis and reading order preservation
    - Table structure recognition
    - Figure detection and classification
    - Formula recognition
    - Bounding box information
    - Support for VLM (Vision Language Model) pipeline
    - Configurable OCR options
    
    Returns PDF document with:
    - content: Full markdown representation
    - pages: List of Page objects with structured data
        - Each page.content: Dict with texts, tables, pictures
        - Each page.metadata: Layout and provenance info
    
    Usage:
        # Basic usage - default standard pipeline
        reader = DoclingReader()
        doc = reader.read("path/to/file.pdf")
        
        # Use VLM pipeline with local model
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.datamodel import vlm_model_specs
        
        vlm_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.SMOLDOCLING_MLX
        )
        reader = DoclingReader(vlm_pipeline_options=vlm_options)
        
        # Use VLM pipeline with remote API (e.g., Alibaba Qwen-VL)
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
        
        vlm_options = VlmPipelineOptions(
            enable_remote_services=True,
            vlm_options=ApiVlmOptions(
                url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                params=dict(
                    model="qwen-vl-max-latest",  # or qwen-vl-plus, qwen2-vl-7b-instruct
                    max_tokens=4096,
                ),
                headers={
                    "Authorization": "Bearer YOUR_API_KEY",  # Replace with your API key
                },
                prompt="Convert this page to markdown.",
                timeout=90,
                response_format=ResponseFormat.MARKDOWN,
            )
        )
        reader = DoclingReader(vlm_pipeline_options=vlm_options)
        
        # Custom standard PDF pipeline options
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        
        pdf_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True
        )
        reader = DoclingReader(pdf_pipeline_options=pdf_options)
        
        # GPU acceleration (configure via pipeline options)
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        
        pdf_options = PdfPipelineOptions()
        pdf_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA  # or AUTO, CPU, MPS
        )
        reader = DoclingReader(pdf_pipeline_options=pdf_options)
    """
    
    @staticmethod
    def _get_supported_types() -> set[FileType]:
        """
        Get supported file types
        
        Note: All formats except PDF use Docling's default pipeline.
        Only PDF format can be customized via pdf_pipeline_options or vlm_pipeline_options.
        """
        return {
            FileType.PDF,
            FileType.WORD,
            FileType.POWERPOINT,
            FileType.EXCEL,
            FileType.HTML,
        }
    
    def __init__(
        self,
        pdf_pipeline_options: Optional[PdfPipelineOptions] = None,
        vlm_pipeline_options: Optional[VlmPipelineOptions] = None,
    ):
        """
        Initialize Docling reader
        
        Args:
            pdf_pipeline_options: Options for standard PDF pipeline
            vlm_pipeline_options: Options for VLM pipeline
            
        Pipeline selection logic:
            - If vlm_pipeline_options is provided, use VLM pipeline
            - Otherwise, use standard PDF pipeline (with pdf_pipeline_options if provided)
            - If both are provided, VLM pipeline takes precedence
        """
        self._pdf_pipeline_options = pdf_pipeline_options
        self._vlm_pipeline_options = vlm_pipeline_options
        
        # Build format options
        format_options = self._build_format_options()
        
        # Initialize converter
        self._converter = DocumentConverter(format_options=format_options)
    
    def _build_format_options(self) -> dict[InputFormat, PdfFormatOption]:
        """
        Build format options for DocumentConverter
        
        Note:
            - Only PDF format is configured here with custom pipeline options
            - Other formats (WORD, POWERPOINT, EXCEL, HTML, IMAGE) use Docling's 
              default pipeline, which is the recommended approach per official examples
            - This matches the official Docling multi-format conversion pattern
        
        Returns:
            Dict of format options (currently only PDF is customized)
        """
        # If VLM options provided, use VLM pipeline
        if self._vlm_pipeline_options is not None:
            return {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=self._vlm_pipeline_options,
                )
            }
        else:
            # Use standard PDF pipeline (with custom options if provided)
            return {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self._pdf_pipeline_options,
                )
            }
    
    def read(self, source: str) -> BaseDocument:
        """
        Read and convert a file to PDF document with structured data
        
        Args:
            source: File path (relative/absolute) or URL
            
        Returns:
            PDF document with markdown content and structured pages
            
        Raises:
            ValueError: If source is invalid or file format is not supported
        """
        # Validate and get complete info
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
        
        # Convert file using Docling
        result = self._converter.convert(source)
        docling_doc = result.document
        
        # Extract markdown content
        markdown_content = docling_doc.export_to_markdown()
        
        # ⚠️ WARNING: Known Issue - Cross-Page Table Splitting
        #    
        #    Docling has a critical limitation when handling tables that span multiple pages:
        #    
        #    Problem:
        #      When a table spans across page boundaries in the PDF, Docling will:
        #      1. Split it into separate tables at the page break
        #      2. Treat the first row of the continuation as a NEW table header
        #      3. Generate a new separator line (|---|---| etc.)
        #    
        #    Impact:
        #      - The most serious issue is NOT the splitting itself, but the MISIDENTIFICATION
        #        of data rows as headers
        #      - This corrupts the semantic meaning of the table
        #      - Example: In usda.md line 10679, a data row becomes a "header" of a "new" table
        #    
        #    Why We Can't Fix It Here:
        #      - By the time we get the Markdown output, table structure is already corrupted
        #      - Cannot distinguish "true separate tables" from "incorrectly split tables"
        #      - Any automatic merging risks combining genuinely separate tables
        #    
        #    Related:
        #      - GitHub Discussion #1104: Multi page table provenance
        #      - This is a known Docling limitation as of 2024
        #    
        #    Future Solution:
        #      - Requires upstream fix in Docling's TableFormer
        #      - May need manual intervention or human-in-the-loop for critical documents
        #      - Could potentially be addressed by analyzing table.prov data BEFORE Markdown export
        #    
        #    Recommendation:
        #      - For documents with critical cross-page tables, consider:
        #        * Manual review and correction
        #        * Alternative PDF parsers (e.g., MinerU with HTML table output)
        #        * Pre-processing PDFs to avoid page breaks in tables
        
        # 🔧 TODO: Remove this patch when Docling fixes Issue #1023
        #          (https://github.com/docling-project/docling/issues/1023)
        #          
        #          Docling currently exports all headers as H2 (##), losing hierarchical structure.
        #          We use SimpleHeaderLevelFixer to infer correct levels based on content patterns.
        #          
        #          This is a temporary workaround and should be removed once upstream is fixed.
        from .patches import SimpleHeaderLevelFixer
        fixer = SimpleHeaderLevelFixer()
        markdown_content = fixer.process(markdown_content)
        
        # Build structured pages
        pages = self._extract_pages(docling_doc)
        
        # Build metadata
        metadata = self._build_metadata(info, markdown_content, docling_doc)
        
        # Create PDF document
        return PDF(
            content=markdown_content,
            metadata=metadata,
            pages=pages
        )
    
    def _extract_pages(self, docling_doc: Any) -> list[Page]:
        """
        Extract structured page data from DoclingDocument
        
        Args:
            docling_doc: The DoclingDocument object
            
        Returns:
            List of Page objects with structured content
        """
        pages = []
        
        # Group items by page number
        page_items = {}
        
        # Process texts
        for text_item in docling_doc.texts:
            page_num = self._get_page_number(text_item)
            if page_num not in page_items:
                page_items[page_num] = {
                    'texts': [],
                    'tables': [],
                    'pictures': []
                }
            page_items[page_num]['texts'].append({
                'text': text_item.text if hasattr(text_item, 'text') else str(text_item),
                'type': text_item.label if hasattr(text_item, 'label') else 'text',
                'bbox': self._extract_bbox(text_item)
            })
        
        # Process tables
        for table_item in docling_doc.tables:
            page_num = self._get_page_number(table_item)
            if page_num not in page_items:
                page_items[page_num] = {
                    'texts': [],
                    'tables': [],
                    'pictures': []
                }
            page_items[page_num]['tables'].append({
                'data': self._extract_table_data(table_item),
                'bbox': self._extract_bbox(table_item)
            })
        
        # Process pictures
        for picture_item in docling_doc.pictures:
            page_num = self._get_page_number(picture_item)
            if page_num not in page_items:
                page_items[page_num] = {
                    'texts': [],
                    'tables': [],
                    'pictures': []
                }
            page_items[page_num]['pictures'].append({
                'caption': picture_item.caption.text if hasattr(picture_item, 'caption') and picture_item.caption else None,
                'bbox': self._extract_bbox(picture_item)
            })
        
        # Create Page objects
        for page_num in sorted(page_items.keys()):
            items = page_items[page_num]
            pages.append(Page(
                page_number=page_num,
                content=items,
                metadata={
                    'text_count': len(items['texts']),
                    'table_count': len(items['tables']),
                    'picture_count': len(items['pictures'])
                }
            ))
        
        return pages
    
    def _get_page_number(self, item: Any) -> int:
        """
        Extract page number from item
        
        Args:
            item: DocItem from DoclingDocument
            
        Returns:
            Page number (1-based), defaults to 1 if not found
        """
        # Try to get page number from prov (provenance)
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'page_no'):
                    return prov.page_no
        
        # Default to page 1
        return 1
    
    def _extract_bbox(self, item: Any) -> dict[str, float] | None:
        """
        Extract bounding box from item
        
        Args:
            item: DocItem from DoclingDocument
            
        Returns:
            Dict with bbox coordinates or None
        """
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'bbox'):
                    bbox = prov.bbox
                    return {
                        'l': bbox.l,
                        't': bbox.t,
                        'r': bbox.r,
                        'b': bbox.b
                    }
        return None
    
    def _extract_table_data(self, table_item: Any) -> dict[str, Any]:
        """
        Extract table data structure
        
        Args:
            table_item: TableItem from DoclingDocument
            
        Returns:
            Dict with table structure and data
        """
        result = {
            'grid': None,
            'markdown': None
        }
        
        # Try to get table grid
        if hasattr(table_item, 'data') and table_item.data:
            result['grid'] = table_item.data.grid if hasattr(table_item.data, 'grid') else None
        
        # Try to export as markdown
        if hasattr(table_item, 'export_to_markdown'):
            result['markdown'] = table_item.export_to_markdown()
        
        return result
    
    def _build_metadata(self, info: SourceInfo, content: str, docling_doc: Any) -> DocumentMetadata:
        """
        Build metadata object from SourceInfo and DoclingDocument
        
        Args:
            info: SourceInfo from validation
            content: Document content
            docling_doc: The DoclingDocument object
            
        Returns:
            DocumentMetadata object
        """
        # Get file size and calculate hash for local files
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
            # For URL sources, we cannot calculate file hash
            # Use a placeholder or skip (depending on requirements)
            # For now, raise an error since md5 is required
            raise ValueError(
                "File hash calculation not supported for URL sources. "
                "Please download the file first and read from local path."
            )
        
        # Extract custom docling metadata
        custom = {
            'text_items_count': len(docling_doc.texts),
            'table_items_count': len(docling_doc.tables),
            'picture_items_count': len(docling_doc.pictures)
        }
        
        # Add document name if available
        if hasattr(docling_doc, 'name') and docling_doc.name:
            custom['doc_name'] = docling_doc.name
        
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
            reader_name="DoclingReader",
            custom=custom
        )
