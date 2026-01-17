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
from ..schemas import BaseDocument, DocumentMetadata, Page
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
        
        # Build structured pages
        pages = self._extract_pages(docling_doc)
        
        # Build metadata
        metadata = self._build_metadata(info, markdown_content, docling_doc)
        
        # Create PDF document with doc_id based on file hash
        # This ensures idempotency: same file -> same doc_id
        pdf_doc = PDF(
            doc_id=metadata.md5,  # Use file hash as doc_id
            content=markdown_content,
            metadata=metadata,
            pages=pages
        )
        
        return pdf_doc
    
    def _extract_pages(self, docling_doc: Any) -> list[Page]:
        """
        Extract pages from DoclingDocument
        
        Simply exports each page's Markdown content using Docling's native export.
        No complex unit construction - just pure page content.
        
        Args:
            docling_doc: The DoclingDocument object
            
        Returns:
            List of Page objects with Markdown content
        """
        pages = []
        
        # Get page count from document
        # Note: doc.pages is a dict in Docling, keys are page numbers
        if hasattr(docling_doc, 'pages') and docling_doc.pages:
            page_numbers = sorted(docling_doc.pages.keys())
        else:
            # Fallback: try to infer from items
            page_numbers = set()
            for item in docling_doc.texts:
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            page_numbers.add(prov.page_no)
            page_numbers = sorted(page_numbers) if page_numbers else [1]
        
        # Create Page objects
        for page_num in page_numbers:
            # Export page content using Docling's native export
            page_content = docling_doc.export_to_markdown(page_no=page_num)
            
            pages.append(Page(
                page_number=page_num,
                content=page_content,
                units=[],  # Empty - no unit construction
                metadata={}
            ))
        
        return pages
    

    
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
