"""PDF reader using LightOnOCR for advanced OCR and table extraction."""

import os
from pathlib import Path
from typing import List, Literal, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF is required. Install with: pip install PyMuPDF"
    )

from zag.schemas import BaseDocument, DocumentMetadata, Page
from zag.schemas.pdf import PDF
from .base import BaseReader
from ..utils.hash import calculate_file_hash


class LightOnOCRReader(BaseReader):
    """
    Reader for PDF files using LightOnOCR for advanced OCR and table extraction.
    
    LightOnOCR is a state-of-the-art vision-language model specialized for:
    - Document OCR (PDF, scans, images)
    - Table structure recognition
    - Multi-column layouts
    - Form processing
    
    Features:
    - End-to-end trainable, no multi-stage pipelines
    - Fast: 5.71 pages/sec on H100 GPU
    - Supports CPU, CUDA, and MPS (Apple Silicon)
    - Lightweight: ~2GB model size
    
    Model variants:
    - lightonai/LightOnOCR-2-1B: OCR only (default)
    - lightonai/LightOnOCR-2-1B-bbox: OCR + bounding box detection
    
    Example:
        >>> # Use default model (OCR only)
        >>> reader = LightOnOCRReader()
        >>> doc = reader.read("document.pdf")
        
        >>> # Use bbox model for image detection
        >>> reader = LightOnOCRReader(
        ...     model_name="lightonai/LightOnOCR-2-1B-bbox"
        ... )
        
        >>> # Force CPU mode (for systems without GPU)
        >>> reader = LightOnOCRReader(device="cpu")
        
        >>> # Use MPS for Apple Silicon
        >>> reader = LightOnOCRReader(device="mps")
    """
    
    def __init__(
        self,
        model_name: str = "lightonai/LightOnOCR-2-1B",
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        dpi: int = 200,
        batch_size: int = 1,
    ):
        """
        Initialize LightOnOCR reader.
        
        Args:
            model_name: HuggingFace model name
                - "lightonai/LightOnOCR-2-1B": OCR only (default, faster)
                - "lightonai/LightOnOCR-2-1B-bbox": OCR + bounding boxes
            device: Device to run inference on
                - "auto": Automatically detect (CUDA > MPS > CPU)
                - "cpu": Force CPU mode (slower but compatible)
                - "cuda": NVIDIA GPU
                - "mps": Apple Silicon GPU
            dpi: DPI for PDF to image conversion (default: 200)
                Higher DPI = better quality but slower
            batch_size: Number of pages to process at once (default: 1)
        
        Note:
            Model will be downloaded on first use (~2GB).
            Set HF_HOME env variable to control cache location.
        """
        self.model_name = model_name
        self.device = device
        self.dpi = dpi
        self.batch_size = batch_size
        
        # Lazy loading - model loaded on first use
        self._model = None
        self._processor = None
        self._actual_device = None
    
    @property
    def supported_formats(self) -> List[str]:
        """Return supported file formats."""
        return [".pdf"]
    
    def _load_model(self):
        """Lazy load model and processor."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required. Install with:\n"
                "  pip install transformers torch pillow"
            )
        
        print(f"Loading LightOnOCR model: {self.model_name}")
        print("This may take a while on first run (downloading ~2GB)...")
        
        # Load processor and model
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForVision2Seq.from_pretrained(self.model_name)
        
        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                self._actual_device = "cuda"
            elif torch.backends.mps.is_available():
                # Check macOS version for MPS compatibility
                import platform
                mac_version = tuple(map(int, platform.mac_ver()[0].split('.')[:2]))
                if mac_version >= (13, 2):
                    self._actual_device = "mps"
                else:
                    print(f"Warning: macOS {platform.mac_ver()[0]} < 13.2, falling back to CPU")
                    self._actual_device = "cpu"
            else:
                self._actual_device = "cpu"
        else:
            self._actual_device = self.device
        
        # Move model to device
        self._model = self._model.to(self._actual_device)
        print(f"Model loaded on device: {self._actual_device}")
    
    def read(self, file_path: str, **kwargs) -> BaseDocument:
        """
        Read PDF file and extract content using LightOnOCR.
        
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
        
        # Lazy load model
        self._load_model()
        
        # Convert PDF to images and process
        print(f"Processing PDF: {file_path.name}")
        pages = self._process_pdf(file_path)
        
        # Build full content
        full_text = "\n\n".join(page.content for page in pages)
        
        # Create metadata
        metadata = self._extract_metadata(file_path, full_text, len(pages))
        
        # Create PDF document
        pdf_doc = PDF(
            doc_id=metadata.md5,
            content=full_text,
            metadata=metadata,
            pages=pages
        )
        
        return pdf_doc
    
    def _process_pdf(self, file_path: Path) -> List[Page]:
        """
        Process PDF file page by page.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            List of Page objects with OCR text
        """
        from PIL import Image
        import torch
        import numpy as np
        
        pages = []
        
        # Open PDF
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        print(f"Processing {total_pages} pages...")
        
        for page_num, page in enumerate(doc, start=1):
            print(f"  Page {page_num}/{total_pages}...", end=" ", flush=True)
            
            # Convert page to image
            # Use matrix to control DPI: dpi = 72 * zoom
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            # Force RGB mode (remove alpha channel if present)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL Image via numpy array to avoid dtype issues
            # PyMuPDF returns samples as bytes, need to convert properly
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img = Image.fromarray(img_array, mode='RGB')
            
            # OCR inference
            inputs = self._processor(images=img, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self._actual_device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_new_tokens=2048)
            
            # Decode output
            text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Create page object
            pages.append(Page(
                page_number=page_num,
                content=text,
                units=[],
                metadata={
                    "ocr_model": self.model_name,
                    "device": self._actual_device,
                    "dpi": self.dpi,
                }
            ))
            
            print("✓")
        
        doc.close()
        print(f"✓ Completed {total_pages} pages")
        
        return pages
    
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
        # Calculate file hash
        try:
            file_size = file_path.stat().st_size
            file_hash = calculate_file_hash(file_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to calculate file hash for {file_path}: {e}"
            )
        
        # Try to get PDF metadata using PyMuPDF
        custom = {
            "doc_name": file_path.name,
            "total_pages": num_pages,
            "ocr_model": self.model_name,
            "ocr_device": self._actual_device,
            "ocr_dpi": self.dpi,
        }
        
        try:
            doc = fitz.open(file_path)
            metadata = doc.metadata
            
            if metadata:
                # Add title if available
                if metadata.get('title'):
                    custom["title"] = metadata.get('title')
                
                # Add author if available
                if metadata.get('author'):
                    custom["author"] = metadata.get('author')
            
            doc.close()
        except Exception:
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
            reader_name="LightOnOCRReader",
            custom=custom
        )
