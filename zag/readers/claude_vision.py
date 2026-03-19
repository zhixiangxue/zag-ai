"""
Claude Vision reader - parse PDF pages as images using Claude Vision API.

Each page is rasterized and sent to Claude for high-quality markdown extraction.
Produces clean HTML tables and preserves document structure.
"""

import base64
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseReader
from ..schemas import DocumentMetadata
from ..schemas.pdf import PDF
from ..schemas.document import Page
from ..utils.hash import calculate_file_hash


def _clean_content(text: str) -> str:
    """Post-process Claude output.

    - Collapse &nbsp; sequences (used as whitespace padding in PDFs) to a single space.
    - Compact HTML tables: strip inter-tag whitespace INSIDE <table>...</table> blocks only.
      (Applying >\\s+< globally would eat the newline between a heading and the next table,
      causing MarkdownHeaderSplitter to swallow the entire table into the heading line.)
    """
    # Collapse one or more &nbsp; (possibly interleaved with spaces) → single space
    text = re.sub(r'(\s*&nbsp;)+', ' ', text)

    # Compact whitespace between HTML tags, but ONLY inside <table>...</table> blocks
    def _compact_table(m: re.Match) -> str:
        return re.sub(r'>\s+<', '><', m.group(0))

    text = re.sub(r'<table[\s\S]*?</table>', _compact_table, text, flags=re.IGNORECASE)
    return text.strip()


PARSE_PROMPT = """Parse this document page into clean markdown. Follow these rules strictly:

1. Preserve all heading levels exactly as they appear (# H1, ## H2, ### H3, etc.)

2. For tables:
   - Use ONLY plain HTML <table> tags. Absolutely no style, class, or CSS attributes.
   - Before writing the table, silently count the total number of columns.
   - Set colspan and rowspan precisely for every merged cell — verify each row adds up to the total column count.
   - Use <th> for header cells, <td> for data cells.
   - Use <br> for line breaks within a cell.
   - Do NOT use markdown pipe tables under any circumstances.

3. Keep all body text, notes, footnotes, and captions verbatim.

4. Output ONLY the markdown content — no preamble, no explanation, no code fences."""


class ClaudeVisionReader(BaseReader):
    """
    Reader using Claude Vision API for high-quality PDF parsing.

    Converts each PDF page to a PNG image and uses Claude to parse it into
    clean markdown. Produces superior output for complex tables and formatted
    documents compared to text-extraction based readers.

    Returns PDF document with:
    - content: Full merged markdown (pages joined by double newline)
    - pages: List of Page objects (one per processed PDF page)
    - metadata: Document and parsing information

    Usage:
        reader = ClaudeVisionReader()
        doc = reader.read("path/to/file.pdf")

        # Use a specific Claude model
        reader = ClaudeVisionReader(model="claude-opus-4-5")
        doc = reader.read("path/to/file.pdf")

        # Specify page range (1-based, inclusive)
        doc = reader.read("path/to/file.pdf", page_range=(1, 50))

    Note:
        Requires ANTHROPIC_API_KEY environment variable or api_key parameter.
        Only supports PDF files.
        Dependency: pip install anthropic pymupdf
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        dpi: int = 250,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ):
        """
        Initialize ClaudeVisionReader.

        Args:
            model: Claude model identifier (default: claude-sonnet-4-6)
            dpi: Resolution for page rasterization (default: 250)
            max_tokens: Max output tokens per page (default: 4096)
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        """
        self.model = model
        self.dpi = dpi
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def read(
        self,
        source: Union[str, Path],
        page_range: Optional[Tuple[int, int]] = None,
    ) -> PDF:
        """
        Read and parse a PDF file using Claude Vision.

        Args:
            source: Local file path (str or Path).
            page_range: Optional (start_page, end_page) tuple, 1-based inclusive.
                       None means process all pages.

        Returns:
            PDF document with markdown content and structured page objects.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If page_range is invalid or ANTHROPIC_API_KEY is missing.
            ImportError: If required dependencies are missing.
        """
        # Validate dependencies
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for ClaudeVisionReader. "
                "Install with: pip install pymupdf"
            )

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic library is required for ClaudeVisionReader. "
                "Install with: pip install anthropic"
            )

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Please set it in your environment or pass api_key parameter."
            )

        pdf_path = Path(source).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Calculate document hash (used as doc_id)
        doc_id = calculate_file_hash(pdf_path)

        # Open PDF and resolve page range
        pdf_doc = fitz.open(str(pdf_path))
        total_pages = len(pdf_doc)

        if page_range is not None:
            start_page, end_page = page_range
            if start_page < 1 or end_page < 1:
                raise ValueError(
                    f"page_range values must be >= 1, got ({start_page}, {end_page})"
                )
            if end_page < start_page:
                raise ValueError(
                    f"end_page ({end_page}) must be >= start_page ({start_page})"
                )
            # Auto-clamp to actual page count
            end_page = min(total_pages, end_page)
            if start_page > total_pages:
                raise ValueError(
                    f"start_page ({start_page}) exceeds total pages ({total_pages})"
                )
        else:
            start_page = 1
            end_page = total_pages

        # Rasterize selected pages to PNG base64
        scale = self.dpi / 72.0
        pages_data: list[tuple[int, str]] = []  # (1-based page number, b64 PNG)

        for page_idx in range(start_page - 1, end_page):  # fitz uses 0-based index
            page = pdf_doc[page_idx]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            pages_data.append((page_idx + 1, b64))  # store with 1-based page number

        pdf_doc.close()

        # Parse each page with Claude Vision
        client = anthropic.Anthropic(api_key=self.api_key)
        page_contents: dict[int, str] = {}
        failed_pages: list[int] = []

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _parse_page(img_b64: str) -> str:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": PARSE_PROMPT},
                    ],
                }],
            )
            return _clean_content(resp.content[0].text)

        for i, (actual_page_num, img_b64) in enumerate(pages_data, 1):
            print(f"  [Claude Vision] page {i}/{len(pages_data)} (PDF page {actual_page_num})...", flush=True)
            try:
                page_contents[actual_page_num] = _parse_page(img_b64)
            except Exception as e:
                print(f"  [Claude Vision] page {actual_page_num} FAILED: {e}", flush=True)
                page_contents[actual_page_num] = ""
                failed_pages.append(actual_page_num)

        # Build Page objects and merged content
        pages: list[Page] = []
        parts: list[str] = []

        for actual_page_num, _ in pages_data:
            content = page_contents.get(actual_page_num, "")
            current_offset = len("\n\n".join(parts)) + (2 if parts else 0)
            pages.append(Page(
                page_number=actual_page_num,
                content=content,
                units=[],
                metadata={
                    "span": (current_offset, current_offset + len(content)),
                },
            ))
            parts.append(content)

        full_content = "\n\n".join(parts)

        # Build DocumentMetadata
        file_size = pdf_path.stat().st_size
        metadata = DocumentMetadata(
            source=str(pdf_path),
            source_type="local",
            file_type="pdf",
            file_name=pdf_path.name,
            file_size=file_size,
            file_extension=".pdf",
            md5=doc_id,
            content_length=len(full_content),
            mime_type=None,
            reader_name="ClaudeVisionReader",
            custom={
                "page_count": len(pages_data),
                "model": self.model,
                "dpi": self.dpi,
                "page_range": [start_page, end_page] if page_range else None,
                "failed_pages": failed_pages if failed_pages else None,
            },
        )

        return PDF(
            doc_id=doc_id,
            content=full_content,
            metadata=metadata,
            pages=pages,
        )
