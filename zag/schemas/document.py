"""
Document-related classes

This module contains base document classes and page structures.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

from .metadata import DocumentMetadata, UnitMetadata
from .unit import BaseUnit, UnitCollection


class BaseDocument(BaseModel, ABC):
    """
    Base class for all document types
    Acts as a container for structured parsed results
    
    Note:
        doc_id must be provided by Reader (typically based on file hash for idempotency)
        metadata is now a structured DocumentMetadata object
    """
    
    doc_id: str  # Required: Reader should provide hash-based ID
    metadata: DocumentMetadata
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }
    
    @abstractmethod
    def split(self, splitter: 'BaseSplitter') -> 'UnitCollection':
        """
        Split document into units using the given splitter
        
        Args:
            splitter: The splitter to use for splitting
            
        Returns:
            UnitCollection containing the split units
        """
        pass


class Page(BaseModel):
    """
    Generic page structure for documents with page-level data
    
    Design:
        - content: Human-readable page content (Markdown text with tables)
        - units: Machine-processable structured units (TextUnit, TableUnit, etc.)
    
    Both fields serve different purposes:
        - content: For display, LLM context, human reading
        - units: For RAG processing, vector indexing, structured operations
    """
    
    page_number: int
    content: str = ""  # Full page text in Markdown format
    units: list['BaseUnit'] = Field(default_factory=list)  # Structured units
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True,
    }


class PageableDocument(BaseDocument):
    """
    Base class for documents with page structure (PDF, DOCX, PPTX, MD, TXT, etc.).
    Carries all shared logic: merge, split, dump/load.
    """

    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)

    content: Any = None
    pages: List[Page] = Field(default_factory=list)
    units: List[BaseUnit] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Page helpers
    # ------------------------------------------------------------------

    def get_page(self, page_num: int) -> Optional[Page]:
        """Return the Page with the given page_number, or None."""
        for page in self.pages:
            if page.page_number == page_num:
                return page
        return None

    def get_page_count(self) -> int:
        """Total number of pages."""
        return len(self.pages)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def __add__(self, other: "PageableDocument") -> "PageableDocument":
        """
        Merge two paged documents into one.
        Used for chunked large-file processing: doc = part1 + part2 + ...

        Rules:
        - content: concatenated with \\n\\n
        - pages: merged (original page numbers preserved), span offsets adjusted
        - units: concatenated
        - metadata: content_length summed; other fields from left
        - doc_id: left's doc_id is kept
        """
        if not isinstance(other, PageableDocument):
            raise TypeError(
                f"Cannot merge {type(self).__name__} with {type(other).__name__}"
            )

        # Merge content
        merged_content = ""
        offset = 0
        if self.content:
            merged_content = str(self.content)
            offset = len(merged_content) + 2  # +2 for "\n\n"
        if other.content:
            merged_content = (merged_content + "\n\n" if merged_content else "") + str(other.content)

        # Merge pages – adjust span offsets for pages from 'other'
        merged_pages: List[Page] = list(self.pages)
        for page in other.pages:
            if page.metadata and isinstance(page.metadata, dict) and "span" in page.metadata:
                old_span = page.metadata["span"]
                if old_span:
                    new_span = (old_span[0] + offset, old_span[1] + offset)
                    page = Page(
                        page_number=page.page_number,
                        content=page.content,
                        units=page.units,
                        metadata={**page.metadata, "span": new_span},
                    )
            merged_pages.append(page)

        # Merge units
        merged_units: List[BaseUnit] = list(self.units) + list(other.units)

        # Merge metadata
        merged_metadata = self.metadata.model_copy(deep=True) if self.metadata else None
        if merged_metadata and other.metadata and other.metadata.content_length:
            merged_metadata.content_length = (
                merged_metadata.content_length or 0
            ) + other.metadata.content_length

        return type(self)(
            doc_id=self.doc_id,
            content=merged_content,
            pages=merged_pages,
            units=merged_units,
            metadata=merged_metadata,
        )

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(self, splitter: "BaseSplitter") -> UnitCollection:
        """
        Split document into units, inject document metadata, infer page numbers.
        """
        units = splitter.split(self)

        for unit in units:
            unit.doc_id = self.doc_id
            if unit.metadata is None:
                unit.metadata = UnitMetadata()
            if self.metadata:
                unit.metadata.document = self.metadata.model_dump(exclude={"custom"})
                if self.metadata.custom:
                    unit.metadata.custom.update(self.metadata.custom)

        from ..utils.page_inference import infer_page_numbers
        infer_page_numbers(list(units), self.pages, full_content=self.content)

        return units

    # ------------------------------------------------------------------
    # Archive: dump / load
    # ------------------------------------------------------------------

    def dump(self, output_dir: Union[Path, str]) -> Path:
        """
        Persist document to an archive directory.

        Structure::

            {output_dir}/{doc_id}/
            ├── manifest.json
            ├── content.md
            ├── metadata.json
            ├── pages/
            │   ├── page_001.md
            │   └── ...
            └── tables/   (only if table units with DataFrames exist)

        Returns:
            Path to the archive directory.
        """
        archive_dir = Path(output_dir) / self.doc_id
        archive_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "doc_id": self.doc_id,
            "doc_type": type(self).__name__,
            "stats": {
                "page_count": len(self.pages),
                "content_length": len(str(self.content)) if self.content else 0,
                "table_count": sum(1 for u in self.units if hasattr(u, "df")),
            },
        }
        with open(archive_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        with open(archive_dir / "content.md", "w", encoding="utf-8") as f:
            f.write(str(self.content) if self.content else "")

        if self.metadata:
            with open(archive_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    self.metadata.model_dump(mode="json"), f,
                    ensure_ascii=False, indent=2
                )

        pages_dir = archive_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        for page in self.pages:
            with open(pages_dir / f"page_{page.page_number:03d}.md", "w", encoding="utf-8") as f:
                f.write(str(page.content) if page.content else "")

        table_units = [u for u in self.units if hasattr(u, "df") and u.df is not None]
        if table_units:
            tables_dir = archive_dir / "tables"
            tables_dir.mkdir(exist_ok=True)
            for i, unit in enumerate(table_units, 1):
                unit.df.to_parquet(tables_dir / f"table_{i:03d}.parquet", index=False)

        return archive_dir

    @classmethod
    def load(cls, archive_dir: Union[Path, str]) -> "PageableDocument":
        """
        Load a document from an archive directory created by :meth:`dump`.

        Raises:
            FileNotFoundError: if manifest.json is missing.
        """
        archive_dir = Path(archive_dir)
        manifest_path = archive_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {archive_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        doc_id = manifest["doc_id"]

        content = ""
        content_path = archive_dir / "content.md"
        if content_path.exists():
            with open(content_path, "r", encoding="utf-8") as f:
                content = f.read()

        metadata = None
        metadata_path = archive_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                md_dict = json.load(f)
            from .metadata import DocumentMetadata
            metadata = DocumentMetadata.model_validate(md_dict)

        pages = []
        pages_dir = archive_dir / "pages"
        if pages_dir.exists():
            for page_file in sorted(pages_dir.glob("page_*.md")):
                page_num = int(page_file.stem.split("_")[1])
                with open(page_file, "r", encoding="utf-8") as f:
                    page_content = f.read()
                pages.append(
                    Page(page_number=page_num, content=page_content, units=[], metadata={})
                )

        return cls(
            doc_id=doc_id,
            content=content,
            pages=pages,
            metadata=metadata,
            units=[],
        )
