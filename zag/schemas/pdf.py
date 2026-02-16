"""
PDF document schema
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Dict
from pydantic import Field, ConfigDict

from .document import PageableDocument, Page
from .unit import UnitCollection, BaseUnit
from .metadata import UnitMetadata

# Try to import zag version from package metadata
try:
    from importlib.metadata import version as get_version
    ZAG_VERSION = get_version("zag-ai")
except Exception:
    ZAG_VERSION = "unknown"


class PDF(PageableDocument):
    """
    PDF document with page structure
    """
    
    model_config = ConfigDict(extra='ignore')  # Ignore unknown fields for version compatibility
    
    content: Any = None  # Raw PDF data or structured content
    units: List[BaseUnit] = Field(default_factory=list)  # Direct units (for table-only readers like Camelot)
    
    def __add__(self, other: "PDF") -> "PDF":
        """
        Merge two PDF documents into one
        
        Usage:
            merged_pdf = pdf1 + pdf2 + pdf3
        
        Merge logic:
            - content: Direct concatenation with newline separator
            - pages: Direct merge (keep original page numbers), adjust span offsets
            - units: Direct merge
            - metadata: Merge (page_count adds up, other fields keep left's values)
            - doc_id: Keep left PDF's doc_id (important for large file processing)
        
        Args:
            other: Another PDF to merge with this one
            
        Returns:
            New merged PDF document
            
        Note:
            This is designed for large file processing where the original file
            is read in page ranges and then merged. Page numbers are preserved
            from the original document (not renumbered).
        """
        if not isinstance(other, PDF):
            raise TypeError(f"Cannot merge PDF with {type(other).__name__}")
        
        # Merge content
        merged_content = ""
        offset = 0  # Character offset for span adjustment
        if self.content:
            merged_content = str(self.content)
            offset = len(merged_content) + 2  # +2 for "\n\n" separator
        if other.content:
            if merged_content:
                merged_content += "\n\n"
            merged_content += str(other.content)
        
        # Merge pages - keep original page numbers (no renumbering)
        # Adjust span offsets for pages from 'other'
        merged_pages: List[Page] = list(self.pages)
        
        for page in other.pages:
            # Adjust span if present
            if page.metadata and isinstance(page.metadata, dict) and 'span' in page.metadata:
                old_span = page.metadata['span']
                if old_span:
                    new_span = (old_span[0] + offset, old_span[1] + offset)
                    # Create new page with adjusted span
                    page = Page(
                        page_number=page.page_number,
                        content=page.content,
                        units=page.units,
                        metadata={**page.metadata, 'span': new_span}
                    )
            merged_pages.append(page)
        
        # Merge units (direct concatenation)
        merged_units: List[BaseUnit] = list(self.units) + list(other.units)
        
        # Merge metadata
        merged_metadata = self.metadata.model_copy(deep=True) if self.metadata else None
        
        if merged_metadata and other.metadata:
            # Add page counts
            other_page_count = other.metadata.custom.get("page_count", 0) if other.metadata.custom else 0
            if merged_metadata.custom:
                current_page_count = merged_metadata.custom.get("page_count", 0)
                merged_metadata.custom["page_count"] = current_page_count + other_page_count
            
            # Add content lengths
            if other.metadata.content_length:
                merged_metadata.content_length = (merged_metadata.content_length or 0) + other.metadata.content_length
        
        # Create merged PDF
        return PDF(
            doc_id=self.doc_id,  # Keep left's doc_id (will be set to original file hash)
            content=merged_content,
            pages=merged_pages,
            units=merged_units,
            metadata=merged_metadata
        )
    
    def split(self, splitter: 'BaseSplitter') -> UnitCollection:
        """
        Split PDF document into units with automatic metadata injection and page inference
        
        This method:
        1. Splits the document using the provided splitter
        2. Injects document metadata into each unit (file info for filtering/tracking)
        3. Automatically infers page numbers for each unit using fuzzy matching
        
        Args:
            splitter: The splitter to use
            
        Returns:
            UnitCollection with complete metadata populated
            
        Note:
            - Document metadata (file_name, md5, etc.) is injected for all splitters uniformly
            - Page numbers are inferred using position-based fuzzy matching
            - This ensures consistent behavior regardless of splitter combination
        """
        # Step 1: Normal splitting
        units = splitter.split(self)
        
        # Step 2: Inject document metadata into each unit
        # Store complete document metadata for filtering/tracking/auditing
        for unit in units:
            # Set source document ID (core relationship)
            unit.doc_id = self.doc_id
            
            # Set metadata
            if unit.metadata is None:
                unit.metadata = UnitMetadata()
            
            # Inject document metadata directly (not in custom)
            if self.metadata:
                unit.metadata.document = self.metadata.model_dump(exclude={'custom'})
                
                # Inject business metadata from document.metadata.custom to unit.metadata.custom
                # This allows business layer to pass custom data through the pipeline
                if self.metadata.custom:
                    unit.metadata.custom.update(self.metadata.custom)
        
        # Step 3: Infer page numbers using fuzzy position matching
        from ..utils.page_inference import infer_page_numbers
        infer_page_numbers(list(units), self.pages, full_content=self.content)
        
        return units
    
    # ========== Archive Methods ==========
    
    def dump(self, output_dir: Union[Path, str]) -> Path:
        """
        Dump PDF to archive format for sharing and version compatibility.
        
        Archive structure:
            {output_dir}/{doc_id}/
            ├── manifest.json      # Core metadata (doc_id, stats, version)
            ├── content.md         # Full markdown content
            ├── metadata.json      # PDF.metadata complete export
            ├── pages/             # Per-page content
            │   ├── page_001.md
            │   └── ...
            └── tables/            # Table data (optional, only if has tables)
                ├── table_001.parquet
                └── ...
        
        Args:
            output_dir: Output directory (archive will be created as {output_dir}/{doc_id}/)
            
        Returns:
            Path to the archive directory
            
        Example:
            >>> pdf = reader.read("document.pdf")
            >>> archive_path = pdf.dump(Path("./archive"))
            >>> # Share archive_path with colleagues
        """
        # Create archive directory
        archive_dir = Path(output_dir) / self.doc_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Write manifest.json
        manifest = {
            "version": "1.0",
            "zag_version": ZAG_VERSION,
            "created_at": datetime.now().isoformat(),
            "doc_id": self.doc_id,
            "stats": {
                "page_count": len(self.pages),
                "content_length": len(str(self.content)) if self.content else 0,
                "table_count": sum(1 for u in self.units if hasattr(u, 'df')),
            }
        }
        with open(archive_dir / "manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        # 2. Write content.md
        with open(archive_dir / "content.md", 'w', encoding='utf-8') as f:
            f.write(str(self.content) if self.content else "")
        
        # 3. Write metadata.json
        if self.metadata:
            metadata_dict = self.metadata.model_dump(mode='json')
            with open(archive_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
        
        # 4. Write pages/
        pages_dir = archive_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        for page in self.pages:
            page_file = pages_dir / f"page_{page.page_number:03d}.md"
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(str(page.content) if page.content else "")
        
        # 5. Write tables/ (optional, only if has tables with DataFrame)
        table_units = [u for u in self.units if hasattr(u, 'df') and u.df is not None]
        if table_units:
            tables_dir = archive_dir / "tables"
            tables_dir.mkdir(exist_ok=True)
            for i, unit in enumerate(table_units, 1):
                table_file = tables_dir / f"table_{i:03d}.parquet"
                unit.df.to_parquet(table_file, index=False)
        
        return archive_dir
    
    @classmethod
    def load(cls, archive_dir: Union[Path, str]) -> "PDF":
        """
        Load PDF from archive format.
        
        Args:
            archive_dir: Path to the archive directory (contains manifest.json)
            
        Returns:
            PDF document with content, pages, and metadata restored
            
        Raises:
            FileNotFoundError: If manifest.json not found
            ValueError: If archive format is invalid
            
        Example:
            >>> pdf = PDF.load(Path("./archive/abc123"))
            >>> print(f"Loaded: {len(pdf.pages)} pages")
        """
        archive_dir = Path(archive_dir)
        
        # Read manifest
        manifest_path = archive_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {archive_dir}")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        doc_id = manifest["doc_id"]
        
        # Read content.md
        content_path = archive_dir / "content.md"
        content = ""
        if content_path.exists():
            with open(content_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Read metadata.json
        metadata = None
        metadata_path = archive_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            # Import here to avoid circular import
            from .metadata import DocumentMetadata
            metadata = DocumentMetadata.model_validate(metadata_dict)
        
        # Read pages/
        pages = []
        pages_dir = archive_dir / "pages"
        if pages_dir.exists():
            for page_file in sorted(pages_dir.glob("page_*.md")):
                # Extract page number from filename: page_001.md -> 1
                page_num = int(page_file.stem.split('_')[1])
                with open(page_file, 'r', encoding='utf-8') as f:
                    page_content = f.read()
                pages.append(Page(
                    page_number=page_num,
                    content=page_content,
                    units=[],
                    metadata={}
                ))
        
        # Tables are NOT loaded back (user said they don't need them)
        # If needed in future, can add table loading logic here
        
        return cls(
            doc_id=doc_id,
            content=content,
            pages=pages,
            metadata=metadata,
            units=[]  # Units not stored in archive
        )
