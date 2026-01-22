"""
PDF document schema
"""

from typing import Any, List
from pydantic import Field

from .document import PageableDocument
from .unit import UnitCollection, BaseUnit
from .metadata import UnitMetadata


class PDF(PageableDocument):
    """
    PDF document with page structure
    """
    
    content: Any = None  # Raw PDF data or structured content
    units: List[BaseUnit] = Field(default_factory=list)  # Direct units (for table-only readers like Camelot)
    
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
        infer_page_numbers(list(units), self.pages)
        
        return units
