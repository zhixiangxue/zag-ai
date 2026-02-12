"""
Chonkie-based text splitter

This module provides ChunkSplitter, a thin wrapper around Chonkie's TokenChunker
for simple and reliable token-based text splitting.
"""

from typing import Union, Optional

from chonkie import TokenChunker

from .base import BaseSplitter
from ..schemas import BaseUnit, UnitMetadata, BaseDocument
from ..schemas.unit import TextUnit


class ChunkSplitter(BaseSplitter):
    """
    Simple token-based text splitter (wraps Chonkie's TokenChunker)
    
    This is a thin wrapper around Chonkie's TokenChunker, providing a simple
    and reliable token-based splitting solution for evaluation and general use.
    
    Why use ChunkSplitter?
    - Simple: Fixed-size chunks with predictable token counts
    - Reliable: Built on Chonkie, a mature chunking library
    - Fast: No complex boundary detection or table protection
    - Universal: Works with any plain text content
    
    Compared to TextSplitter:
    - ChunkSplitter: Simple, fixed-size chunks (uses Chonkie)
                     Ideal for evaluation scenarios where predictability matters
    - TextSplitter: Smart semantic splitting with table protection
                    Ideal for production with complex documents
    
    Works with ANY plain text content:
    - Markdown documents
    - Plain text files
    - Extracted text from PDFs
    - HTML content (after stripping tags)
    
    Use cases:
    - Evaluation/testing scenarios (simple, predictable chunk sizes)
    - General RAG applications without complex requirements
    - Quick prototyping and experimentation
    
    Args:
        chunk_size: Target chunk size in TOKENS (default: 512)
                   Uses tiktoken (cl100k_base encoding) for token counting
        chunk_overlap: Number of TOKENS to overlap between chunks (default: 50)
                      Helps maintain context across chunk boundaries
    
    Example:
        >>> # Basic usage
        >>> splitter = ChunkSplitter(chunk_size=512)
        >>> units = doc.split(splitter)
        >>> 
        >>> # With overlap for better context
        >>> splitter = ChunkSplitter(chunk_size=512, chunk_overlap=50)
        >>> units = doc.split(splitter)
    
    Note:
        This splitter depends on the 'chonkie' package. Install it with:
        pip install chonkie
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize ChunkSplitter
        
        Args:
            chunk_size: Target chunk size in tokens (default: 512)
            chunk_overlap: Overlap between chunks in tokens (default: 50)
        
        Raises:
            ValueError: If parameters are invalid
            ImportError: If chonkie is not installed
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create Chonkie TokenChunker
        try:
            self.chunker = TokenChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except Exception as e:
            raise ImportError(
                "Failed to initialize Chonkie TokenChunker. "
                "Make sure 'chonkie' is installed: pip install chonkie"
            ) from e
    
    def _do_split(self, input_data: Union[BaseDocument, list[BaseUnit]]) -> list[BaseUnit]:
        """
        Split document or units into chunks
        
        Supports two input types:
        1. Document - Split the document content
        2. list[BaseUnit] - Split each unit's content individually
        
        Args:
            input_data: Document or units to split
            
        Returns:
            List of TextUnits with metadata
        """
        # Check input type
        if isinstance(input_data, list):
            # Process units: split each unit individually
            all_units = []
            for unit in input_data:
                # Split this unit's content
                sub_units = self._split_content(unit.content, parent_unit=unit)
                all_units.extend(sub_units)
            return all_units
        else:
            # Process document: split the document content
            content = input_data.content if hasattr(input_data, 'content') else ""
            return self._split_content(content)
    
    def _split_content(
        self,
        content: str,
        parent_unit: Optional[BaseUnit] = None
    ) -> list[TextUnit]:
        """
        Split text content into chunks using Chonkie
        
        Args:
            content: Text content to split
            parent_unit: Parent unit (if splitting from unit)
            
        Returns:
            List of TextUnits with chunk metadata
        """
        if not content or not content.strip():
            return []
        
        # Use Chonkie to split text into chunks
        chonkie_chunks = self.chunker.chunk(content)
        
        # Convert Chonkie chunks to our TextUnit format
        units = []
        for i, chunk in enumerate(chonkie_chunks):
            unit = self._build_unit_from_chunk(
                chunk_text=chunk.text,
                chunk_index=i,
                token_count=chunk.token_count,
                parent_unit=parent_unit
            )
            units.append(unit)
        
        # Build chain relationships (prev/next unit IDs)
        for i in range(len(units)):
            if i > 0:
                units[i].prev_unit_id = units[i - 1].unit_id
            if i < len(units) - 1:
                units[i].next_unit_id = units[i + 1].unit_id
        
        return units
    
    def _build_unit_from_chunk(
        self,
        chunk_text: str,
        chunk_index: int,
        token_count: int,
        parent_unit: Optional[BaseUnit] = None
    ) -> TextUnit:
        """
        Build TextUnit from Chonkie chunk
        
        Args:
            chunk_text: Content of this chunk
            chunk_index: Index of this chunk (0-based)
            token_count: Token count from Chonkie
            parent_unit: Parent unit to inherit metadata from
            
        Returns:
            TextUnit with complete metadata
        """
        # Inherit metadata from parent if exists
        if parent_unit and parent_unit.metadata:
            # Copy parent metadata
            metadata = parent_unit.metadata.model_copy()
        else:
            # Create new metadata
            metadata = UnitMetadata()
        
        # Create TextUnit
        unit = TextUnit(
            unit_id=self.generate_unit_id(),
            content=chunk_text,
            metadata=metadata
        )
        
        return unit
    
    def __repr__(self) -> str:
        """String representation"""
        return f"ChunkSplitter(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"
