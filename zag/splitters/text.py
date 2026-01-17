""" 
Text splitter with semantic boundary awareness
Split large text into chunks while respecting paragraph and sentence boundaries
Protects tables from being split - tables are always kept intact
"""

import re
from typing import Optional, Any, Union

import tiktoken

from .base import BaseSplitter
from ..schemas import BaseUnit, UnitMetadata, BaseDocument
from ..schemas.unit import TextUnit


class TextSplitter(BaseSplitter):
    """
    Split large text units by semantic boundaries
    
    Strategy (priority order):
    1. Detect and protect tables (never split tables)
    2. Split non-table text by paragraphs (\\n\\n)
    3. If paragraph still too large, split by sentences
    4. If sentence still too large, split by fixed size (last resort)
    
    Table Protection:
    - Markdown tables are detected and kept intact regardless of size
    - Tables are never split across chunks
    - Ensures data integrity for structured content
    
    This ensures semantic completeness while controlling chunk size.
    Designed to work in pipelines to handle oversized chunks from
    header-based splitting.
    
    Args:
        max_chunk_tokens: Maximum token count per chunk (default: 1200)
        tokenizer: Token counter (defaults to tiktoken cl100k_base)
    
    Example:
        >>> from zag.splitters import MarkdownHeaderSplitter, TextSplitter
        >>> 
        >>> # Use in pipeline to handle large sections
        >>> pipeline = (
        ...     MarkdownHeaderSplitter()
        ...     | TextSplitter(max_chunk_tokens=1200)
        ... )
        >>> units = doc.split(pipeline)
    
    Notes:
        - Preserves context_path from parent units
        - Maintains semantic integrity by respecting paragraph/sentence boundaries
        - Tables are protected and never split
        - Only falls back to hard splitting for extremely long sentences
    """
    
    # Markdown table pattern (same as TableParser)
    TABLE_PATTERN = re.compile(
        r'(\|.+\|[\r\n]+\|[\s\-:|]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)',
        re.MULTILINE
    )
    
    def __init__(
        self,
        max_chunk_tokens: int = 1200,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize text splitter
        
        Args:
            max_chunk_tokens: Maximum tokens per chunk
            tokenizer: Custom tokenizer (defaults to tiktoken cl100k_base)
        """
        self.max_chunk_tokens = max_chunk_tokens
        
        # Default to tiktoken (OpenAI's tokenizer)
        if tokenizer is None:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
            except Exception as e:
                raise ImportError(
                    "tiktoken is required for TextSplitter. "
                    "Install it with: pip install tiktoken"
                ) from e
        else:
            self.tokenizer = tokenizer
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Text to count
        
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def _do_split(self, input_data: Union[BaseDocument, list[BaseUnit]]) -> list[BaseUnit]:
        """
        Split document/unit by semantic boundaries
        
        Supports two input types:
        1. Document - Split the document content
        2. list[BaseUnit] - Split each unit individually
        
        Args:
            input_data: Document or units to split
            
        Returns:
            List of text units
        """
        # Check input type
        if isinstance(input_data, list):
            # Process units: split each unit that exceeds max size
            all_units = []
            for unit in input_data:
                # Check if this unit needs splitting
                token_count = self._count_tokens(unit.content)
                if token_count <= self.max_chunk_tokens:
                    # No splitting needed
                    all_units.append(unit)
                else:
                    # Split this unit
                    sub_units = self._split_by_semantic_boundaries(
                        unit.content, 
                        unit.metadata if hasattr(unit, 'metadata') else None
                    )
                    # Inherit source_doc_id from parent
                    for sub_unit in sub_units:
                        if hasattr(unit, 'source_doc_id'):
                            sub_unit.source_doc_id = unit.source_doc_id
                    all_units.extend(sub_units)
            return all_units
        else:
            # Process document
            content = input_data.content if hasattr(input_data, 'content') else ""
            original_metadata = None
            
            # Always use semantic boundaries splitting
            # This ensures tables are detected and protected even in small documents
            return self._split_by_semantic_boundaries(content, original_metadata)
    
    def _detect_tables(self, content: str) -> list[tuple[int, int, str]]:
        """
        Detect all tables in content and return their positions
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of (start_pos, end_pos, table_text) tuples
        """
        tables = []
        for match in self.TABLE_PATTERN.finditer(content):
            tables.append((match.start(), match.end(), match.group(0)))
        return tables
    
    def _split_by_semantic_boundaries(
        self,
        content: str,
        original_metadata: Optional[UnitMetadata]
    ) -> list[TextUnit]:
        """
        Split by paragraph -> sentence -> fixed size
        Protects tables - they are never split regardless of size
        
        Args:
            content: Text content to split
            original_metadata: Metadata to preserve
            
        Returns:
            List of text units
        """
        # 0. Detect tables first
        tables = self._detect_tables(content)
        
        # If no tables, use original logic
        if not tables:
            return self._split_text_without_tables(content, original_metadata)
        
        # If has tables, split around them
        return self._split_text_with_tables(content, tables, original_metadata)
    
    def _split_text_without_tables(
        self,
        content: str,
        original_metadata: Optional[UnitMetadata]
    ) -> list[TextUnit]:
        """
        Original splitting logic for text without tables
        
        Args:
            content: Text content to split
            original_metadata: Metadata to preserve
            
        Returns:
            List of text units
        """
        # Early return: if content fits in one chunk, return as is
        token_count = self._count_tokens(content)
        if token_count <= self.max_chunk_tokens:
            unit = TextUnit(
                unit_id=self.generate_unit_id(),
                content=content,
                metadata=original_metadata.model_copy() if original_metadata else UnitMetadata()
            )
            return [unit]
        
        # 1. Try splitting by paragraphs
        paragraphs = content.split('\n\n')
        
        # If only one paragraph, go straight to sentence splitting
        if len(paragraphs) == 1:
            return self._split_by_sentences(content, original_metadata)
        
        # Group paragraphs into chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            para_tokens = self._count_tokens(para)
            
            # Single paragraph exceeds max - need to split it further
            if para_tokens > self.max_chunk_tokens:
                # Save current accumulated chunks
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split this oversized paragraph
                sub_units = self._split_by_sentences(para, original_metadata)
                chunks.extend([u.content for u in sub_units])
                continue
            
            # Check if we can add to current chunk
            if current_tokens + para_tokens <= self.max_chunk_tokens:
                current_chunk.append(para)
                current_tokens += para_tokens
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
        
        # Save last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Convert to TextUnits
        return self._chunks_to_units(chunks, original_metadata)
    
    def _split_text_with_tables(
        self,
        content: str,
        tables: list[tuple[int, int, str]],
        original_metadata: Optional[UnitMetadata]
    ) -> list[TextUnit]:
        """
        Split text while protecting tables
        
        Strategy:
        1. Check if entire content (with tables) fits in one chunk
        2. If yes: keep as single unit (preserve context)
        3. If no: split into segments [text, table, text, table, ...]
        4. Split text segments normally, keep table segments intact
        
        Args:
            content: Text content to split
            tables: List of (start_pos, end_pos, table_text)
            original_metadata: Metadata to preserve
            
        Returns:
            List of text units
        """
        # Early check: if entire content fits, keep it together
        # This preserves semantic context (text + tables as a whole)
        total_tokens = self._count_tokens(content)
        if total_tokens <= self.max_chunk_tokens:
            unit = TextUnit(
                unit_id=self.generate_unit_id(),
                content=content,
                metadata=original_metadata.model_copy() if original_metadata else UnitMetadata()
            )
            return [unit]
        
        # Content too large: split into segments
        segments = []
        last_pos = 0
        
        # Segment content into [text, table, text, table, ...]
        for start_pos, end_pos, table_text in tables:
            # Add text before table
            if start_pos > last_pos:
                text_before = content[last_pos:start_pos]
                if text_before.strip():
                    segments.append(('text', text_before))
            
            # Add table (protected)
            segments.append(('table', table_text))
            last_pos = end_pos
        
        # Add remaining text after last table
        if last_pos < len(content):
            text_after = content[last_pos:]
            if text_after.strip():
                segments.append(('text', text_after))
        
        # Process segments
        all_units = []
        
        for seg_type, seg_content in segments:
            if seg_type == 'table':
                # Table: keep as single unit, no splitting
                unit = TextUnit(
                    unit_id=self.generate_unit_id(),
                    content=seg_content,
                    metadata=original_metadata.model_copy() if original_metadata else UnitMetadata()
                )
                all_units.append(unit)
            else:
                # Text: split normally
                text_units = self._split_text_without_tables(seg_content, original_metadata)
                all_units.extend(text_units)
        
        return all_units
    
    def _split_by_sentences(
        self,
        content: str,
        original_metadata: Optional[UnitMetadata]
    ) -> list[TextUnit]:
        """
        Split by sentence boundaries
        
        Supports both English and Chinese sentence endings.
        
        Args:
            content: Text content to split
            original_metadata: Metadata to preserve
            
        Returns:
            List of text units
        """
        # Split by sentence endings (English and Chinese)
        # Pattern matches: . ! ? 。 ! ? followed by space or end of string
        sentence_pattern = r'(?<=[.!?。!?])\s+'
        sentences = re.split(sentence_pattern, content)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            sentence_tokens = self._count_tokens(sentence)
            
            # Single sentence exceeds max (rare) - hard split it
            if sentence_tokens > self.max_chunk_tokens:
                # Save current chunks
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Hard split this oversized sentence
                hard_chunks = self._hard_split(sentence)
                chunks.extend(hard_chunks)
                continue
            
            # Check if we can add to current chunk
            if current_tokens + sentence_tokens <= self.max_chunk_tokens:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Save current and start new
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
        
        # Save last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return self._chunks_to_units(chunks, original_metadata)
    
    def _hard_split(self, text: str) -> list[str]:
        """
        Last resort: split by fixed token size
        
        Used only when a single sentence exceeds max_chunk_tokens.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        tokens = self.tokenizer.encode(text)
        
        for i in range(0, len(tokens), self.max_chunk_tokens):
            chunk_tokens = tokens[i:i + self.max_chunk_tokens]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        
        return chunks
    
    def _chunks_to_units(
        self,
        chunks: list[str],
        original_metadata: Optional[UnitMetadata]
    ) -> list[TextUnit]:
        """
        Convert text chunks to TextUnits, preserving metadata
        
        Args:
            chunks: List of text chunks
            original_metadata: Metadata from original unit/document
            
        Returns:
            List of TextUnits
        """
        units = []
        
        for chunk in chunks:
            # Create new metadata, preserving context_path if it exists
            if original_metadata:
                metadata = original_metadata.model_copy()
            else:
                metadata = UnitMetadata()
            
            unit = TextUnit(
                unit_id=self.generate_unit_id(),
                content=chunk,
                metadata=metadata
            )
            units.append(unit)
        
        return units
    
    def __repr__(self) -> str:
        """String representation"""
        return f"TextSplitter(max_chunk_tokens={self.max_chunk_tokens})"
