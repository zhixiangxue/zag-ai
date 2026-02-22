""" 
Table splitter for splitting large tables by token size
Split tables while preserving headers for each chunk
"""

import re
import tiktoken
from typing import Optional, Union

from .base import BaseSplitter
from ..schemas import BaseDocument, BaseUnit, UnitMetadata
from ..schemas.unit import TextUnit


class TableSplitter(BaseSplitter):
    """
    Split large tables by token size while preserving headers
    
    Strategy:
    1. Detect tables in content
    2. For tables exceeding max_chunk_tokens, split into chunks by rows
    3. Each chunk keeps the original header rows
    4. Accumulate rows until reaching max_chunk_tokens
    5. Each chunk contains at least min_rows (even if exceeds token limit)
    
    This ensures large tables (like glossaries, TOCs) are split into
    manageable chunks while maintaining table structure integrity.
    
    Args:
        max_chunk_tokens: Maximum tokens per table chunk (default: 1500)
        min_rows: Minimum rows per chunk (default: 1)
                  Even if single row exceeds max_chunk_tokens, it will be kept
    
    Example:
        >>> from zag.splitters import TableSplitter
        >>> 
        >>> # Use in pipeline to handle large tables
        >>> pipeline = (
        ...     MarkdownHeaderSplitter()
        ...     | TextSplitter(max_chunk_size=1200)
        ...     | TableSplitter(max_chunk_tokens=1500)
        ...     | RecursiveMergingSplitter(target_token_size=800)
        ... )
        >>> units = doc.split(pipeline)
    
    Notes:
        - Splits by token count, not row count (more intelligent)
        - Only splits tables that exceed max_chunk_tokens
        - Preserves table headers in each chunk
        - Maintains row order and continuity
        - Works seamlessly in pipelines
        - Does not modify non-table content
    """
    
    # Markdown table pattern (same as TextSplitter)
    TABLE_PATTERN = re.compile(
        r'(\|.+\|[\r\n]+\|[\s\-:|]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)',
        re.MULTILINE
    )
    
    # HTML table pattern
    HTML_TABLE_PATTERN = re.compile(
        r'<table[^>]*>.*?</table>',
        re.DOTALL | re.IGNORECASE
    )
    
    def __init__(self, max_chunk_tokens: int = 1500, min_rows: int = 1):
        """
        Initialize table splitter
        
        Args:
            max_chunk_tokens: Maximum tokens per chunk (excluding headers)
            min_rows: Minimum rows per chunk (default: 1)
                     Ensures at least 1 row even if it exceeds token limit
        """
        if max_chunk_tokens < 100:
            raise ValueError("max_chunk_tokens must be at least 100")
        if min_rows < 1:
            raise ValueError("min_rows must be at least 1")
        
        self.max_chunk_tokens = max_chunk_tokens
        self.min_rows = min_rows
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            raise ImportError(
                "tiktoken is required for TableSplitter. "
                "Install it with: pip install tiktoken"
            ) from e
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def _do_split(self, input_data: Union[BaseDocument, list[BaseUnit]]) -> list[BaseUnit]:
        """
        Split large tables in document/units
        
        Supports two input types:
        1. Document - Process the document content
        2. list[BaseUnit] - Process each unit individually
        
        Args:
            input_data: Document or units to process
            
        Returns:
            List of units with large tables split
        """
        if isinstance(input_data, list):
            # Process units: split tables in each unit
            all_units = []
            for unit in input_data:
                # Check if unit contains tables
                if self._has_table(unit.content):
                    # Split tables in this unit
                    sub_units = self._split_tables_in_content(
                        unit.content,
                        unit.metadata if hasattr(unit, 'metadata') else None
                    )
                    # Inherit source_doc_id from parent
                    for sub_unit in sub_units:
                        if hasattr(unit, 'source_doc_id'):
                            sub_unit.source_doc_id = unit.source_doc_id
                    all_units.extend(sub_units)
                else:
                    # No tables, keep as is
                    all_units.append(unit)
            return all_units
        else:
            # Process document
            content = input_data.content if hasattr(input_data, 'content') else ""
            return self._split_tables_in_content(content, None)
    
    def _has_table(self, content: str) -> bool:
        """Check if content contains Markdown or HTML tables"""
        return bool(
            self.TABLE_PATTERN.search(content) or 
            self.HTML_TABLE_PATTERN.search(content)
        )
    
    def _split_tables_in_content(
        self,
        content: str,
        original_metadata: Optional[UnitMetadata]
    ) -> list[TextUnit]:
        """
        Split tables in content while preserving non-table parts
        
        Args:
            content: Text content to process
            original_metadata: Metadata to preserve
            
        Returns:
            List of text units
        """
        # Find all markdown tables
        markdown_tables = [
            {
                'start': match.start(),
                'end': match.end(),
                'text': match.group(0),
                'type': 'markdown'
            }
            for match in self.TABLE_PATTERN.finditer(content)
        ]
        
        # Find all HTML tables
        html_tables = [
            {
                'start': match.start(),
                'end': match.end(),
                'text': match.group(0),
                'type': 'html'
            }
            for match in self.HTML_TABLE_PATTERN.finditer(content)
        ]
        
        # Merge and sort by position
        tables = sorted(markdown_tables + html_tables, key=lambda x: x['start'])
        
        if not tables:
            # No tables, return as single unit
            unit = TextUnit(
                unit_id=self.generate_unit_id(),
                content=content,
                metadata=original_metadata.model_copy() if original_metadata else UnitMetadata()
            )
            return [unit]
        
        # Split content into segments: [text, table, text, table, ...]
        segments = []
        last_pos = 0
        
        for table_info in tables:
            # Add text before table
            if table_info['start'] > last_pos:
                text_before = content[last_pos:table_info['start']]
                if text_before.strip():
                    segments.append(('text', text_before))
            
            # Add table (may be split into multiple chunks)
            table_chunks = self._split_table(table_info['text'])
            for chunk in table_chunks:
                segments.append(('table', chunk))
            
            last_pos = table_info['end']
        
        # Add remaining text
        if last_pos < len(content):
            text_after = content[last_pos:]
            if text_after.strip():
                segments.append(('text', text_after))
        
        # Convert segments to units
        units = []
        for seg_type, seg_content in segments:
            unit = TextUnit(
                unit_id=self.generate_unit_id(),
                content=seg_content,
                metadata=original_metadata.model_copy() if original_metadata else UnitMetadata()
            )
            units.append(unit)
        
        return units
    
    def _split_table(self, table_text: str) -> list[str]:
        """
        Split a table into chunks by token count
        
        Routes to markdown or HTML splitter based on format.
        
        Args:
            table_text: Markdown or HTML table as string
            
        Returns:
            List of table chunks (each with headers)
        """
        # Detect table type and route to appropriate splitter
        if table_text.strip().startswith('<table'):
            return self._split_html_table(table_text)
        else:
            return self._split_markdown_table(table_text)
    
    def _split_markdown_table(self, table_text: str) -> list[str]:
        """
        Split a markdown table into chunks by token count
        
        Strategy:
        1. Count header tokens
        2. Accumulate rows until reaching max_chunk_tokens
        3. Each chunk has at least min_rows
        
        Args:
            table_text: Markdown table as string
            
        Returns:
            List of table chunks (each with headers)
        """
        lines = table_text.strip().split('\n')
        
        if len(lines) < 3:
            # Too small to split (need header + separator + at least 1 row)
            return [table_text]
        
        # Parse table structure
        header_line = lines[0]
        separator_line = lines[1]
        data_lines = lines[2:]
        
        # Count header tokens
        header_text = f"{header_line}\n{separator_line}"
        header_tokens = self._count_tokens(header_text)
        
        # Check if table needs splitting
        total_tokens = self._count_tokens(table_text)
        if total_tokens <= self.max_chunk_tokens:
            # Small enough, no split needed
            return [table_text]
        
        # Split into chunks by token count
        chunks = []
        current_chunk_rows = []
        current_tokens = header_tokens
        
        for row in data_lines:
            row_tokens = self._count_tokens(row)
            
            # Check if adding this row would exceed limit
            if current_chunk_rows and current_tokens + row_tokens > self.max_chunk_tokens:
                # Save current chunk
                chunk_lines = [header_line, separator_line] + current_chunk_rows
                chunks.append('\n'.join(chunk_lines))
                
                # Start new chunk
                current_chunk_rows = [row]
                current_tokens = header_tokens + row_tokens
            else:
                # Add row to current chunk
                current_chunk_rows.append(row)
                current_tokens += row_tokens
                
                # Ensure at least min_rows even if first row exceeds limit
                if len(current_chunk_rows) >= self.min_rows and current_tokens > self.max_chunk_tokens:
                    # Save chunk with min_rows satisfied
                    chunk_lines = [header_line, separator_line] + current_chunk_rows
                    chunks.append('\n'.join(chunk_lines))
                    current_chunk_rows = []
                    current_tokens = header_tokens
        
        # Save last chunk if any
        if current_chunk_rows:
            chunk_lines = [header_line, separator_line] + current_chunk_rows
            chunks.append('\n'.join(chunk_lines))
        
        return chunks
    
    def _split_html_table(self, table_html: str) -> list[str]:
        """
        Split HTML table into chunks by token count
        
        Strategy:
        1. Parse HTML with BeautifulSoup
        2. Extract header rows (<thead> or first <tr>)
        3. Split data rows by token count
        4. Each chunk keeps the header
        
        Args:
            table_html: HTML table as string
            
        Returns:
            List of HTML table chunks (each with headers)
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # If bs4 not available, return as single chunk
            return [table_html]
        
        soup = BeautifulSoup(table_html, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return [table_html]
        
        # Extract header rows
        thead = table.find('thead')
        if thead:
            header_rows = thead.find_all('tr')
            tbody = table.find('tbody')
            data_rows = tbody.find_all('tr') if tbody else []
        else:
            # Use first row as header
            all_rows = table.find_all('tr')
            if not all_rows:
                return [table_html]
            header_rows = [all_rows[0]]
            data_rows = all_rows[1:]
        
        # Build header HTML
        header_html = ''.join(str(row) for row in header_rows)
        header_tokens = self._count_tokens(header_html)
        
        # Check if needs splitting
        total_tokens = self._count_tokens(table_html)
        if total_tokens <= self.max_chunk_tokens:
            return [table_html]
        
        # Extract table attributes for reconstruction
        table_attrs = ' '.join(f'{k}="{v}"' for k, v in table.attrs.items())
        table_opening = f'<table {table_attrs}>' if table_attrs else '<table>'
        
        # Split by rows
        chunks = []
        current_chunk_rows = []
        current_tokens = header_tokens
        
        for row in data_rows:
            row_html = str(row)
            row_tokens = self._count_tokens(row_html)
            
            # Check if adding this row would exceed limit
            if current_chunk_rows and current_tokens + row_tokens > self.max_chunk_tokens:
                # Save current chunk
                chunk_html = f"{table_opening}{header_html}{''.join(current_chunk_rows)}</table>"
                chunks.append(chunk_html)
                
                # Start new chunk
                current_chunk_rows = [row_html]
                current_tokens = header_tokens + row_tokens
            else:
                # Add row to current chunk
                current_chunk_rows.append(row_html)
                current_tokens += row_tokens
                
                # Ensure at least min_rows even if first row exceeds limit
                if len(current_chunk_rows) >= self.min_rows and current_tokens > self.max_chunk_tokens:
                    # Save chunk with min_rows satisfied
                    chunk_html = f"{table_opening}{header_html}{''.join(current_chunk_rows)}</table>"
                    chunks.append(chunk_html)
                    current_chunk_rows = []
                    current_tokens = header_tokens
        
        # Save last chunk if any
        if current_chunk_rows:
            chunk_html = f"{table_opening}{header_html}{''.join(current_chunk_rows)}</table>"
            chunks.append(chunk_html)
        
        return chunks if chunks else [table_html]
    
    def __repr__(self) -> str:
        """String representation"""
        return f"TableSplitter(max_chunk_tokens={self.max_chunk_tokens})"
