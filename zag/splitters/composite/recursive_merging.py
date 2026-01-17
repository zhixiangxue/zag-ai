"""
Recursive merging splitter
Merges small chunks into larger ones based on token count
"""

from typing import Optional, Any, Union
import tiktoken

from ..base import BaseSplitter
from ...schemas import BaseUnit, BaseDocument, UnitMetadata
from ...schemas.unit import TextUnit


class RecursiveMergingSplitter(BaseSplitter):
    """
    Recursive merging splitter
    
    Intelligently merges small chunks into larger ones until reaching
    the target token size. Designed to work in pipelines with other splitters.
    
    Workflow:
    1. Receives units from previous splitter (or document)
    2. Merge adjacent chunks sequentially using TextUnit's + operator
    3. Stop merging when target token size is reached, start a new group
    4. Preserve chain relationships (prev_unit_id / next_unit_id)
    
    Features:
    - Preserves original header hierarchy information
    - Smart merging to avoid semantic fragmentation
    - Simplified merge logic using TextUnit.__add__()
    - Works seamlessly in pipeline composition
    
    Args:
        target_token_size: Target token count per merged chunk (default: 500)
        tokenizer: Token counter (defaults to tiktoken cl100k_base)
    
    Example:
        >>> from zag.splitters import MarkdownHeaderSplitter, RecursiveMergingSplitter
        >>> 
        >>> # Use in pipeline
        >>> pipeline = (
        ...     MarkdownHeaderSplitter()
        ...     | RecursiveMergingSplitter(target_token_size=800)
        ... )
        >>> units = doc.split(pipeline)
        >>> # Result: each unit has ~800 tokens with complete semantics
    
    Notes:
        - Requires tiktoken: pip install tiktoken
        - Uses cl100k_base encoder (GPT-4 tokenizer)
        - Merged units record merge info in metadata.custom (merged_from, merged_count)
    """
    
    def __init__(
        self,
        target_token_size: int = 500,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize recursive merging splitter
        
        Args:
            target_token_size: Target token count per chunk (stops merging at this size)
            tokenizer: Custom tokenizer (defaults to tiktoken cl100k_base)
        """
        self.target_token_size = target_token_size
        
        # Default to tiktoken (OpenAI's tokenizer)
        if tokenizer is None:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
            except Exception as e:
                raise ImportError(
                    "tiktoken is required for RecursiveMergingSplitter. "
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
        Execute merging on units
        
        Supports two input types:
        1. Document - Convert to single unit then return
        2. list[BaseUnit] - Merge units based on token size
        
        Args:
            input_data: Document or units to process
        
        Returns:
            List of merged units
        """
        # Check input type
        if isinstance(input_data, list):
            # Process units: merge them
            return self._merge_units(input_data)
        else:
            # Process document: create single unit
            content = input_data.content if hasattr(input_data, 'content') else ""
            
            if not content:
                return []
            
            # Create a TextUnit from the document content
            # Document metadata injection is handled by Document.split() later
            unit = TextUnit(
                unit_id=self.generate_unit_id(),
                content=content,
                metadata=UnitMetadata()  # Empty metadata, will be populated by Document.split()
            )
            
            return [unit]
    
    def _merge_units(self, units: list[BaseUnit]) -> list[BaseUnit]:
        """
        Merge units based on token size
        
        This is called by CompositeSplitter in pipeline mode.
        Merges adjacent TextUnits until reaching target token size.
        
        Args:
            units: List of units to merge
            
        Returns:
            List of merged units
        """
        if not units:
            return []
        
        # Filter to TextUnits only (skip TableUnit, ImageUnit, etc.)
        text_units = [u for u in units if isinstance(u, TextUnit)]
        
        if not text_units:
            return units
        
        # Merge adjacent units
        merged_units = []
        current_merged: Optional[TextUnit] = None
        current_tokens = 0
        
        for unit in text_units:
            unit_tokens = self._count_tokens(unit.content)
            
            # Start new group if no current
            if current_merged is None:
                current_merged = unit
                current_tokens = unit_tokens
                continue
            
            # Try to merge with current group
            potential_tokens = current_tokens + unit_tokens
            
            if potential_tokens <= self.target_token_size:
                # Can merge - use + operator!
                current_merged = current_merged + unit
                current_tokens = potential_tokens
            else:
                # Exceeds target, save current and start new
                merged_units.append(current_merged)
                current_merged = unit
                current_tokens = unit_tokens
        
        # Save last merged unit
        if current_merged:
            merged_units.append(current_merged)
        
        # Rebuild chain relationships
        for i in range(len(merged_units)):
            if i > 0:
                merged_units[i].prev_unit_id = merged_units[i - 1].unit_id
            else:
                merged_units[i].prev_unit_id = None
            
            if i < len(merged_units) - 1:
                merged_units[i].next_unit_id = merged_units[i + 1].unit_id
            else:
                merged_units[i].next_unit_id = None
        
        return merged_units
    
    def __repr__(self) -> str:
        """String representation"""
        return f"RecursiveMergingSplitter(target={self.target_token_size})"
