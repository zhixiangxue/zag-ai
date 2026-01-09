"""
Recursive merging splitter
Merges small chunks into larger ones based on token count
"""

from typing import Optional, Any
import tiktoken

from ..base import BaseSplitter
from ...schemas.base import BaseUnit
from ...schemas.unit import TextUnit


class RecursiveMergingSplitter(BaseSplitter):
    """
    Recursive merging splitter
    
    Wraps an existing splitter (e.g., MarkdownHeaderSplitter) and intelligently
    merges small chunks into larger ones until reaching the target token size.
    
    Workflow:
    1. Use base_splitter to split document into small chunks
    2. Merge adjacent chunks sequentially using TextUnit's + operator
    3. Stop merging when target token size is reached, start a new group
    4. Preserve chain relationships (prev_unit_id / next_unit_id)
    
    Features:
    - Does not modify base splitter behavior
    - Preserves original header hierarchy information
    - Smart merging to avoid semantic fragmentation
    - Simplified merge logic using TextUnit.__add__()
    
    Args:
        base_splitter: Base splitter (e.g., MarkdownHeaderSplitter)
        target_token_size: Target token count per merged chunk (default: 500)
        tokenizer: Token counter (defaults to tiktoken cl100k_base)
    
    Example:
        >>> from zag.splitters.markdown import MarkdownHeaderSplitter
        >>> from zag.splitters.composite import RecursiveMergingSplitter
        >>> 
        >>> # Create base splitter
        >>> base_splitter = MarkdownHeaderSplitter()
        >>> 
        >>> # Wrap with recursive merging
        >>> merger = RecursiveMergingSplitter(
        ...     base_splitter=base_splitter,
        ...     target_token_size=800  # Merge until ~800 tokens
        ... )
        >>> 
        >>> # Split document
        >>> units = doc.split(merger)
        >>> # Result: each unit has ~800 tokens with complete semantics
    
    Notes:
        - Requires tiktoken: pip install tiktoken
        - Uses cl100k_base encoder (GPT-4 tokenizer)
        - Merged units record merge info in metadata.custom (merged_from, merged_count)
    """
    
    def __init__(
        self,
        base_splitter: BaseSplitter,
        target_token_size: int = 500,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize recursive merging splitter
        
        Args:
            base_splitter: Base splitter to wrap
            target_token_size: Target token count per chunk (stops merging at this size)
            tokenizer: Custom tokenizer (defaults to tiktoken cl100k_base)
        """
        self.base_splitter = base_splitter
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
    
    def _do_split(self, document) -> list[BaseUnit]:
        """
        Execute splitting + merging
        
        Flow:
        1. Use base_splitter to split into small chunks
        2. Merge adjacent chunks until reaching target token size
        3. Preserve chain relationships and metadata
        
        Args:
            document: Document to split
        
        Returns:
            List of merged units
        """
        # 1. Base splitting
        base_units = self.base_splitter.split(document)
        
        if not base_units:
            return []
        
        # Filter to TextUnits only (skip TableUnit, ImageUnit, etc.)
        text_units = [u for u in base_units if isinstance(u, TextUnit)]
        
        if not text_units:
            return base_units
        
        # 2. Recursive merging
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
        
        # 3. Rebuild chain relationships
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
        return (
            f"RecursiveMergingSplitter("
            f"base={self.base_splitter.__class__.__name__}, "
            f"target={self.target_token_size})"
        )
