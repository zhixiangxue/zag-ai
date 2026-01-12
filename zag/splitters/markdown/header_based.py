"""
Header-based markdown splitter
Split markdown content by headers (H1-H6)
Inspired by LlamaIndex's MarkdownNodeParser
"""

import re
from typing import Optional, Union

from ..base import BaseSplitter
from ...schemas.base import BaseUnit, UnitMetadata, BaseDocument
from ...schemas.unit import TextUnit


class MarkdownHeaderSplitter(BaseSplitter):
    """
    Split markdown content by headers (H1-H6)
    
    Splits a document into units using header-based splitting logic.
    Each unit contains its text content and the hierarchical path of headers.
    
    Inspired by LlamaIndex's MarkdownNodeParser implementation.
    
    Args:
        header_path_separator: Separator for header path (default: " > ")
        include_header_in_content: Whether to include header in unit content (default: True)
    
    Example:
        >>> splitter = MarkdownHeaderSplitter()
        >>> doc = Markdown(content="# Intro\\ntext\\n## Background\\nmore text")
        >>> units = doc.split(splitter)
        >>> units[0].metadata.context_path
        "Intro"
        >>> units[1].metadata.context_path
        "Intro > Background"
    """
    
    def __init__(
        self,
        header_path_separator: str = " > ",
        include_header_in_content: bool = True
    ):
        """
        Initialize splitter
        
        Args:
            header_path_separator: Separator for context path (default: " > ")
            include_header_in_content: Whether to include header line in content
        """
        self.header_path_separator = header_path_separator
        self.include_header_in_content = include_header_in_content
    
    def _do_split(self, input_data: Union[BaseDocument, list[BaseUnit]]) -> list[BaseUnit]:
        """
        Split markdown document by headers
        
        Supports two input types:
        1. Document - Split the document content
        2. list[BaseUnit] - Split each unit's content individually
        
        Args:
            input_data: Document or units to split
            
        Returns:
            List of TextUnits with context_path metadata
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
            return self._split_content(content, document=input_data)
    
    def _split_content(
        self, 
        content: str, 
        document=None,
        parent_unit: Optional[BaseUnit] = None
    ) -> list[TextUnit]:
        """
        Split markdown content by headers
        
        Args:
            content: Markdown content to split
            document: Source document (if available)
            parent_unit: Parent unit (if splitting from unit)
            
        Returns:
            List of TextUnits
        """
        
        units = []
        lines = content.split("\n")
        current_section = ""
        # Keep track of (header_level, header_text) for headers
        header_stack: list[tuple[int, str]] = []
        code_block = False
        
        for line in lines:
            # Track if we're inside a code block to avoid parsing headers in code
            if line.lstrip().startswith("```"):
                code_block = not code_block
                current_section += line + "\n"
                continue
            
            # Only parse headers if we're not in a code block
            if not code_block:
                header_match = re.match(r"^(#+)\s(.*)", line)
                if header_match:
                    # Save the previous section before starting a new one
                    if current_section.strip():
                        units.append(
                            self._build_unit_from_split(
                                current_section.strip(),
                                document,
                                header_stack,
                            )
                        )
                    
                    header_level = len(header_match.group(1))
                    header_text = header_match.group(2)
                    
                    # Pop headers of equal or higher level
                    while header_stack and header_stack[-1][0] >= header_level:
                        header_stack.pop()
                    
                    # Add the new header
                    header_stack.append((header_level, header_text))
                    
                    # Start new section with header
                    if self.include_header_in_content:
                        current_section = "#" * header_level + f" {header_text}\n"
                    else:
                        current_section = ""
                    continue
            
            current_section += line + "\n"
        
        # Add the final section
        if current_section.strip():
            units.append(
                self._build_unit_from_split(
                    current_section.strip(),
                    document,
                    header_stack,
                )
            )
        
        # Build chain relationships
        for i in range(len(units)):
            if i > 0:
                units[i].prev_unit_id = units[i - 1].unit_id
            if i < len(units) - 1:
                units[i].next_unit_id = units[i + 1].unit_id
            
            # Set source document ID
            if document and hasattr(document, 'doc_id'):
                units[i].source_doc_id = document.doc_id
            elif parent_unit:
                # Inherit from parent unit
                units[i].source_doc_id = parent_unit.source_doc_id
                # Also inherit context_path if parent has one
                if parent_unit.metadata and parent_unit.metadata.context_path:
                    # Prepend parent's context_path
                    if units[i].metadata.context_path:
                        units[i].metadata.context_path = f"{parent_unit.metadata.context_path}{self.header_path_separator}{units[i].metadata.context_path}"
                    else:
                        units[i].metadata.context_path = parent_unit.metadata.context_path
        
        return units
    
    def _build_unit_from_split(
        self,
        text_split: str,
        document,
        header_stack: list[tuple[int, str]],
    ) -> TextUnit:
        """
        Build unit from single text split
        
        Args:
            text_split: Content of this section
            document: Source document
            header_stack: Stack of headers (including current)
            
        Returns:
            TextUnit with context_path metadata
        """
        # Build context path from header stack
        context_path = self.header_path_separator.join(
            h[1] for h in header_stack
        ) if header_stack else None
        
        # Create unit
        unit = TextUnit(
            unit_id=self.generate_unit_id(),
            content=text_split,
            metadata=UnitMetadata(
                context_path=context_path
            )
        )
        
        return unit
