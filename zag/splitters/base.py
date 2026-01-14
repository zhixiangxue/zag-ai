"""   
Base splitter class for all splitters
"""

from abc import ABC, abstractmethod
import uuid
from typing import Union

from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from ..schemas.base import BaseUnit, UnitCollection, BaseDocument


class BaseSplitter(ABC):
    """
    Base class for all splitters
    Splitters split documents into units with automatic chain relationship setup
    
    Supports pipeline composition via | operator:
        splitter = MarkdownHeaderSplitter() | TextSplitter() | RecursiveMergingSplitter()
    """
    
    @abstractmethod
    def _do_split(self, input_data: Union[BaseDocument, list[BaseUnit]]) -> list[BaseUnit]:
        """
        Internal method to perform the actual splitting logic
        Subclasses should implement this method
        
        IMPORTANT: This method must support both input types:
        1. BaseDocument - Initial splitting from document
        2. list[BaseUnit] - Re-processing existing units (for pipeline)
        
        Args:
            input_data: Either a document to split, or units to process
            
        Returns:
            List of units (without chain relationships set)
        
        Implementation pattern:
            def _do_split(self, input_data):
                # Check input type
                if isinstance(input_data, list):
                    # Process units
                    return self._process_units(input_data)
                else:
                    # Process document
                    return self._process_document(input_data)
        """
        pass
    
    def __or__(self, other: 'BaseSplitter') -> 'CompositeSplitter':
        """
        Enable pipeline syntax: splitter1 | splitter2
        
        Example:
            >>> pipeline = (
            ...     MarkdownHeaderSplitter()
            ...     | TextSplitter(max_chunk_size=1200)
            ...     | RecursiveMergingSplitter(target_token_size=800)
            ... )
            >>> units = doc.split(pipeline)
        
        Args:
            other: Another splitter to chain
            
        Returns:
            CompositeSplitter that executes both splitters in sequence
        """
        if not isinstance(other, BaseSplitter):
            raise TypeError(f"Can only pipe to BaseSplitter, got {type(other)}")
        
        return CompositeSplitter(splitters=[self, other])
    
    def split(self, document: BaseDocument, show_progress: bool = True) -> UnitCollection:
        """
        Split document into units and establish chain relationships
        
        Args:
            document: The document to split
            show_progress: Whether to show progress bar (default: True)
            
        Returns:
            UnitCollection with units having prev/next relationships
        """
        if show_progress:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(
                    "Processing...",
                    total=100
                )
                
                # Perform actual splitting
                units = self._do_split(document)
                progress.update(task, advance=50)
                
                # Establish chain relationships
                for i, unit in enumerate(units):
                    # Set source document
                    unit.doc_id = document.doc_id
                    
                    # Set prev/next relationships
                    if i > 0:
                        unit.prev_unit_id = units[i - 1].unit_id
                    if i < len(units) - 1:
                        unit.next_unit_id = units[i + 1].unit_id
                
                progress.update(task, advance=50)
        else:
            # No progress bar
            units = self._do_split(document)
            
            # Establish chain relationships
            for i, unit in enumerate(units):
                # Set source document
                unit.doc_id = document.doc_id
                
                # Set prev/next relationships
                if i > 0:
                    unit.prev_unit_id = units[i - 1].unit_id
                if i < len(units) - 1:
                    unit.next_unit_id = units[i + 1].unit_id
        
        return UnitCollection(units)
    
    @staticmethod
    def generate_unit_id() -> str:
        """Generate a unique unit ID"""
        return str(uuid.uuid4())


class CompositeSplitter(BaseSplitter):
    """
    Composite splitter for pipeline processing
    Automatically created by | operator
    
    Executes multiple splitters in sequence, where each splitter
    processes the output of the previous one.
    
    Args:
        splitters: List of splitters to execute in order
    
    Example:
        >>> # Created automatically via | operator
        >>> pipeline = splitter1 | splitter2 | splitter3
        >>> 
        >>> # Or create manually
        >>> pipeline = CompositeSplitter(splitters=[splitter1, splitter2])
    """
    
    def __init__(self, splitters: list[BaseSplitter]):
        self.splitters = splitters
    
    def __or__(self, other: BaseSplitter) -> 'CompositeSplitter':
        """Support chaining: s1 | s2 | s3"""
        if not isinstance(other, BaseSplitter):
            raise TypeError(f"Can only pipe to BaseSplitter, got {type(other)}")
        return CompositeSplitter(splitters=self.splitters + [other])
    
    def _do_split(self, input_data: Union[BaseDocument, list[BaseUnit]]) -> list[BaseUnit]:
        """
        Execute pipeline sequentially
        
        Each splitter processes the output of the previous one.
        All splitters now support both Document and list[BaseUnit] input.
        
        Args:
            input_data: Document or units to process
            
        Returns:
            Final list of units after all splitters
        """
        current_data = input_data
        
        for splitter in self.splitters:
            # Each splitter's _do_split() handles both Document and list[BaseUnit]
            # We pass the current_data directly to _do_split()
            current_data = splitter._do_split(current_data)
        
        return current_data
    
    def _unit_to_document(self, unit: BaseUnit, original_doc: BaseDocument) -> BaseDocument:
        """
        Convert unit back to document for next splitter
        
        Args:
            unit: Unit to convert
            original_doc: Original document (for metadata reference)
            
        Returns:
            Document with unit's content
        """
        # Create a minimal document-like object with unit's content
        # This allows subsequent splitters to process it
        from ..schemas.markdown import Markdown
        
        return Markdown(
            content=unit.content,
            metadata=original_doc.metadata,
            doc_id=unit.unit_id  # Use unit ID as doc ID for traceability
        )
    
    def __repr__(self) -> str:
        """String representation"""
        splitter_names = ' | '.join(s.__class__.__name__ for s in self.splitters)
        return f"CompositeSplitter({splitter_names})"
