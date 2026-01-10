"""   
Base splitter class for all splitters
"""

from abc import ABC, abstractmethod
import uuid

from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from ..schemas.base import BaseUnit, UnitCollection, BaseDocument


class BaseSplitter(ABC):
    """
    Base class for all splitters
    Splitters split documents into units with automatic chain relationship setup
    """
    
    @abstractmethod
    def _do_split(self, document: BaseDocument) -> list[BaseUnit]:
        """
        Internal method to perform the actual splitting logic
        Subclasses should implement this method
        
        Args:
            document: The document to split
            
        Returns:
            List of units (without chain relationships set)
        """
        pass
    
    def split(self, document: BaseDocument) -> UnitCollection:
        """
        Split document into units and establish chain relationships
        
        Args:
            document: The document to split
            
        Returns:
            UnitCollection with units having prev/next relationships
        """
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
                unit.source_doc_id = document.doc_id
                
                # Set prev/next relationships
                if i > 0:
                    unit.prev_unit_id = units[i - 1].unit_id
                if i < len(units) - 1:
                    unit.next_unit_id = units[i + 1].unit_id
            
            progress.update(task, advance=50)
            
            return UnitCollection(units)
    
    @staticmethod
    def generate_unit_id() -> str:
        """Generate a unique unit ID"""
        return str(uuid.uuid4())
