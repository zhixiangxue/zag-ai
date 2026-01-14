"""
Base extractor class for extracting metadata from units
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Sequence
import asyncio

from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn


class BaseExtractor(ABC):
    """
    Base class for all extractors.
    
    Extractors process units to extract additional metadata.
    Subclasses must implement _extract_from_unit() method.
    
    Note: These extractors are async-only and require LLM calls.
    Use 'await extractor.aextract(units)' in async context.
    
    Design Pattern:
    - Subclasses implement _extract_from_unit(unit) for single unit extraction
    - Base class provides aextract(units) with progress bar and concurrency control
    - Subclasses can override aextract() if custom batch processing is needed
    """
    
    @abstractmethod
    async def _extract_from_unit(self, unit) -> Dict:
        """
        Extract metadata from a single unit (async).
        
        This is the core method that subclasses must implement.
        
        Args:
            unit: A single unit to process
            
        Returns:
            Dictionary of extracted metadata for this unit
        """
        pass
    
    async def aextract(self, units: Sequence, max_concurrent: int = 3) -> List[Dict]:
        """
        Extract metadata from units and write to unit.metadata (async with progress bar).
        
        This method provides:
        - Progress bar visualization (shows count: X/Y)
        - Concurrency control to avoid overwhelming the LLM API
        - Automatic batching and gathering of results
        - Auto-write extracted metadata to units
        
        Metadata writing rules:
        - 'keywords' field → unit.metadata.keywords (framework field)
        - 'embedding_content' field → unit.embedding_content (top-level unit field)
        - Other fields → unit.metadata.custom (business fields)
        
        Args:
            units: Sequence of units to process
            max_concurrent: Maximum concurrent LLM requests (default: 3)
            
        Returns:
            List of metadata dictionaries, one per unit
        """
        if not units:
            return []
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "Processing...",
                total=len(units)
            )
            
            # Use semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_limit(unit):
                async with semaphore:
                    result = await self._extract_from_unit(unit)
                    progress.update(task, advance=1)
                    return result
            
            tasks = [process_with_limit(unit) for unit in units]
            results = await asyncio.gather(*tasks)
        
        # Write results to unit.metadata and unit fields
        for unit, result in zip(units, results):
            if not result:
                continue
            
            # Special handling: keywords field goes to unit.metadata.keywords
            if 'keywords' in result:
                unit.metadata.keywords = result.pop('keywords')
            
            # Special handling: embedding_content goes to unit.embedding_content (top-level)
            if 'embedding_content' in result:
                unit.embedding_content = result.pop('embedding_content')
            
            # All other fields go to custom
            if result:  # If there are remaining fields
                unit.metadata.custom.update(result)
        
        return results
