"""
Chain postprocessor - execute multiple postprocessors sequentially
"""

from ..base import BasePostprocessor
from ...schemas.base import BaseUnit


class ChainPostprocessor(BasePostprocessor):
    """
    Chain postprocessor
    
    Executes multiple postprocessors sequentially, where the output of one
    becomes the input of the next. Similar to Unix pipes: p1 | p2 | p3
    
    Characteristics:
        - Sequential execution
        - Short-circuit: stops early if any step returns empty list
        - Flexible composition
    
    Examples:
        >>> from zag.postprocessors import (
        ...     ChainPostprocessor,
        ...     CrossEncoderReranker,
        ...     SimilarityFilter,
        ...     TokenCompressor
        ... )
        >>> 
        >>> chain = ChainPostprocessor([
        ...     CrossEncoderReranker(),
        ...     SimilarityFilter(threshold=0.7),
        ...     TokenCompressor(max_tokens=4000),
        ... ])
        >>> 
        >>> results = chain.process(query, units)
    """
    
    def __init__(self, processors: list[BasePostprocessor]):
        """
        Initialize chain postprocessor
        
        Args:
            processors: List of postprocessors to execute in order
            
        Raises:
            ValueError: If processors list is empty
        """
        if not processors:
            raise ValueError("Must provide at least one processor")
        self.processors = processors
    
    def process(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Execute all postprocessors sequentially
        
        Args:
            query: Original query text
            units: Units to process
            
        Returns:
            Units after all processors
        """
        result = units
        
        for i, processor in enumerate(self.processors):
            if not result:
                # Short-circuit: stop if any step returns empty
                break
            
            result = processor.process(query, result)
        
        return result
    
    async def aprocess(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Async version of chain processing
        
        Args:
            query: Original query text
            units: Units to process
            
        Returns:
            Units after all processors
        """
        result = units
        
        for processor in self.processors:
            if not result:
                break
            result = await processor.aprocess(query, result)
        
        return result
