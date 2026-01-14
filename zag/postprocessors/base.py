"""
Base postprocessor abstract class
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..schemas.base import BaseUnit


class BasePostprocessor(ABC):
    """
    Abstract base class for all postprocessors
    
    All postprocessors (including rerankers, filters, augmentors, etc.) 
    implement this unified interface. This allows flexible composition 
    and nesting of postprocessors.
    
    Design Philosophy:
        - Single responsibility: each postprocessor focuses on one task
        - Composability: postprocessors can be combined using | operator
        - Uniformity: all postprocessors use the same interface
    
    Examples:
        >>> class MyFilter(BasePostprocessor):
        ...     def process(self, query: str, units: list[BaseUnit]) -> list[BaseUnit]:
        ...         return [u for u in units if u.score > 0.7]
        >>> 
        >>> # Chain postprocessors using | operator
        >>> pipeline = Reranker(...) | SimilarityFilter(0.7) | TokenCompressor(4000)
        >>> result = pipeline.process(query, units)
    """
    
    @abstractmethod
    def process(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Post-process retrieved units
        
        Args:
            query: Original query text
            units: Units to process
            
        Returns:
            Processed units (may change count, order, or content)
        """
        pass
    
    async def aprocess(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Async version of process (optional implementation)
        
        Args:
            query: Original query text
            units: Units to process
            
        Returns:
            Processed units
        """
        # Default implementation: call sync version
        return self.process(query, units)
    
    def __or__(self, other: BasePostprocessor) -> "PostprocessorPipeline":
        """
        Chain postprocessors using | operator
        
        Args:
            other: Next postprocessor in the chain
            
        Returns:
            Pipeline that executes both postprocessors sequentially
            
        Examples:
            >>> pipeline = reranker | filter | compressor
            >>> result = pipeline.process(query, units)
        """
        if isinstance(self, PostprocessorPipeline):
            # self is already a pipeline, extend it
            return PostprocessorPipeline(self.processors + [other])
        else:
            # Create new pipeline
            return PostprocessorPipeline([self, other])


class PostprocessorPipeline(BasePostprocessor):
    """
    Pipeline of postprocessors created by | operator
    
    This is created automatically when using | to chain postprocessors.
    Users typically don't need to instantiate this class directly.
    
    Examples:
        >>> # These are equivalent:
        >>> pipeline = Reranker(...) | SimilarityFilter(0.7) | TokenCompressor(4000)
        >>> pipeline = PostprocessorPipeline([Reranker(...), SimilarityFilter(0.7), TokenCompressor(4000)])
    """
    
    def __init__(self, processors: list[BasePostprocessor]):
        """
        Initialize pipeline
        
        Args:
            processors: List of postprocessors to execute in order
        """
        if not processors:
            raise ValueError("Pipeline must have at least one processor")
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
        
        for processor in self.processors:
            if not result:  # Short-circuit if empty
                break
            result = processor.process(query, result)
        
        return result
    
    async def aprocess(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Async version of pipeline execution
        
        Args:
            query: Original query text
            units: Units to process
            
        Returns:
            Units after all processors
        """
        result = units
        
        for processor in self.processors:
            if not result:  # Short-circuit if empty
                break
            result = await processor.aprocess(query, result)
        
        return result
    
    def __repr__(self) -> str:
        processor_names = " | ".join(p.__class__.__name__ for p in self.processors)
        return f"Pipeline({processor_names})"
