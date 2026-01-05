"""
Conditional postprocessor - choose different postprocessors based on conditions
"""

from typing import Callable, Optional

from ..base import BasePostprocessor
from ...schemas.base import BaseUnit


class ConditionalPostprocessor(BasePostprocessor):
    """
    Conditional postprocessor
    
    Chooses different postprocessors to execute based on conditions.
    Similar to if-else logic.
    
    Use cases:
        - Choose different strategies based on result count
        - Choose different processing based on query type
        - Dynamically adjust post-processing flow
    
    Examples:
        >>> from zag.postprocessors import ConditionalPostprocessor
        >>> 
        >>> def need_reranking(query: str, units: list) -> bool:
        ...     return len(units) > 20
        >>> 
        >>> conditional = ConditionalPostprocessor(
        ...     condition=need_reranking,
        ...     true_processor=CrossEncoderReranker(),
        ...     false_processor=None,  # No processing if not needed
        ... )
        >>> 
        >>> results = conditional.process(query, units)
    """
    
    def __init__(
        self,
        condition: Callable[[str, list[BaseUnit]], bool],
        true_processor: BasePostprocessor,
        false_processor: Optional[BasePostprocessor] = None,
    ):
        """
        Initialize conditional postprocessor
        
        Args:
            condition: Condition function that takes query and units, returns bool
            true_processor: Processor to use when condition is True
            false_processor: Processor to use when condition is False (None = return as-is)
        """
        self.condition = condition
        self.true_processor = true_processor
        self.false_processor = false_processor
    
    def process(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Choose processor based on condition
        
        Args:
            query: Original query text
            units: Units to process
            
        Returns:
            Processed units
        """
        if self.condition(query, units):
            return self.true_processor.process(query, units)
        elif self.false_processor:
            return self.false_processor.process(query, units)
        else:
            return units
