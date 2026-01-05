"""
Context augmentor - get adjacent units to provide more complete context
"""

from ..base import BasePostprocessor
from ...schemas.base import BaseUnit


class ContextAugmentor(BasePostprocessor):
    """
    Context augmentor
    
    Retrieves adjacent units (prev/next) of the search results to provide
    more complete context.
    
    Use cases:
        - Need more context for understanding
        - Avoid information fragmentation
        - Improve LLM comprehension
    
    Examples:
        >>> from zag.postprocessors.augmentors import ContextAugmentor
        >>> 
        >>> augmentor = ContextAugmentor(window_size=1)
        >>> augmented = augmentor.process(query, units)
    """
    
    def __init__(
        self, 
        window_size: int = 1,
        deduplicate: bool = True
    ):
        """
        Initialize context augmentor
        
        Args:
            window_size: Window size (number of units before and after)
            deduplicate: Whether to remove duplicates
        """
        self.window_size = window_size
        self.deduplicate = deduplicate
    
    def process(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Augment units with surrounding context
        
        Args:
            query: Original query text (not used by this augmentor)
            units: Units to augment
            
        Returns:
            Units with surrounding context
        """
        augmented = []
        
        for unit in units:
            # Get previous units
            current = unit
            prev_units = []
            for _ in range(self.window_size):
                if prev := current.get_prev():
                    prev_units.insert(0, prev)  # Insert at beginning
                    current = prev
                else:
                    break
            
            augmented.extend(prev_units)
            
            # Add current unit
            augmented.append(unit)
            
            # Get next units
            current = unit
            for _ in range(self.window_size):
                if next_unit := current.get_next():
                    augmented.append(next_unit)
                    current = next_unit
                else:
                    break
        
        # Deduplicate (optional)
        if self.deduplicate:
            seen = set()
            result = []
            for unit in augmented:
                if unit.unit_id not in seen:
                    seen.add(unit.unit_id)
                    result.append(unit)
            return result
        
        return augmented
