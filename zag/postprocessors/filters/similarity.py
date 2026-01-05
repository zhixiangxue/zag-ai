"""
Similarity filter - filter units by similarity score threshold
"""

from ..base import BasePostprocessor
from ...schemas.base import BaseUnit


class SimilarityFilter(BasePostprocessor):
    """
    Similarity filter
    
    Keeps only units with similarity scores above the threshold.
    
    Use cases:
        - Filter low-quality results
        - Ensure minimum relevance requirement
        - Control result quality
    
    Examples:
        >>> from zag.postprocessors.filters import SimilarityFilter
        >>> 
        >>> filter = SimilarityFilter(threshold=0.7)
        >>> filtered = filter.process(query, units)
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize similarity filter
        
        Args:
            threshold: Similarity threshold (0-1), results below this will be filtered
            
        Raises:
            ValueError: If threshold is not between 0 and 1
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
    
    def process(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Filter low similarity results
        
        Args:
            query: Original query text (not used by this filter)
            units: Units to filter
            
        Returns:
            Units with similarity >= threshold
        """
        return [
            unit for unit in units 
            if hasattr(unit, 'score') and unit.score is not None 
            and unit.score >= self.threshold
        ]
