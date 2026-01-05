"""
Deduplicator - remove duplicate or highly similar units
"""

from ..base import BasePostprocessor
from ...schemas.base import BaseUnit


class Deduplicator(BasePostprocessor):
    """
    Deduplicator
    
    Removes duplicate or highly similar units.
    
    Deduplication strategies:
        - exact: Exact same content (based on unit_id)
        - content: Same content (based on content hash)
        - semantic: Semantically similar (based on embedding similarity)
    
    Examples:
        >>> from zag.postprocessors.filters import Deduplicator
        >>> 
        >>> dedup = Deduplicator(strategy="exact")
        >>> unique = dedup.process(query, units)
    """
    
    def __init__(
        self, 
        strategy: str = "exact",
        similarity_threshold: float = 0.95
    ):
        """
        Initialize deduplicator
        
        Args:
            strategy: Deduplication strategy ("exact", "content", "semantic")
            similarity_threshold: Similarity threshold (only for semantic mode)
            
        Raises:
            ValueError: If strategy is unknown
        """
        if strategy not in ["exact", "content", "semantic"]:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
    
    def process(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Remove duplicate units
        
        Args:
            query: Original query text
            units: Units to deduplicate
            
        Returns:
            Deduplicated units
        """
        if self.strategy == "exact":
            return self._exact_dedup(units)
        elif self.strategy == "content":
            return self._content_dedup(units)
        else:  # semantic
            return self._semantic_dedup(units)
    
    def _exact_dedup(self, units: list[BaseUnit]) -> list[BaseUnit]:
        """Deduplicate based on unit_id"""
        seen = set()
        result = []
        for unit in units:
            if unit.unit_id not in seen:
                seen.add(unit.unit_id)
                result.append(unit)
        return result
    
    def _content_dedup(self, units: list[BaseUnit]) -> list[BaseUnit]:
        """Deduplicate based on content hash"""
        import hashlib
        seen = set()
        result = []
        for unit in units:
            content_hash = hashlib.md5(str(unit.content).encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                result.append(unit)
        return result
    
    def _semantic_dedup(self, units: list[BaseUnit]) -> list[BaseUnit]:
        """Deduplicate based on semantic similarity"""
        # TODO: Implement semantic deduplication (requires embedding similarity computation)
        raise NotImplementedError("Semantic deduplication not yet implemented")
