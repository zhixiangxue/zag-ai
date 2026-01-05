"""
Token compressor - limit total token count to avoid exceeding LLM context window
"""

from ..base import BasePostprocessor
from ...schemas.base import BaseUnit


class TokenCompressor(BasePostprocessor):
    """
    Token compressor
    
    Limits total token count to avoid exceeding LLM context window.
    
    Compression strategies:
        - truncate: Simply truncate (keep first N units)
        - smart: Smart compression (prioritize high-scoring results)
    
    Examples:
        >>> from zag.postprocessors.compressors import TokenCompressor
        >>> 
        >>> compressor = TokenCompressor(max_tokens=4000)
        >>> compressed = compressor.process(query, units)
    """
    
    def __init__(
        self, 
        max_tokens: int = 4000,
        strategy: str = "smart",
        chars_per_token: float = 4.0
    ):
        """
        Initialize token compressor
        
        Args:
            max_tokens: Maximum token count
            strategy: Compression strategy ("truncate" or "smart")
            chars_per_token: Average characters per token (for estimation)
        """
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.chars_per_token = chars_per_token
    
    def process(
        self, 
        query: str,
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        Compress to specified token count
        
        Args:
            query: Original query text
            units: Units to compress
            
        Returns:
            Compressed units
        """
        if self.strategy == "truncate":
            return self._truncate(units)
        else:  # smart
            return self._smart_compress(units)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return int(len(text) / self.chars_per_token)
    
    def _truncate(self, units: list[BaseUnit]) -> list[BaseUnit]:
        """Simple truncation"""
        total_tokens = 0
        result = []
        
        for unit in units:
            tokens = self._estimate_tokens(str(unit.content))
            if total_tokens + tokens <= self.max_tokens:
                result.append(unit)
                total_tokens += tokens
            else:
                break
        
        return result
    
    def _smart_compress(self, units: list[BaseUnit]) -> list[BaseUnit]:
        """Smart compression: prioritize high-scoring results"""
        # Sort by score if available
        if units and hasattr(units[0], 'score') and units[0].score is not None:
            sorted_units = sorted(units, key=lambda x: x.score or 0, reverse=True)
        else:
            sorted_units = units
        
        # Then truncate
        return self._truncate(sorted_units)
