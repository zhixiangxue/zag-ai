"""
Keyword extractor for extracting keywords from units
"""

from typing import List, Dict, Sequence
from pydantic import BaseModel, Field


from .base import BaseExtractor


DEFAULT_TEMPLATE = """Extract {num_keywords} most important keywords from the following text.

Text:
{text}

Requirements:
1. Keywords should uniquely identify the core topic of this text
2. Use the same language as the source text

Please extract the keywords:"""


class KeywordList(BaseModel):
    """Keyword list structure"""
    keywords: List[str] = Field(description="Extracted keywords from the text")


class KeywordExtractor(BaseExtractor):
    """
    Keyword extractor that extracts keywords from units using LLM.
    
    Stores keywords in unit.metadata.keywords for easy access.
    
    Args:
        llm_uri: LLM URI in format: provider/model
        api_key: API key for the LLM provider
        num_keywords: Number of keywords to extract, default 5
    
    Example:
        >>> extractor = KeywordExtractor(
        ...     llm_uri="openai/gpt-4o-mini",
        ...     api_key="sk-xxx",
        ...     num_keywords=5
        ... )
        >>> units = extractor(units)
        >>> 
        >>> # Access results directly from metadata
        >>> print(units[0].metadata.keywords)
        >>> # ["fixed rate", "30-year", "mortgage", "refinance", "conventional loan"]
    """
    
    def __init__(
        self,
        llm_uri: str,
        api_key: str = None,
        num_keywords: int = 5,
    ):
        self.llm_uri = llm_uri
        self.api_key = api_key
        self.num_keywords = num_keywords
        
        # Use chak (for general dialogue)
        import chak
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit) -> Dict:
        """Extract keywords from a single unit and store in metadata.keywords."""
        content = unit.content if hasattr(unit, 'content') else str(unit)
        
        prompt = DEFAULT_TEMPLATE.format(
            text=content,
            num_keywords=self.num_keywords
        )
        
        try:
            # Use structured output
            response = await self._conv.asend(prompt, returns=KeywordList)
            
            # Check if response is valid
            if response is None:
                return {"keywords": []}
            
            # response is already a KeywordList object, access keywords directly
            keywords = response.keywords[:self.num_keywords]
            
            return {"keywords": keywords}
        
        except Exception as e:
            # If extraction fails, return empty list
            print(f"Warning: Keyword extraction failed: {e}")
            return {"keywords": []}
