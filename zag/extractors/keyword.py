"""
Keyword extractor for extracting keywords from units
"""

from typing import List, Dict, Sequence


from .base import BaseExtractor


DEFAULT_TEMPLATE = """从以下文本中提取 {num_keywords} 个最重要的关键词。

文本：
{text}

要求：
1. 关键词应能唯一标识这段文本的核心主题
2. 返回JSON 数组格式，如：["keyword1", "keyword2", "keyword3"]
3. 使用原文语言

关键词数组："""


class KeywordExtractor(BaseExtractor):
    """
    Keyword extractor that extracts keywords from units using LLM.
    
    Returns keywords in list format for easy processing.
    
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
        >>> # Access results (list format)
        >>> print(units[0].metadata["excerpt_keywords"])
        >>> # ["fixed rate", "30-year", "mortgage", "refinance", "conventional loan"]
    """
    
    def __init__(
        self,
        llm_uri: str,
        api_key: str,
        num_keywords: int = 5,
    ):
        self.llm_uri = llm_uri
        self.api_key = api_key
        self.num_keywords = num_keywords
        
        # Use chak (for general dialogue)
        import chak
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit) -> Dict:
        """Extract keywords from a single unit."""
        content = unit.content if hasattr(unit, 'content') else str(unit)
        
        prompt = DEFAULT_TEMPLATE.format(
            text=content,
            num_keywords=self.num_keywords
        )
        
        response = await self._conv.asend(prompt)
        
        # Parse JSON array
        import json
        try:
            keywords = json.loads(response.content.strip())
            if not isinstance(keywords, list):
                # Fallback: try splitting by comma
                keywords = [k.strip() for k in response.content.strip().split(',')]
        except json.JSONDecodeError:
            # Fallback: split by comma
            keywords = [k.strip() for k in response.content.strip().split(',')]
        
        return {"excerpt_keywords": keywords[:self.num_keywords]}
    
    async def aextract(self, units: Sequence) -> List[Dict]:
        """Batch extract metadata from units."""
        import asyncio
        tasks = [self._extract_from_unit(unit) for unit in units]
        return await asyncio.gather(*tasks)
