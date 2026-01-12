"""
Keyword extractor for extracting keywords from units
"""

from typing import List, Dict, Sequence
from pydantic import BaseModel, Field


from .base import BaseExtractor


DEFAULT_TEMPLATE = """从以下文本中提取 {num_keywords} 个最重要的关键词。

文本：
{text}

要求：
1. 关键词应能唯一标识这段文本的核心主题
2. 使用原文语言

请提取关键词："""


class KeywordList(BaseModel):
    """关键词列表结构"""
    keywords: List[str] = Field(description="Extracted keywords from the text")


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
        """Extract keywords from a single unit."""
        content = unit.content if hasattr(unit, 'content') else str(unit)
        
        prompt = DEFAULT_TEMPLATE.format(
            text=content,
            num_keywords=self.num_keywords
        )
        
        # 使用结构化输出
        response = await self._conv.asend(prompt, returns=KeywordList)
        
        # response.content 已经是 KeywordList 对象
        keywords = response.content.keywords[:self.num_keywords]
        
        return {"excerpt_keywords": keywords}
