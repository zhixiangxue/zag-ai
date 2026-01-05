"""
Table extractor for generating natural language descriptions of tables
"""

from typing import List, Dict, Sequence

from .base import BaseExtractor
from ..schemas.unit import TableUnit


class TableExtractor(BaseExtractor):
    """
    Table extractor that generates natural language descriptions for TableUnits.
    
    Key design:
    - Reuses TableUnit.json_data (already structured)
    - Does not re-parse tables
    - Only generates table_summary for vector retrieval
    - Does not duplicate json_data storage
    
    Args:
        llm_uri: LLM URI in format: provider/model
                 e.g., "openai/gpt-4o-mini", "bailian/qwen-plus"
        api_key: API key for the LLM provider
    
    Example:
        >>> extractor = TableExtractor(
        ...     llm_uri="openai/gpt-4o-mini",
        ...     api_key="sk-xxx"
        ... )
        >>> units = extractor(units)
        >>> # Access results
        >>> for unit in units:
        ...     if isinstance(unit, TableUnit):
        ...         print(f"Summary: {unit.metadata['table_summary']}")
        ...         print(f"Data: {unit.json_data}")
    """
    
    def __init__(self, llm_uri: str, api_key: str):
        self.llm_uri = llm_uri
        self.api_key = api_key
        
        # Create chak conversation (for general dialogue)
        import chak
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit) -> Dict:
        """Extract table information from a single unit."""
        if not isinstance(unit, TableUnit):
            return {}
        
        # Reuse structured data from TableUnit
        json_data = unit.json_data
        if not json_data:
            return {}
        
        # Generate natural language description
        prompt = f"""以下是一个表格的结构化数据：

{json_data}

请用 2-3 句话总结这个表格的内容，突出关键数据和对比关系。
要求：使用完整的句子，便于向量检索。

摘要："""
        
        response = await self._conv.asend(prompt)
        
        # Only return summary, do not duplicate json_data
        return {"table_summary": response.content.strip()}
    
    async def aextract(self, units: Sequence) -> List[Dict]:
        """Batch extract metadata from units."""
        import asyncio
        tasks = [self._extract_from_unit(unit) for unit in units]
        return await asyncio.gather(*tasks)
