"""
Table extractor for generating natural language descriptions of tables
"""

from typing import List, Dict, Sequence
import re

from .base import BaseExtractor
from ..schemas.unit import TableUnit, TextUnit


class TableExtractor(BaseExtractor):
    """
    Table extractor that generates natural language descriptions for tables.
    
    Supports:
    1. TableUnit: Generate embedding_content (summary) from json_data
    2. TextUnit: Replace tables in content with summaries for embedding_content
    
    Key design:
    - Reuses TableUnit.json_data (already structured)
    - Does not re-parse tables
    - Only generates embedding_content for vector retrieval
    - Does not modify the unit itself (returns Dict with extracted data)
    - Automatically detects and uses source language
    
    Args:
        llm_uri: LLM URI in format: provider/model
                 e.g., "openai/gpt-4o-mini", "bailian/qwen-plus"
        api_key: API key for the LLM provider
    
    Example:
        >>> # For TableUnits
        >>> extractor = TableExtractor(
        ...     llm_uri="openai/gpt-4o-mini",
        ...     api_key="sk-xxx"
        ... )
        >>> results = await extractor.aextract(table_units)
        >>> for unit, metadata in zip(table_units, results):
        ...     unit.embedding_content = metadata.get("embedding_content")
        >>> 
        >>> # For TextUnits with tables
        >>> results = await extractor.aextract(text_units)
        >>> for unit, metadata in zip(text_units, results):
        ...     if metadata.get("embedding_content"):
        ...         unit.embedding_content = metadata["embedding_content"]
    """
    
    def __init__(self, llm_uri: str, api_key: str):
        self.llm_uri = llm_uri
        self.api_key = api_key
        
        # Create chak conversation (for general dialogue)
        import chak
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit) -> Dict:
        """
        Extract table information from a single unit.
        
        Returns:
            Dict with extracted metadata (does not modify unit)
        """
        # Case 1: TableUnit - generate summary
        if isinstance(unit, TableUnit):
            json_data = unit.json_data
            if not json_data:
                return {}
            
            summary = await self._generate_table_summary(json_data)
            return {"embedding_content": summary}
        
        # Case 2: TextUnit - replace tables with summaries
        if isinstance(unit, TextUnit):
            if self._has_markdown_tables(unit.content):
                processed = await self._process_text_with_tables(unit)
                return {"embedding_content": processed}
        
        return {}
    
    async def _process_text_with_tables(self, text_unit: TextUnit) -> str:
        """
        Replace tables in TextUnit.content with summaries.
        
        Args:
            text_unit: TextUnit containing Markdown tables
        
        Returns:
            Processed text with tables replaced by natural language summaries
        """
        from ..parsers import TableParser
        
        content = text_unit.content
        parser = TableParser()
        
        # Find all tables
        matches = parser.TABLE_PATTERN.findall(content)
        if not matches:
            return content
        
        # Parse all tables and prepare tasks for concurrent processing
        import asyncio
        tasks = []
        valid_matches = []
        
        for table_text in matches:
            json_data = parser._parse_table_text(table_text)
            if json_data:
                tasks.append(self._generate_table_summary(json_data))
                valid_matches.append(table_text)
        
        if not tasks:
            return content
        
        # Generate summaries concurrently
        summaries = await asyncio.gather(*tasks)
        
        # Replace tables with summaries
        modified_content = content
        for table_text, summary in zip(valid_matches, summaries):
            modified_content = modified_content.replace(table_text, summary)
        
        return modified_content
    
    def _has_markdown_tables(self, content: str) -> bool:
        """Check if content contains Markdown tables"""
        from ..parsers import TableParser
        return bool(TableParser.TABLE_PATTERN.search(content))
    
    async def _generate_table_summary(self, json_data: dict) -> str:
        """
        Generate natural language description from table data.
        
        Converts table to complete natural language without losing information.
        LLM will automatically detect and use the source language.
        """
        prompt = f"""Convert the following table into natural language description. Use the SAME LANGUAGE as the table content.

Table data:
{json_data}

Requirements:
- Use the same language as the table (e.g., if table is in Chinese, respond in Chinese)
- Include ALL data from the table - do NOT summarize or omit any information
- Describe each row with complete details from all columns
- Use complete sentences suitable for vector search
- Organize by rows for clarity (e.g., "Row 1: ..., Row 2: ..." or "Product A: ..., Product B: ...")
- Output ONLY the table description, do NOT include any surrounding text or context
- End with a newline to separate from following content

Description:"""
        
        response = await self._conv.asend(prompt)
        # Add newlines to clearly separate from surrounding content
        return "\n" + response.content.strip() + "\n\n"
    
    async def aextract(self, units: Sequence) -> List[Dict]:
        """Batch extract metadata from units."""
        import asyncio
        tasks = [self._extract_from_unit(unit) for unit in units]
        return await asyncio.gather(*tasks)
