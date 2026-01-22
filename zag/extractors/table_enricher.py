"""
TableUnit enricher for generating embedding content, caption, and schema
"""

from typing import Dict, List
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import BaseExtractor
from ..schemas.unit import TableUnit


class TableEnricher(BaseExtractor):
    """
    TableUnit enricher for generating embedding content and caption
    
    Generates:
        - embedding_content: Detailed natural language description (for vector search)
        - caption: Intelligent title based on content and context (for display/organization)
    
    Note:
        - Does NOT generate schema (df already contains all schema info: df.columns, len(df))
        - Each LLM call uses independent conversation to avoid context pollution
        - Caption generation uses TextUnit context for better accuracy
    
    Args:
        llm_uri: LLM URI in format: provider/model (e.g., "bailian/qwen-plus")
        api_key: API key for the LLM provider
    
    Example:
        >>> enricher = TableEnricher(
        ...     llm_uri="bailian/qwen-plus",
        ...     api_key="sk-xxx"
        ... )
        >>> await enricher.aextract(table_units)
        >>> # Fields auto-populated:
        >>> print(table_units[0].caption)
        >>> print(table_units[0].embedding_content)
    """
    
    def __init__(self, llm_uri: str, api_key: str = None):
        self.llm_uri = llm_uri
        self.api_key = api_key
        # Do NOT create shared conversation here - it will cause context pollution
        # Each table should use its own conversation instance
    
    async def aextract(self, units: list, max_concurrent: int = 3) -> list:
        """
        Extract enrichment data and auto-populate TableUnit fields
        
        Override base class to handle caption field specially:
        - embedding_content → unit.embedding_content (handled by base class)
        - caption → unit.caption (special handling for TableUnit)
        
        Args:
            units: List of TableUnits
            max_concurrent: Maximum concurrent LLM requests
            
        Returns:
            List of result dicts (mostly empty after auto-population)
        """
        # Call parent's aextract (handles embedding_content automatically)
        results = await super().aextract(units, max_concurrent)
        
        # Special handling for caption: move from metadata.custom to unit.caption
        for unit in units:
            if isinstance(unit, TableUnit):
                if unit.metadata.custom and "caption" in unit.metadata.custom:
                    caption_value = unit.metadata.custom.pop("caption")
                    # Only set if not empty (LLM might return empty string)
                    if caption_value and caption_value.strip():
                        unit.caption = caption_value
        
        return results
    
    async def _extract_from_unit(self, unit) -> Dict:
        """
        Extract enrichment data from a single TableUnit
        
        Returns:
            Dict with enrichment data:
            {
                "embedding_content": str,  # Detailed NL description
                "caption": str,            # Intelligent caption
            }
        """
        if not isinstance(unit, TableUnit):
            return {}
        
        if unit.df is None or unit.df.empty:
            return {
                "embedding_content": "",
                "caption": ""
            }
        
        # 1. Get context from referenced TextUnit
        source_text_units = unit.get_referenced_by()
        context = source_text_units[0].content if source_text_units else ""
        
        # 2. Generate caption (with context)
        # Each LLM call uses its own conversation to avoid context pollution
        caption = await self._generate_caption(unit.df, context)
        
        # 3. Generate embedding_content (detailed NL description)
        # Independent conversation - no shared context with caption generation
        embedding_content = await self._generate_embedding_content(unit.df, caption)
        
        return {
            "embedding_content": embedding_content,
            "caption": caption
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def _generate_caption(self, df: 'pd.DataFrame', context: str = "") -> str:
        """
        Generate intelligent caption for table using LLM
        
        Uses context from TextUnit to generate more accurate caption.
        Creates independent conversation for isolation.
        
        Args:
            df: pandas DataFrame
            context: Content of the source TextUnit (for better accuracy)
        
        Returns:
            Intelligent caption, or empty string if failed
        """
        # Create independent conversation for this call
        import chak
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
        # Prepare table data for LLM
        table_preview = self._dataframe_to_preview(df, max_rows=3)
        
        # Build prompt with context
        if context:
            prompt = f"""Based on the following context and table data, generate a concise and accurate table caption.

Context (from document):
{context[:500]}...

Table preview:
{table_preview}

Requirements:
- Generate a SHORT, descriptive caption (10-15 words max)
- Use the SAME LANGUAGE as the table content
- Capture the main topic of this table
- DO NOT include surrounding explanation, ONLY output the caption

Caption:"""
        else:
            prompt = f"""Generate a concise caption for the following table:

Table preview:
{table_preview}

Requirements:
- Generate a SHORT, descriptive caption (10-15 words max)
- Use the SAME LANGUAGE as the table content
- Capture the main topic of this table
- DO NOT include surrounding explanation, ONLY output the caption

Caption:"""
        
        try:
            response = await conv.asend(prompt)
            caption = response.content.strip()
            # Remove quotes if LLM added them
            caption = caption.strip('"').strip("'")
            # Ensure caption is not empty
            if not caption or not caption.strip():
                # Fallback if LLM returned empty
                return f"Table with columns: {', '.join(df.columns[:3])}"
            return caption
        except Exception:
            # Fallback: generate simple caption from columns
            return f"Table with columns: {', '.join(df.columns[:3])}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def _generate_embedding_content(
        self, 
        df: 'pd.DataFrame', 
        caption: str
    ) -> str:
        """
        Generate detailed natural language description for embedding
        
        Includes: caption, all columns, sample data
        Pure natural language, suitable for vector search
        Creates independent conversation for isolation.
        
        Args:
            df: pandas DataFrame
            caption: Table caption (passed as parameter, not from conversation history)
        
        Returns:
            Detailed NL description
        """
        # Create independent conversation for this call
        import chak
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
        # Prepare comprehensive table data
        table_data = self._dataframe_to_dict(df, max_rows=5)
        
        prompt = f"""Convert the following table into a DETAILED natural language description suitable for semantic search.

Table Caption: {caption}

Table Data:
{table_data}

Requirements:
- Use the SAME LANGUAGE as the table content (do NOT translate)
- Start with the caption
- List ALL column names
- Describe the FIRST 3-5 ROWS with COMPLETE details from ALL columns
- Use natural, flowing sentences
- Make it suitable for vector search (semantic matching)
- Include ALL data values, do NOT omit or summarize
- Format example: "Table: [Caption]. This table contains columns: [列1], [列2], [列3]. The data includes: Row 1 - [列1]: [值], [列2]: [值]..."

Description:"""
        
        # Create new conversation for this table (avoid context pollution)
        import chak
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
        
        try:
            response = await conv.asend(prompt)
            return response.content.strip()
        except Exception:
            # Fallback: simple description
            columns = ", ".join(df.columns)
            sample_row = df.iloc[0].to_dict() if len(df) > 0 else {}
            sample_desc = ", ".join([f"{k}: {v}" for k, v in sample_row.items()])
            return f"{caption}. Columns: {columns}. Sample: {sample_desc}"
    
    def _dataframe_to_preview(self, df: 'pd.DataFrame', max_rows: int = 3) -> str:
        """Convert DataFrame to preview string for LLM"""
        preview = f"Columns: {list(df.columns)}\n"
        preview += f"Rows: {len(df)}\n"
        preview += f"Sample data (first {max_rows} rows):\n"
        preview += df.head(max_rows).to_string(index=False)
        return preview
    
    def _dataframe_to_dict(self, df: 'pd.DataFrame', max_rows: int = 5) -> str:
        """Convert DataFrame to dict format for LLM (preserves duplicate columns)"""
        sample_df = df.head(max_rows)
        data = {
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "sample_rows": sample_df.values.tolist()
        }
        import json
        return json.dumps(data, ensure_ascii=False, indent=2)
