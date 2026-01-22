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
    
    def __init__(self, llm_uri: str, api_key: str = None, normalize_table: bool = False):
        """
        Initialize TableEnricher
        
        Args:
            llm_uri: LLM URI for generation
            api_key: API key for LLM provider
            normalize_table: Whether to normalize/fix table structure before enrichment.
                           Useful for complex tables with merged cells, multi-level headers.
        """
        self.llm_uri = llm_uri
        self.api_key = api_key
        self.normalize_table = normalize_table
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
        
        # 0. Normalize table structure if enabled (fix merged cells, multi-level headers)
        if self.normalize_table:
            try:
                normalized_df = await self._normalize_table_structure(unit.df)
                # Update both df and content (markdown representation)
                unit.df = normalized_df
                unit.content = self._df_to_markdown(normalized_df)
            except Exception as e:
                # If normalization fails, continue with original df
                # Print error for debugging
                print(f"[DEBUG] Table normalization failed: {e}")
                import traceback
                traceback.print_exc()
        
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
    async def _normalize_table_structure(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Normalize complex table structure using LLM
        
        Fixes:
        - Multi-level merged headers
        - Empty cells from merged cells
        - Misaligned rows and columns
        - Complex table layouts
        
        Args:
            df: Original DataFrame with structural issues
        
        Returns:
            Normalized DataFrame with complete rows/columns
        """
        import chak
        from pydantic import BaseModel
        import pandas as pd
        
        # Define output schema
        class NormalizedTable(BaseModel):
            columns: list[str]
            rows: list[list]
        
        # Serialize DataFrame to JSON for LLM
        table_data = {
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "rows": df.values.tolist()
        }
        
        import json
        table_json = json.dumps(table_data, ensure_ascii=False, indent=2)
        
        prompt = f"""You are a table structure normalization expert. The table below was extracted from a PDF and may have structural issues due to merged cells.

Original Table Data:
{table_json}

Your Task:
Generate a FULLY NORMALIZED table where EVERY cell has a value. The table should be analysis-ready with consistent structure.

IMPORTANT - Column/Row Consistency Rule:
- **The number of values in EVERY row must EXACTLY match the number of columns**
- Count carefully: if you have N columns, each row must have N values
- This is CRITICAL - mismatched column counts will break the table

Normalization Strategy:
1. **Identify the table structure**:
   - Look at the data pattern - is it a wide table with many similar column groups?
   - If yes, consider if rows should be expanded instead of creating too many columns
   - Prefer having more rows with fewer columns over fewer rows with many columns

2. **Fill ALL empty cells**: 
   - If a cell is empty due to merged cells, copy the value from above (for row spans) or from left (for column spans)
   - NEVER leave any cell empty - fill with the appropriate merged value
   - Information redundancy is acceptable and encouraged

3. **Flatten multi-level headers**:
   - If headers span multiple rows, combine them into single descriptive column names
   - Use "|" separator for hierarchy (e.g., "Transaction Type|Occupancy|Attribute")
   - Make column names clear and complete
   - Keep column count reasonable (prefer 5-15 columns, not 50+)

4. **Ensure data alignment**:
   - Verify that each data row has the EXACT same number of values as there are columns
   - If original table has repeating column patterns, consider expanding rows instead
   - Remove any structural rows that are just formatting (if any)

Output Requirements:
- columns: List of clear, flattened column names
- rows: List of complete data rows where EVERY cell has a value
- **CRITICAL**: len(row) == len(columns) for every single row
- NO empty strings unless the original data is genuinely missing (not due to merged cells)

Example - Converting wide table to normalized:
Before (wide, hard to read):
Columns: ["A|B|X", "A|B|Y", "A|C|X", "A|C|Y"]
Rows: [["1", "2", "3", "4"]]

After (normalized, analysis-ready):
Columns: ["Category", "Type", "Value"]
Rows: [["A", "B", "1"], ["A", "B", "2"], ["A", "C", "3"], ["A", "C", "4"]]

Verification Checklist:
✓ Does every row have exactly len(columns) values?
✓ Are there no empty cells (except genuine missing data)?
✓ Is the column count reasonable (<20 columns)?
✓ Would this table be easy to analyze in Excel/Pandas?

IMPORTANT: Prioritize creating a correctly structured table (matching column/row counts) over preserving the exact original layout.
"""
        
        # Create independent conversation
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
        
        try:
            # Get structured response from LLM
            print(f"[DEBUG] Sending table to LLM for normalization...")
            print(f"[DEBUG] Original table shape: {df.shape}")
            print(f"[DEBUG] Original columns: {df.columns.tolist()[:3]}...")
            
            response = await conv.asend(prompt, returns=NormalizedTable)
            
            print(f"[DEBUG] LLM response received")
            print(f"[DEBUG] Response columns count: {len(response.columns)}")
            print(f"[DEBUG] Response rows count: {len(response.rows)}")
            print(f"[DEBUG] Response columns: {response.columns[:3]}...")
            
            # Validate response
            if not response.columns or not response.rows:
                raise ValueError("LLM returned empty table structure")
            
            # Ensure all rows have same length as columns
            validated_rows = []
            for row in response.rows:
                if len(row) == len(response.columns):
                    validated_rows.append(row)
                elif len(row) < len(response.columns):
                    # Pad with empty strings
                    validated_rows.append(row + [""] * (len(response.columns) - len(row)))
                else:
                    # Truncate
                    validated_rows.append(row[:len(response.columns)])
            
            # Create new DataFrame
            normalized_df = pd.DataFrame(validated_rows, columns=response.columns)
            
            return normalized_df
            
        except Exception as e:
            # If normalization fails, return original df
            raise Exception(f"Failed to normalize table: {e}")
    
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
    
    def _df_to_markdown(self, df: 'pd.DataFrame') -> str:
        """Convert DataFrame to markdown table format"""
        if df is None or df.empty:
            return "*[Empty table]*"
        
        import pandas as pd
        
        lines = []
        
        # Header row
        header = [str(col).strip() if pd.notna(col) and str(col).strip() else "" for col in df.columns]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Data rows
        for idx in range(len(df)):
            row = df.iloc[idx].tolist()
            row = [str(cell).strip() if pd.notna(cell) and str(cell).strip() else "" for cell in row]
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
