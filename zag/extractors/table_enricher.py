"""
TableUnit enricher for generating embedding content, caption, and criticality judgment
"""

from typing import Dict, List, Any
from enum import Enum
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import BaseExtractor
from ..schemas.unit import TableUnit


class TableEnrichMode(Enum):
    """Table enrichment mode
    
    Defines how TableEnricher processes tables: judge criticality + generate enrichment content
    """
    
    ALL = "all"
    """Judge all tables for is_data_critical + enrich all tables
    
    Use cases:
    - Development/debugging phase
    - Treating all tables equally
    
    Cost: Higher (all tables call LLM for caption / embedding_content)
    """
    
    CRITICAL_ONLY = "critical_only"
    """Judge all tables for is_data_critical + enrich only critical tables (recommended)
    
    Use cases:
    - Production environment, cost-sensitive
    - Only important tables need independent indexing
    - Non-critical tables can be embedded with TextUnit
    
    Cost: Medium (only critical tables call LLM for enrichment)
    """


class TableEnricher(BaseExtractor):
    """
    TableUnit enricher for generating embedding content, caption, and criticality judgment
    
    Generates:
        - is_data_critical: Whether table contains critical business data (always judged)
        - embedding_content: Detailed natural language description (optional, based on mode)
        - caption: Intelligent title based on content and context (optional, based on mode)
    
    Note:
        - Does NOT generate schema (df already contains all schema info: df.columns, len(df))
        - Each LLM call uses independent conversation to avoid context pollution
        - Caption generation uses TextUnit context for better accuracy
    
    Args:
        llm_uri: LLM URI in format: provider/model (e.g., "bailian/qwen-plus")
        api_key: API key for the LLM provider
    
    Example:
        >>> from zag.extractors import TableEnricher, TableEnrichMode
        >>> 
        >>> enricher = TableEnricher(
        ...     llm_uri="bailian/qwen-plus",
        ...     api_key="sk-xxx"
        ... )
        >>> 
        >>> # Production (recommended) - judge all + enrich critical only
        >>> await enricher.aextract(tables, mode=TableEnrichMode.CRITICAL_ONLY)
        >>> 
        >>> # Development - judge all + enrich all
        >>> await enricher.aextract(tables, mode=TableEnrichMode.ALL)
        >>> 
        >>> # Filter critical tables
        >>> critical = [t for t in tables 
        ...             if t.metadata.custom.get("table", {}).get("is_data_critical")]
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
    
    async def aextract(
        self, 
        units: list, 
        mode: TableEnrichMode = TableEnrichMode.CRITICAL_ONLY,
        max_concurrent: int = 3
    ) -> list:
        """
        Enrich tables: judge is_data_critical + generate caption / embedding_content
        
        Workflow:
        1. Judge all tables for is_data_critical (write to metadata.custom["table"])
        2. Based on mode, decide enrichment scope:
           - TableEnrichMode.ALL: Enrich all tables
           - TableEnrichMode.CRITICAL_ONLY: Only enrich data-critical tables (recommended)
        
        Args:
            units: List of TableUnits
            mode: Enrichment mode, default CRITICAL_ONLY (recommended for production)
            max_concurrent: Maximum concurrent LLM requests
            
        Returns:
            Processed TableUnit list (all tables, but only some enriched)
            
        Note:
            - All tables will be judged for is_data_critical (written to metadata.custom["table"])
            - Users can filter tables based on metadata.custom["table"]["is_data_critical"]
            
        Usage:
            # Production (recommended)
            from zag.extractors import TableEnricher, TableEnrichMode
            enricher = TableEnricher(llm_uri="bailian/qwen-plus", api_key="sk-xxx")
            tables = await enricher.aextract(tables)  # Default CRITICAL_ONLY
            
            # Development/debugging
            tables = await enricher.aextract(tables, mode=TableEnrichMode.ALL)
            
            # Filter critical tables
            critical_tables = [
                t for t in tables 
                if t.metadata.custom.get("table", {}).get("is_data_critical")
            ]
        """
        # 1. Judge all tables for is_data_critical
        await self._batch_judge_critical(units, max_concurrent)
        
        # 2. Based on mode, decide which tables to enrich
        if mode == TableEnrichMode.CRITICAL_ONLY:
            units_to_enrich = [
                u for u in units 
                if isinstance(u, TableUnit) and 
                u.metadata.custom.get("table", {}).get("is_data_critical", False)
            ]
        else:  # TableEnrichMode.ALL
            units_to_enrich = [u for u in units if isinstance(u, TableUnit)]
        
        # 3. Enrich selected tables (caption + embedding_content)
        if units_to_enrich:
            await self._batch_enrich(units_to_enrich, max_concurrent)
        
        return units
    
    async def _batch_judge_critical(self, units: list, max_concurrent: int):
        """Batch judge all tables for is_data_critical
        
        Args:
            units: List of TableUnits
            max_concurrent: Maximum concurrent LLM requests
        """
        table_units = [u for u in units if isinstance(u, TableUnit)]
        if not table_units:
            return
        
        # Process in batches
        for i in range(0, len(table_units), max_concurrent):
            batch = table_units[i : i + max_concurrent]
            tasks = [self._judge_critical(unit) for unit in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _batch_enrich(self, units: list, max_concurrent: int):
        """Batch generate caption + embedding_content for tables
        
        Args:
            units: List of TableUnits to enrich
            max_concurrent: Maximum concurrent LLM requests
        """
        # Use parent's aextract to generate caption + embedding_content
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
    
    async def _judge_critical(self, table_unit: TableUnit) -> bool:
        """Judge whether a single table is data-critical using LLM
        
        Args:
            table_unit: TableUnit to analyze
            
        Returns:
            True if table is data-critical, False otherwise
            
        Notes:
            Data-critical tables are those containing:
            - Important numerical data (rates, prices, metrics)
            - Key business information (products, features, specifications)
            - Comparison data between entities (product vs product, option vs option)
            
            NOT data-critical:
            - Formatting/layout tables
            - Simple lists that could be bullet points
            - Metadata/navigation tables
        """
        if table_unit.df is None or table_unit.df.empty:
            # Empty table is not critical
            table_meta = table_unit.metadata.custom.setdefault("table", {})
            table_meta["is_data_critical"] = False
            table_meta["criticality_reason"] = "Empty table"
            return False
        
        try:
            import chak
            from pydantic import BaseModel, Field
            
            table_preview = self._format_table_for_llm(table_unit.df, max_rows=3)
            
            class TableCriticalityAnalysis(BaseModel):
                is_critical: bool = Field(
                    description="Whether this table contains critical data",
                )
                reason: str = Field(
                    description="Brief explanation of why it's critical or not",
                )
            
            prompt = f"""Analyze whether this table contains CRITICAL DATA that needs special processing.

Table preview:
{table_preview}

A table is DATA-CRITICAL if it contains:
✅ Quantitative data that users would query (numbers, percentages, amounts, measurements)
✅ Structured information requiring precise retrieval (specifications, features, attributes)
✅ Comparison data between entities (product vs product, option vs option)
✅ Reference information with business value (rules, thresholds, limits, requirements)

A table is NOT data-critical if:
❌ Document navigation or structure (table of contents, page numbers, section indexes)
❌ Pure metadata without business value (version info, timestamps, author names)
❌ Status or progress indicators (checkmarks, completion markers, read/unread flags)
❌ Decorative or formatting elements (simple lists that could be bullets, layout tables)

Key question: Would a user specifically search for and retrieve this data to answer a business question?
- If YES → Mark as critical
- If NO (just navigation/structure/metadata) → Mark as NOT critical

Based on the table preview above, is this table data-critical?
"""
            
            conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
            analysis = await conv.asend(prompt, returns=TableCriticalityAnalysis)
            
            is_critical = analysis.is_critical
            try:
                if table_unit.metadata is not None:
                    table_meta = table_unit.metadata.custom.setdefault("table", {})
                    table_meta["is_data_critical"] = is_critical
                    table_meta["criticality_reason"] = analysis.reason
            except Exception:
                pass
            
            return is_critical
            
        except Exception:
            # On error, assume critical to be safe
            try:
                table_meta = table_unit.metadata.custom.setdefault("table", {})
                table_meta["is_data_critical"] = True
                table_meta["criticality_reason"] = "Error during judgment, assumed critical"
            except Exception:
                pass
            return True
    
    def _format_table_for_llm(self, df: Any, max_rows: int = 3) -> str:
        """Format DataFrame into a JSON-like preview string for LLM prompts"""
        import json
        
        try:
            sample_df = df.head(max_rows)
            preview = {
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "sample_rows": sample_df.values.tolist(),
            }
            return json.dumps(preview, ensure_ascii=False, indent=2)
        except Exception:
            return "{}"
    
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
                col_names = [str(c) for c in df.columns[:3]]
                return f"Table with columns: {', '.join(col_names)}"
            return caption
        except Exception:
            # Fallback: generate simple caption from columns
            col_names = [str(c) for c in df.columns[:3]]
            return f"Table with columns: {', '.join(col_names)}"
    
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
        
        prompt = f"""Create a search-optimized description of this table for semantic retrieval.

TABLE CAPTION: {caption}

TABLE DATA PREVIEW (first 3 rows):
{table_data}

INSTRUCTIONS:
1. **Purpose** (1 sentence): State what this table compares or shows.

2. **Structure** (1 sentence): Briefly explain column meanings and groupings.

3. **Key Values** (MOST IMPORTANT - extract 8-12 critical data points):
   For each important data point, use this format: "[Row context] for [Column context] is [exact value]"
   
   Examples:
   - "Owner occupied purchase loan has 90% maximum LTV"
   - "Minimum FICO score for investment property cash-out refinance is 700"
   - "Maximum loan amount for second home is $1.0MM"
   
   CRITICAL: Include specific numbers, percentages, scores, amounts, thresholds.
   CRITICAL: Link row context + column context + exact value in ONE sentence.

4. **Patterns** (1-2 sentences): Note important relationships or trends.

REQUIREMENTS:
- Use the SAME LANGUAGE as the table content
- Keep all original values unchanged
- Focus on data points most likely to be queried
- Make explicit connections between conditions and values

OUTPUT FORMAT:
Purpose: [1 sentence about what this table shows]

Structure: [1 sentence about column organization]

Key Data:
- [Row meaning] for [Column meaning] is [exact value]
- [Another critical data point with full context]
- [Continue for 8-12 most important points]

Patterns: [1-2 sentences about trends or rules]

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
