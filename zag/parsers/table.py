"""
Table parser for extracting and converting Markdown tables to TableUnits
"""

from typing import Optional
import re
from uuid import uuid4

from ..schemas.unit import TextUnit, TableUnit
from ..schemas import UnitMetadata


class TableParser:
    """
    Parse Markdown tables from TextUnit and convert to TableUnit
    
    Features:
    - Extract Markdown tables from TextUnit.content
    - Convert to pandas DataFrame (mandatory)
    - Inherit full metadata from TextUnit
    - Establish bidirectional relations using add_reference
    - Optional: Filter data-critical tables using LLM
    
    Example:
        >>> # Basic usage (no filtering)
        >>> parser = TableParser()
        >>> table_units = parser.parse([text_unit])
        
        >>> # With data-critical filtering (async)
        >>> parser = TableParser(
        ...     llm_uri="bailian/qwen-plus",
        ...     api_key="sk-xxx"
        ... )
        >>> critical_tables = await parser.aparse(
        ...     text_units=[text_unit],
        ...     filter_critical=True
        ... )
        
        >>> # Sync filtering will raise error
        >>> parser.parse([text_unit], filter_critical=True)  # ValueError!
    
    Notes:
        - Only parses standard Markdown tables (with header separator row)
        - Does not generate embedding_content or caption (use TableEnricher)
        - TextUnit content remains unchanged (no replacement)
        - filter_critical requires async execution (use aparse)
    """
    
    # Markdown table pattern
    TABLE_PATTERN = re.compile(
        r'(\|.+\|[\r\n]+\|[\s\-:|]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)',
        re.MULTILINE
    )
    
    def __init__(self, llm_uri: str = None, api_key: str = None):
        """
        Initialize TableParser
        
        Args:
            llm_uri: LLM URI for filtering (e.g., "bailian/qwen-plus")
            api_key: API key for LLM provider
        """
        self.llm_uri = llm_uri
        self.api_key = api_key
    
    def parse_from_unit(self, text_unit: TextUnit) -> list[TableUnit]:
        """
        Parse tables from TextUnit and create TableUnits with complete information
        
        Args:
            text_unit: TextUnit containing Markdown tables
        
        Returns:
            List of TableUnits with:
            - content: Original Markdown table
            - df: pandas DataFrame (mandatory)
            - metadata: Full inheritance from TextUnit
            - Bidirectional relations established via add_reference
        
        Note:
            TextUnit.content remains unchanged (no replacement)
        """
        content = text_unit.content
        if not content:
            return []
        
        # Find all tables
        matches = self.TABLE_PATTERN.findall(content)
        if not matches:
            return []
        
        table_units = []
        
        for table_text in matches:
            # Parse to DataFrame
            df = self._parse_markdown_to_dataframe(table_text)
            if df is None or df.empty:
                continue
            
            # Create TableUnit with full metadata inheritance
            table_unit = TableUnit(
                unit_id=str(uuid4()),
                content=table_text.strip(),  # Original Markdown
                df=df,  # Mandatory pandas DataFrame
                metadata=text_unit.metadata.model_copy(deep=True) if text_unit.metadata else UnitMetadata()
            )
            
            # Inherit doc_id
            if hasattr(text_unit, 'doc_id') and text_unit.doc_id:
                table_unit.doc_id = text_unit.doc_id
            
            # Establish bidirectional relation
            text_unit.add_reference(table_unit)
            
            table_units.append(table_unit)
        
        return table_units
    
    def _parse_markdown_to_dataframe(self, table_text: str) -> Optional['pd.DataFrame']:
        """
        Parse Markdown table text to pandas DataFrame
        
        Args:
            table_text: Markdown table as string
        
        Returns:
            pandas DataFrame, or None if invalid
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for TableParser. "
                "Install it with: pip install pandas"
            )
        
        lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]
        
        if len(lines) < 3:  # Need at least header, separator, and one data row
            return None
        
        # Parse header row
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split('|')[1:-1]]  # Remove leading/trailing |
        
        if not headers:
            return None
        
        # Skip separator row (lines[1])
        
        # Parse data rows
        rows = []
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) == len(headers):  # Only include valid rows
                rows.append(cells)
        
        if not rows:
            return None
        
        # Create DataFrame
        try:
            df = pd.DataFrame(rows, columns=headers)
            return df
        except Exception:
            return None
    
    def parse(self, text_units: list[TextUnit], filter_critical: bool = False) -> list[TableUnit]:
        """
        Parse tables from multiple TextUnits (synchronous)
        
        Args:
            text_units: List of TextUnits
            filter_critical: If True, raises error (use aparse() instead)
        
        Returns:
            Flattened list of all TableUnits
        
        Raises:
            ValueError: If filter_critical=True (async required for LLM filtering)
        
        Note:
            For data-critical filtering, use aparse() with filter_critical=True
        """
        if filter_critical:
            raise ValueError(
                "filter_critical=True requires async execution. "
                "Please use aparse() instead of parse()."
            )
        
        all_tables = []
        for unit in text_units:
            if isinstance(unit, TextUnit):
                tables = self.parse_from_unit(unit)
                all_tables.extend(tables)
        return all_tables
    
    async def aparse(self, text_units: list[TextUnit], filter_critical: bool = False, max_concurrent: int = 3) -> list[TableUnit]:
        """
        Parse tables from multiple TextUnits (asynchronous with optional filtering)
        
        Args:
            text_units: List of TextUnits
            filter_critical: If True, use LLM to filter data-critical tables only
            max_concurrent: Maximum concurrent LLM requests (only used if filter_critical=True)
        
        Returns:
            Flattened list of TableUnits (filtered if filter_critical=True)
        
        Note:
            - If filter_critical=False, returns all parsed tables
            - If filter_critical=True, uses LLM to filter data-critical tables only
            - Requires llm_uri and api_key to be set for filtering
        """
        import asyncio
        
        all_tables = []
        
        # First, parse all tables synchronously
        for unit in text_units:
            if isinstance(unit, TextUnit):
                tables = self.parse_from_unit(unit)
                all_tables.extend(tables)
        
        # If filtering is disabled or LLM not configured, return all tables
        if not filter_critical or not self.llm_uri:
            return all_tables
        
        # Filter data-critical tables using LLM (with concurrency control)
        async def filter_batch(batch: list[TableUnit]) -> list[TableUnit]:
            tasks = [self._is_data_critical(table) for table in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Keep only data-critical tables (handle exceptions gracefully)
            filtered = []
            for table, is_critical in zip(batch, results):
                if isinstance(is_critical, Exception):
                    # If LLM fails, keep the table (fail-open)
                    filtered.append(table)
                elif is_critical:
                    filtered.append(table)
            
            return filtered
        
        # Process in batches to control concurrency
        critical_tables = []
        for i in range(0, len(all_tables), max_concurrent):
            batch = all_tables[i:i + max_concurrent]
            batch_results = await filter_batch(batch)
            critical_tables.extend(batch_results)
        
        return critical_tables
    
    async def _is_data_critical(self, table_unit: TableUnit) -> bool:
        """
        Use LLM to determine if a table contains critical data
        
        Args:
            table_unit: TableUnit to analyze
        
        Returns:
            True if table is data-critical, False otherwise
        
        Note:
            Data-critical tables are those containing:
            - Important numerical data (rates, prices, metrics)
            - Key business information (products, features, specifications)
            - Comparison data between options
            
            NOT data-critical:
            - Formatting/layout tables
            - Simple lists that could be bullet points
            - Metadata/navigation tables
        """
        # Check entry conditions
        if not self.llm_uri:
            return True  # Fail-open: keep table if LLM not configured
        
        if table_unit.df is None or table_unit.df.empty:
            return True  # Fail-open: keep table if no data
        
        try:
            import chak
            from pydantic import BaseModel, Field
            
            # Prepare table preview
            table_preview = self._format_table_for_llm(table_unit.df, max_rows=3)
            
            # Define structured output
            class TableCriticalityAnalysis(BaseModel):
                is_critical: bool = Field(description="Whether this table contains critical data")
                reason: str = Field(description="Brief explanation of why it's critical or not")
            
            # Build prompt
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
            
            # Call LLM
            conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
            analysis = await conv.asend(prompt, returns=TableCriticalityAnalysis)
            
            return analysis.is_critical
            
        except Exception as e:
            # Fail-open: if LLM fails, keep the table
            return True
    
    def _format_table_for_llm(self, df: 'pd.DataFrame', max_rows: int = 3) -> str:
        """
        Format DataFrame for LLM analysis
        
        Args:
            df: pandas DataFrame
            max_rows: Maximum rows to include in preview
        
        Returns:
            Formatted string representation
        """
        import json
        
        preview = {
            "columns": list(df.columns),
            "row_count": len(df),
            "sample_rows": df.head(max_rows).to_dict(orient='records')
        }
        
        return json.dumps(preview, ensure_ascii=False, indent=2)
