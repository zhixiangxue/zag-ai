"""
Table parser for extracting and converting Markdown tables to TableUnits
"""

from typing import Optional
import re
from uuid import uuid4

from ..schemas.unit import TextUnit, TableUnit
from ..schemas.base import UnitMetadata


class TableParser:
    """
    Parse Markdown tables from TextUnit and convert to TableUnit
    
    Features:
    - Extract Markdown tables from TextUnit.content
    - Convert to TableUnit with structured json_data
    - Optionally replace table in TextUnit with placeholder
    - Establish relations between TextUnit and TableUnits
    
    Example:
        >>> parser = TableParser()
        >>> text_unit = TextUnit(
        ...     content="# Rates\\n| Product | Rate |\\n|---------|------|\\n| 30Y | 6.5% |"
        ... )
        >>> table_units = parser.parse_from_unit(text_unit)
        >>> len(table_units)
        1
        >>> table_units[0].json_data
        {'headers': ['Product', 'Rate'], 'rows': [['30Y', '6.5%']]}
    
    Notes:
        - Only parses standard Markdown tables (with header separator row)
        - Does not generate table summaries (use TableExtractor for that)
        - Preserves original table text in TableUnit.content
    """
    
    # Markdown table pattern
    TABLE_PATTERN = re.compile(
        r'(\|.+\|[\r\n]+\|[\s\-:|]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)',
        re.MULTILINE
    )
    
    def __init__(self):
        pass
    
    def parse_from_unit(
        self,
        text_unit: TextUnit,
        replace_with_placeholder: bool = False,
        establish_relation: bool = True
    ) -> list[TableUnit]:
        """
        Parse tables from TextUnit
        
        Args:
            text_unit: TextUnit containing Markdown tables
            replace_with_placeholder: If True, replace tables in content with {{TABLE:id}}
            establish_relation: If True, add table_ids to unit metadata
        
        Returns:
            List of TableUnits parsed from the text
        """
        content = text_unit.content
        if not content:
            return []
        
        # Find all tables
        matches = self.TABLE_PATTERN.findall(content)
        if not matches:
            return []
        
        table_units = []
        table_ids = []
        modified_content = content
        
        for table_text in matches:
            # Parse table to structured data
            json_data = self._parse_table_text(table_text)
            if not json_data:
                continue
            
            # Create TableUnit
            table_id = str(uuid4())
            table_unit = TableUnit(
                unit_id=table_id,
                content=table_text.strip(),  # Original Markdown table
                json_data=json_data,
                metadata=UnitMetadata(
                    context_path=text_unit.metadata.context_path if text_unit.metadata else None
                )
            )
            
            # Set source doc
            if hasattr(text_unit, 'source_doc_id') and text_unit.source_doc_id:
                table_unit.source_doc_id = text_unit.source_doc_id
            
            table_units.append(table_unit)
            table_ids.append(table_id)
            
            # Replace with placeholder if requested
            if replace_with_placeholder:
                placeholder = f"{{{{TABLE:{table_id}}}}}"
                modified_content = modified_content.replace(table_text, placeholder)
        
        # Update TextUnit content if replaced
        if replace_with_placeholder and table_units:
            text_unit.content = modified_content
        
        # Establish relation
        if establish_relation and table_units:
            if not text_unit.metadata:
                text_unit.metadata = UnitMetadata()
            if not text_unit.metadata.custom:
                text_unit.metadata.custom = {}
            text_unit.metadata.custom['table_ids'] = table_ids
        
        return table_units
    
    def _parse_table_text(self, table_text: str) -> Optional[dict]:
        """
        Parse Markdown table text to structured data
        
        Args:
            table_text: Markdown table as string
        
        Returns:
            Dict with 'headers' and 'rows', or None if invalid
        """
        lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]
        
        if len(lines) < 3:  # Need at least header, separator, and one data row
            return None
        
        # Parse header row
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split('|')[1:-1]]  # Remove leading/trailing |
        
        # Skip separator row (lines[1])
        
        # Parse data rows
        rows = []
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) == len(headers):  # Only include valid rows
                rows.append(cells)
        
        if not rows:
            return None
        
        return {
            'headers': headers,
            'rows': rows
        }
    
    def parse_all_from_units(
        self,
        text_units: list[TextUnit],
        replace_with_placeholder: bool = False,
        establish_relation: bool = True
    ) -> list[TableUnit]:
        """
        Parse tables from multiple TextUnits
        
        Args:
            text_units: List of TextUnits
            replace_with_placeholder: If True, replace tables with placeholders
            establish_relation: If True, establish relations
        
        Returns:
            Flattened list of all TableUnits
        """
        all_tables = []
        for unit in text_units:
            if isinstance(unit, TextUnit):
                tables = self.parse_from_unit(
                    unit,
                    replace_with_placeholder=replace_with_placeholder,
                    establish_relation=establish_relation
                )
                all_tables.extend(tables)
        return all_tables
