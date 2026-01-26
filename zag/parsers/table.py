"""Table parser for extracting tables from raw text into TableUnits.

This module defines TableParser, which is responsible for:
- Detecting tables from raw text (Markdown / HTML)
- Converting table text into a pandas DataFrame
- Wrapping results into TableUnit objects

Design principles:
- Core input is always plain text (str)
- Optional metadata/doc_id can be passed explicitly from callers
- Parser does NOT depend on TextUnit for its core logic
- A thin adapter is provided for TextUnit-based pipelines
"""

from typing import Any, Optional
import re
from uuid import uuid4

from ..schemas.unit import TextUnit, TableUnit
from ..schemas import UnitMetadata


class TableParser:
    """Parse tables from text and convert to TableUnit.

    Core responsibilities:
    - Accept raw text (Markdown/HTML/mixed)
    - Extract table segments from the text
    - Parse each table into a pandas DataFrame
    - Create TableUnit instances with optional metadata/doc_id

    Usage (string-based):
        >>> parser = TableParser()
        >>> tables = parser.parse(text, metadata=unit_metadata, doc_id=doc_id)

    Usage (TextUnit-based convenience):
        >>> parser = TableParser()
        >>> tables = parser.parse_from_unit(text_unit)

    Notes:
        - Only parses standard Markdown tables (with header separator row)
        - HTML tables are detected via BeautifulSoup if available
        - Does not generate embedding_content or caption (use TableEnricher)
        - Does not judge is_data_critical (use TableEnricher)
        - Does not modify original text or manage cross-unit references
    """

    # Markdown table pattern (same as TextSplitter, TableSplitter)
    TABLE_PATTERN = re.compile(
        r"(\|.+\|[\r\n]+\|[\s\-:|]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)",
        re.MULTILINE,
    )

    # ------------------------------------------------------------------
    # Core text-based parsing API
    # ------------------------------------------------------------------
    def parse(
        self,
        text: str,
        metadata: UnitMetadata | None = None,
        doc_id: str | None = None,
    ) -> list[TableUnit]:
        """Parse tables from raw text and return TableUnits.

        This is the core API which works purely on strings. It can be used
        independently of any zag-specific unit types. Callers may optionally
        provide metadata/doc_id to be attached to each created TableUnit.

        Args:
            text: Raw text that may contain Markdown and/or HTML tables
            metadata: Optional UnitMetadata to be copied to each TableUnit
            doc_id: Optional document id to attach to each TableUnit

        Returns:
            List of TableUnit instances extracted from the text
        """

        if not text:
            return []

        table_units: list[TableUnit] = []

        # 1. Markdown tables
        md_matches = self.TABLE_PATTERN.findall(text)
        for table_text in md_matches:
            df = self._parse_markdown_to_dataframe(table_text)
            if df is None or df.empty:
                continue

            unit_metadata = (
                metadata.model_copy(deep=True) if metadata is not None else UnitMetadata()
            )
            table_meta = unit_metadata.custom.setdefault("table", {})
            table_meta["source_format"] = "markdown"
            table_meta["row_count"] = len(df)
            table_meta["column_count"] = len(df.columns)

            table_unit = TableUnit(
                unit_id=str(uuid4()),
                content=table_text.strip(),
                df=df,
                metadata=unit_metadata,
            )
            if doc_id:
                table_unit.doc_id = doc_id

            table_units.append(table_unit)

        # 2. HTML tables (only if BeautifulSoup is available)
        html_tables = self._extract_html_tables(text)
        for table_html in html_tables:
            is_complex = self._detect_html_table_complexity(table_html)
            df = self._convert_html_to_dataframe(table_html)
            if df is None or df.empty:
                continue

            unit_metadata = (
                metadata.model_copy(deep=True) if metadata is not None else UnitMetadata()
            )
            table_meta = unit_metadata.custom.setdefault("table", {})
            table_meta["source_format"] = "html"
            table_meta["row_count"] = len(df)
            table_meta["column_count"] = len(df.columns)
            table_meta["is_complex"] = is_complex

            table_unit = TableUnit(
                unit_id=str(uuid4()),
                content=table_html.strip(),
                df=df,
                metadata=unit_metadata,
            )
            if doc_id:
                table_unit.doc_id = doc_id

            table_units.append(table_unit)

        return table_units

    # ------------------------------------------------------------------
    # TextUnit-based convenience API
    # ------------------------------------------------------------------
    def parse_from_unit(self, text_unit: TextUnit) -> list[TableUnit]:
        """Parse tables from a TextUnit.

        This is a thin convenience wrapper around :meth:`parse` that
        preserves metadata/doc_id from the TextUnit.

        Args:
            text_unit: TextUnit containing text with tables

        Returns:
            List of TableUnits created from the TextUnit content
        """

        return self.parse(
            text=text_unit.content or "",
            metadata=text_unit.metadata,
            doc_id=getattr(text_unit, "doc_id", None),
        )

    # ------------------------------------------------------------------
    # Low-level parsing helpers
    # ------------------------------------------------------------------
    def _extract_html_tables(self, text: str) -> list[str]:
        """Extract all <table>...</table> snippets from text.

        BeautifulSoup is used when available. If it is not installed, this
        method returns an empty list and HTML tables will simply be ignored.
        """

        try:
            from bs4 import BeautifulSoup
        except ImportError:  # pragma: no cover - optional dependency
            return []

        soup = BeautifulSoup(text, "html.parser")
        return [str(table) for table in soup.find_all("table")]

    def _parse_markdown_to_dataframe(self, table_text: str) -> "Any | None":
        """Parse a Markdown table into a pandas DataFrame.

        Args:
            table_text: Markdown table as string

        Returns:
            pandas DataFrame, or None if parsing fails
        """

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "pandas is required for TableParser. "
                "Install it with: pip install pandas"
            ) from exc

        lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
        if len(lines) < 3:
            # Need at least header, separator, and one data row
            return None

        # Parse header row
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split("|")[1:-1]]
        if not headers:
            return None

        # Parse data rows (skip separator row at index 1)
        rows: list[list[str]] = []
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if len(cells) == len(headers):
                rows.append(cells)

        if not rows:
            return None

        try:
            df = pd.DataFrame(rows, columns=headers)
            return df
        except Exception:
            return None

    def _detect_html_table_complexity(self, table_html: str) -> bool:
        """Detect whether an HTML table is structurally complex.

        Complexity here is defined as the presence of merged cells
        (rowspan/colspan > 1) or multi-row headers. This is a purely
        structural notion and does not involve any semantic reasoning.
        """

        try:
            from bs4 import BeautifulSoup
        except ImportError:  # pragma: no cover - optional dependency
            return False

        soup = BeautifulSoup(table_html, "html.parser")

        # 1) Check for merged cells
        for cell in soup.find_all(["td", "th"]):
            rowspan = cell.get("rowspan")
            colspan = cell.get("colspan")
            try:
                if (rowspan and int(rowspan) > 1) or (colspan and int(colspan) > 1):
                    return True
            except ValueError:
                # Non-integer attributes are treated as complex to be safe
                return True

        # 2) Check for multi-row header (thead with multiple tr)
        thead = soup.find("thead")
        if thead is not None:
            header_rows = thead.find_all("tr", recursive=False)
            if len(header_rows) > 1:
                return True

        return False

    def _convert_html_to_dataframe(self, table_html: str) -> "Any | None":
        """Convert an HTML <table>...</table> snippet into a pandas DataFrame.

        This is a best-effort conversion based on pandas.read_html, which in
        turn relies on an HTML parser such as lxml or html5lib. For complex
        tables with multiple header rows or merged cells (rowspan/colspan),
        the resulting DataFrame may be a flattened approximation rather than
        a perfect structural reconstruction. The original HTML is always
        preserved in TableUnit.content for full-fidelity use cases.

        Args:
            table_html: HTML table snippet as string

        Returns:
            pandas DataFrame, or None if parsing fails
        """

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "pandas is required for TableParser. "
                "Install it with: pip install pandas"
            ) from exc

        try:
            # read_html returns a list of DataFrames; take the first one
            dfs = pd.read_html(table_html)
            if not dfs:
                return None
            return dfs[0]
        except Exception:
            return None
