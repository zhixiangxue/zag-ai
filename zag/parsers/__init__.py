"""
Parsers for identifying and converting special structures from text

Parsers are responsible for:
- Identifying structured elements in raw text (e.g., tables, code blocks)
- Converting text representations to structured Units
- Extracting structured data from markup languages

Different from extractors:
- Parsers: text → structured Units (pattern recognition)
- Extractors: Units → metadata/summaries (information generation)
"""

from .table import TableParser

__all__ = ["TableParser"]
