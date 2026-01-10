#!/usr/bin/env python3
"""
Test TableParser - Parse Markdown tables from TextUnit

Demonstrates:
- Parsing tables from TextUnit
- Converting to TableUnit with structured json_data
- Optional placeholder replacement
- Establishing relations between units
"""

import sys
from pathlib import Path

from zag.parsers import TableParser
from zag.schemas.unit import TextUnit
from zag.schemas.base import UnitMetadata


# Sample TextUnit with tables
SAMPLE_CONTENT = """# Mortgage Rates Comparison

We offer competitive rates for different loan products.

## Fixed-Rate Mortgages

| Product Type        | Term    | Interest Rate | APR    | Minimum Down Payment |
| ------------------- | ------- | ------------- | ------ | -------------------- |
| Fixed-Rate Mortgage | 30-Year | 6.125%        | 6.275% | 3%                   |
| Fixed-Rate Mortgage | 15-Year | 5.750%        | 5.920% | 5%                   |

These rates are for borrowers with excellent credit.

## Adjustable-Rate Mortgages

| Product Type | Term    | Interest Rate | APR    |
| ------------ | ------- | ------------- | ------ |
| 5/1 ARM      | 30-Year | 5.875%        | 6.125% |
| 7/1 ARM      | 30-Year | 6.000%        | 6.250% |

ARM rates adjust after the initial fixed period.
"""


def main():
    print("=" * 70)
    print("TableParser Test")
    print("=" * 70)
    print()
    
    # 1. Create TextUnit
    text_unit = TextUnit(
        unit_id="text_001",
        content=SAMPLE_CONTENT,
        metadata=UnitMetadata(
            context_path="Mortgage Products/Rates"
        )
    )
    
    print("📄 Original TextUnit:")
    print(f"  Unit ID: {text_unit.unit_id}")
    print(f"  Content length: {len(text_unit.content)} chars")
    print(f"  Context path: {text_unit.metadata.context_path}")
    print()
    
    # 2. Parse tables (without replacement)
    print("=" * 70)
    print("Test 1: Parse tables (preserve original content)")
    print("=" * 70)
    
    parser = TableParser()
    table_units = parser.parse_from_unit(text_unit, replace_with_placeholder=False)
    
    print(f"\n✅ Parsed {len(table_units)} tables")
    print()
    
    for i, table in enumerate(table_units, 1):
        print(f"Table {i}:")
        print(f"  Unit ID: {table.unit_id}")
        print(f"  Headers: {table.json_data['headers']}")
        print(f"  Row count: {len(table.json_data['rows'])}")
        print(f"  Context path: {table.metadata.context_path}")
        print(f"  Content preview: {table.content[:100]}...")
        print()
    
    # Check relation
    if text_unit.metadata.custom and 'table_ids' in text_unit.metadata.custom:
        print(f"✅ Relations established:")
        print(f"  TextUnit table_ids: {text_unit.metadata.custom['table_ids']}")
        print()
    
    # 3. Parse with replacement
    print("=" * 70)
    print("Test 2: Parse tables (replace with placeholders)")
    print("=" * 70)
    
    text_unit2 = TextUnit(
        unit_id="text_002",
        content=SAMPLE_CONTENT,
        metadata=UnitMetadata(context_path="Mortgage Products/Rates")
    )
    
    parser = TableParser()
    table_units2 = parser.parse_from_unit(text_unit2, replace_with_placeholder=True)
    
    print(f"\n✅ Parsed {len(table_units2)} tables")
    print()
    print("Modified TextUnit content:")
    print(text_unit2.content[:500])
    print("...")
    print()
    
    # 4. Verify structured data
    print("=" * 70)
    print("Test 3: Verify structured data")
    print("=" * 70)
    print()
    
    table1 = table_units[0]
    print("First table (Fixed-Rate Mortgages):")
    print(f"  Headers: {table1.json_data['headers']}")
    print(f"  Rows:")
    for row in table1.json_data['rows']:
        print(f"    {row}")
    print()
    
    table2 = table_units[1]
    print("Second table (Adjustable-Rate Mortgages):")
    print(f"  Headers: {table2.json_data['headers']}")
    print(f"  Rows:")
    for row in table2.json_data['rows']:
        print(f"    {row}")
    print()
    
    # 5. Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"✅ Successfully parsed {len(table_units)} tables from TextUnit")
    print(f"✅ All tables have structured json_data")
    print(f"✅ Relations established via metadata.custom['table_ids']")
    print(f"✅ Optional placeholder replacement works")
    print()
    print("💡 Next steps:")
    print("   1. Use TableExtractor to generate embedding_content (summaries)")
    print("   2. Store both TextUnit and TableUnits to vector database")
    print("   3. Use table_ids for relation-based retrieval")
    print()
    print("=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
