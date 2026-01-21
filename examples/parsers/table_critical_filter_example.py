#!/usr/bin/env python3
"""
TableParser with Data-Critical Filtering Example

This example demonstrates how to use TableParser with LLM-based filtering
to extract only data-critical tables from documents.

Before running:
1. Install dependencies: pandas, chak
2. Set environment variable: BAILIAN_API_KEY
3. Prepare test document with mixed tables (critical + non-critical)
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from zag.parsers import TableParser
from zag.schemas.unit import TextUnit

# Load environment
load_dotenv()

# Configuration
API_KEY = os.getenv("BAILIAN_API_KEY")
LLM_URI = "bailian/qwen-plus"

# Test data: TextUnit with multiple tables
# Table 1: Critical (mortgage rates - important numerical data)
# Table 2: Non-critical (formatting/layout table)
# Table 3: Critical (product comparison)
test_content = """
# Mortgage Products

## Interest Rates

| Product Type | Interest Rate | APR | Term |
|-------------|--------------|-----|------|
| Fixed-Rate  | 6.125%       | 6.275% | 30-Year |
| ARM         | 5.875%       | 6.125% | 5/1 |
| FHA Loan    | 6.250%       | 6.450% | 30-Year |

## Document Navigation

| Chapter | Title |
|---------|-------|
| Chapter 1 | Introduction |
| Chapter 2 | Overview |
| Chapter 3 | Main Content |
| Chapter 4 | Summary |
| Chapter 5 | References |

## Product Features Comparison

| Feature | Basic | Premium | Enterprise |
|---------|-------|---------|------------|
| Price | $99/mo | $299/mo | Custom |
| Support | Email | Phone | 24/7 Dedicated |
| Users | 10 | 50 | Unlimited |
| Storage | 100GB | 1TB | Custom |
"""


async def example_without_filtering():
    """Example 1: Parse all tables (no filtering)"""
    print("\n" + "=" * 70)
    print("  Example 1: Parse ALL Tables (No Filtering)")
    print("=" * 70 + "\n")
    
    # Create parser
    parser = TableParser()
    
    # Create TextUnit
    text_unit = TextUnit(
        content=test_content,
        unit_id="test_unit_1"
    )
    
    # Parse tables (synchronous)
    table_units = parser.parse([text_unit])
    
    print(f"✅ Parsed {len(table_units)} tables:\n")
    for i, table in enumerate(table_units, 1):
        print(f"Table {i}:")
        print(f"  Shape: {table.df.shape}")
        print(f"  Columns: {list(table.df.columns)}")
        print(f"  Content preview: {table.content[:60]}...")
        print()


async def example_with_filtering():
    """Example 2: Parse only data-critical tables (with LLM filtering)"""
    print("\n" + "=" * 70)
    print("  Example 2: Parse ONLY Data-Critical Tables (LLM Filtering)")
    print("=" * 70 + "\n")
    
    if not API_KEY:
        print("❌ BAILIAN_API_KEY not found. Skipping LLM filtering example.")
        return
    
    # Create parser with LLM configured
    parser = TableParser(
        llm_uri=LLM_URI,
        api_key=API_KEY
    )
    
    # Create TextUnit
    text_unit = TextUnit(
        content=test_content,
        unit_id="test_unit_2"
    )
    
    print("🤖 Using LLM to filter data-critical tables...")
    print(f"   LLM: {LLM_URI}\n")
    
    # Parse and filter tables (async with filter_critical=True)
    critical_tables = await parser.aparse([text_unit], filter_critical=True)
    
    print(f"✅ Found {len(critical_tables)} data-critical tables:\n")
    for i, table in enumerate(critical_tables, 1):
        print(f"Critical Table {i}:")
        print(f"  Shape: {table.df.shape}")
        print(f"  Columns: {list(table.df.columns)}")
        print(f"  First row data: {table.df.iloc[0].to_dict()}")
        print()


async def example_comparison():
    """Example 3: Compare results with/without filtering"""
    print("\n" + "=" * 70)
    print("  Example 3: Comparison (All vs Critical)")
    print("=" * 70 + "\n")
    
    if not API_KEY:
        print("❌ BAILIAN_API_KEY not found. Skipping comparison.")
        return
    
    text_unit = TextUnit(content=test_content, unit_id="test_unit_3")
    
    # Parse all
    parser_all = TableParser()
    all_tables = parser_all.parse([text_unit])
    
    # Parse critical only
    parser_critical = TableParser(
        llm_uri=LLM_URI,
        api_key=API_KEY
    )
    critical_tables = await parser_critical.aparse([text_unit], filter_critical=True)
    
    print(f"📊 Results:")
    print(f"   All tables: {len(all_tables)}")
    print(f"   Critical tables: {len(critical_tables)}")
    print(f"   Filtered out: {len(all_tables) - len(critical_tables)}")
    print()
    
    if len(critical_tables) < len(all_tables):
        print("✅ Filtering worked! Non-critical tables were removed.")
        print(f"   Efficiency gain: {(1 - len(critical_tables)/len(all_tables)) * 100:.1f}%")
    else:
        print("⚠️  All tables were marked as critical (or filtering failed).")


async def example_sync_error():
    """Example 4: Show error when using sync with filter_critical=True"""
    print("\n" + "=" * 70)
    print("  Example 4: Sync Filtering Error Demo")
    print("=" * 70 + "\n")
    
    parser = TableParser(llm_uri=LLM_URI, api_key=API_KEY)
    text_unit = TextUnit(content=test_content, unit_id="test_unit_4")
    
    print("Attempting to use filter_critical=True with sync parse()...\n")
    
    try:
        # This should raise ValueError
        tables = parser.parse([text_unit], filter_critical=True)
        print("❌ ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"✅ Caught expected error:\n   {e}")
        print("\n💡 Solution: Use aparse() instead of parse() for filtering.")


async def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("  🎯 TableParser Data-Critical Filtering Examples")
    print("=" * 70)
    
    # Example 1: No filtering
    await example_without_filtering()
    
    # Example 2: With filtering
    await example_with_filtering()
    
    # Example 3: Comparison
    await example_comparison()
    
    # Example 4: Error handling
    await example_sync_error()
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
