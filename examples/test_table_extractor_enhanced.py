#!/usr/bin/env python3
"""
Test Enhanced TableExtractor - Generate embedding_content for both TableUnit and TextUnit

Demonstrates:
- TableUnit: Generate embedding_content from json_data
- TextUnit: Replace tables with summaries in embedding_content
- Source language detection and preservation
- Extractor does not modify units (returns Dict)
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zag.extractors import TableExtractor
from zag.parsers import TableParser
from zag.schemas.unit import TextUnit, TableUnit
from zag.schemas.base import UnitMetadata


# Sample TextUnit with mixed language tables
SAMPLE_CONTENT_EN = """# Mortgage Rates

We offer competitive rates for different products.

| Product Type        | Term    | Interest Rate | APR    |
| ------------------- | ------- | ------------- | ------ |
| Fixed-Rate Mortgage | 30-Year | 6.125%        | 6.275% |
| Fixed-Rate Mortgage | 15-Year | 5.750%        | 5.920% |

These rates are for borrowers with excellent credit.
"""

SAMPLE_CONTENT_CN = """# 贷款利率

我们为不同产品提供有竞争力的利率。

| 产品类型 | 期限   | 利率   | APR    |
| -------- | ------ | ------ | ------ |
| 固定利率 | 30年   | 6.125% | 6.275% |
| 固定利率 | 15年   | 5.750% | 5.920% |

这些利率适用于信用优秀的借款人。
"""


async def test_tableunit_extraction():
    """Test 1: Extract from TableUnit"""
    print("=" * 70)
    print("Test 1: TableUnit - Generate embedding_content")
    print("=" * 70)
    print()
    
    # Create TableUnit with English table
    table_unit = TableUnit(
        unit_id="table_001",
        content="| Product | Rate |\n|---------|------|\n| 30Y | 6.5% |",
        json_data={
            'headers': ['Product', 'Rate'],
            'rows': [['30Y', '6.5%'], ['15Y', '5.8%']]
        },
        metadata=UnitMetadata(context_path="Rates/Fixed")
    )
    
    print(f"📊 Original TableUnit:")
    print(f"  Unit ID: {table_unit.unit_id}")
    print(f"  Headers: {table_unit.json_data['headers']}")
    print(f"  Rows: {table_unit.json_data['rows']}")
    print(f"  embedding_content: {table_unit.embedding_content}")
    print()
    
    # Extract
    extractor = TableExtractor(
        llm_uri="openai/gpt-4o-mini",
        api_key="YOUR_API_KEY"  # Replace with actual key
    )
    
    results = await extractor.aextract([table_unit])
    
    print("✅ Extraction result:")
    print(f"  Returned metadata: {results[0]}")
    print()
    
    # Update unit (caller's responsibility)
    table_unit.embedding_content = results[0].get("embedding_content")
    
    print("✅ After update:")
    print(f"  embedding_content: {table_unit.embedding_content}")
    print(f"  content: {table_unit.content[:50]}...")  # Original unchanged
    print()


async def test_textunit_extraction_en():
    """Test 2: Extract from TextUnit (English)"""
    print("=" * 70)
    print("Test 2: TextUnit with English tables")
    print("=" * 70)
    print()
    
    text_unit = TextUnit(
        unit_id="text_001",
        content=SAMPLE_CONTENT_EN,
        metadata=UnitMetadata(context_path="Mortgage/Rates")
    )
    
    print(f"📄 Original TextUnit:")
    print(f"  Unit ID: {text_unit.unit_id}")
    print(f"  content length: {len(text_unit.content)} chars")
    print(f"  embedding_content: {text_unit.embedding_content}")
    print()
    print("Content preview:")
    print(text_unit.content[:200])
    print("...")
    print()
    
    # Extract
    extractor = TableExtractor(
        llm_uri="openai/gpt-4o-mini",
        api_key="YOUR_API_KEY"
    )
    
    results = await extractor.aextract([text_unit])
    
    print("✅ Extraction result:")
    if results[0].get("embedding_content"):
        print(f"  embedding_content generated (length: {len(results[0]['embedding_content'])} chars)")
        print()
        print("embedding_content preview:")
        print(results[0]['embedding_content'][:300])
        print("...")
    else:
        print("  No embedding_content generated (no tables found)")
    print()
    
    # Update unit
    if results[0].get("embedding_content"):
        text_unit.embedding_content = results[0]["embedding_content"]
    
    print("✅ After update:")
    print(f"  content: UNCHANGED (still {len(text_unit.content)} chars)")
    print(f"  embedding_content: SET ({len(text_unit.embedding_content or '')} chars)")
    print()


async def test_textunit_extraction_cn():
    """Test 3: Extract from TextUnit (Chinese)"""
    print("=" * 70)
    print("Test 3: TextUnit with Chinese tables")
    print("=" * 70)
    print()
    
    text_unit = TextUnit(
        unit_id="text_002",
        content=SAMPLE_CONTENT_CN,
        metadata=UnitMetadata(context_path="贷款/利率")
    )
    
    print(f"📄 Original TextUnit:")
    print(f"  Unit ID: {text_unit.unit_id}")
    print(f"  content length: {len(text_unit.content)} chars")
    print()
    
    # Extract
    extractor = TableExtractor(
        llm_uri="openai/gpt-4o-mini",
        api_key="YOUR_API_KEY"
    )
    
    results = await extractor.aextract([text_unit])
    
    print("✅ Extraction result:")
    if results[0].get("embedding_content"):
        print(f"  embedding_content (should be in Chinese):")
        print(f"  {results[0]['embedding_content'][:200]}...")
    print()


async def test_mixed_units():
    """Test 4: Process mixed units (TextUnit + TableUnit)"""
    print("=" * 70)
    print("Test 4: Mixed units (TextUnit + TableUnit)")
    print("=" * 70)
    print()
    
    # Create TextUnit
    text_unit = TextUnit(
        unit_id="text_003",
        content=SAMPLE_CONTENT_EN,
        metadata=UnitMetadata(context_path="Rates")
    )
    
    # Parse tables from TextUnit
    parser = TableParser()
    table_units = parser.parse_from_unit(text_unit)
    
    print(f"📦 Prepared units:")
    print(f"  TextUnit: 1")
    print(f"  TableUnit: {len(table_units)}")
    print()
    
    # Extract from all units
    extractor = TableExtractor(
        llm_uri="openai/gpt-4o-mini",
        api_key="YOUR_API_KEY"
    )
    
    all_units = [text_unit] + table_units
    results = await extractor.aextract(all_units)
    
    print("✅ Extraction results:")
    for i, (unit, metadata) in enumerate(zip(all_units, results)):
        unit_type = type(unit).__name__
        has_embedding = "embedding_content" in metadata
        print(f"  {i+1}. {unit_type}: {'✅ generated' if has_embedding else '❌ skipped'}")
    print()
    
    # Update all units
    for unit, metadata in zip(all_units, results):
        if metadata.get("embedding_content"):
            unit.embedding_content = metadata["embedding_content"]
    
    print("✅ All units updated successfully!")
    print()


async def main():
    print("=" * 70)
    print("Enhanced TableExtractor Test")
    print("=" * 70)
    print()
    print("⚠️  Note: This test requires a valid API key")
    print("    Replace 'YOUR_API_KEY' in the code with your actual key")
    print()
    
    try:
        # Run all tests
        await test_tableunit_extraction()
        await test_textunit_extraction_en()
        await test_textunit_extraction_cn()
        await test_mixed_units()
        
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print()
        print("✅ TableExtractor now supports:")
        print("   1. TableUnit → embedding_content (from json_data)")
        print("   2. TextUnit → embedding_content (replace tables with summaries)")
        print("   3. Source language detection (Chinese, English, etc.)")
        print("   4. Does not modify units (returns Dict)")
        print()
        print("💡 Key features:")
        print("   - Caller decides whether to update units")
        print("   - Original content preserved in unit.content")
        print("   - Processed content in unit.embedding_content")
        print()
        print("=" * 70)
        print("✅ All tests complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nMake sure to:")
        print("  1. Replace 'YOUR_API_KEY' with actual API key")
        print("  2. Install required dependencies: pip install chak")


if __name__ == "__main__":
    asyncio.run(main())
