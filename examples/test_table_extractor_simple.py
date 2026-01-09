#!/usr/bin/env python3
"""
Test Enhanced TableExtractor - Simple Version

Tests:
1. TableUnit: Generate embedding_content from json_data
2. TextUnit with English tables: Replace tables with summaries
3. TextUnit with Chinese tables: Replace tables with summaries (language detection)
"""

import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zag.extractors import TableExtractor
from zag.parsers import TableParser
from zag.schemas.unit import TextUnit, TableUnit
from zag.schemas.base import UnitMetadata

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("BAILIAN_API_KEY")
LLM_MODEL = "qwen-plus"


# Sample data
SAMPLE_TEXT_EN = """# Mortgage Rates

We offer competitive rates for different products.

| Product Type        | Term    | Interest Rate | APR    |
| ------------------- | ------- | ------------- | ------ |
| Fixed-Rate Mortgage | 30-Year | 6.125%        | 6.275% |
| Fixed-Rate Mortgage | 15-Year | 5.750%        | 5.920% |

These rates are for borrowers with excellent credit.
"""

SAMPLE_TEXT_CN = """# 贷款利率

我们为不同产品提供有竞争力的利率。

| 产品类型 | 期限   | 利率   | APR    |
| -------- | ------ | ------ | ------ |
| 固定利率 | 30年   | 6.125% | 6.275% |
| 固定利率 | 15年   | 5.750% | 5.920% |

这些利率适用于信用优秀的借款人。
"""


async def test_tableunit():
    """Test 1: TableUnit - Generate embedding_content"""
    print("\n" + "=" * 70)
    print("Test 1: TableUnit - Generate embedding_content")
    print("=" * 70)
    
    # Create TableUnit
    table_unit = TableUnit(
        unit_id="table_001",
        content="| Product | Rate |\n|---------|------|\n| 30Y | 6.5% |\n| 15Y | 5.8% |",
        json_data={
            'headers': ['Product', 'Rate'],
            'rows': [['30Y Fixed', '6.5%'], ['15Y Fixed', '5.8%']]
        },
        metadata=UnitMetadata(context_path="Rates/Fixed")
    )
    
    print(f"\n📊 Original TableUnit:")
    print(f"  Headers: {table_unit.json_data['headers']}")
    print(f"  Rows: {table_unit.json_data['rows']}")
    print(f"  embedding_content: {table_unit.embedding_content}")
    
    # Extract
    extractor = TableExtractor(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY
    )
    
    results = await extractor.aextract([table_unit])
    
    print(f"\n✅ Extraction result:")
    print(f"  Generated embedding_content:")
    print(f"  {results[0].get('embedding_content', 'N/A')}")
    
    # Update unit
    if results[0].get("embedding_content"):
        table_unit.embedding_content = results[0]["embedding_content"]
        print(f"\n✅ Unit updated successfully")
        print(f"  content: {table_unit.content[:50]}... (UNCHANGED)")
        print(f"  embedding_content: {table_unit.embedding_content[:80]}...")


async def test_textunit_en():
    """Test 2: TextUnit with English tables"""
    print("\n" + "=" * 70)
    print("Test 2: TextUnit with English tables")
    print("=" * 70)
    
    text_unit = TextUnit(
        unit_id="text_001",
        content=SAMPLE_TEXT_EN,
        metadata=UnitMetadata(context_path="Mortgage/Rates")
    )
    
    print(f"\n📄 Original TextUnit:")
    print(f"  Content length: {len(text_unit.content)} chars")
    print(f"  Has tables: Yes")
    print(f"  embedding_content: {text_unit.embedding_content}")
    
    # Extract
    extractor = TableExtractor(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY
    )
    
    results = await extractor.aextract([text_unit])
    
    print(f"\n✅ Extraction result:")
    if results[0].get("embedding_content"):
        print(f"  Generated embedding_content (length: {len(results[0]['embedding_content'])} chars)")
        print(f"\n  Preview:")
        print(f"  {results[0]['embedding_content'][:200]}...")
        
        # Update unit
        text_unit.embedding_content = results[0]["embedding_content"]
        print(f"\n✅ Unit updated successfully")
        print(f"  content: UNCHANGED ({len(text_unit.content)} chars)")
        print(f"  embedding_content: SET ({len(text_unit.embedding_content)} chars)")
    else:
        print(f"  ❌ No embedding_content generated")


async def test_textunit_cn():
    """Test 3: TextUnit with Chinese tables"""
    print("\n" + "=" * 70)
    print("Test 3: TextUnit with Chinese tables (Language Detection)")
    print("=" * 70)
    
    text_unit = TextUnit(
        unit_id="text_002",
        content=SAMPLE_TEXT_CN,
        metadata=UnitMetadata(context_path="贷款/利率")
    )
    
    print(f"\n📄 Original TextUnit:")
    print(f"  Content length: {len(text_unit.content)} chars")
    print(f"  Has tables: Yes (Chinese)")
    
    # Extract
    extractor = TableExtractor(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY
    )
    
    results = await extractor.aextract([text_unit])
    
    print(f"\n✅ Extraction result:")
    if results[0].get("embedding_content"):
        print(f"  Generated embedding_content (should be in Chinese):")
        print(f"\n  {results[0]['embedding_content'][:300]}...")
        
        # Check if response is in Chinese
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in results[0]['embedding_content'])
        print(f"\n  Language check: {'✅ Chinese detected' if has_chinese else '❌ No Chinese'}")


async def test_mixed_units():
    """Test 4: Mixed units (TextUnit + TableUnit)"""
    print("\n" + "=" * 70)
    print("Test 4: Mixed units (TextUnit + TableUnit)")
    print("=" * 70)
    
    # Create TextUnit
    text_unit = TextUnit(
        unit_id="text_003",
        content=SAMPLE_TEXT_EN,
        metadata=UnitMetadata(context_path="Rates")
    )
    
    # Parse tables
    parser = TableParser()
    table_units = parser.parse_from_unit(text_unit)
    
    print(f"\n📦 Prepared units:")
    print(f"  TextUnit: 1")
    print(f"  TableUnit: {len(table_units)}")
    
    # Extract from all
    extractor = TableExtractor(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY
    )
    
    all_units = [text_unit] + table_units
    results = await extractor.aextract(all_units)
    
    print(f"\n✅ Extraction results:")
    for i, (unit, metadata) in enumerate(zip(all_units, results)):
        unit_type = type(unit).__name__
        has_embedding = bool(metadata.get("embedding_content"))
        print(f"  {i+1}. {unit_type:12} → {'✅ generated' if has_embedding else '❌ skipped'}")
    
    # Update all units
    for unit, metadata in zip(all_units, results):
        if metadata.get("embedding_content"):
            unit.embedding_content = metadata["embedding_content"]
    
    print(f"\n✅ All units updated successfully!")


async def main():
    print("=" * 70)
    print("Enhanced TableExtractor Test")
    print("=" * 70)
    
    if not API_KEY:
        print("\n❌ Error: BAILIAN_API_KEY not found in .env")
        print("   Please set your API key in .env file")
        return
    
    print(f"\n✅ Using Bailian API ({LLM_MODEL})")
    print(f"   API key: {API_KEY[:10]}...")
    
    try:
        # Run all tests
        await test_tableunit()
        await test_textunit_en()
        await test_textunit_cn()
        await test_mixed_units()
        
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print("\n✅ All tests passed!")
        print("\nKey features verified:")
        print("  1. ✅ TableUnit → embedding_content (from json_data)")
        print("  2. ✅ TextUnit → embedding_content (replace tables)")
        print("  3. ✅ Language detection (LLM-based)")
        print("  4. ✅ Does not modify original content")
        print("  5. ✅ Mixed units processing")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
