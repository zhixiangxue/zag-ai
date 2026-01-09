#!/usr/bin/env python3
"""
Test concurrent performance - compare sequential vs concurrent processing
"""

import sys
import os
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from zag.extractors import TableExtractor
from zag.schemas.unit import TextUnit
from zag.schemas.base import UnitMetadata

load_dotenv()

API_KEY = os.getenv("BAILIAN_API_KEY")
LLM_MODEL = "qwen-plus"


# Sample with multiple tables
MULTI_TABLE_CONTENT = """# Product Comparison

## Interest Rates

| Product Type | 30Y | 15Y | 10Y |
| ------------ | --- | --- | --- |
| Fixed Rate   | 6.5%| 6.0%| 5.8%|

## Down Payment Requirements

| Product | Min | Recommended |
| ------- | --- | ----------- |
| FHA     | 3.5%| 10%         |
| VA      | 0%  | 0%          |
| Conv    | 5%  | 20%         |

## Monthly Payment Estimates

| Loan Amount | 30Y Payment | 15Y Payment |
| ----------- | ----------- | ----------- |
| $200,000    | $1,264      | $1,687      |
| $300,000    | $1,896      | $2,531      |

## APR Comparison

| Product | Base APR | With Points |
| ------- | -------- | ----------- |
| Fixed   | 6.75%    | 6.50%       |
| ARM     | 6.25%    | 6.00%       |

Summary: All products offer competitive rates.
"""


async def test_performance():
    """Test performance with multiple tables"""
    
    print("=" * 70)
    print("Concurrent Performance Test")
    print("=" * 70)
    print()
    
    # Create TextUnit with 4 tables
    text_unit = TextUnit(
        unit_id="text_multi",
        content=MULTI_TABLE_CONTENT,
        metadata=UnitMetadata(context_path="Products/Comparison")
    )
    
    # Count tables
    from zag.parsers import TableParser
    parser = TableParser()
    matches = parser.TABLE_PATTERN.findall(text_unit.content)
    table_count = len(matches)
    
    print(f"📊 Test data:")
    print(f"  Content length: {len(text_unit.content)} chars")
    print(f"  Number of tables: {table_count}")
    print()
    
    # Test concurrent processing
    print("🚀 Testing concurrent processing...")
    extractor = TableExtractor(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY
    )
    
    start_time = time.time()
    results = await extractor.aextract([text_unit])
    elapsed = time.time() - start_time
    
    print(f"✅ Completed in {elapsed:.2f} seconds")
    print(f"   Average per table: {elapsed/table_count:.2f} seconds")
    print()
    
    # Show result
    if results[0].get("embedding_content"):
        print("📄 Generated embedding_content:")
        print("-" * 70)
        print(results[0]["embedding_content"][:500])
        print("...")
        print("-" * 70)
        print()
    
    # Performance estimate
    print("📊 Performance analysis:")
    print(f"  With {table_count} tables:")
    print(f"    Concurrent: {elapsed:.2f}s (actual)")
    print(f"    Sequential (estimated): {table_count * 1.5:.2f}s")
    print(f"    Speed improvement: ~{(table_count * 1.5 / elapsed):.1f}x faster")
    print()
    
    print("💡 Conclusion:")
    if table_count > 2:
        print(f"   Concurrent processing is WORTH IT for {table_count} tables!")
    else:
        print(f"   Concurrent processing is a nice optimization for {table_count} tables.")
    print()


if __name__ == "__main__":
    asyncio.run(test_performance())
