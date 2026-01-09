#!/usr/bin/env python3
"""
Test table boundary issue - verify LLM doesn't include surrounding text
"""

import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from zag.extractors import TableExtractor
from zag.parsers import TableParser
from zag.schemas.unit import TextUnit
from zag.schemas.base import UnitMetadata

load_dotenv()

API_KEY = os.getenv("BAILIAN_API_KEY")
LLM_MODEL = "qwen-plus"


async def test_boundary():
    """Test if LLM output includes surrounding text"""
    
    content = """# 贷款利率

我们为不同产品提供有竞争力的利率。

| 产品类型 | 期限   | 利率   | APR    |
| -------- | ------ | ------ | ------ |
| 固定利率 | 30年   | 6.125% | 6.275% |
| 固定利率 | 15年   | 5.750% | 5.920% |

这些利率适用于信用优秀的借款人。
"""
    
    print("=" * 70)
    print("Testing Table Boundary Issue")
    print("=" * 70)
    print()
    
    # 1. Parse table to see what regex captures
    parser = TableParser()
    matches = parser.TABLE_PATTERN.findall(content)
    
    print("1️⃣ Regex captured table:")
    print("-" * 70)
    print(matches[0])
    print("-" * 70)
    print()
    
    # 2. Parse to json_data
    json_data = parser._parse_table_text(matches[0])
    print("2️⃣ Parsed json_data:")
    print(json_data)
    print()
    
    # 3. Generate summary
    extractor = TableExtractor(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY
    )
    
    summary = await extractor._generate_table_summary(json_data)
    
    print("3️⃣ LLM generated summary:")
    print("-" * 70)
    print(summary)
    print("-" * 70)
    print()
    
    # 4. Check if summary contains surrounding text
    before_text = "我们为不同产品提供有竞争力的利率"
    after_text = "这些利率适用于信用优秀的借款人"
    
    print("4️⃣ Boundary check:")
    print(f"  Contains before text ('{before_text[:20]}...'): {before_text in summary}")
    print(f"  Contains after text ('{after_text}'): {after_text in summary}")
    print()
    
    # 5. Show replacement result
    result = content.replace(matches[0], summary)
    print("5️⃣ Final replacement result:")
    print("-" * 70)
    print(result)
    print("-" * 70)
    
    # Verify
    if after_text in summary:
        print("\n❌ PROBLEM FOUND: LLM output includes text AFTER the table!")
        print("   This causes semantic confusion.")
    else:
        print("\n✅ OK: LLM output does NOT include surrounding text")


if __name__ == "__main__":
    asyncio.run(test_boundary())
