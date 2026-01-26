"""
Test Extractors with Bailian (DashScope)

This example demonstrates how to use the extractors with Bailian provider.

Before running:
1. Install dependencies: pip install python-dotenv
2. Set your API key in .env file: BAILIAN_API_KEY=your-key
"""

import os
import uuid
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from zag.schemas.unit import TextUnit, TableUnit
from zag.extractors import TableSummarizer, StructuredExtractor, KeywordExtractor

# Load environment variables
load_dotenv()

# ============ Configuration ============
API_KEY = os.getenv("BAILIAN_API_KEY")
LLM_MODEL = "qwen-plus"  # Bailian model


async def test_keyword_extractor():
    """Test KeywordExtractor"""
    print("\n" + "=" * 60)
    print("Test 1: Keyword Extractor")
    print("=" * 60)
    
    # Create extractor
    extractor = KeywordExtractor(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY,
        num_keywords=5
    )
    
    # Create test units
    units = [
        TextUnit(
            unit_id=str(uuid.uuid4()),
            content="This document describes a 30-year fixed-rate mortgage product with competitive APR rates and flexible down payment options."
        ),
        TextUnit(
            unit_id=str(uuid.uuid4()),
            content="Python is a high-level programming language known for its simplicity and readability, widely used in data science and machine learning."
        ),
    ]
    
    print(f"Processing {len(units)} units...")
    results = await extractor.aextract(units)
    
    # Update units with extracted metadata
    for unit, metadata in zip(units, results):
        unit.metadata.custom.update(metadata)
    
    print("\nResults:")
    for i, unit in enumerate(units, 1):
        print(f"\n  Unit {i}:")
        print(f"  Text: {unit.content[:80]}...")
        print(f"  Keywords: {unit.metadata.custom.get('excerpt_keywords', [])}")


async def test_structured_extractor():
    """Test StructuredExtractor"""
    print("\n" + "=" * 60)
    print("Test 2: Structured Extractor")
    print("=" * 60)
    
    # Define custom schema
    class LoanProduct(BaseModel):
        product_name: str = Field(description="贷款产品名称")
        loan_type: str = Field(description="贷款类型，如固定利率、浮动利率")
        term_years: int = Field(description="贷款期限（年）")
        min_down_payment: float = Field(description="最低首付比例（百分比）")
    
    # Create extractor
    extractor = StructuredExtractor(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY,
        schema=LoanProduct
    )
    
    # Create test unit
    units = [
        TextUnit(
            unit_id=str(uuid.uuid4()),
            content="""
            我们的30年期固定利率抵押贷款产品提供稳定的月供。
            贷款期限为30年，最低首付要求为房价的20%。
            这是一款传统的固定利率产品，利率在整个贷款期间保持不变。
            """
        ),
    ]
    
    print(f"Processing {len(units)} units...")
    results = await extractor.aextract(units)
    
    # Update units with extracted metadata
    for unit, metadata in zip(units, results):
        unit.metadata.custom.update(metadata)
    
    print("\nResults:")
    for i, unit in enumerate(units, 1):
        print(f"\n  Unit {i}:")
        print(f"  Text: {unit.content[:80]}...")
        print(f"  Extracted data:")
        print(f"    - Product Name: {unit.metadata.custom.get('product_name', 'N/A')}")
        print(f"    - Loan Type: {unit.metadata.custom.get('loan_type', 'N/A')}")
        print(f"    - Term: {unit.metadata.custom.get('term_years', 'N/A')} years")
        print(f"    - Min Down Payment: {unit.metadata.custom.get('min_down_payment', 'N/A')}%")


async def test_table_extractor():
    """Test TableSummarizer"""
    print("\n" + "=" * 60)
    print("Test 3: Table Summarizer")
    print("=" * 60)
    
    # Create extractor
    extractor = TableSummarizer(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY
    )
    
    # Create test table unit
    units = [
        TableUnit(
            unit_id=str(uuid.uuid4()),
            content="| Product | APR | Term | Min Amount |\n|---------|-----|------|------------|\n| Fixed-30 | 6.5% | 30Y | $100K |\n| Fixed-15 | 6.0% | 15Y | $150K |",
            json_data={
                "headers": ["Product", "APR", "Term", "Min Amount"],
                "rows": [
                    ["Fixed-30", "6.5%", "30Y", "$100K"],
                    ["Fixed-15", "6.0%", "15Y", "$150K"]
                ]
            }
        ),
    ]
    
    print(f"Processing {len(units)} units...")
    results = await extractor.aextract(units)
    
    # Update units with extracted metadata
    for unit, metadata in zip(units, results):
        unit.metadata.custom.update(metadata)
        if metadata.get("embedding_content"):
            unit.embedding_content = metadata["embedding_content"]
    
    print("\nResults:")
    for i, unit in enumerate(units, 1):
        print(f"\n  Unit {i}:")
        print(f"  Table Data: {unit.json_data}")
        print(f"  Generated Summary: {unit.metadata.custom.get('table_summary', 'N/A')}")


async def test_chaining_extractors():
    """Test chaining multiple extractors"""
    print("\n" + "=" * 60)
    print("Test 4: Chaining Extractors")
    print("=" * 60)
    
    # Define schema
    class DocumentInfo(BaseModel):
        topic: str = Field(description="文档主题")
        category: str = Field(description="文档类别")
    
    # Create extractors
    keyword_extractor = KeywordExtractor(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY,
        num_keywords=3
    )
    
    structured_extractor = StructuredExtractor(
        llm_uri=f"bailian/{LLM_MODEL}",
        api_key=API_KEY,
        schema=DocumentInfo
    )
    
    # Create test unit
    units = [
        TextUnit(
            unit_id=str(uuid.uuid4()),
            content="本文档介绍了人工智能在医疗诊断领域的应用，包括图像识别、疾病预测和个性化治疗方案推荐。"
        ),
    ]
    
    print(f"Processing with chained extractors...")
    
    # Chain extractors
    results1 = await keyword_extractor.aextract(units)
    for unit, metadata in zip(units, results1):
        unit.metadata.custom.update(metadata)
    
    results2 = await structured_extractor.aextract(units)
    for unit, metadata in zip(units, results2):
        unit.metadata.custom.update(metadata)
    
    print("\nResults:")
    for i, unit in enumerate(units, 1):
        print(f"\n  Unit {i}:")
        print(f"  Text: {unit.content}")
        print(f"  Keywords: {unit.metadata.custom.get('excerpt_keywords', [])}")
        print(f"  Topic: {unit.metadata.custom.get('topic', 'N/A')}")
        print(f"  Category: {unit.metadata.custom.get('category', 'N/A')}")


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Extractors Tests with Bailian")
    print("=" * 60)
    
    try:
        # Run all test functions
        await test_keyword_extractor()
        
        # StructuredExtractor requires 'instructor' library
        try:
            await test_structured_extractor()
        except ImportError as e:
            print(f"\n⚠️  Skipping StructuredExtractor test: {e}")
        
        await test_table_extractor()
        
        # Chaining test also uses StructuredExtractor
        try:
            await test_chaining_extractors()
        except ImportError as e:
            print(f"\n⚠️  Skipping chaining test: {e}")
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Error occurred: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
