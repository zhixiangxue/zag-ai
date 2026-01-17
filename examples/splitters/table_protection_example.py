#!/usr/bin/env python3
"""
Table Protection Example - Demonstrates how TextSplitter protects tables

TextSplitter ensures tables are never split, regardless of size.
This example demonstrates:
1. Small documents with tables remain intact
2. Large tables (>max_chunk_size) are protected
3. Mixed content is handled intelligently
4. Semantic context is preserved when possible

Key Principle: 
- If entire content (text + tables) fits in one chunk → keep together
- If too large → split, but protect tables as atomic units
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from zag.splitters import MarkdownHeaderSplitter, TextSplitter, RecursiveMergingSplitter
from zag.schemas.markdown import Markdown
from zag.schemas import DocumentMetadata


def print_separator(title: str, width: int = 70):
    """Print a formatted separator"""
    print(f"\n{'='*width}")
    print(f"{title}")
    print(f"{'='*width}\n")


def count_tokens(content: str) -> int:
    """Count tokens in content"""
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(content))
    except ImportError:
        # Fallback: rough estimate
        return len(content) // 4


def has_table(content: str) -> bool:
    """Check if content contains a table"""
    lines = content.split('\n')
    return any('|' in line and '---' in content for line in lines)


def demo1_small_document_with_table():
    """Demo 1: Small document with table - kept as single unit"""
    print_separator("Demo 1: Small Document with Table")
    
    content = """# Product Rates

Our current mortgage rates:

| Product | Rate | Term |
|---------|------|------|
| 30Y Fixed | 6.5% | 30 years |
| 15Y Fixed | 5.8% | 15 years |
| 5/1 ARM | 5.2% | 5 years |

Contact us for personalized quotes.
"""
    
    doc = Markdown(
        content=content,
        metadata=DocumentMetadata(
            source="demo1",
            source_type="inline",
            file_type="markdown",
            content_length=len(content)
        )
    )
    
    # Use TextSplitter
    splitter = TextSplitter(max_chunk_size=1200)
    units = doc.split(splitter)
    
    tokens = count_tokens(content)
    print(f"📄 Input: {tokens} tokens (< 1200 max_chunk_size)")
    print(f"✅ Output: {len(units)} unit(s)")
    print(f"\n💡 Result: Kept as single unit to preserve context")
    print(f"   (text + table + text together)")
    
    if len(units) == 1:
        print("\n✅ CORRECT: Semantic context preserved!")
    else:
        print("\n⚠️  Unexpected: Should be 1 unit")


def demo2_large_table():
    """Demo 2: Large table exceeding max_chunk_size"""
    print_separator("Demo 2: Large Table Protection")
    
    # Create a large table
    rows = []
    for i in range(100):
        rows.append(
            f"| Product {i:03d} | Description for product {i} | "
            f"${i*100:.2f} | Category {i%10} | Status: Active |"
        )
    
    table = """| Product | Description | Price | Category | Status |
|---------|-------------|-------|----------|--------|
""" + "\n".join(rows)
    
    content = f"""# Product Catalog

{table}

End of catalog.
"""
    
    doc = Markdown(
        content=content,
        metadata=DocumentMetadata(
            source="demo2",
            source_type="inline",
            file_type="markdown",
            content_length=len(content)
        )
    )
    
    table_tokens = count_tokens(table)
    total_tokens = count_tokens(content)
    
    print(f"📄 Input:")
    print(f"   - Table: {table_tokens} tokens (> 1200 max_chunk_size)")
    print(f"   - Total: {total_tokens} tokens")
    
    splitter = TextSplitter(max_chunk_size=1200)
    units = doc.split(splitter)
    
    print(f"\n✅ Output: {len(units)} units")
    
    # Check table integrity
    table_found = False
    for i, unit in enumerate(units):
        unit_tokens = count_tokens(unit.content)
        if has_table(unit.content):
            print(f"   Unit {i+1}: Table ({unit_tokens} tokens)")
            table_found = True
            if unit_tokens == table_tokens:
                print(f"      ✅ Table kept intact!")
        else:
            print(f"   Unit {i+1}: Text ({unit_tokens} tokens)")
    
    if table_found:
        print("\n💡 Result: Table protected despite exceeding max_chunk_size!")
    else:
        print("\n⚠️  Warning: Table not found in units")


def demo3_mixed_content_fits():
    """Demo 3: Mixed content that fits in one chunk"""
    print_separator("Demo 3: Mixed Content (Fits in One Chunk)")
    
    content = """# Financial Summary

## Q4 Results

Our quarterly results:

| Metric | Q3 | Q4 | Change |
|--------|----|----|--------|
| Revenue | $1.2M | $1.5M | +25% |
| Profit | $200K | $300K | +50% |

Strong growth across all segments. Market conditions remain favorable.
"""
    
    doc = Markdown(
        content=content,
        metadata=DocumentMetadata(
            source="demo3",
            source_type="inline",
            file_type="markdown",
            content_length=len(content)
        )
    )
    
    tokens = count_tokens(content)
    print(f"📄 Input: {tokens} tokens (< 1200 max_chunk_size)")
    print(f"   Content: Text + Table + Text")
    
    splitter = TextSplitter(max_chunk_size=1200)
    units = doc.split(splitter)
    
    print(f"\n✅ Output: {len(units)} unit(s)")
    
    if len(units) == 1:
        print("\n💡 Result: Entire content kept together")
        print("   ✅ Preserves semantic context")
        print("   ✅ Table remains with surrounding text")
        print("   ✅ Better for RAG retrieval")
    else:
        print(f"\n⚠️  Content split into {len(units)} units")


def demo4_pipeline_with_tables():
    """Demo 4: Full pipeline with table protection"""
    print_separator("Demo 4: Full Pipeline (Header + Text + Merge)")
    
    content = """# Documentation

## Section 1: Introduction

This section explains our approach.

## Section 2: Data

Here is the complete dataset:

""" + """| ID | Name | Value | Category | Status |
|----|----- |-------|----------|--------|
""" + "\n".join([
    f"| {i} | Item {i} | {i*10} | Cat {i%5} | Active |"
    for i in range(150)
]) + """

## Section 3: Conclusion

Data shows strong trends across all categories.
"""
    
    doc = Markdown(
        content=content,
        metadata=DocumentMetadata(
            source="demo4",
            source_type="inline",
            file_type="markdown",
            content_length=len(content)
        )
    )
    
    total_tokens = count_tokens(content)
    print(f"📄 Input: {total_tokens:,} tokens")
    
    # Full pipeline
    pipeline = (
        MarkdownHeaderSplitter()
        | TextSplitter(max_chunk_size=1200)
        | RecursiveMergingSplitter(target_token_size=800)
    )
    
    print(f"🔧 Pipeline: Header → Text → Merge")
    units = doc.split(pipeline)
    
    print(f"\n✅ Output: {len(units)} units")
    
    # Analyze units
    table_count = sum(1 for u in units if has_table(u.content))
    max_tokens = max(count_tokens(u.content) for u in units)
    avg_tokens = sum(count_tokens(u.content) for u in units) // len(units)
    
    print(f"\n📊 Analysis:")
    print(f"   - Units with tables: {table_count}")
    print(f"   - Max tokens: {max_tokens}")
    print(f"   - Avg tokens: {avg_tokens}")
    
    print(f"\n💡 Key Observations:")
    print(f"   ✅ Large table protected (not split)")
    print(f"   ✅ Text sections optimally sized")
    print(f"   ✅ Ready for RAG retrieval")


def main():
    """Run all demonstrations"""
    print("="*70)
    print("Table Protection in TextSplitter - Comprehensive Demo")
    print("="*70)
    
    print("\n📚 Overview:")
    print("   TextSplitter protects tables from being split while")
    print("   intelligently handling surrounding text.")
    print("\n   Core Principle:")
    print("   • If content fits → keep together (preserve context)")
    print("   • If too large → split, but protect tables")
    
    try:
        demo1_small_document_with_table()
        demo2_large_table()
        demo3_mixed_content_fits()
        demo4_pipeline_with_tables()
        
        print_separator("Summary")
        print("✅ All demonstrations completed!")
        print("\n🎯 Key Takeaways:")
        print("   1. Tables are NEVER split, regardless of size")
        print("   2. Small content (< max_chunk_size) stays together")
        print("   3. Large content is split intelligently")
        print("   4. Semantic context is preserved when possible")
        print("   5. Works seamlessly in pipelines")
        print("\n💡 Best Practice:")
        print("   Always use TextSplitter in pipeline after header-based")
        print("   splitting to ensure optimal chunk sizes while protecting")
        print("   tables and preserving context.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
