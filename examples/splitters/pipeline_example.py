#!/usr/bin/env python3
"""
Splitter Pipeline Example - Pipeline Composition

Demonstrates the powerful pipeline syntax for composing multiple splitters:
    pipeline = Splitter1() | Splitter2() | Splitter3()

Features:
- Elegant pipeline composition using | operator
- Automatic chain relationship management
- Semantic boundary-aware splitting
- Smart token-based merging

Pipeline workflow:
1. MarkdownHeaderSplitter: Split by headers (H1-H6)
2. TextSplitter: Break down oversized sections by paragraphs/sentences
3. RecursiveMergingSplitter: Merge small chunks to target size
"""

import sys
from pathlib import Path

from zag.splitters import MarkdownHeaderSplitter, TextSplitter, RecursiveMergingSplitter
from zag.schemas.markdown import Markdown
from zag.schemas import DocumentMetadata


def print_separator(title: str):
    """Print a formatted separator"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")


def analyze_units(units, title: str):
    """Analyze and print unit statistics"""
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    token_counts = [len(tokenizer.encode(u.content)) for u in units]
    
    print(f"{title}:")
    print(f"  - Total units: {len(units)}")
    print(f"  - Token range: {min(token_counts)}-{max(token_counts)}")
    print(f"  - Average tokens: {sum(token_counts) // len(token_counts)}")
    print(f"  - Total tokens: {sum(token_counts)}")
    
    # Show distribution
    small = sum(1 for t in token_counts if t < 300)
    medium = sum(1 for t in token_counts if 300 <= t <= 1000)
    large = sum(1 for t in token_counts if t > 1000)
    print(f"  - Distribution: {small} small (<300) | {medium} medium (300-1000) | {large} large (>1000)")


def main():
    print_separator("Splitter Pipeline Example")
    
    # Load sample document
    sample_file = Path(__file__).parent.parent / "files" / "mortgage_products.md"
    
    if not sample_file.exists():
        print(f"❌ File not found: {sample_file}")
        sys.exit(1)
    
    content = sample_file.read_text(encoding="utf-8")
    
    # Create document
    metadata = DocumentMetadata(
        source=str(sample_file),
        source_type="local",
        file_type="markdown",
        file_name=sample_file.name,
        file_size=len(content),
        content_length=len(content),
    )
    
    doc = Markdown(content=content, metadata=metadata)
    print(f"📄 Document: {sample_file.name}")
    print(f"📄 Length: {len(content)} characters\n")
    
    # Example 1: Single splitter (baseline)
    print_separator("Example 1: Single Splitter (Baseline)")
    
    splitter1 = MarkdownHeaderSplitter()
    units1 = doc.split(splitter1)
    analyze_units(units1, "MarkdownHeaderSplitter only")
    
    # Example 2: Pipeline with merging
    print_separator("Example 2: Pipeline - Header Split + Merge")
    
    pipeline2 = (
        MarkdownHeaderSplitter()
        | RecursiveMergingSplitter(target_token_size=800)
    )
    
    print(f"Pipeline: {pipeline2}\n")
    units2 = doc.split(pipeline2)
    analyze_units(units2, "With merging")
    
    # Example 3: Full pipeline (split large + merge small)
    print_separator("Example 3: Full Pipeline - Split + Merge")
    
    pipeline3 = (
        MarkdownHeaderSplitter()
        | TextSplitter(max_chunk_size=1200)
        | RecursiveMergingSplitter(target_token_size=800)
    )
    
    print(f"Pipeline: {pipeline3}\n")
    units3 = doc.split(pipeline3)
    analyze_units(units3, "Full pipeline")
    
    # Comparison
    print_separator("Comparison Summary")
    
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    results = [
        ("Baseline (Header only)", units1),
        ("Header + Merge", units2),
        ("Full Pipeline", units3),
    ]
    
    print(f"{'Strategy':<25} {'Units':<8} {'Avg Tokens':<12} {'Range'}")
    print(f"{'-'*70}")
    
    for name, units in results:
        tokens = [len(tokenizer.encode(u.content)) for u in units]
        avg = sum(tokens) // len(tokens)
        range_str = f"{min(tokens)}-{max(tokens)}"
        print(f"{name:<25} {len(units):<8} {avg:<12} {range_str}")
    
    # Show sample units from full pipeline
    print_separator("Sample Units (Full Pipeline)")
    
    for i, unit in enumerate(units3[:3]):
        token_count = len(tokenizer.encode(unit.content))
        print(f"\n📦 Unit {i+1}")
        print(f"   ID: {unit.unit_id}")
        print(f"   Context: {unit.metadata.context_path if unit.metadata else 'N/A'}")
        print(f"   Tokens: {token_count}")
        print(f"   Preview: {unit.content[:150]}...")
    
    # Verify chain relationships
    print_separator("Chain Relationship Verification")
    
    errors = []
    for i, unit in enumerate(units3):
        # Check prev
        if i == 0 and unit.prev_unit_id is not None:
            errors.append(f"Unit {i}: First unit has prev_unit_id")
        elif i > 0 and unit.prev_unit_id != units3[i-1].unit_id:
            errors.append(f"Unit {i}: Wrong prev_unit_id")
        
        # Check next
        if i == len(units3)-1 and unit.next_unit_id is not None:
            errors.append(f"Unit {i}: Last unit has next_unit_id")
        elif i < len(units3)-1 and unit.next_unit_id != units3[i+1].unit_id:
            errors.append(f"Unit {i}: Wrong next_unit_id")
    
    if errors:
        print("❌ Chain errors found:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ All chain relationships are correct!")
    
    # Summary
    print_separator("Key Takeaways")
    
    print("✅ Pipeline Syntax:")
    print("   - Elegant composition: splitter1 | splitter2 | splitter3")
    print("   - Automatic unit chaining and metadata preservation")
    print("   - Easy to adjust and extend")
    print()
    print("✅ TextSplitter:")
    print("   - Splits oversized chunks by semantic boundaries")
    print("   - Priority: paragraphs > sentences > hard split")
    print("   - Preserves context_path from parent units")
    print()
    print("✅ RecursiveMergingSplitter:")
    print("   - Merges small chunks to optimal size")
    print("   - Uses TextUnit.__add__() for seamless merging")
    print("   - Maintains chain relationships")
    print()
    print("💡 Best Practice:")
    print("   For RAG applications, aim for 500-1000 tokens per chunk")
    print("   to balance context richness and retrieval precision.")


if __name__ == "__main__":
    # Check dependencies
    try:
        import tiktoken
    except ImportError:
        print("❌ tiktoken is required")
        print("Run: pip install tiktoken")
        sys.exit(1)
    
    main()
