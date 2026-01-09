#!/usr/bin/env python3
"""
Test RecursiveMergingSplitter - Recursive Merging Splitter

Demonstrates how to use RecursiveMergingSplitter to wrap MarkdownHeaderSplitter,
merging small chunks into larger, semantically complete chunks.

Features:
- Uses TextUnit.__add__() operator for merging
- Automatic token counting (using tiktoken)
- Preserves chain relationships (prev_unit_id / next_unit_id)
- Records merge info (merged_from, merged_count)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zag.splitters import MarkdownHeaderSplitter, RecursiveMergingSplitter
from zag.schemas.markdown import Markdown
from zag.schemas.base import DocumentMetadata


def print_unit_info(unit, index):
    """Print unit details"""
    print(f"\n{'='*70}")
    print(f"Unit {index + 1}")
    print(f"{'='*70}")
    print(f"Unit ID: {unit.unit_id}")
    print(f"Context Path: {unit.metadata.context_path if unit.metadata else 'N/A'}")
    
    # Calculate token count
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_count = len(tokenizer.encode(unit.content))
    print(f"Token Count: {token_count}")
    
    # Show merge info
    if unit.metadata and unit.metadata.custom:
        merged_from = unit.metadata.custom.get("merged_from")
        merged_count = unit.metadata.custom.get("merged_count")
        if merged_from:
            print(f"Merged From: {merged_count} units")
            print(f"  Original IDs: {merged_from[:3]}{'...' if len(merged_from) > 3 else ''}")
    
    # Show chain relationships
    print(f"Previous Unit: {unit.prev_unit_id or 'None'}")
    print(f"Next Unit: {unit.next_unit_id or 'None'}")
    
    # Show content preview
    print(f"\nContent Preview (first 200 chars):")
    print(f"{unit.content[:200]}...")


def verify_chain_relationships(units):
    """
    Verify chain relationships are correct
    This is CRITICAL for context expansion!
    """
    print("\n" + "=" * 70)
    print("🔍 CRITICAL: Chain Relationship Verification")
    print("=" * 70)
    
    errors = []
    
    for i, unit in enumerate(units):
        # Check prev_unit_id
        if i == 0:
            # First unit should have no previous
            if unit.prev_unit_id is not None:
                errors.append(f"Unit {i}: First unit should have prev_unit_id=None, got {unit.prev_unit_id}")
        else:
            # Should point to previous unit
            expected_prev = units[i - 1].unit_id
            if unit.prev_unit_id != expected_prev:
                errors.append(f"Unit {i}: prev_unit_id should be {expected_prev}, got {unit.prev_unit_id}")
        
        # Check next_unit_id
        if i == len(units) - 1:
            # Last unit should have no next
            if unit.next_unit_id is not None:
                errors.append(f"Unit {i}: Last unit should have next_unit_id=None, got {unit.next_unit_id}")
        else:
            # Should point to next unit
            expected_next = units[i + 1].unit_id
            if unit.next_unit_id != expected_next:
                errors.append(f"Unit {i}: next_unit_id should be {expected_next}, got {unit.next_unit_id}")
    
    # Print results
    if errors:
        print("\n❌ CHAIN RELATIONSHIP ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ All chain relationships are CORRECT!")
        print("\nVerification Details:")
        for i, unit in enumerate(units):
            prev_status = "✅" if (i == 0 and unit.prev_unit_id is None) or (i > 0 and unit.prev_unit_id == units[i-1].unit_id) else "❌"
            next_status = "✅" if (i == len(units)-1 and unit.next_unit_id is None) or (i < len(units)-1 and unit.next_unit_id == units[i+1].unit_id) else "❌"
            print(f"  Unit {i}: {prev_status} prev | {next_status} next")
        return True


def main():
    print("=" * 70)
    print("RecursiveMergingSplitter Test - Sample Guideline")
    print("=" * 70)
    print()
    
    # 1. Load Markdown document from file
    sample_file = Path(__file__).parent.parent / "tmp" / "sample-guideline.md"
    
    if not sample_file.exists():
        print(f"❌ File not found: {sample_file}")
        sys.exit(1)
    
    content = sample_file.read_text(encoding="utf-8")
    
    # Create document with metadata
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
    print(f"📄 Document length: {len(content)} characters")
    print()
    
    # 2. Base splitting with MarkdownHeaderSplitter
    print("=" * 70)
    print("Step 1: Base Splitting (MarkdownHeaderSplitter)")
    print("=" * 70)
    
    base_splitter = MarkdownHeaderSplitter()
    base_units = doc.split(base_splitter)
    
    print(f"\n✅ Base splitting result: {len(base_units)} units")
    
    # Show base splitting token stats
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    base_tokens = [len(tokenizer.encode(u.content)) for u in base_units]
    print(f"\nToken Statistics:")
    print(f"  Min: {min(base_tokens)} tokens")
    print(f"  Max: {max(base_tokens)} tokens")
    print(f"  Avg: {sum(base_tokens) // len(base_tokens)} tokens")
    print(f"  Total: {sum(base_tokens)} tokens")
    
    print(f"\nToken count per unit: {base_tokens}")
    
    # 3. Recursive merging
    print("\n" + "=" * 70)
    print("Step 2: Recursive Merging (RecursiveMergingSplitter)")
    print("=" * 70)
    
    merger = RecursiveMergingSplitter(
        base_splitter=base_splitter,
        target_token_size=800  # Target 800 tokens per chunk
    )
    
    merged_units = doc.split(merger)
    
    print(f"\n✅ Merged result: {len(merged_units)} units")
    
    # Show merged token stats
    merged_tokens = [len(tokenizer.encode(u.content)) for u in merged_units]
    print(f"\nToken Statistics:")
    print(f"  Min: {min(merged_tokens)} tokens")
    print(f"  Max: {max(merged_tokens)} tokens")
    print(f"  Avg: {sum(merged_tokens) // len(merged_tokens)} tokens")
    print(f"  Total: {sum(merged_tokens)} tokens")
    
    print(f"\nToken count per unit: {merged_tokens}")
    
    # 4. CRITICAL: Verify chain relationships
    chain_valid = verify_chain_relationships(merged_units)
    
    if not chain_valid:
        print("\n⚠️  WARNING: Chain relationship verification FAILED!")
        print("This will break context expansion functionality!")
        sys.exit(1)
    
    # 5. Show detailed info for first 3 units
    print("\n" + "=" * 70)
    print("Step 4: Detailed Info (First 3 Units)")
    print("=" * 70)
    
    for i, unit in enumerate(merged_units[:3]):
        print_unit_info(unit, i)
    
    # 6. Compare results
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    
    print(f"\nBase Splitting:")
    print(f"  - Number of units: {len(base_units)}")
    print(f"  - Avg tokens: {sum(base_tokens) // len(base_tokens)}")
    print(f"  - Semantic completeness: Low (fragmented)")
    
    print(f"\nAfter Merging:")
    print(f"  - Number of units: {len(merged_units)}")
    print(f"  - Avg tokens: {sum(merged_tokens) // len(merged_tokens)}")
    print(f"  - Semantic completeness: High (follows RAG best practices)")
    
    print(f"\nImprovements:")
    print(f"  - Units reduced: {len(base_units) - len(merged_units)} ({(1 - len(merged_units)/len(base_units))*100:.1f}%)")
    print(f"  - Avg tokens increased: {(sum(merged_tokens) // len(merged_tokens)) - (sum(base_tokens) // len(base_tokens))} per unit")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)
    
    print("\n💡 Key Findings:")
    print("   - TextUnit.__add__() operator simplifies merge logic")
    print("   - Merged chunks have better semantic completeness")
    print("   - Follows RAG best practices (500-1000 tokens)")
    print("   - Preserves chain relationships and metadata")


if __name__ == "__main__":
    # Check if tiktoken is installed
    try:
        import tiktoken
    except ImportError:
        print("❌ tiktoken is required")
        print("Run: pip install tiktoken")
        sys.exit(1)
    
    main()
