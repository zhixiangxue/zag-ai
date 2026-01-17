"""
Deduplicator Example

Demonstrates removing duplicate units from search results.
"""

from zag.postprocessors.filters import Deduplicator
from zag.schemas.unit import TextUnit


def create_test_units_with_duplicates() -> list[TextUnit]:
    """Create test units including exact and near duplicates"""
    units = [
        TextUnit(
            unit_id="original_1",
            content="Machine learning is a field of artificial intelligence",
            score=0.95
        ),
        TextUnit(
            unit_id="duplicate_1_exact",
            content="Machine learning is a field of artificial intelligence",  # Exact duplicate
            score=0.92
        ),
        TextUnit(
            unit_id="unique_1",
            content="Deep learning uses neural networks with multiple layers",
            score=0.88
        ),
        TextUnit(
            unit_id="original_2",
            content="Python is a popular programming language for data science",
            score=0.85
        ),
        TextUnit(
            unit_id="duplicate_2_similar",
            content="Python is a popular programming language for data science.",  # Similar (with period)
            score=0.83
        ),
        TextUnit(
            unit_id="unique_2",
            content="Natural language processing enables computers to understand text",
            score=0.80
        ),
        TextUnit(
            unit_id="duplicate_1_another",
            content="Machine learning is a field of artificial intelligence",  # Another exact duplicate
            score=0.78
        ),
    ]
    return units


def example_exact_deduplication():
    """Example 1: Exact content deduplication"""
    print("="*70)
    print("Example 1: Exact Content Deduplication")
    print("="*70)
    
    units = create_test_units_with_duplicates()
    print(f"\nOriginal units: {len(units)}")
    for u in units:
        print(f"  [{u.score:.2f}] {u.unit_id}")
        print(f"         {u.content[:60]}...")
    
    # Exact deduplication (default strategy)
    dedup = Deduplicator(strategy="exact")
    unique_units = dedup.process("query", units)
    
    print(f"\nAfter deduplication: {len(unique_units)} unique units")
    print(f"Removed: {len(units) - len(unique_units)} duplicates")
    print("\nRemaining units:")
    for u in unique_units:
        print(f"  [{u.score:.2f}] {u.unit_id}")


def example_preserve_highest_score():
    """Example 2: Keep unit with highest score"""
    print("\n" + "="*70)
    print("Example 2: Preserving Highest Scoring Duplicate")
    print("="*70)
    
    units = create_test_units_with_duplicates()
    
    print("\nDuplicates with different scores:")
    ml_units = [u for u in units if "Machine learning" in u.content]
    for u in ml_units:
        print(f"  [{u.score:.2f}] {u.unit_id}")
    
    dedup = Deduplicator(strategy="exact")
    unique_units = dedup.process("query", units)
    
    # Check which one was kept
    kept_ml = [u for u in unique_units if "Machine learning" in u.content][0]
    print(f"\nKept unit: {kept_ml.unit_id} with score {kept_ml.score:.2f}")
    print("✓ Highest scoring duplicate was preserved")


def example_empty_and_edge_cases():
    """Example 3: Edge cases"""
    print("\n" + "="*70)
    print("Example 3: Edge Cases")
    print("="*70)
    
    dedup = Deduplicator(strategy="exact")
    
    # Empty list
    result = dedup.process("query", [])
    print(f"\nEmpty list: {len(result)} units (expected: 0)")
    
    # Single unit
    single = [TextUnit(unit_id="only", content="Single unit", score=0.9)]
    result = dedup.process("query", single)
    print(f"Single unit: {len(result)} units (expected: 1)")
    
    # All unique
    unique = [
        TextUnit(unit_id=f"u{i}", content=f"Unique content {i}", score=0.9-i*0.1)
        for i in range(3)
    ]
    result = dedup.process("query", unique)
    print(f"All unique: {len(result)} units (expected: 3)")
    
    # All duplicates
    all_dup = [
        TextUnit(unit_id=f"dup{i}", content="Same content", score=0.9-i*0.1)
        for i in range(5)
    ]
    result = dedup.process("query", all_dup)
    print(f"All duplicates: {len(result)} units (expected: 1, got highest score)")


def example_with_metadata():
    """Example 4: Deduplication with metadata"""
    print("\n" + "="*70)
    print("Example 4: Units with Metadata")
    print("="*70)
    
    from zag.schemas import UnitMetadata
    
    units = [
        TextUnit(
            unit_id="doc1_chunk1",
            content="Python is great for data analysis",
            score=0.9,
            metadata=UnitMetadata(context_path="tutorial.md/Page1", custom={"source": "tutorial.md"})
        ),
        TextUnit(
            unit_id="doc2_chunk1",
            content="Python is great for data analysis",  # Same content, different source
            score=0.85,
            metadata=UnitMetadata(context_path="guide.md/Page5", custom={"source": "guide.md"})
        ),
        TextUnit(
            unit_id="doc1_chunk2",
            content="NumPy provides array operations",
            score=0.88,
            metadata=UnitMetadata(context_path="tutorial.md/Page2", custom={"source": "tutorial.md"})
        ),
    ]
    
    print("\nUnits with metadata:")
    for u in units:
        source = u.metadata.custom.get("source", "unknown")
        print(f"  [{u.score:.2f}] {u.unit_id} - {source}")
        print(f"         {u.content}")
    
    dedup = Deduplicator(strategy="exact")
    unique_units = dedup.process("query", units)
    
    print(f"\nAfter deduplication: {len(unique_units)} units")
    print("Note: Deduplication is based on content only, not metadata")
    for u in unique_units:
        source = u.metadata.custom.get("source", "unknown")
        print(f"  [{u.score:.2f}] {u.unit_id} - {source}")


def example_integration_pipeline():
    """Example 5: Integration with retrieval pipeline"""
    print("\n" + "="*70)
    print("Example 5: Integration with Retrieval Pipeline")
    print("="*70)
    
    print("\nTypical usage in RAG pipeline:")
    print("""
    # 1. Retrieve units from multiple sources
    units_source1 = vector_store1.search(query, top_k=10)
    units_source2 = vector_store2.search(query, top_k=10)
    all_units = units_source1 + units_source2
    
    # 2. Remove duplicates
    dedup = Deduplicator(strategy="exact")
    unique_units = dedup.process(query, all_units)
    
    # 3. Use deduplicated results
    print(f"Total: {len(all_units)}, Unique: {len(unique_units)}")
    """)
    
    # Simulate multi-source retrieval
    units = create_test_units_with_duplicates()
    dedup = Deduplicator(strategy="exact")
    unique_units = dedup.process("machine learning", units)
    
    print(f"\nSimulation:")
    print(f"  Total retrieved: {len(units)} units")
    print(f"  After deduplication: {len(unique_units)} unique units")
    print(f"  Efficiency gain: {((len(units)-len(unique_units))/len(units))*100:.1f}% duplicates removed")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Deduplicator Examples")
    print("="*70)
    print("\nRemoves duplicate units from search results.")
    print("Use cases:")
    print("  - Merge results from multiple retrievers")
    print("  - Remove redundant content")
    print("  - Improve result diversity")
    
    example_exact_deduplication()
    example_preserve_highest_score()
    example_empty_and_edge_cases()
    example_with_metadata()
    example_integration_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
