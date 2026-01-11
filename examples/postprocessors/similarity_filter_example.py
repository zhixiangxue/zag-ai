"""
SimilarityFilter Example

Demonstrates filtering units based on similarity scores.
"""

from zag.postprocessors.filters import SimilarityFilter
from zag.schemas.unit import TextUnit


def create_test_units() -> list[TextUnit]:
    """Create test units with various scores"""
    units = [
        TextUnit(
            unit_id="high_1",
            content="Machine learning is a subset of artificial intelligence",
            score=0.95
        ),
        TextUnit(
            unit_id="high_2",
            content="Deep learning uses neural networks with multiple layers",
            score=0.88
        ),
        TextUnit(
            unit_id="medium_1",
            content="Python is a popular programming language",
            score=0.72
        ),
        TextUnit(
            unit_id="medium_2",
            content="Data science involves statistics and programming",
            score=0.68
        ),
        TextUnit(
            unit_id="low_1",
            content="The weather is nice today",
            score=0.45
        ),
        TextUnit(
            unit_id="low_2",
            content="I like to eat pizza",
            score=0.32
        ),
    ]
    return units


def example_basic_filtering():
    """Example 1: Basic threshold filtering"""
    print("="*70)
    print("Example 1: Basic Threshold Filtering")
    print("="*70)
    
    units = create_test_units()
    print(f"\nOriginal units: {len(units)}")
    for u in units:
        print(f"  [{u.score:.2f}] {u.unit_id}: {u.content[:50]}...")
    
    # Filter with threshold 0.7
    filter = SimilarityFilter(threshold=0.7)
    filtered = filter.process("machine learning query", units)
    
    print(f"\nAfter filtering (threshold=0.7): {len(filtered)} units")
    for u in filtered:
        print(f"  [{u.score:.2f}] {u.unit_id}: {u.content[:50]}...")


def example_aggressive_filtering():
    """Example 2: Aggressive filtering (high threshold)"""
    print("\n" + "="*70)
    print("Example 2: Aggressive Filtering (High Threshold)")
    print("="*70)
    
    units = create_test_units()
    
    # Only keep high-quality results
    filter = SimilarityFilter(threshold=0.85)
    filtered = filter.process("machine learning query", units)
    
    print(f"\nOriginal: {len(units)} units")
    print(f"After filtering (threshold=0.85): {len(filtered)} units")
    print("\nHigh-quality results only:")
    for u in filtered:
        print(f"  [{u.score:.2f}] {u.unit_id}: {u.content}")


def example_lenient_filtering():
    """Example 3: Lenient filtering (low threshold)"""
    print("\n" + "="*70)
    print("Example 3: Lenient Filtering (Low Threshold)")
    print("="*70)
    
    units = create_test_units()
    
    # Keep more results, filter out only very poor matches
    filter = SimilarityFilter(threshold=0.5)
    filtered = filter.process("machine learning query", units)
    
    print(f"\nOriginal: {len(units)} units")
    print(f"After filtering (threshold=0.5): {len(filtered)} units")
    print(f"Removed: {len(units) - len(filtered)} low-quality units")


def example_no_filtering_needed():
    """Example 4: All units above threshold"""
    print("\n" + "="*70)
    print("Example 4: All Units Pass Threshold")
    print("="*70)
    
    # Create units with all high scores
    high_quality_units = [
        TextUnit(unit_id=f"unit_{i}", content=f"High quality content {i}", score=0.9 + i*0.01)
        for i in range(5)
    ]
    
    filter = SimilarityFilter(threshold=0.85)
    filtered = filter.process("query", high_quality_units)
    
    print(f"\nOriginal: {len(high_quality_units)} units (all high-quality)")
    print(f"After filtering (threshold=0.85): {len(filtered)} units")
    print("✓ All units passed the threshold")


def example_integration_with_retrieval():
    """Example 5: Integration with retrieval pipeline"""
    print("\n" + "="*70)
    print("Example 5: Integration with Retrieval Pipeline")
    print("="*70)
    
    print("\nTypical usage in RAG pipeline:")
    print("""
    # 1. Retrieve units from vector store
    units = vector_store.search(query, top_k=20)
    
    # 2. Filter low-quality results
    filter = SimilarityFilter(threshold=0.7)
    filtered_units = filter.process(query, units)
    
    # 3. Use filtered results
    print(f"Retrieved {len(units)} units, kept {len(filtered_units)} after filtering")
    """)
    
    # Simulate the pipeline
    units = create_test_units()
    filter = SimilarityFilter(threshold=0.7)
    filtered_units = filter.process("machine learning", units)
    
    print(f"\nSimulation:")
    print(f"  Retrieved: {len(units)} units")
    print(f"  Filtered: {len(filtered_units)} units (threshold=0.7)")
    print(f"  Quality improvement: {(len(filtered_units)/len(units))*100:.1f}% of results kept")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("SimilarityFilter Examples")
    print("="*70)
    print("\nFilters units based on their similarity scores.")
    print("Use cases:")
    print("  - Remove low-quality retrieval results")
    print("  - Improve precision of search results")
    print("  - Control result quality in RAG pipelines")
    
    example_basic_filtering()
    example_aggressive_filtering()
    example_lenient_filtering()
    example_no_filtering_needed()
    example_integration_with_retrieval()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
