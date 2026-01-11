"""
ConditionalPostprocessor Example

Demonstrates conditional postprocessing based on dynamic conditions.
"""

from zag.postprocessors import ConditionalPostprocessor
from zag.postprocessors.filters import SimilarityFilter
from zag.postprocessors.compressors import TokenCompressor
from zag.schemas.unit import TextUnit
import random


def create_test_units(count: int) -> list[TextUnit]:
    """Create test units"""
    return [
        TextUnit(
            unit_id=f"unit_{i}",
            content=f"Content for unit {i}. " * random.randint(5, 10),
            score=random.uniform(0.5, 1.0)
        )
        for i in range(count)
    ]


def example_basic_conditional():
    """Example 1: Basic conditional processing"""
    print("="*70)
    print("Example 1: Basic Conditional Processing")
    print("="*70)
    
    # Condition: Only filter if we have many results
    def need_filtering(query: str, units: list) -> bool:
        return len(units) > 10
    
    conditional = ConditionalPostprocessor(
        condition=need_filtering,
        true_processor=SimilarityFilter(threshold=0.8),
        false_processor=None,  # Do nothing if condition is False
    )
    
    # Test with many units
    many_units = create_test_units(15)
    print(f"\nTest 1: {len(many_units)} units (> 10)")
    result1 = conditional.process("query", many_units)
    print(f"  Condition: True → Applied filtering")
    print(f"  Result: {len(result1)} units")
    
    # Test with few units
    few_units = create_test_units(5)
    print(f"\nTest 2: {len(few_units)} units (<= 10)")
    result2 = conditional.process("query", few_units)
    print(f"  Condition: False → No filtering")
    print(f"  Result: {len(result2)} units")


def example_query_based_condition():
    """Example 2: Condition based on query"""
    print("\n" + "="*70)
    print("Example 2: Query-Based Conditional")
    print("="*70)
    
    # Condition: More aggressive filtering for technical queries
    def is_technical_query(query: str, units: list) -> bool:
        technical_keywords = ["programming", "algorithm", "database", "machine learning"]
        return any(keyword in query.lower() for keyword in technical_keywords)
    
    conditional = ConditionalPostprocessor(
        condition=is_technical_query,
        true_processor=SimilarityFilter(threshold=0.85),  # High threshold for technical
        false_processor=SimilarityFilter(threshold=0.65),  # Lower threshold for general
    )
    
    units = create_test_units(10)
    
    # Technical query
    print("\nTechnical query: 'machine learning algorithms'")
    result1 = conditional.process("machine learning algorithms", units)
    print(f"  → High-threshold filtering applied")
    print(f"  Result: {len(result1)} units")
    
    # General query
    print("\nGeneral query: 'interesting facts'")
    result2 = conditional.process("interesting facts", units)
    print(f"  → Low-threshold filtering applied")
    print(f"  Result: {len(result2)} units")


def example_score_based_condition():
    """Example 3: Condition based on result quality"""
    print("\n" + "="*70)
    print("Example 3: Score-Based Conditional")
    print("="*70)
    
    # Condition: Only compress if average score is low
    def low_quality_results(query: str, units: list) -> bool:
        if not units:
            return False
        avg_score = sum(u.score for u in units) / len(units)
        return avg_score < 0.7
    
    conditional = ConditionalPostprocessor(
        condition=low_quality_results,
        true_processor=TokenCompressor(max_tokens=500, strategy="smart"),  # Aggressive compression
        false_processor=None,  # Keep all if high quality
    )
    
    # Low-quality results
    low_quality = [
        TextUnit(unit_id=f"low_{i}", content=f"Content {i}", score=random.uniform(0.3, 0.6))
        for i in range(10)
    ]
    
    print(f"\nLow-quality results (avg score < 0.7):")
    avg1 = sum(u.score for u in low_quality) / len(low_quality)
    print(f"  Average score: {avg1:.2f}")
    result1 = conditional.process("query", low_quality)
    print(f"  → Compression applied: {len(low_quality)} → {len(result1)} units")
    
    # High-quality results
    high_quality = [
        TextUnit(unit_id=f"high_{i}", content=f"Content {i}", score=random.uniform(0.8, 1.0))
        for i in range(10)
    ]
    
    print(f"\nHigh-quality results (avg score >= 0.7):")
    avg2 = sum(u.score for u in high_quality) / len(high_quality)
    print(f"  Average score: {avg2:.2f}")
    result2 = conditional.process("query", high_quality)
    print(f"  → No compression: {len(high_quality)} units kept")


def example_nested_conditionals():
    """Example 4: Nested conditional logic"""
    print("\n" + "="*70)
    print("Example 4: Nested Conditional Logic")
    print("="*70)
    
    # First condition: Check result count
    def many_results(query: str, units: list) -> bool:
        return len(units) > 15
    
    # Second condition: Check quality
    def low_quality(query: str, units: list) -> bool:
        if not units:
            return False
        avg_score = sum(u.score for u in units) / len(units)
        return avg_score < 0.75  # Only filter if LOW quality
    
    # Inner conditional: filter only if low quality
    quality_conditional = ConditionalPostprocessor(
        condition=low_quality,
        true_processor=SimilarityFilter(threshold=0.7),  # Filter if low quality
        false_processor=None,  # Keep all if high quality
    )
    
    # Outer conditional
    count_conditional = ConditionalPostprocessor(
        condition=many_results,
        true_processor=quality_conditional,  # Check quality if many results
        false_processor=None,  # Keep all if few results
    )
    
    print("\nNested logic:")
    print("  IF many_results (>15):")
    print("    IF low_quality (<0.75): filter (threshold=0.7)")
    print("    ELSE: keep all")
    print("  ELSE: keep all")
    
    # Test case 1: Many results, high quality
    units1 = [
        TextUnit(unit_id=f"u{i}", content=f"Content {i}", score=random.uniform(0.8, 1.0))
        for i in range(20)
    ]
    result1 = count_conditional.process("query", units1)
    print(f"\nCase 1: {len(units1)} units, avg score {sum(u.score for u in units1)/len(units1):.2f}")
    print(f"  Result: {len(result1)} units (high quality, kept all)")
    
    # Test case 2: Many results, low quality
    units2 = [
        TextUnit(unit_id=f"u{i}", content=f"Content {i}", score=random.uniform(0.4, 0.7))
        for i in range(20)
    ]
    result2 = count_conditional.process("query", units2)
    print(f"\nCase 2: {len(units2)} units, avg score {sum(u.score for u in units2)/len(units2):.2f}")
    print(f"  Result: {len(result2)} units (low quality, filtered)")


def example_integration_pipeline():
    """Example 5: Integration in production pipeline"""
    print("\n" + "="*70)
    print("Example 5: Production Pipeline Integration")
    print("="*70)
    
    print("\nTypical production use case:")
    print("""
    # Adaptive processing based on retrieval results
    
    def need_aggressive_filtering(query: str, units: list) -> bool:
        # Filter more aggressively if many low-quality results
        if len(units) < 5:
            return False
        avg_score = sum(u.score for u in units) / len(units)
        return avg_score < 0.75 or len(units) > 30
    
    adaptive_processor = ConditionalPostprocessor(
        condition=need_aggressive_filtering,
        true_processor=ChainPostprocessor([
            SimilarityFilter(threshold=0.8),
            TokenCompressor(max_tokens=1000)
        ]),
        false_processor=SimilarityFilter(threshold=0.6)
    )
    
    # Process retrieval results
    units = vector_store.search(query, top_k=50)
    processed = adaptive_processor.process(query, units)
    """)
    
    print("\nBenefits:")
    print("  1. Adapts to result quality automatically")
    print("  2. Balances precision and recall dynamically")
    print("  3. Handles edge cases gracefully")
    print("  4. Improves system robustness")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("ConditionalPostprocessor Examples")
    print("="*70)
    print("\nApplies postprocessing conditionally based on runtime conditions.")
    print("Use cases:")
    print("  - Adaptive processing based on result quality")
    print("  - Query-specific processing strategies")
    print("  - Handle edge cases gracefully")
    print("  - Build robust, intelligent pipelines")
    
    example_basic_conditional()
    example_query_based_condition()
    example_score_based_condition()
    example_nested_conditionals()
    example_integration_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
