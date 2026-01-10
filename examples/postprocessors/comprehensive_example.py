"""Test postprocessors - comprehensive functionality testing

This test demonstrates the complete postprocessor architecture with real data.
"""

import sys
from pathlib import Path

from zag.postprocessors import (
    # Rerankers
    Reranker,
    # Filters
    SimilarityFilter,
    Deduplicator,
    # Compressors
    TokenCompressor,
    # Augmentors
    ContextAugmentor,
    # Composite
    ChainPostprocessor,
    ConditionalPostprocessor,
)
from zag.schemas.unit import TextUnit
import random


def create_test_units(count: int = 10, with_chain: bool = False) -> list[TextUnit]:
    """Create test units for demonstration"""
    units = []
    
    for i in range(count):
        unit = TextUnit(
            unit_id=f"unit_{i}",
            content=f"This is content for unit {i}. " * 5  # Repeat for more content
        )
        unit.score = random.uniform(0.5, 1.0)
        units.append(unit)
    
    # Set up chain relationships if requested
    if with_chain:
        for i in range(len(units)):
            if i > 0:
                units[i].prev_unit_id = units[i-1].unit_id
            if i < len(units) - 1:
                units[i].next_unit_id = units[i+1].unit_id
    
    return units


def test_similarity_filter():
    print("\n" + "="*70)
    print("Test 1: SimilarityFilter")
    print("="*70)
    
    units = create_test_units(10)
    print(f"Original: {len(units)} units")
    for i, u in enumerate(units[:3], 1):
        print(f"  {i}. [Score: {u.score:.3f}] {u.content[:50]}...")
    
    # Filter with threshold 0.75
    filter = SimilarityFilter(threshold=0.75)
    filtered = filter.process("test query", units)
    
    print(f"\nAfter filtering (threshold=0.75): {len(filtered)} units")
    for i, u in enumerate(filtered, 1):
        print(f"  {i}. [Score: {u.score:.3f}] {u.content[:50]}...")


def test_deduplicator():
    print("\n" + "="*70)
    print("Test 2: Deduplicator")
    print("="*70)
    
    # Create units with duplicates
    units = create_test_units(5)
    units.append(units[0])  # Add duplicate
    units.append(units[1])  # Add another duplicate
    
    print(f"Original: {len(units)} units (with duplicates)")
    
    # Deduplicate
    dedup = Deduplicator(strategy="exact")
    unique = dedup.process("test query", units)
    
    print(f"After deduplication: {len(unique)} unique units")


def test_token_compressor():
    print("\n" + "="*70)
    print("Test 3: TokenCompressor")
    print("="*70)
    
    units = create_test_units(20)
    print(f"Original: {len(units)} units")
    
    # Compress to 1000 tokens (roughly 10 units)
    compressor = TokenCompressor(max_tokens=1000, strategy="smart")
    compressed = compressor.process("test query", units)
    
    print(f"After compression (max_tokens=1000): {len(compressed)} units")


def test_context_augmentor():
    print("\n" + "="*70)
    print("Test 4: ContextAugmentor")
    print("="*70)
    
    # Create units with chain relationships
    units = create_test_units(10, with_chain=True)
    
    # Select a few units (simulate retrieval results)
    selected = [units[3], units[7]]
    print(f"Selected: {len(selected)} units")
    for u in selected:
        print(f"  - {u.unit_id}")
    
    # Augment with context (window_size=1)
    augmentor = ContextAugmentor(window_size=1)
    augmented = augmentor.process("test query", selected)
    
    print(f"\nAfter augmentation (window_size=1): {len(augmented)} units")
    for u in augmented:
        print(f"  - {u.unit_id}")


def test_chain_postprocessor():
    print("\n" + "="*70)
    print("Test 5: ChainPostprocessor")
    print("="*70)
    
    units = create_test_units(20)
    print(f"Original: {len(units)} units")
    
    # Create a processing chain
    chain = ChainPostprocessor([
        SimilarityFilter(threshold=0.65),
        Deduplicator(strategy="exact"),
        TokenCompressor(max_tokens=500, strategy="smart"),
    ])
    
    print("\nProcessing chain:")
    print("  1. SimilarityFilter(threshold=0.65)")
    print("  2. Deduplicator(strategy='exact')")
    print("  3. TokenCompressor(max_tokens=500)")
    
    result = chain.process("test query", units)
    
    print(f"\nFinal result: {len(result)} units")
    for i, u in enumerate(result[:5], 1):
        print(f"  {i}. [Score: {u.score:.3f}] {u.content[:50]}...")


def test_conditional_postprocessor():
    print("\n" + "="*70)
    print("Test 6: ConditionalPostprocessor")
    print("="*70)
    
    # Define condition
    def need_filtering(query: str, units: list) -> bool:
        """Only filter if we have many results"""
        return len(units) > 10
    
    # Create conditional processor
    conditional = ConditionalPostprocessor(
        condition=need_filtering,
        true_processor=SimilarityFilter(threshold=0.8),
        false_processor=None,
    )
    
    # Test with many units
    many_units = create_test_units(15)
    print(f"Test with {len(many_units)} units (> 10)")
    result1 = conditional.process("test", many_units)
    print(f"  → Filtered to {len(result1)} units (condition=True)")
    
    # Test with few units
    few_units = create_test_units(5)
    print(f"\nTest with {len(few_units)} units (<= 10)")
    result2 = conditional.process("test", few_units)
    print(f"  → Kept {len(result2)} units (condition=False)")


def test_reranker():
    print("\n" + "="*70)
    print("Test 7: Reranker (Requires sentence-transformers)")
    print("="*70)
    
    print("Note: Reranker requires sentence-transformers package.")
    print("Install with: pip install sentence-transformers")
    print("\nExample usage:")
    print("  from zag.postprocessors import Reranker")
    print("  reranker = Reranker('local/cross-encoder/ms-marco-MiniLM-L-12-v2')")
    print("  reranked = reranker.rerank(query, units, top_k=10)")


def test_nested_combination():
    print("\n" + "="*70)
    print("Test 8: Nested Combination")
    print("="*70)
    
    units = create_test_units(30)
    print(f"Original: {len(units)} units")
    
    # Create nested processors
    # Inner chain: filter + deduplicate
    inner_chain = ChainPostprocessor([
        SimilarityFilter(threshold=0.7),
        Deduplicator(strategy="exact"),
    ])
    
    # Outer conditional: compress if still too many
    def need_compression(query: str, units: list) -> bool:
        return len(units) > 10
    
    outer_conditional = ConditionalPostprocessor(
        condition=need_compression,
        true_processor=TokenCompressor(max_tokens=800),
        false_processor=None,
    )
    
    # Combine them in a chain
    final_chain = ChainPostprocessor([
        inner_chain,
        outer_conditional,
    ])
    
    print("\nNested structure:")
    print("  ChainPostprocessor([")
    print("    ChainPostprocessor([")
    print("      SimilarityFilter(0.7),")
    print("      Deduplicator(),")
    print("    ]),")
    print("    ConditionalPostprocessor(")
    print("      if len > 10: TokenCompressor(800)")
    print("    ),")
    print("  ])")
    
    result = final_chain.process("test query", units)
    
    print(f"\nFinal result: {len(result)} units")


def main():
    print("\n" + "="*70)
    print("Postprocessor Module - Comprehensive Testing")
    print("="*70)
    
    test_similarity_filter()
    test_deduplicator()
    test_token_compressor()
    test_context_augmentor()
    test_chain_postprocessor()
    test_conditional_postprocessor()
    test_reranker()
    test_nested_combination()
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
