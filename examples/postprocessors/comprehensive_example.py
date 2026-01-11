"""Comprehensive Postprocessor Examples

Demonstrates complex combinations and real-world RAG pipeline scenarios.

For individual postprocessor examples, see:
- similarity_filter_example.py
- deduplicator_example.py
- token_compressor_example.py
- context_augmentor_example.py
- reranker_example.py
- chain_example.py
- conditional_example.py
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


def test_chain_postprocessor():
    print("\n" + "="*70)
    print("Example 1: ChainPostprocessor - Sequential Pipeline")
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
    print("Example 2: ConditionalPostprocessor - Adaptive Processing")
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
    print("Example 3: Reranker - Two-Stage Retrieval")
    print("="*70)
    
    try:
        units = create_test_units(10)
        print(f"Original: {len(units)} units")
        for i, u in enumerate(units[:3], 1):
            print(f"  {i}. [Score: {u.score:.3f}] {u.content[:50]}...")
        
        # Create reranker
        reranker = Reranker("sentence_transformers/cross-encoder/ms-marco-MiniLM-L-12-v2")
        
        query = "machine learning"
        reranked = reranker.rerank(query, units, top_k=5)
        
        print(f"\nReranked (top 5): {len(reranked)} units")
        for i, u in enumerate(reranked, 1):
            print(f"  {i}. [Score: {u.score:.3f}] {u.content[:50]}...")
        
    except ImportError:
        print("\n⚠️  sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        print("\nExample usage:")
        print("  from zag.postprocessors import Reranker")
        print("  reranker = Reranker('sentence_transformers/cross-encoder/ms-marco-MiniLM-L-12-v2')")
        print("  reranked = reranker.rerank(query, units, top_k=10)")


def test_nested_combination():
    print("\n" + "="*70)
    print("Example 4: Nested Combination - Complex Pipeline")
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
    print("Comprehensive Postprocessor Examples")
    print("="*70)
    print("\nThese examples demonstrate complex combinations of postprocessors.")
    print("For individual postprocessor usage, see their dedicated example files.")
    
    test_chain_postprocessor()
    test_conditional_postprocessor()
    test_reranker()
    test_nested_combination()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  - Check individual example files for detailed usage")
    print("  - Combine postprocessors to build your RAG pipeline")
    print("="*70)


if __name__ == "__main__":
    main()
