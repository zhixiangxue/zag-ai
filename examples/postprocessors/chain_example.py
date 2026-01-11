"""
ChainPostprocessor Example

Demonstrates chaining multiple postprocessors into a pipeline.
"""

from zag.postprocessors import ChainPostprocessor
from zag.postprocessors.filters import SimilarityFilter, Deduplicator
from zag.postprocessors.compressors import TokenCompressor
from zag.schemas.unit import TextUnit
import random


def create_test_units(count: int = 20) -> list[TextUnit]:
    """Create test units with some duplicates and varying scores"""
    units = []
    
    for i in range(count):
        # Create some duplicates
        if i > 0 and random.random() < 0.2:  # 20% chance of duplicate
            content = units[random.randint(0, i-1)].content
        else:
            content = f"Content for unit {i}. " * random.randint(3, 8)
        
        units.append(TextUnit(
            unit_id=f"unit_{i}",
            content=content,
            score=random.uniform(0.4, 1.0)
        ))
    
    return units


def example_basic_chain():
    """Example 1: Basic postprocessor chain"""
    print("="*70)
    print("Example 1: Basic Postprocessor Chain")
    print("="*70)
    
    units = create_test_units(20)
    print(f"\nOriginal: {len(units)} units")
    
    # Create a processing chain
    chain = ChainPostprocessor([
        SimilarityFilter(threshold=0.6),
        Deduplicator(strategy="exact"),
        TokenCompressor(max_tokens=500, strategy="smart"),
    ])
    
    print("\nProcessing chain:")
    print("  1. SimilarityFilter(threshold=0.6)  - Remove low-quality results")
    print("  2. Deduplicator(strategy='exact')   - Remove duplicates")
    print("  3. TokenCompressor(max_tokens=500)  - Fit within token limit")
    
    result = chain.process("test query", units)
    
    print(f"\nFinal result: {len(result)} units")
    print("Chain processing completed successfully!")


def example_step_by_step():
    """Example 2: Observing each step"""
    print("\n" + "="*70)
    print("Example 2: Step-by-Step Processing")
    print("="*70)
    
    units = create_test_units(15)
    
    print(f"\nStarting with: {len(units)} units")
    
    # Manual step-by-step (to show what chain does)
    print("\nStep 1: SimilarityFilter(threshold=0.7)")
    filter = SimilarityFilter(threshold=0.7)
    step1 = filter.process("query", units)
    print(f"  Result: {len(step1)} units (filtered out {len(units) - len(step1)} low-scoring)")
    
    print("\nStep 2: Deduplicator()")
    dedup = Deduplicator(strategy="exact")
    step2 = dedup.process("query", step1)
    print(f"  Result: {len(step2)} units (removed {len(step1) - len(step2)} duplicates)")
    
    print("\nStep 3: TokenCompressor(max_tokens=300)")
    compressor = TokenCompressor(max_tokens=300, strategy="smart")
    step3 = compressor.process("query", step2)
    print(f"  Result: {len(step3)} units (compressed by {len(step2) - len(step3)} units)")
    
    print(f"\nTotal reduction: {len(units)} → {len(step3)} units")
    
    # Now do the same with chain
    print("\nUsing ChainPostprocessor:")
    chain = ChainPostprocessor([filter, dedup, compressor])
    result = chain.process("query", units)
    print(f"  Result: {len(result)} units (same as manual steps)")


def example_flexible_ordering():
    """Example 3: Different chain orderings"""
    print("\n" + "="*70)
    print("Example 3: Flexible Chain Ordering")
    print("="*70)
    
    units = create_test_units(15)
    
    print(f"\nStarting with: {len(units)} units\n")
    
    # Order 1: Filter → Dedup → Compress
    chain1 = ChainPostprocessor([
        SimilarityFilter(threshold=0.7),
        Deduplicator(),
        TokenCompressor(max_tokens=400, strategy="smart"),
    ])
    result1 = chain1.process("query", units)
    print(f"Order 1 (Filter → Dedup → Compress): {len(result1)} units")
    
    # Order 2: Dedup → Filter → Compress
    chain2 = ChainPostprocessor([
        Deduplicator(),
        SimilarityFilter(threshold=0.7),
        TokenCompressor(max_tokens=400, strategy="smart"),
    ])
    result2 = chain2.process("query", units)
    print(f"Order 2 (Dedup → Filter → Compress): {len(result2)} units")
    
    print("\nNote: Different orderings can produce different results")
    print("Best practice: Filter → Dedup → Compress")


def example_reusable_chain():
    """Example 4: Reusable chain configuration"""
    print("\n" + "="*70)
    print("Example 4: Reusable Chain Configuration")
    print("="*70)
    
    # Create a standard processing chain
    standard_chain = ChainPostprocessor([
        SimilarityFilter(threshold=0.65),
        Deduplicator(strategy="exact"),
        TokenCompressor(max_tokens=1000, strategy="smart"),
    ])
    
    print("\nStandard processing chain created")
    print("Can be reused for multiple queries:\n")
    
    # Use for different queries
    for query_num in range(1, 4):
        units = create_test_units(15)
        result = standard_chain.process(f"query_{query_num}", units)
        print(f"Query {query_num}: {len(units)} units → {len(result)} units")


def example_integration_pipeline():
    """Example 5: Full RAG pipeline integration"""
    print("\n" + "="*70)
    print("Example 5: Full RAG Pipeline Integration")
    print("="*70)
    
    print("\nTypical RAG pipeline with postprocessor chain:")
    print("""
    # 1. Retrieve from vector store
    raw_units = vector_store.search(query, top_k=50)
    
    # 2. Apply postprocessor chain
    postprocessor = ChainPostprocessor([
        SimilarityFilter(threshold=0.7),    # Quality control
        Deduplicator(strategy="exact"),     # Remove duplicates
        TokenCompressor(max_tokens=2000),   # Fit LLM context
    ])
    
    processed_units = postprocessor.process(query, raw_units)
    
    # 3. Generate response
    context = "\\n".join(u.content for u in processed_units)
    response = llm.generate(query, context=context)
    """)
    
    # Simulate
    raw_units = create_test_units(50)
    
    postprocessor = ChainPostprocessor([
        SimilarityFilter(threshold=0.7),
        Deduplicator(strategy="exact"),
        TokenCompressor(max_tokens=2000, strategy="smart"),
    ])
    
    processed = postprocessor.process("machine learning", raw_units)
    
    print(f"\nSimulation:")
    print(f"  Retrieved: {len(raw_units)} units")
    print(f"  After processing: {len(processed)} units")
    print(f"  Reduction: {((len(raw_units)-len(processed))/len(raw_units))*100:.1f}%")
    print(f"  ✓ Ready for LLM consumption")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("ChainPostprocessor Examples")
    print("="*70)
    print("\nChains multiple postprocessors into a sequential pipeline.")
    print("Use cases:")
    print("  - Build complex processing pipelines")
    print("  - Combine multiple postprocessing strategies")
    print("  - Create reusable processing configurations")
    
    example_basic_chain()
    example_step_by_step()
    example_flexible_ordering()
    example_reusable_chain()
    example_integration_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
