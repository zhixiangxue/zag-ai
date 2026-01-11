"""
TokenCompressor Example

Demonstrates compressing units to fit within token limits.
"""

from zag.postprocessors.compressors import TokenCompressor
from zag.schemas.unit import TextUnit


def create_test_units(count: int = 20) -> list[TextUnit]:
    """Create test units with varying content lengths"""
    units = []
    for i in range(count):
        # Vary content length
        repeat_count = (i % 5) + 1
        content = f"This is unit {i} with some content. " * repeat_count * 10
        
        units.append(TextUnit(
            unit_id=f"unit_{i}",
            content=content,
            score=1.0 - (i * 0.02)  # Descending scores
        ))
    return units


def example_basic_compression():
    """Example 1: Basic token compression"""
    print("="*70)
    print("Example 1: Basic Token Compression")
    print("="*70)
    
    units = create_test_units(15)
    total_chars = sum(len(u.content) for u in units)
    
    print(f"\nOriginal: {len(units)} units")
    print(f"Total characters: {total_chars}")
    print(f"Average per unit: {total_chars//len(units)}")
    
    # Compress to 1000 tokens (roughly 4000 characters)
    compressor = TokenCompressor(max_tokens=1000, strategy="smart")
    compressed = compressor.process("query", units)
    
    compressed_chars = sum(len(u.content) for u in compressed)
    print(f"\nAfter compression (max_tokens=1000): {len(compressed)} units")
    print(f"Total characters: {compressed_chars}")
    print(f"Compression ratio: {(compressed_chars/total_chars)*100:.1f}%")


def example_smart_vs_truncate():
    """Example 2: Smart vs Truncate strategy"""
    print("\n" + "="*70)
    print("Example 2: Smart vs Truncate Strategy")
    print("="*70)
    
    units = create_test_units(10)
    
    # Strategy 1: Smart (keeps highest scoring units)
    smart_comp = TokenCompressor(max_tokens=500, strategy="smart")
    smart_result = smart_comp.process("query", units)
    
    print(f"\nSmart strategy (max_tokens=500):")
    print(f"  Kept: {len(smart_result)} units")
    print(f"  Score range: {smart_result[-1].score:.2f} - {smart_result[0].score:.2f}")
    
    # Strategy 2: Truncate (keeps first N units)
    trunc_comp = TokenCompressor(max_tokens=500, strategy="truncate")
    trunc_result = trunc_comp.process("query", units)
    
    print(f"\nTruncate strategy (max_tokens=500):")
    print(f"  Kept: {len(trunc_result)} units")
    print(f"  Units: {[u.unit_id for u in trunc_result]}")


def example_no_compression_needed():
    """Example 3: Content already within limit"""
    print("\n" + "="*70)
    print("Example 3: No Compression Needed")
    print("="*70)
    
    # Create small units
    units = [
        TextUnit(unit_id=f"small_{i}", content=f"Short content {i}", score=0.9-i*0.1)
        for i in range(3)
    ]
    
    total_chars = sum(len(u.content) for u in units)
    
    # Large token limit
    compressor = TokenCompressor(max_tokens=10000, strategy="smart")
    result = compressor.process("query", units)
    
    print(f"\nOriginal: {len(units)} units ({total_chars} chars)")
    print(f"Token limit: 10000 (much larger than content)")
    print(f"Result: {len(result)} units")
    print("✓ All units preserved (no compression needed)")


def example_extreme_compression():
    """Example 4: Extreme compression"""
    print("\n" + "="*70)
    print("Example 4: Extreme Compression")
    print("="*70)
    
    units = create_test_units(20)
    total_chars = sum(len(u.content) for u in units)
    
    print(f"\nOriginal: {len(units)} units ({total_chars} chars)")
    
    # Very small token limit
    compressor = TokenCompressor(max_tokens=100, strategy="smart")
    compressed = compressor.process("query", units)
    
    compressed_chars = sum(len(u.content) for u in compressed)
    
    print(f"After extreme compression (max_tokens=100): {len(compressed)} units")
    print(f"Characters: {compressed_chars} (from {total_chars})")
    print(f"Kept only top-scoring units")
    for u in compressed:
        print(f"  [{u.score:.2f}] {u.unit_id}")


def example_integration_pipeline():
    """Example 5: Integration with RAG pipeline"""
    print("\n" + "="*70)
    print("Example 5: Integration with RAG Pipeline")
    print("="*70)
    
    print("\nTypical usage in RAG pipeline:")
    print("""
    # 1. Retrieve many units
    units = vector_store.search(query, top_k=50)
    
    # 2. Compress to fit LLM context window
    # Assuming 4K context, reserve 1K for prompt + response
    compressor = TokenCompressor(max_tokens=3000, strategy="smart")
    compressed_units = compressor.process(query, units)
    
    # 3. Use compressed results in LLM
    context = "\\n".join(u.content for u in compressed_units)
    llm_response = llm.generate(query, context=context)
    """)
    
    # Simulate
    units = create_test_units(30)
    compressor = TokenCompressor(max_tokens=2000, strategy="smart")
    compressed = compressor.process("machine learning", units)
    
    print(f"\nSimulation:")
    print(f"  Retrieved: {len(units)} units")
    print(f"  Compressed: {len(compressed)} units (max_tokens=2000)")
    print(f"  Reduction: {((len(units)-len(compressed))/len(units))*100:.1f}%")
    print(f"  Fits in LLM context window: ✓")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("TokenCompressor Examples")
    print("="*70)
    print("\nCompresses units to fit within token limits.")
    print("Use cases:")
    print("  - Fit results into LLM context windows")
    print("  - Control computational costs")
    print("  - Prioritize high-quality content")
    
    example_basic_compression()
    example_smart_vs_truncate()
    example_no_compression_needed()
    example_extreme_compression()
    example_integration_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
