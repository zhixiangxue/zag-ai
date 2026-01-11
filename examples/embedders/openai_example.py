"""
OpenAI Embedder Example

This example demonstrates how to use the OpenAI embedder provider.

Before running:
1. Set your OpenAI API key:
   - Windows: set OPENAI_API_KEY=sk-...
   - Linux/Mac: export OPENAI_API_KEY=sk-...
   Or pass api_key directly in the code

2. Install OpenAI SDK:
   pip install openai
"""

from zag import Embedder


def main():
    print("=" * 60)
    print("OpenAI Embedder Example")
    print("=" * 60)
    
    # Example 1: Basic usage with default model (text-embedding-3-small)
    print("\n1. Basic Usage (text-embedding-3-small)")
    print("-" * 60)
    
    embedder = Embedder("openai/text-embedding-3-small")
    print(f"✓ Created OpenAI embedder")
    print(f"  Model: text-embedding-3-small")
    print(f"  Dimension: {embedder.dimension}")
    
    # Embed single text
    text = "OpenAI provides powerful embedding models"
    vector = embedder.embed(text)
    print(f"\n✓ Embedded text: '{text}'")
    print(f"  Vector dimension: {len(vector)}")
    print(f"  First 5 values: {vector[:5]}")
    
    # Embed batch
    texts = [
        "OpenAI embeddings are great",
        "Vector search is powerful",
        "Machine learning is fascinating"
    ]
    vectors = embedder.embed_batch(texts)
    print(f"\n✓ Embedded {len(texts)} texts")
    print(f"  Batch size: {len(vectors)}")
    print(f"  Vector dimensions: {[len(v) for v in vectors]}")
    
    # Example 2: High-quality model (text-embedding-3-large)
    print("\n\n2. High-Quality Model (text-embedding-3-large)")
    print("-" * 60)
    
    embedder_large = Embedder("openai/text-embedding-3-large")
    print(f"✓ Created OpenAI embedder with large model")
    print(f"  Model: text-embedding-3-large")
    print(f"  Dimension: {embedder_large.dimension}")
    
    vector_large = embedder_large.embed("High quality embeddings")
    print(f"\n✓ Embedded text with large model")
    print(f"  Vector dimension: {len(vector_large)}")
    
    # Example 3: Dimension reduction (text-embedding-3 only)
    print("\n\n3. Dimension Reduction")
    print("-" * 60)
    
    # Reduce from 1536 to 512 dimensions
    embedder_reduced = Embedder(
        "openai/text-embedding-3-small",
        dimensions=512
    )
    print(f"✓ Created OpenAI embedder with dimension reduction")
    print(f"  Original dimension: 1536")
    print(f"  Reduced dimension: {embedder_reduced.dimension}")
    
    vector_reduced = embedder_reduced.embed("Compact embeddings")
    print(f"\n✓ Embedded text with reduced dimensions")
    print(f"  Vector dimension: {len(vector_reduced)}")
    print(f"  Storage savings: {(1 - 512/1536) * 100:.1f}%")
    
    # Example 4: Explicit API key (if not using environment variable)
    print("\n\n4. Explicit API Key Configuration")
    print("-" * 60)
    
    try:
        embedder_with_key = Embedder(
            "openai/text-embedding-3-small",
            api_key="your-api-key-here"  # Replace with actual key
        )
        print(f"✓ Created embedder with explicit API key")
        print(f"  Note: In production, use environment variables!")
    except Exception as e:
        print(f"⚠ Skipped (API key not provided): {e}")
    
    # Example 5: Azure OpenAI
    print("\n\n5. Azure OpenAI Configuration")
    print("-" * 60)
    
    try:
        embedder_azure = Embedder(
            "openai/text-embedding-3-small",
            api_key="your-azure-key",  # Replace with actual key
            base_url="https://your-resource.openai.azure.com/"  # Replace with actual endpoint
        )
        print(f"✓ Created Azure OpenAI embedder")
        print(f"  Note: This example requires valid Azure credentials")
    except Exception as e:
        print(f"⚠ Skipped (Azure credentials not provided): {e}")
    
    # Performance comparison
    print("\n\n6. Performance & Cost Comparison")
    print("-" * 60)
    print("""
Model Comparison:

1. text-embedding-3-small
   - Dimensions: 1536 (default) or custom (e.g., 512)
   - Cost: $0.02 / 1M tokens
   - Use case: Cost-effective, good quality
   
2. text-embedding-3-large
   - Dimensions: 3072 (default) or custom
   - Cost: $0.13 / 1M tokens
   - Use case: Best quality, higher cost
   
3. text-embedding-ada-002 (legacy)
   - Dimensions: 1536
   - Cost: $0.10 / 1M tokens
   - Use case: Legacy support only

Recommendation:
- Default: text-embedding-3-small (best cost/performance)
- High quality: text-embedding-3-large
- Storage constrained: text-embedding-3-small with dimensions=512
    """)
    
    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
