"""
Ollama Reranker Example

Demonstrates reranking search results using Ollama's local reranking models.

Requirements:
    pip install ollama
    
Setup:
    1. Install Ollama from https://ollama.com/
    2. Pull a reranking model: ollama pull bge-reranker-large
    3. Start Ollama service
"""

from zag.postprocessors import Reranker
from zag.schemas.unit import TextUnit


def create_test_units() -> list[TextUnit]:
    """Create test units with initial retrieval scores"""
    units = [
        TextUnit(
            unit_id="doc_1",
            content="Python is a high-level programming language used for web development, data analysis, and machine learning",
            score=0.75
        ),
        TextUnit(
            unit_id="doc_2",
            content="Java is an object-oriented programming language commonly used for enterprise applications",
            score=0.82
        ),
        TextUnit(
            unit_id="doc_3",
            content="Machine learning algorithms can learn patterns from data without explicit programming",
            score=0.68
        ),
        TextUnit(
            unit_id="doc_4",
            content="The Python programming language was created by Guido van Rossum in 1991",
            score=0.79
        ),
        TextUnit(
            unit_id="doc_5",
            content="Neural networks are a fundamental component of deep learning systems",
            score=0.71
        ),
    ]
    return units


def example_basic_reranking():
    """Example 1: Basic reranking with Ollama"""
    print("="*70)
    print("Example 1: Basic Ollama Reranking")
    print("="*70)
    
    try:
        units = create_test_units()
        query = "What is Python programming?"
        
        print(f"\nQuery: {query}")
        print(f"\nOriginal ranking (by vector similarity):")
        for i, u in enumerate(units, 1):
            print(f"  {i}. [score={u.score:.2f}] {u.unit_id}")
            print(f"      {u.content[:60]}...")
        
        # Rerank using Ollama
        reranker = Reranker("ollama/qllama/bge-reranker-large:latest")
        reranked = reranker.rerank(query, units)
        
        print(f"\nAfter reranking (by Ollama):")
        for i, u in enumerate(reranked, 1):
            print(f"  {i}. [score={u.score:.3f}] {u.unit_id}")
            print(f"      {u.content[:60]}...")
        
        print("\nNote: Ollama runs completely local, no API key needed")
        
    except ImportError:
        print("\n⚠️  ollama library not installed")
        print("Install with: pip install ollama")
        return
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("Make sure Ollama is running and the model is pulled:")
        print("  ollama pull bge-reranker-large")
        return


def example_top_k_reranking():
    """Example 2: Rerank with top_k"""
    print("\n" + "="*70)
    print("Example 2: Rerank with Top-K Selection")
    print("="*70)
    
    try:
        units = create_test_units()
        query = "machine learning"
        
        print(f"\nQuery: {query}")
        print(f"Original units: {len(units)}")
        
        # Rerank and keep top 3
        reranker = Reranker("ollama/qllama/bge-reranker-large:latest")
        top_results = reranker.rerank(query, units, top_k=3)
        
        print(f"\nTop 3 after reranking:")
        for i, u in enumerate(top_results, 1):
            print(f"  {i}. [score={u.score:.3f}] {u.unit_id}")
            print(f"      {u.content}")
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        return


def example_parameter_configuration():
    """Example 3: Different parameter configurations"""
    print("\n" + "="*70)
    print("Example 3: Configuration Options")
    print("="*70)
    
    print("\nURI format (recommended):")
    print('  reranker = Reranker("ollama/qllama/bge-reranker-large:latest")')
    
    print("\nParameter format (with custom settings):")
    print("""
  reranker = Reranker(
      provider="ollama",
      model="bge-reranker-large",
      base_url="http://localhost:11434",  # Custom Ollama server
      timeout=60.0                         # Request timeout
  )
    """)
    
    print("\nAvailable models (pull with 'ollama pull <model>'):")
    print("  - bge-reranker-large (recommended, best quality)")
    print("  - bge-reranker-base (faster, good quality)")
    print("  - jina-reranker-v1-base-en (English only)")


def example_comparison():
    """Example 4: Compare Ollama vs other rerankers"""
    print("\n" + "="*70)
    print("Example 4: Ollama vs Other Rerankers")
    print("="*70)
    
    print("\nOllama Reranker:")
    print("  ✅ Completely local (privacy)")
    print("  ✅ No API key required")
    print("  ✅ Free, unlimited use")
    print("  ✅ Good quality")
    print("  ⚠️  Slower than API-based (Cohere)")
    print("  ⚠️  Requires local resources")
    
    print("\nCohere Reranker:")
    print("  ✅ Highest quality")
    print("  ✅ Very fast (API-based)")
    print("  ❌ Requires API key")
    print("  ❌ Cost per request")
    print("  ❌ Data sent to cloud")
    
    print("\nSentence-Transformers:")
    print("  ✅ Local, free")
    print("  ✅ Good offline support")
    print("  ⚠️  Requires GPU for speed")
    print("  ⚠️  Medium quality")
    
    print("\nChoose based on your needs:")
    print("  - Privacy-sensitive → Ollama or sentence-transformers")
    print("  - Best quality → Cohere")
    print("  - Balanced (local + good) → Ollama")
    print("  - Budget unlimited → Ollama or sentence-transformers")


def example_integration_pipeline():
    """Example 5: Integration with RAG pipeline"""
    print("\n" + "="*70)
    print("Example 5: Integration with RAG Pipeline")
    print("="*70)
    
    print("\nTypical RAG pipeline with Ollama reranking:")
    print("""
    # 1. Retrieve many candidates (recall-focused)
    candidates = vector_store.search(query, top_k=50)
    
    # 2. Rerank locally with Ollama
    reranker = Reranker("ollama/qllama/bge-reranker-large:latest")
    reranked = reranker.rerank(query, candidates, top_k=10)
    
    # 3. Use top reranked results
    context = "\\n".join(u.content for u in reranked)
    response = llm.generate(query, context=context)
    """)
    
    print("\nWhy Ollama for reranking:")
    print("  1. Privacy: All data stays local")
    print("  2. Cost: Free, unlimited reranking")
    print("  3. Quality: Good quality with bge-reranker models")
    print("  4. Simple: Easy setup with Ollama")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Ollama Reranker Examples")
    print("="*70)
    print("\nUses Ollama's local reranking models for privacy-preserving reranking.")
    print("\nPrerequisites:")
    print("  1. Install Ollama: https://ollama.com/")
    print("  2. Pull model: ollama pull bge-reranker-large")
    print("  3. Start service: Ollama should be running")
    
    example_basic_reranking()
    example_top_k_reranking()
    example_parameter_configuration()
    example_comparison()
    example_integration_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
