"""
Cohere Reranker Example

Demonstrates reranking search results using Cohere's Rerank API.

Requirements:
    pip install cohere
    
Setup:
    1. Get API key from https://dashboard.cohere.com/api-keys
    2. Set environment variable: export COHERE_API_KEY="your-api-key"
    3. Or pass api_key parameter directly
"""

import os
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
    """Example 1: Basic reranking with Cohere"""
    print("="*70)
    print("Example 1: Basic Cohere Reranking")
    print("="*70)
    
    try:
        units = create_test_units()
        query = "What is Python programming?"
        
        print(f"\nQuery: {query}")
        print(f"\nOriginal ranking (by vector similarity):")
        for i, u in enumerate(units, 1):
            print(f"  {i}. [score={u.score:.2f}] {u.unit_id}")
            print(f"      {u.content[:60]}...")
        
        # Rerank using Cohere
        # API key will be read from COHERE_API_KEY environment variable
        reranker = Reranker("cohere/rerank-english-v3.0")
        reranked = reranker.rerank(query, units)
        
        print(f"\nAfter reranking (by Cohere):")
        for i, u in enumerate(reranked, 1):
            print(f"  {i}. [score={u.score:.3f}] {u.unit_id}")
            print(f"      {u.content[:60]}...")
        
        print("\nNote: Cohere provides production-grade reranking")
        
    except ImportError:
        print("\n⚠️  cohere library not installed")
        print("Install with: pip install cohere")
        return
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("Make sure COHERE_API_KEY environment variable is set")
        return


def example_multilingual_reranking():
    """Example 2: Multilingual reranking"""
    print("\n" + "="*70)
    print("Example 2: Multilingual Reranking")
    print("="*70)
    
    try:
        # Create multilingual test units
        units = [
            TextUnit(
                unit_id="doc_1",
                content="Machine learning is a subset of artificial intelligence",
                score=0.80
            ),
            TextUnit(
                unit_id="doc_2",
                content="机器学习是人工智能的一个分支,它使计算机能够从数据中学习",
                score=0.75
            ),
            TextUnit(
                unit_id="doc_3",
                content="El aprendizaje automático permite a las computadoras aprender de los datos",
                score=0.70
            ),
        ]
        
        query = "What is machine learning?"
        
        print(f"\nQuery: {query}")
        print(f"\nOriginal ranking:")
        for i, u in enumerate(units, 1):
            print(f"  {i}. [score={u.score:.2f}] {u.unit_id}")
            print(f"      {u.content[:60]}...")
        
        # Use multilingual model
        reranker = Reranker("cohere/rerank-multilingual-v3.0")
        reranked = reranker.rerank(query, units, top_k=3)
        
        print(f"\nAfter multilingual reranking:")
        for i, u in enumerate(reranked, 1):
            print(f"  {i}. [score={u.score:.3f}] {u.unit_id}")
            print(f"      {u.content[:60]}...")
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        return


def example_with_api_key():
    """Example 3: Passing API key directly"""
    print("\n" + "="*70)
    print("Example 3: Passing API Key Directly")
    print("="*70)
    
    print("\nYou can pass API key directly instead of using environment variable:")
    print("""
    reranker = Reranker(
        provider="cohere",
        model="rerank-english-v3.0",
        api_key="your-cohere-api-key"
    )
    
    # Or using URI with parameters
    reranker = Reranker(
        "cohere/rerank-english-v3.0",
        api_key="your-cohere-api-key"
    )
    """)


def example_parameter_configuration():
    """Example 4: Different parameter configurations"""
    print("\n" + "="*70)
    print("Example 4: Configuration Options")
    print("="*70)
    
    print("\nURI format (recommended):")
    print('  reranker = Reranker("cohere/rerank-english-v3.0")')
    
    print("\nParameter format (with custom settings):")
    print("""
  reranker = Reranker(
      provider="cohere",
      model="rerank-english-v3.0",
      api_key="your-key",              # Optional if env var set
      max_chunks_per_doc=10            # For long documents
  )
    """)
    
    print("\nAvailable models:")
    print("  - rerank-english-v3.0 (latest, best quality)")
    print("  - rerank-multilingual-v3.0 (supports 100+ languages)")
    print("  - rerank-english-v2.0 (legacy)")
    print("  - rerank-multilingual-v2.0 (legacy)")


def example_comparison():
    """Example 5: Compare Cohere vs local models"""
    print("\n" + "="*70)
    print("Example 5: Cohere vs Local Models")
    print("="*70)
    
    print("\nCohere Reranker:")
    print("  ✅ Production-grade quality")
    print("  ✅ Fast (API-based, no local compute)")
    print("  ✅ Multilingual support")
    print("  ✅ No GPU required")
    print("  ❌ Requires API key and internet")
    print("  ❌ Cost per request")
    
    print("\nLocal Models (sentence-transformers):")
    print("  ✅ Free, unlimited use")
    print("  ✅ Works offline")
    print("  ✅ Data privacy (stays local)")
    print("  ❌ Requires GPU for good performance")
    print("  ❌ Lower quality than Cohere")
    
    print("\nChoose based on your needs:")
    print("  - Production RAG systems → Cohere")
    print("  - Privacy-sensitive data → Local models")
    print("  - High volume, low budget → Local models")
    print("  - Best quality needed → Cohere")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Cohere Reranker Examples")
    print("="*70)
    print("\nUses Cohere's production-grade Rerank API for high-quality reranking.")
    
    # Check if API key is set
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("\n⚠️  Warning: COHERE_API_KEY environment variable not set")
        print("Some examples may fail. Get your API key from:")
        print("https://dashboard.cohere.com/api-keys")
    
    example_basic_reranking()
    example_multilingual_reranking()
    example_with_api_key()
    example_parameter_configuration()
    example_comparison()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
