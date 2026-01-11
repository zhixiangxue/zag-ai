"""
Reranker Example

Demonstrates reranking search results using cross-encoder models.

Requirements:
    pip install sentence-transformers
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
    """Example 1: Basic reranking"""
    print("="*70)
    print("Example 1: Basic Reranking")
    print("="*70)
    
    try:
        units = create_test_units()
        query = "What is Python programming?"
        
        print(f"\nQuery: {query}")
        print(f"\nOriginal ranking (by vector similarity):")
        for i, u in enumerate(units, 1):
            print(f"  {i}. [score={u.score:.2f}] {u.unit_id}")
            print(f"      {u.content[:60]}...")
        
        # Rerank using cross-encoder
        reranker = Reranker("sentence_transformers/cross-encoder/ms-marco-MiniLM-L-12-v2")
        reranked = reranker.rerank(query, units)
        
        print(f"\nAfter reranking (by cross-encoder):")
        for i, u in enumerate(reranked, 1):
            print(f"  {i}. [score={u.score:.3f}] {u.unit_id}")
            print(f"      {u.content[:60]}...")
        
        print("\nNote: Reranking provides more accurate relevance scores")
        
    except ImportError:
        print("\n⚠️  sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
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
        reranker = Reranker("sentence_transformers/cross-encoder/ms-marco-MiniLM-L-12-v2")
        top_results = reranker.rerank(query, units, top_k=3)
        
        print(f"\nTop 3 after reranking:")
        for i, u in enumerate(top_results, 1):
            print(f"  {i}. [score={u.score:.3f}] {u.unit_id}")
            print(f"      {u.content}")
        
    except ImportError:
        print("\n⚠️  sentence-transformers not installed")
        return


def example_parameter_configuration():
    """Example 3: Different parameter configurations"""
    print("\n" + "="*70)
    print("Example 3: Parameter Configuration")
    print("="*70)
    
    print("\nURI format (recommended):")
    print('  reranker = Reranker("sentence_transformers/cross-encoder/ms-marco-MiniLM-L-12-v2")')
    
    print("\nParameter format (with custom settings):")
    print("""
  reranker = Reranker(
      provider="sentence_transformers",
      model="cross-encoder/ms-marco-MiniLM-L-12-v2",
      device="cuda",      # Use GPU if available
      batch_size=16       # Process in batches
  )
    """)
    
    print("\nDifferent models:")
    print("  - ms-marco-MiniLM-L-12-v2 (fast, good quality)")
    print("  - ms-marco-electra-base (slower, better quality)")
    print("  - ms-marco-TinyBERT-L-2-v2 (very fast, lower quality)")


def example_comparison_with_without():
    """Example 4: Compare results with/without reranking"""
    print("\n" + "="*70)
    print("Example 4: Impact of Reranking")
    print("="*70)
    
    try:
        units = create_test_units()
        query = "Python programming language"
        
        # Without reranking (just sort by original score)
        by_original = sorted(units, key=lambda u: u.score, reverse=True)
        
        print(f"\nQuery: {query}")
        print("\nWithout reranking (vector similarity only):")
        print(f"  Top result: {by_original[0].unit_id}")
        print(f"  Content: {by_original[0].content[:80]}...")
        
        # With reranking
        reranker = Reranker("sentence_transformers/cross-encoder/ms-marco-MiniLM-L-12-v2")
        reranked = reranker.rerank(query, units, top_k=1)
        
        print("\nWith reranking (cross-encoder):")
        print(f"  Top result: {reranked[0].unit_id}")
        print(f"  Content: {reranked[0].content[:80]}...")
        
        if by_original[0].unit_id != reranked[0].unit_id:
            print("\n✓ Reranking changed the top result (more accurate!)")
        
    except ImportError:
        print("\n⚠️  sentence-transformers not installed")
        return


def example_integration_pipeline():
    """Example 5: Integration with RAG pipeline"""
    print("\n" + "="*70)
    print("Example 5: Integration with RAG Pipeline")
    print("="*70)
    
    print("\nTypical RAG pipeline with reranking:")
    print("""
    # 1. Retrieve many candidates (recall-focused)
    candidates = vector_store.search(query, top_k=50)
    
    # 2. Rerank to improve precision
    reranker = Reranker("sentence_transformers/cross-encoder/ms-marco-MiniLM-L-12-v2")
    reranked = reranker.rerank(query, candidates, top_k=10)
    
    # 3. Use top reranked results
    context = "\\n".join(u.content for u in reranked)
    response = llm.generate(query, context=context)
    """)
    
    print("\nWhy reranking improves RAG:")
    print("  1. Vector search optimizes for fast retrieval (recall)")
    print("  2. Cross-encoder provides accurate relevance (precision)")
    print("  3. Best of both: fast retrieval + accurate ranking")
    print("  4. Significantly improves answer quality")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Reranker Examples")
    print("="*70)
    print("\nReranks search results using cross-encoder models.")
    print("Use cases:")
    print("  - Improve relevance ranking in RAG systems")
    print("  - Two-stage retrieval (fast vector search + accurate reranking)")
    print("  - Enhance answer quality")
    
    example_basic_reranking()
    example_top_k_reranking()
    example_parameter_configuration()
    example_comparison_with_without()
    example_integration_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
