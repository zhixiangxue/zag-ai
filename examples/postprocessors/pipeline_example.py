"""
Pipeline Postprocessor Example

Demonstrates chaining postprocessors using the | operator.
This is the modern, Pythonic way to compose postprocessing pipelines.
"""

from zag.postprocessors import (
    SimilarityFilter,
    Deduplicator,
    TokenCompressor,
)
from zag.schemas.unit import TextUnit
import random


def create_test_units(count: int = 20) -> list[TextUnit]:
    """Create test units with varying scores and some duplicates"""
    units = []
    
    for i in range(count):
        # Create some duplicates (20% chance)
        if i > 0 and random.random() < 0.2:
            content = units[random.randint(0, i-1)].content
        else:
            content = f"Document about topic {i % 5}. " * random.randint(5, 10)
        
        units.append(TextUnit(
            unit_id=f"doc_{i}",
            content=content,
            score=random.uniform(0.3, 0.95)
        ))
    
    return units


def example_basic_pipeline():
    """Example 1: Basic postprocessor pipeline using | operator"""
    print("="*70)
    print("Example 1: Basic Pipeline with | Operator")
    print("="*70)
    
    units = create_test_units(20)
    print(f"\nOriginal: {len(units)} units")
    print(f"  Score range: [{min(u.score for u in units):.2f}, {max(u.score for u in units):.2f}]")
    
    # Create pipeline using | operator (elegant!)
    pipeline = (
        SimilarityFilter(threshold=0.6) | 
        Deduplicator() | 
        TokenCompressor(max_tokens=500)
    )
    
    print(f"\nPipeline: {pipeline}")
    print("\nProcessing steps:")
    print("  1. SimilarityFilter(0.6)  → Remove low-quality results")
    print("  2. Deduplicator()         → Remove duplicate content")
    print("  3. TokenCompressor(500)   → Fit within token limit")
    
    result = pipeline.process("test query", units)
    
    print(f"\nResult: {len(result)} units")
    print(f"  Reduced by: {len(units) - len(result)} units ({(len(units)-len(result))/len(units)*100:.1f}%)")
    print("\n✅ Pipeline executed successfully!")


def example_reranker_pipeline():
    """Example 2: Pipeline concept (without real reranker)"""
    print("\n" + "="*70)
    print("Example 2: Multi-Step Pipeline")
    print("="*70)
    
    units = create_test_units(30)
    query = "machine learning algorithms"
    
    print(f"\nQuery: '{query}'")
    print(f"Initial: {len(units)} units")
    
    # Typical RAG pipeline: filter → dedup → compress
    pipeline = (
        SimilarityFilter(threshold=0.6) |
        Deduplicator() |
        SimilarityFilter(threshold=0.75) |  # Progressive filtering
        TokenCompressor(max_tokens=1500)
    )
    
    print(f"\nPipeline: {pipeline}")
    
    result = pipeline.process(query, units)
    
    print(f"\nFinal: {len(result)} units")
    print("  ✓ First filter removed low-quality results")
    print("  ✓ Deduplicator removed duplicates")
    print("  ✓ Second filter was more aggressive")
    print("  ✓ Compressor fit to context window")


def example_progressive_filtering():
    """Example 3: Progressive filtering pipeline"""
    print("\n" + "="*70)
    print("Example 3: Progressive Filtering")
    print("="*70)
    
    units = create_test_units(50)
    
    print(f"\nStarting with: {len(units)} units\n")
    
    # Progressive filtering: each step reduces results
    step1 = SimilarityFilter(threshold=0.5)
    step2 = Deduplicator()
    step3 = SimilarityFilter(threshold=0.7)  # More aggressive
    step4 = TokenCompressor(max_tokens=1000)
    
    # Build pipeline step by step
    pipeline = step1 | step2 | step3 | step4
    
    print(f"Pipeline: {pipeline}\n")
    
    # Manual step-by-step to show progress
    result1 = step1.process("query", units)
    print(f"After step 1 (filter 0.5):  {len(result1)} units")
    
    result2 = step2.process("query", result1)
    print(f"After step 2 (dedup):       {len(result2)} units")
    
    result3 = step3.process("query", result2)
    print(f"After step 3 (filter 0.7):  {len(result3)} units")
    
    result4 = step4.process("query", result3)
    print(f"After step 4 (compress):    {len(result4)} units")
    
    print(f"\nTotal reduction: {len(units)} → {len(result4)} units")
    print(f"Kept: {len(result4)/len(units)*100:.1f}% of original")


def example_reusable_pipelines():
    """Example 4: Reusable pipeline configurations"""
    print("\n" + "="*70)
    print("Example 4: Reusable Pipeline Configurations")
    print("="*70)
    
    # Define standard pipelines for different scenarios
    
    # 1. Light processing (few results)
    light_pipeline = SimilarityFilter(0.5) | TokenCompressor(3000)
    
    # 2. Standard processing (moderate results)
    standard_pipeline = (
        SimilarityFilter(0.65) |
        Deduplicator() |
        TokenCompressor(2000)
    )
    
    # 3. Aggressive processing (many results)
    aggressive_pipeline = (
        SimilarityFilter(0.75) |
        Deduplicator() |
        SimilarityFilter(0.85) |  # Double filtering
        TokenCompressor(1000)
    )
    
    print("\nDefined 3 standard pipelines:\n")
    print(f"1. Light:      {light_pipeline}")
    print(f"2. Standard:   {standard_pipeline}")
    print(f"3. Aggressive: {aggressive_pipeline}")
    
    # Apply to different scenarios
    units = create_test_units(40)
    
    print(f"\n\nApplying to {len(units)} units:\n")
    
    result1 = light_pipeline.process("query", units)
    print(f"Light:      {len(units)} → {len(result1)} units")
    
    result2 = standard_pipeline.process("query", units)
    print(f"Standard:   {len(units)} → {len(result2)} units")
    
    result3 = aggressive_pipeline.process("query", units)
    print(f"Aggressive: {len(units)} → {len(result3)} units")


def example_rag_integration():
    """Example 5: Complete RAG pipeline"""
    print("\n" + "="*70)
    print("Example 5: Complete RAG Pipeline Integration")
    print("="*70)
    
    print("\nTypical RAG workflow with postprocessor pipeline:")
    print("""
    from zag.retrievers import VectorRetriever
    from zag.postprocessors import SimilarityFilter, Deduplicator, TokenCompressor
    
    # 1. Define retrieval
    retriever = VectorRetriever(vector_store)
    
    # 2. Define postprocessing pipeline using | operator
    postprocessor = (
        SimilarityFilter(threshold=0.7) |
        Deduplicator() |
        TokenCompressor(max_tokens=2000)
    )
    
    # 3. Execute
    def rag_search(query: str) -> list[TextUnit]:
        # Retrieve
        units = retriever.retrieve(query, top_k=50)
        
        # Postprocess with pipeline
        processed = postprocessor.process(query, units)
        
        return processed
    
    # 4. Use in your application
    results = rag_search("What is machine learning?")
    context = "\\n".join(u.content for u in results)
    response = llm.generate(query, context=context)
    """)
    
    # Simulate
    print("\nSimulation:")
    raw_units = create_test_units(50)
    print(f"  1. Retrieved: {len(raw_units)} units")
    
    pipeline = (
        SimilarityFilter(threshold=0.7) |
        Deduplicator() |
        TokenCompressor(max_tokens=2000)
    )
    
    processed = pipeline.process("query", raw_units)
    print(f"  2. After pipeline: {len(processed)} units")
    print(f"  3. Reduction: {(len(raw_units)-len(processed))/len(raw_units)*100:.1f}%")
    print(f"  ✓ Ready for LLM consumption")


def example_async_pipeline():
    """Example 6: Async pipeline execution"""
    print("\n" + "="*70)
    print("Example 6: Async Pipeline (for async postprocessors)")
    print("="*70)
    
    print("""
Pipelines support async execution automatically:

    import asyncio
    from zag.postprocessors import LLMSelector, Reranker, TokenCompressor
    
    # Pipeline with async components (LLMSelector is async)
    async_pipeline = (
        Reranker("...") |
        LLMSelector(llm_uri="bailian/qwen-plus") |  # Async
        TokenCompressor(max_tokens=1000)
    )
    
    # Use with asyncio
    async def process():
        result = await async_pipeline.aprocess(query, units)
        return result
    
    # Run
    result = asyncio.run(process())

Note: All postprocessors support both sync and async execution.
The pipeline will use async when needed automatically.
    """)


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Postprocessor Pipeline Examples")
    print("="*70)
    print("\nModern Pythonic way to compose postprocessing pipelines using | operator.")
    print("\nKey benefits:")
    print("  ✓ Elegant and readable syntax")
    print("  ✓ Easy to compose and reorder")
    print("  ✓ Type-safe and IDE-friendly")
    print("  ✓ Supports both sync and async execution")
    
    example_basic_pipeline()
    example_reranker_pipeline()
    example_progressive_filtering()
    example_reusable_pipelines()
    example_rag_integration()
    example_async_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  - Try composing your own pipelines")
    print("  - Experiment with different orderings")
    print("  - Build reusable configurations for your use cases")
    print("="*70)


if __name__ == "__main__":
    main()
