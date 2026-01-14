"""
LLM Selector Example

Demonstrates extracting relevant passages from long documents using LLM.

Requirements:
    pip install chakpy
"""

import asyncio
from zag.postprocessors import LLMSelector
from zag.schemas.unit import TextUnit


def create_test_units() -> list[TextUnit]:
    """Create test units with long content"""
    units = [
        TextUnit(
            unit_id="doc_1",
            content="""
Python Programming Language Overview

Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.
It emphasizes code readability with significant whitespace. Python supports multiple programming
paradigms including procedural, object-oriented, and functional programming.

Key Features:
- Simple and easy to learn syntax
- Extensive standard library
- Cross-platform compatibility
- Dynamic typing
- Automatic memory management

Popular Use Cases:
Python is widely used in web development (Django, Flask), data science (pandas, NumPy),
machine learning (TensorFlow, PyTorch), automation, and scripting. Its versatility makes
it one of the most popular programming languages in the world.

Community and Ecosystem:
Python has a large and active community with millions of developers worldwide. The Python
Package Index (PyPI) hosts hundreds of thousands of third-party packages.
            """.strip(),
            score=0.85
        ),
        TextUnit(
            unit_id="doc_2",
            content="""
Java Enterprise Application Development

Java is a robust, object-oriented programming language designed for enterprise applications.
It was developed by James Gosling at Sun Microsystems in 1995. Java's "write once, run anywhere"
philosophy makes it highly portable.

Enterprise Features:
- Strong type safety
- Extensive libraries for enterprise integration
- Multi-threading support
- Mature ecosystem with Spring Framework
- JDBC for database connectivity

Java is commonly used in:
- Large-scale enterprise systems
- Android mobile development
- Web servers and application servers
- Financial services applications
- Big data processing with Hadoop

The Java Virtual Machine (JVM) allows Java bytecode to run on any platform that has a JVM
implementation, ensuring true platform independence.
            """.strip(),
            score=0.78
        ),
        TextUnit(
            unit_id="doc_3",
            content="""
History of Python Programming Language

Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI)
in the Netherlands. The language was officially released in 1991 as Python 0.9.0.

Etymology:
The name "Python" was inspired by the British comedy series "Monty Python's Flying Circus",
not by the snake. Guido van Rossum was reading scripts from the show while implementing Python.

Major Milestones:
- Python 1.0 (1994): First official release with lambda, map, filter, reduce
- Python 2.0 (2000): Introduced list comprehensions and garbage collection
- Python 3.0 (2008): Major revision, not backward compatible with Python 2.x
- Python 2 End of Life (2020): Python 2.7 support officially ended

Design Philosophy:
The Zen of Python, written by Tim Peters, encapsulates Python's design principles:
"Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex."

Guido van Rossum served as Python's "Benevolent Dictator For Life" (BDFL) until 2018, when
he stepped down and a steering council was formed to guide Python's development.
            """.strip(),
            score=0.92
        ),
    ]
    return units


async def example_basic_selection():
    """Example 1: Basic passage selection"""
    print("="*70)
    print("Example 1: Basic LLM Selection")
    print("="*70)
    
    units = create_test_units()
    query = "Who created Python programming language?"
    
    print(f"\nQuery: {query}")
    print(f"\nOriginal units: {len(units)} documents")
    for i, u in enumerate(units, 1):
        print(f"  {i}. {u.unit_id} ({len(u.content)} chars)")
    
    # Create selector
    selector = LLMSelector(llm_uri="ollama/qwen2.5:7b")
    
    # Process
    print("\n🔄 Processing with LLM...")
    selected = await selector.aprocess(query, units)
    
    print(f"\n✅ Selected units: {len(selected)}")
    for i, u in enumerate(selected, 1):
        print(f"\n  {i}. {u.unit_id} (score: {u.score:.2f})")
        print(f"     Extracted content ({len(u.content)} chars):")
        print(f"     ---")
        print(f"     {u.content}")
        print(f"     ---")


async def example_filtering():
    """Example 2: Filtering irrelevant documents"""
    print("\n" + "="*70)
    print("Example 2: Filtering Irrelevant Documents")
    print("="*70)
    
    units = create_test_units()
    query = "Tell me about Java programming"
    
    print(f"\nQuery: {query}")
    print(f"Original units: {len(units)}")
    
    selector = LLMSelector(llm_uri="ollama/qwen2.5:7b")
    selected = await selector.aprocess(query, units)
    
    print(f"\nFiltered to: {len(selected)} units")
    print("\nRemaining units:")
    for u in selected:
        print(f"  - {u.unit_id}")


async def example_pipeline():
    """Example 3: Integration with pipeline"""
    print("\n" + "="*70)
    print("Example 3: Pipeline Integration")
    print("="*70)
    
    print("\nTypical RAG pipeline with LLMSelector:")
    print("""
    # 1. Retrieve candidates
    candidates = vector_store.search(query, top_k=50)
    
    # 2. Rerank
    reranker = Reranker("cohere/rerank-english-v3.0")
    reranked = reranker.rerank(query, candidates, top_k=10)
    
    # 3. Extract relevant passages
    selector = LLMSelector(llm_uri="ollama/qwen2.5:7b")
    selected = selector.process(query, reranked)
    
    # 4. Use extracted passages for generation
    context = "\\n\\n".join(u.content for u in selected)
    response = llm.generate(query, context=context)
    """)
    
    print("\nBenefits:")
    print("  ✅ Removes irrelevant content")
    print("  ✅ Extracts precise passages")
    print("  ✅ Reduces token usage")
    print("  ✅ Improves answer quality")


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("LLM Selector Examples")
    print("="*70)
    print("\nExtracts relevant passages from long documents using LLM.")
    
    await example_basic_selection()
    await example_filtering()
    await example_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
