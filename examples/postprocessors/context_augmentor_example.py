"""
ContextAugmentor Example

Demonstrates augmenting units with surrounding context.
"""

from zag.postprocessors.augmentors import ContextAugmentor
from zag.schemas.unit import TextUnit


def create_document_chain() -> list[TextUnit]:
    """Create a chain of units representing a document"""
    content_parts = [
        "Introduction to Machine Learning",
        "Machine learning is a subset of artificial intelligence.",
        "It enables computers to learn from data without explicit programming.",
        "Types of Machine Learning",
        "There are three main types: supervised, unsupervised, and reinforcement learning.",
        "Supervised learning uses labeled data to train models.",
        "Applications of Machine Learning",
        "ML is used in recommendation systems, image recognition, and natural language processing.",
        "These applications have transformed many industries.",
        "Conclusion",
        "Machine learning continues to evolve and impact our daily lives.",
    ]
    
    units = []
    for i, content in enumerate(content_parts):
        unit = TextUnit(
            unit_id=f"chunk_{i}",
            content=content,
            score=0.8
        )
        
        # Set up chain links
        if i > 0:
            unit.prev_unit_id = f"chunk_{i-1}"
        if i < len(content_parts) - 1:
            unit.next_unit_id = f"chunk_{i+1}"
        
        units.append(unit)
    
    return units


def example_basic_augmentation():
    """Example 1: Basic context augmentation"""
    print("="*70)
    print("Example 1: Basic Context Augmentation")
    print("="*70)
    
    all_units = create_document_chain()
    
    # Simulate retrieval: got chunks 2, 5, 7
    retrieved = [all_units[2], all_units[5], all_units[7]]
    
    print(f"\nRetrieved units: {len(retrieved)}")
    for u in retrieved:
        print(f"  {u.unit_id}: {u.content}")
    
    # Augment with neighboring chunks (window_size=1)
    augmentor = ContextAugmentor(window_size=1)
    augmented = augmentor.process("machine learning", retrieved)
    
    print(f"\nAfter augmentation (window_size=1): {len(augmented)} units")
    for u in sorted(augmented, key=lambda x: x.unit_id):
        marker = "★" if u in retrieved else "+"
        print(f"  {marker} {u.unit_id}: {u.content}")
    
    print("\n★ = original retrieval, + = added context")


def example_different_window_sizes():
    """Example 2: Different window sizes"""
    print("\n" + "="*70)
    print("Example 2: Different Window Sizes")
    print("="*70)
    
    all_units = create_document_chain()
    retrieved = [all_units[5]]  # Just one middle chunk
    
    print(f"\nOriginal retrieval: {retrieved[0].unit_id}")
    print(f"Content: {retrieved[0].content}")
    
    for window in [0, 1, 2, 3]:
        augmentor = ContextAugmentor(window_size=window)
        augmented = augmentor.process("query", retrieved)
        
        unit_ids = sorted([u.unit_id for u in augmented])
        print(f"\nWindow size {window}: {len(augmented)} units")
        print(f"  Units: {', '.join(unit_ids)}")


def example_at_boundaries():
    """Example 3: Augmentation at document boundaries"""
    print("\n" + "="*70)
    print("Example 3: Augmentation at Document Boundaries")
    print("="*70)
    
    all_units = create_document_chain()
    
    # Test at start of document
    start_unit = [all_units[0]]
    augmentor = ContextAugmentor(window_size=2)
    result = augmentor.process("query", start_unit)
    
    print(f"\nAt document start (chunk_0):")
    print(f"  Window size: 2")
    print(f"  Result: {len(result)} units")
    print(f"  Units: {', '.join(sorted([u.unit_id for u in result]))}")
    print("  Note: Only forward context available")
    
    # Test at end of document
    end_unit = [all_units[-1]]
    result = augmentor.process("query", end_unit)
    
    print(f"\nAt document end (chunk_{len(all_units)-1}):")
    print(f"  Window size: 2")
    print(f"  Result: {len(result)} units")
    print(f"  Units: {', '.join(sorted([u.unit_id for u in result]))}")
    print("  Note: Only backward context available")


def example_multiple_retrievals():
    """Example 4: Multiple retrieved units with overlap"""
    print("\n" + "="*70)
    print("Example 4: Multiple Retrievals with Overlapping Context")
    print("="*70)
    
    all_units = create_document_chain()
    
    # Retrieve close-by units
    retrieved = [all_units[3], all_units[5]]
    
    print(f"\nRetrieved units:")
    for u in retrieved:
        print(f"  {u.unit_id}: {u.content}")
    
    augmentor = ContextAugmentor(window_size=1)
    augmented = augmentor.process("query", retrieved)
    
    print(f"\nAfter augmentation (window_size=1): {len(augmented)} units")
    print("Note: Overlapping context is deduplicated")
    for u in sorted(augmented, key=lambda x: x.unit_id):
        marker = "★" if u in retrieved else "+"
        print(f"  {marker} {u.unit_id}")


def example_integration_pipeline():
    """Example 5: Integration with RAG pipeline"""
    print("\n" + "="*70)
    print("Example 5: Integration with RAG Pipeline")
    print("="*70)
    
    print("\nTypical usage in RAG pipeline:")
    print("""
    # 1. Retrieve relevant chunks
    units = vector_store.search(query, top_k=5)
    
    # 2. Augment with surrounding context for better understanding
    augmentor = ContextAugmentor(window_size=1)
    augmented_units = augmentor.process(query, units)
    
    # 3. Use augmented context in LLM
    context = "\\n".join(u.content for u in augmented_units)
    llm_response = llm.generate(query, context=context)
    """)
    
    # Simulate
    all_units = create_document_chain()
    retrieved = [all_units[1], all_units[7]]
    
    augmentor = ContextAugmentor(window_size=1)
    augmented = augmentor.process("machine learning types", retrieved)
    
    print(f"\nSimulation:")
    print(f"  Retrieved: {len(retrieved)} chunks")
    print(f"  After augmentation: {len(augmented)} chunks")
    print(f"  Added context: {len(augmented) - len(retrieved)} chunks")
    print(f"  Benefit: More complete information for LLM")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("ContextAugmentor Examples")
    print("="*70)
    print("\nAugments retrieved units with surrounding context.")
    print("Use cases:")
    print("  - Provide more complete information to LLM")
    print("  - Recover context lost during chunking")
    print("  - Improve answer quality in RAG systems")
    
    example_basic_augmentation()
    example_different_window_sizes()
    example_at_boundaries()
    example_multiple_retrievals()
    example_integration_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
