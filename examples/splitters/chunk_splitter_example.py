#!/usr/bin/env python3
"""
ChunkSplitter Example - Simple Token-Based Splitting for Evaluation

This example demonstrates ChunkSplitter, a simple and reliable token-based
splitter built on Chonkie for evaluation scenarios and general-purpose use.

Why use ChunkSplitter?
- Simple: One line to configure, predictable chunk sizes in TOKENS
- Reliable: Built on Chonkie, a mature chunking library
- Token-aligned: Uses tiktoken (GPT-4 tokenizer) to align with LLM context windows
- Universal: Works with any plain text (Markdown, TXT, PDF extracts, etc.)

Compared to TextSplitter:
- ChunkSplitter: Simple, fixed-size chunks (wraps Chonkie)
- TextSplitter: Smart semantic splitting with table protection (for production)

Use cases:
- Building evaluation/testing pipelines for RAG systems
- General RAG applications without complex requirements
- Quick prototyping and experimentation
"""

from zag.splitters import ChunkSplitter
from zag.schemas.markdown import Markdown
from zag.schemas import UnitMetadata, DocumentMetadata


# Sample document (can be ANY plain text)
SAMPLE_CONTENT = """# AI in Healthcare

Artificial Intelligence is revolutionizing the healthcare industry by providing innovative solutions for diagnosis, treatment, and patient care. Machine learning algorithms can analyze medical images with remarkable accuracy, often matching or exceeding human expert performance.

## Early Detection and Diagnosis

One of the most promising applications of AI in healthcare is early disease detection. Deep learning models trained on millions of medical images can identify patterns that might be invisible to the human eye. For example, AI systems can detect early signs of diabetic retinopathy, various cancers, and cardiovascular diseases from routine screening tests.

These AI-powered diagnostic tools are not meant to replace doctors but to augment their capabilities. By handling routine screenings and flagging potential issues, AI allows healthcare professionals to focus their expertise where it's needed most.

## Personalized Treatment Plans

AI is also transforming how we approach treatment. By analyzing vast amounts of patient data, genetic information, and clinical trial results, AI systems can help doctors create personalized treatment plans. This precision medicine approach considers each patient's unique characteristics, leading to more effective treatments with fewer side effects.

Machine learning models can predict how individual patients might respond to different medications or therapies, helping doctors make more informed decisions about treatment options.

## Challenges and Future Directions

While AI shows tremendous promise in healthcare, important challenges remain. Data privacy and security are paramount concerns, especially when dealing with sensitive medical information. There are also questions about algorithmic bias and ensuring that AI systems work equitably across diverse patient populations.

Looking ahead, the integration of AI in healthcare will likely accelerate. As more data becomes available and algorithms improve, we can expect even more sophisticated applications. The key will be maintaining a human-centered approach, where AI serves as a powerful tool in the hands of skilled healthcare professionals.
"""


def create_sample_doc():
    """Helper function to create a sample document"""
    import uuid
    return Markdown(
        doc_id=str(uuid.uuid4()),
        content=SAMPLE_CONTENT,
        metadata=DocumentMetadata(
            source="evaluation_test.md",
            source_type="local",
            file_type="markdown",
            md5="test"
        )
    )


def example_basic_usage():
    """Example 1: Basic usage for evaluation"""
    print("=" * 70)
    print("Example 1: Basic Usage (Evaluation Scenario)")
    print("=" * 70)
    print()
    
    # Create document
    doc = create_sample_doc()
    print(f"📄 Original document: {len(doc.content)} characters")
    print()
    
    # Create splitter (simple configuration)
    splitter = ChunkSplitter(chunk_size=512)
    
    print("⚙️  Configuration: chunk_size=512 tokens (default overlap=50 tokens)")
    print()
    
    # Split document
    units = doc.split(splitter)
    
    print(f"✅ Split into {len(units)} chunks")
    print()
    
    # Show chunk details
    for i, unit in enumerate(units, 1):
        actual_tokens = unit.metadata.custom.get('actual_tokens', 'N/A')
        print(f"Chunk {i}:")
        print(f"  Characters: {len(unit.content)}")
        print(f"  Tokens: {actual_tokens}")
        print(f"  Chunk index: {unit.metadata.custom.get('chunk_index')}")
        print(f"  Preview: {unit.content[:80]}...")
        print()
    
    print("💡 Note: ChunkSplitter uses Chonkie for reliable token-based splitting")
    print()


def example_production_config():
    """Example 2: Production configuration with overlap"""
    print("=" * 70)
    print("Example 2: Production Configuration (with Overlap)")
    print("=" * 70)
    print()
    
    doc = create_sample_doc()
    
    # Create splitter with overlap for better context
    splitter = ChunkSplitter(
        chunk_size=512,
        chunk_overlap=50,  # 50 tokens overlap between chunks
    )
    
    print("⚙️  Configuration:")
    print("   - chunk_size=512 tokens")
    print("   - chunk_overlap=50 tokens")
    print()
    
    units = doc.split(splitter)
    
    print(f"✅ Split into {len(units)} chunks")
    print()
    
    # Show token counts
    print("📊 Token distribution:")
    for i, unit in enumerate(units, 1):
        actual_tokens = unit.metadata.custom.get('actual_tokens', 'N/A')
        print(f"  Chunk {i}: {actual_tokens} tokens")
    print()
    
    # Show overlap effect
    if len(units) >= 2:
        print("💡 Overlap helps maintain context across chunks")
    print()


def example_different_sizes():
    """Example 3: Comparing different chunk sizes"""
    print("=" * 70)
    print("Example 3: Effect of Different Chunk Sizes")
    print("=" * 70)
    print()
    
    doc = create_sample_doc()
    print(f"📄 Document: {len(doc.content)} characters")
    print()
    
    sizes = [256, 512, 1024]
    
    print("⚙️  Testing different chunk sizes:")
    print()
    
    for size in sizes:
        splitter = ChunkSplitter(chunk_size=size)
        units = doc.split(splitter)
        
        # Calculate actual token counts
        actual_tokens = [u.metadata.custom.get('actual_tokens', 0) for u in units]
        avg_tokens = sum(actual_tokens) / len(actual_tokens) if actual_tokens else 0
        min_tokens = min(actual_tokens) if actual_tokens else 0
        max_tokens = max(actual_tokens) if actual_tokens else 0
        
        print(f"  chunk_size={size} tokens:")
        print(f"    → Generated {len(units)} chunks")
        print(f"    → Actual tokens: avg={avg_tokens:.0f}, min={min_tokens}, max={max_tokens}")
        print()
    
    print("💡 Recommendation:")
    print("   - Small (256-384): More precise retrieval, more chunks to search")
    print("   - Medium (512-768): Balanced for most RAG applications")
    print("   - Large (1024+): More context per chunk, fewer chunks")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("=" * 70)
    print("ChunkSplitter Examples (Built on Chonkie)")
    print("=" * 70)
    print()
    print("ChunkSplitter is a simple, reliable TOKEN-BASED text splitter:")
    print("  • Built on Chonkie for reliability")
    print("  • Evaluation and testing scenarios")
    print("  • General-purpose RAG applications")
    print("  • Any plain text content (Markdown, TXT, PDF extracts, etc.)")
    print()
    print("Note: Uses tiktoken (cl100k_base - GPT-4 tokenizer)")
    print()
    
    try:
        example_basic_usage()
        example_production_config()
        example_different_sizes()
        
        print("=" * 70)
        print("✅ All Examples Complete!")
        print("=" * 70)
        print()
        print("💡 Quick Start for Evaluation:")
        print("   from zag.splitters import ChunkSplitter")
        print("   splitter = ChunkSplitter(chunk_size=512)  # 512 tokens")
        print("   units = doc.split(splitter)")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
