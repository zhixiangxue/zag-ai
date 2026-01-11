"""
LanceDB Vector Store Example

This example demonstrates how to use LanceDB as a vector store.
LanceDB is a lightweight, embedded vector database built on Lance format.

Key features:
- Embedded database (no server needed, like SQLite)
- Fast vector search with IVF-PQ index
- Columnar storage format (Apache Arrow)
- Zero-copy reads for high performance
- Multi-modal support

Installation:
    pip install lancedb

References:
    - Quickstart: https://docs.lancedb.com/quickstart
    - Python SDK: https://lancedb.github.io/lancedb/python/python/
"""

from zag.schemas.unit import TextUnit
from zag.schemas.base import UnitMetadata
from zag.storages.vector import LanceDBVectorStore
from zag import Embedder


def main():
    """Main example function"""
    
    # Initialize embedder (using Ollama)
    print("Initializing Ollama embedder...")
    embedder = Embedder("ollama/jina/jina-embeddings-v2-base-en:latest")
    print(f"✓ Embedder dimension: {embedder.dimension}\n")
    
    # ============================================================
    # EXAMPLE 1: Local Mode (Recommended)
    # ============================================================
    print("="*70)
    print("EXAMPLE 1: Local Mode (Embedded Database)")
    print("="*70)
    
    # Create local persistent store
    store = LanceDBVectorStore.local(
        path="./tmp/lancedb_data",
        table_name="example_docs",
        embedder=embedder
    )
    print("✓ Created local LanceDB store at ./tmp/lancedb_data\n")
    
    # Create sample documents
    units = [
        TextUnit(
            unit_id="doc1",
            content="LanceDB is a fast embedded vector database built on Lance columnar format.",
            metadata=UnitMetadata(
                context_path="docs/lancedb.md/intro"
            )
        ),
        TextUnit(
            unit_id="doc2",
            content="Apache Arrow provides zero-copy reads for efficient data processing.",
            metadata=UnitMetadata(
                context_path="docs/arrow.md/features"
            )
        ),
        TextUnit(
            unit_id="doc3",
            content="Vector databases enable semantic search using embedding models.",
            metadata=UnitMetadata(
                context_path="docs/vectors.md/concepts"
            )
        ),
    ]
    
    # Add documents
    print("Adding documents to LanceDB...")
    store.add(units)
    print(f"✓ Added {len(units)} documents")
    print(f"✓ Total vectors in store: {store.count()}\n")
    
    # Search
    print("Searching for 'vector database'...")
    results = store.search("vector database", top_k=2)
    print(f"Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}] {unit.content[:60]}...")
        if hasattr(unit, 'score'):
            print(f"     Score: {unit.score:.4f}")
    print()
    
    # Get specific document
    print("Getting document by ID...")
    retrieved = store.get(["doc2"])
    if retrieved:
        print(f"✓ Retrieved: [{retrieved[0].unit_id}] {retrieved[0].content}")
    print()
    
    # Update document
    print("Updating document...")
    updated_unit = TextUnit(
        unit_id="doc1",
        content="LanceDB is an ultra-fast embedded vector database with columnar storage.",
        metadata=UnitMetadata(
            context_path="docs/lancedb_v2.md/intro/updated"
        )
    )
    store.update(updated_unit)
    print("✓ Document updated\n")
    
    # Search again
    print("Searching after update...")
    results = store.search("fast database", top_k=1)
    if results:
        print(f"✓ Found: {results[0].content}")
    print()
    
    # Delete document
    print("Deleting document...")
    store.delete(["doc3"])
    print(f"✓ Document deleted")
    print(f"✓ Remaining vectors: {store.count()}\n")
    
    # ============================================================
    # EXAMPLE 2: Batch Operations
    # ============================================================
    print("="*70)
    print("EXAMPLE 2: Batch Operations")
    print("="*70)
    
    # Create new store for batch demo
    batch_store = LanceDBVectorStore.local(
        path="./tmp/lancedb_data",
        table_name="batch_demo",
        embedder=embedder
    )
    
    # Batch add
    print("Batch adding 100 documents...")
    batch_units = []
    for i in range(100):
        unit = TextUnit(
            unit_id=f"batch_{i}",
            content=f"This is batch document number {i} about vector search and embeddings.",
            metadata=UnitMetadata(
                context_path=f"batch/doc_{i}"
            )
        )
        batch_units.append(unit)
    
    batch_store.add(batch_units)
    print(f"✓ Added {len(batch_units)} documents in batch")
    print(f"✓ Total vectors: {batch_store.count()}\n")
    
    # Batch search
    print("Performing batch search...")
    results = batch_store.search("vector search", top_k=5)
    print(f"✓ Found top {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. {unit.unit_id}: {unit.content[:50]}...")
    print()
    
    # Clear store
    print("Clearing batch store...")
    batch_store.clear()
    print(f"✓ Store cleared, remaining vectors: {batch_store.count()}\n")
    
    # ============================================================
    # EXAMPLE 3: Cloud Mode (Optional)
    # ============================================================
    print("="*70)
    print("EXAMPLE 3: Cloud Mode (LanceDB Cloud)")
    print("="*70)
    print("To use LanceDB Cloud:")
    print("  1. Sign up at https://cloud.lancedb.com/")
    print("  2. Get your API key (format: ldb_...)")
    print("  3. Use the following code:")
    print()
    print("  store = LanceDBVectorStore.cloud(")
    print("      url='db://my_database',")
    print("      api_key='ldb_...',")
    print("      table_name='my_table',")
    print("      embedder=embedder")
    print("  )")
    print()
    
    print("="*70)
    print("✓ All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
