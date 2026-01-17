""" 
Milvus Vector Store Example

This example demonstrates how to use Milvus as a vector store.
Milvus is a high-performance, cloud-native vector database built for scale.

Key features:
- High-performance vector search at scale
- Distributed architecture (separates compute and storage)
- Hardware acceleration (CPU/GPU)
- Multi-tenancy support
- Advanced metadata filtering
- Hybrid search (dense + sparse vectors)
- Full-text search with BM25

Deployment modes:
- Milvus Lite: Embedded mode (Linux/macOS only, < 1M vectors)
- Standalone: Single-node server (Docker recommended)
- Cluster: Distributed deployment
- Zilliz Cloud: Fully managed service

**Windows Users**: Milvus Lite is NOT supported on Windows!
Please use Docker (server mode) or Zilliz Cloud instead.

Installation:
    pip install pymilvus
    
    # For Docker:
    docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

References:
    - Official Site: https://milvus.io/
    - Python SDK: https://github.com/milvus-io/pymilvus
    - Docs: https://milvus.io/docs
"""

from zag.schemas.unit import TextUnit
from zag.schemas import UnitMetadata
from zag.storages.vector import MilvusVectorStore
from zag import Embedder


def main():
    """Main example function"""
    import platform
    
    # Initialize embedder (using Ollama)
    print("Initializing Ollama embedder...")
    embedder = Embedder("ollama/jina/jina-embeddings-v2-base-en:latest")
    print(f"✓ Embedder dimension: {embedder.dimension}\n")
    
    # ============================================================
    # EXAMPLE 1: Server Mode (Recommended for Windows)
    # ============================================================
    print("="*70)
    print("EXAMPLE 1: Server Mode (Docker) - Works on All Platforms")
    print("="*70)
    
    # Check if running on Windows
    if platform.system() == "Windows":
        print("\n⚠️  Note: You are on Windows. Milvus Lite is NOT supported.")
        print("Please ensure Milvus server is running.\n")
        print("💡 Recommended: Use WSL2 (much easier than Docker Desktop!):\n")
        print("  # In WSL2 (Ubuntu):")
        print("  curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh")
        print("  bash standalone_embed.sh start\n")
        print("Alternative: Docker Desktop (if you prefer):")
        print("  docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest\n")
    
    # Try to connect to server
    try:
        store = MilvusVectorStore.server(
            host="localhost",
            port=19530,
            collection_name="example_docs",
            embedder=embedder
        )
        print("✓ Connected to Milvus server at localhost:19530\n")
        
        # Clear old data
        print("Clearing old data...")
        store.clear()
        print("✓ Data cleared\n")
        
    except Exception as e:
        print(f"\n❌ Failed to connect to Milvus server: {e}")
        print("\n💡 Start Milvus server first:\n")
        print("Recommended (WSL2):")
        print("  curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh")
        print("  bash standalone_embed.sh start")
        print("\nAlternative (Docker Desktop):")
        print("  docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest\n")
        print("Skipping to Example 2 (Local Mode - Linux/macOS only)...\n")
        store = None
    
    # Create sample documents
    units = [
        TextUnit(
            unit_id="doc1",
            content="Milvus is a high-performance vector database built for AI applications.",
            metadata=UnitMetadata(
                context_path="docs/milvus.md/intro"
            )
        ),
        TextUnit(
            unit_id="doc2",
            content="Vector databases enable semantic search using embedding models for AI.",
            metadata=UnitMetadata(
                context_path="docs/vectors.md/concepts"
            )
        ),
        TextUnit(
            unit_id="doc3",
            content="Milvus supports distributed architecture for large-scale deployments.",
            metadata=UnitMetadata(
                context_path="docs/milvus.md/architecture"
            )
        ),
    ]
    
    if store:
        # Add documents
        print("Adding documents to Milvus...")
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
            content="Milvus is an ultra-fast, cloud-native vector database for AI at scale.",
            metadata=UnitMetadata(
                context_path="docs/milvus_v2.md/intro/updated"
            )
        )
        store.update(updated_unit)
        print("✓ Document updated\n")
        
        # Search again
        print("Searching after update...")
        results = store.search("AI database", top_k=1)
        if results:
            print(f"✓ Found: {results[0].content}")
        print()
        
        # Delete document
        print("Deleting document...")
        store.delete(["doc3"])
        print(f"✓ Document deleted")
        print(f"✓ Remaining vectors: {store.count()}\n")
    
    # ============================================================
    # EXAMPLE 2: Local Mode (Milvus Lite) - Linux/macOS Only
    # ============================================================
    print("="*70)
    print("EXAMPLE 2: Local Mode (Milvus Lite) - Linux/macOS Only")
    print("="*70)
    
    if platform.system() == "Windows":
        print("\n⚠️  Milvus Lite is NOT supported on Windows.")
        print("Skipping this example...\n")
        print("On Linux/macOS, you can use:")
        print("  store = MilvusVectorStore.local(")
        print("      path='./tmp/milvus_data.db',")
        print("      collection_name='my_collection',")
        print("      embedder=embedder")
        print("  )\n")
    else:
        # Local mode only works on Linux/macOS
        try:
            batch_store = MilvusVectorStore.local(
                path="./tmp/milvus_data.db",
                collection_name="batch_demo",
                embedder=embedder
            )
            print("✓ Created Milvus Lite store\n")
            
            # Batch add
            print("Batch adding 100 documents...")
            batch_units = []
            for i in range(100):
                unit = TextUnit(
                    unit_id=f"batch_{i}",
                    content=f"This is batch document number {i} about AI vector search and embeddings.",
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
        except RuntimeError as e:
            print(f"\n❌ Error: {e}\n")
    
    # ============================================================
    # EXAMPLE 3: Cloud Mode (Zilliz Cloud)
    # ============================================================
    print("="*70)
    print("EXAMPLE 3: Cloud Mode (Zilliz Cloud)")
    print("="*70)
    print("To use Zilliz Cloud (fully managed Milvus):")
    print("  1. Sign up at https://zilliz.com/ (free tier available)")
    print("  2. Create a cluster and get your endpoint URL and API key")
    print("  3. Connect to Zilliz Cloud:")
    print()
    print("  store = MilvusVectorStore.cloud(")
    print("      url='https://xxx.api.gcp-us-west1.zillizcloud.com:443',")
    print("      api_key='your_zilliz_api_key',")
    print("      collection_name='my_collection',")
    print("      embedder=embedder")
    print("  )")
    print()
    print("Benefits:")
    print("  - Fully managed (no ops)")
    print("  - Auto-scaling")
    print("  - High availability")
    print("  - Global deployment\n")
    
    print("="*70)
    print("✓ All examples completed!")
    print("="*70)
    print()
    print("Performance Notes:")
    print("- Milvus Lite: Good for < 1M vectors, embedded mode")
    print("- Standalone: Good for millions of vectors, single server")
    print("- Cluster: Good for billions of vectors, distributed")
    print("- Hardware acceleration: GPU support for even faster search")
    print()


if __name__ == "__main__":
    main()
