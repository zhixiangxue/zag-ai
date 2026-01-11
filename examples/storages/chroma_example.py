"""
ChromaDB Vector Store Examples

This example demonstrates all deployment modes of ChromaVectorStore:
1. In-memory mode - for testing
2. Local persistent mode - for development
3. Server mode - for production

Before running:
1. Install dependencies: pip install chromadb
2. Set BAILIAN_API_KEY in environment or .env file
3. (Optional) Start Chroma server for server mode:
   chroma run --host localhost --port 18000
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from zag.schemas.unit import TextUnit
from zag.schemas.base import UnitMetadata
from zag.storages.vector import ChromaVectorStore
from zag import Embedder

load_dotenv()


def example_inmemory():
    """Example 1: In-memory mode (for testing)"""
    print("=" * 70)
    print("Example 1: ChromaDB In-Memory Mode")
    print("=" * 70)
    
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    print("\n1. Creating in-memory store...")
    print("   Usage: ChromaVectorStore.in_memory()")
    
    store = ChromaVectorStore.in_memory(
        collection_name="test_collection",
        embedder=embedder
    )
    print(f"   ✓ Store created: {store.collection_name}")
    print("   ✓ Mode: In-memory (data lost on exit)")
    
    # Add and search
    units = [
        TextUnit(
            unit_id="unit_1",
            content="Vector databases enable semantic search.",
            metadata=UnitMetadata(context_path="Concepts/Vector DB")
        ),
        TextUnit(
            unit_id="unit_2",
            content="Embeddings represent text as numerical vectors.",
            metadata=UnitMetadata(context_path="Concepts/Embeddings")
        ),
    ]
    
    store.add(units)
    print(f"\n2. Added {len(units)} units, total: {store.count()}")
    
    results = store.search("What are embeddings?", top_k=2)
    print(f"\n3. Search found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"   {i}. [{unit.unit_id}] {unit.content[:50]}... (score: {unit.score:.4f})")
    
    print("\n✅ Use cases: Unit testing, quick prototyping, CI/CD")
    print("⚠️  Data is lost when process exits!\n")


def example_local():
    """Example 2: Local persistent mode (for development)"""
    print("=" * 70)
    print("Example 2: ChromaDB Local Persistent Mode")
    print("=" * 70)
    
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    db_path = "./tmp/chroma_local_example"
    print(f"\n1. Creating local persistent store...")
    print(f"   Usage: ChromaVectorStore.local(path='{db_path}')")
    
    store = ChromaVectorStore.local(
        path=db_path,
        collection_name="dev_docs",
        embedder=embedder
    )
    print(f"   ✓ Store created: {store.collection_name}")
    print(f"   ✓ Data directory: {Path(db_path).absolute()}")
    
    existing_count = store.count()
    if existing_count > 0:
        print(f"   ✓ Found {existing_count} existing units (from previous run)")
    
    # Add units
    units = [
        TextUnit(
            unit_id="unit_dev_1",
            content="Local mode is perfect for development.",
            metadata=UnitMetadata(context_path="Development/Storage")
        ),
        TextUnit(
            unit_id="unit_dev_2",
            content="Data persists across application restarts.",
            metadata=UnitMetadata(context_path="Development/Persistence")
        ),
    ]
    
    store.add(units)
    print(f"\n2. Added {len(units)} new units, total: {store.count()}")
    
    results = store.search("How does persistence work?", top_k=2)
    print(f"\n3. Search found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"   {i}. [{unit.unit_id}] {unit.content[:50]}... (score: {unit.score:.4f})")
    
    print("\n✅ Use cases: Local development, single-node deployment")
    print(f"💡 Tip: Delete '{db_path}' to reset\n")


def example_server():
    """Example 3: Server mode (for production)"""
    print("=" * 70)
    print("Example 3: ChromaDB Server Mode")
    print("=" * 70)
    
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    print("\n1. Connecting to Chroma server...")
    print("   Usage: ChromaVectorStore.server(host='localhost', port=18000)")
    print("\n   ⚠️  Make sure server is running:")
    print("   $ chroma run --host localhost --port 18000")
    
    try:
        store = ChromaVectorStore.server(
            host="localhost",
            port=18000,
            collection_name="production_docs",
            embedder=embedder
        )
        print(f"   ✓ Connected to server at localhost:18000")
        print(f"   ✓ Collection: {store.collection_name}")
        
        existing_count = store.count()
        print(f"   ✓ Existing units: {existing_count}")
        
        # Add units
        units = [
            TextUnit(
                unit_id="unit_prod_1",
                content="Server mode enables multi-client access.",
                metadata=UnitMetadata(context_path="Production/Architecture")
            ),
        ]
        
        store.add(units)
        print(f"\n2. Added {len(units)} units, total: {store.count()}")
        
        results = store.search("multi-client", top_k=1)
        print(f"\n3. Search found {len(results)} results:")
        for i, unit in enumerate(results, 1):
            print(f"   {i}. [{unit.unit_id}] {unit.content[:50]}... (score: {unit.score:.4f})")
        
        print("\n✅ Use cases: Production, multi-client, microservices")
        print("💡 Server management:")
        print("   Start: chroma run --host localhost --port 18000")
        print("   Stop:  Ctrl+C in server terminal\n")
        
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        print("\n   💡 Start Chroma server first:")
        print("   $ chroma run --host localhost --port 18000")
        print("\n   Skipping server mode example...\n")


def main():
    print("\n" + "=" * 70)
    print("  ChromaDB Vector Store - All Deployment Modes")
    print("=" * 70)
    print()
    
    try:
        # Example 1: In-memory
        example_inmemory()
        
        # Example 2: Local persistent
        example_local()
        
        # Example 3: Server (optional)
        example_server()
        
        print("=" * 70)
        print("✅ All ChromaDB examples completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
