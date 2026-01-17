"""
Qdrant Vector Store Examples

This example demonstrates all deployment modes of QdrantVectorStore:
1. In-memory mode - for testing
2. Local persistent mode - for development
3. Server mode - for production
4. Cloud mode - for managed service

Before running:
1. Install dependencies: pip install qdrant-client
2. Set BAILIAN_API_KEY in environment or .env file
3. (Optional) Start Qdrant server for server mode:
   qdrant.exe --http-port 16333 --grpc-port 16334
4. (Optional) Set QDRANT_CLOUD_URL and QDRANT_API_KEY for cloud mode
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from zag.schemas.unit import TextUnit
from zag.schemas import UnitMetadata
from zag.storages.vector import QdrantVectorStore
from zag import Embedder

load_dotenv()


def example_inmemory():
    """Example 1: In-memory mode (for testing)"""
    print("=" * 70)
    print("Example 1: Qdrant In-Memory Mode")
    print("=" * 70)
    
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    print("\n1. Creating in-memory store...")
    print("   Usage: QdrantVectorStore.in_memory()")
    
    store = QdrantVectorStore.in_memory(
        collection_name="test_collection",
        embedder=embedder
    )
    print(f"   ✓ Store created: {store.collection_name}")
    print("   ✓ Mode: In-memory (data lost on exit)")
    
    # Add and search
    units = [
        TextUnit(
            unit_id="qdrant_1",
            content="Qdrant is a vector similarity search engine.",
            metadata=UnitMetadata(context_path="Intro/Qdrant")
        ),
        TextUnit(
            unit_id="qdrant_2",
            content="Qdrant supports filtering and payload indexing.",
            metadata=UnitMetadata(context_path="Features/Filtering")
        ),
    ]
    
    store.add(units)
    print(f"\n2. Added {len(units)} units, total: {store.count()}")
    
    results = store.search("What are Qdrant's features?", top_k=2)
    print(f"\n3. Search found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"   {i}. [{unit.unit_id}] {unit.content[:50]}... (score: {unit.score:.4f})")
    
    print("\n✅ Use cases: Unit testing, quick prototyping, CI/CD")
    print("⚠️  Data is lost when process exits!\n")


def example_local():
    """Example 2: Local persistent mode (for development)"""
    print("=" * 70)
    print("Example 2: Qdrant Local Persistent Mode")
    print("=" * 70)
    
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    db_path = "./tmp/qdrant_local_example"
    print(f"\n1. Creating local persistent store...")
    print(f"   Usage: QdrantVectorStore.local(path='{db_path}')")
    
    store = QdrantVectorStore.local(
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
            unit_id="qdrant_dev_1",
            content="Local Qdrant is perfect for development.",
            metadata=UnitMetadata(context_path="Development/Setup")
        ),
        TextUnit(
            unit_id="qdrant_dev_2",
            content="Data persists to local directory automatically.",
            metadata=UnitMetadata(context_path="Development/Persistence")
        ),
    ]
    
    store.add(units)
    print(f"\n2. Added {len(units)} new units, total: {store.count()}")
    
    results = store.search("How does local persistence work?", top_k=2)
    print(f"\n3. Search found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"   {i}. [{unit.unit_id}] {unit.content[:50]}... (score: {unit.score:.4f})")
    
    print("\n✅ Use cases: Local development, single-node deployment")
    print(f"💡 Tip: Delete '{db_path}' to reset\n")


def example_server():
    """Example 3: Server mode (for production)"""
    print("=" * 70)
    print("Example 3: Qdrant Server Mode")
    print("=" * 70)
    
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    print("\n1. Connecting to Qdrant server...")
    print("   Usage: QdrantVectorStore.server(host='localhost', port=16333)")
    print("\n   ⚠️  Make sure server is running:")
    print("   $ qdrant.exe --http-port 16333 --grpc-port 16334")
    
    try:
        store = QdrantVectorStore.server(
            host="localhost",
            port=16333,
            collection_name="production_docs",
            embedder=embedder
        )
        print(f"   ✓ Connected to server at localhost:16333")
        print(f"   ✓ Collection: {store.collection_name}")
        
        existing_count = store.count()
        print(f"   ✓ Existing units: {existing_count}")
        
        # Add units
        units = [
            TextUnit(
                unit_id="qdrant_prod_1",
                content="Qdrant server enables multi-client access.",
                metadata=UnitMetadata(context_path="Production/Architecture")
            ),
        ]
        
        store.add(units)
        print(f"\n2. Added {len(units)} units, total: {store.count()}")
        
        results = store.search("How does Qdrant handle performance?", top_k=1)
        print(f"\n3. Search found {len(results)} results:")
        for i, unit in enumerate(results, 1):
            print(f"   {i}. [{unit.unit_id}] {unit.content[:50]}... (score: {unit.score:.4f})")
        
        print("\n✅ Use cases: Production, distributed systems, microservices")
        print("💡 Server management:")
        print("   Start: qdrant.exe --http-port 16333 --grpc-port 16334")
        print("   Stop:  Ctrl+C in server terminal")
        print("   UI:    http://localhost:16333/dashboard\n")
        
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        print("\n   💡 Start Qdrant server first:")
        print("   $ qdrant.exe --http-port 16333 --grpc-port 16334")
        print("\n   Skipping server mode example...\n")


def example_cloud():
    """Example 4: Cloud mode (for managed service)"""
    print("=" * 70)
    print("Example 4: Qdrant Cloud Mode")
    print("=" * 70)
    
    cloud_url = os.getenv('QDRANT_CLOUD_URL')
    cloud_api_key = os.getenv('QDRANT_API_KEY')
    
    if not cloud_url or not cloud_api_key:
        print("\n⚠️  Qdrant Cloud credentials not configured")
        print("\nTo run this example, set:")
        print("   QDRANT_CLOUD_URL=https://xxx-yyy-zzz.qdrant.io")
        print("   QDRANT_API_KEY=your_api_key_here")
        print("\n💡 Sign up: https://cloud.qdrant.io/")
        print("\nSkipping cloud mode example...\n")
        return
    
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    print("\n1. Connecting to Qdrant Cloud...")
    print(f"   Usage: QdrantVectorStore.cloud(url='...', api_key='...')")
    
    try:
        store = QdrantVectorStore.cloud(
            url=cloud_url,
            api_key=cloud_api_key,
            collection_name="cloud_docs",
            embedder=embedder
        )
        print(f"   ✓ Connected to Qdrant Cloud")
        print(f"   ✓ Collection: {store.collection_name}")
        
        existing_count = store.count()
        print(f"   ✓ Existing units: {existing_count}")
        
        # Add units
        units = [
            TextUnit(
                unit_id="qdrant_cloud_1",
                content="Qdrant Cloud is a fully managed vector database.",
                metadata=UnitMetadata(context_path="Cloud/Overview")
            ),
        ]
        
        store.add(units)
        print(f"\n2. Added {len(units)} units, total: {store.count()}")
        
        results = store.search("What are the benefits of Qdrant Cloud?", top_k=1)
        print(f"\n3. Search found {len(results)} results:")
        for i, unit in enumerate(results, 1):
            print(f"   {i}. [{unit.unit_id}] {unit.content[:50]}... (score: {unit.score:.4f})")
        
        print("\n✅ Use cases: Production without DevOps, global apps, enterprise")
        print("💡 Management:")
        print("   Dashboard: https://cloud.qdrant.io/")
        print("   Docs:      https://qdrant.tech/documentation/cloud/\n")
        
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        print("\n   💡 Check your credentials and network connection")
        print("\n   Skipping cloud mode example...\n")


def main():
    print("\n" + "=" * 70)
    print("  Qdrant Vector Store - All Deployment Modes")
    print("=" * 70)
    print()
    
    try:
        # Example 1: In-memory
        example_inmemory()
        
        # Example 2: Local persistent
        example_local()
        
        # Example 3: Server (optional)
        example_server()
        
        # Example 4: Cloud (optional)
        example_cloud()
        
        print("=" * 70)
        print("✅ All Qdrant examples completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
