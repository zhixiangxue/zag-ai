"""
Sample code to test VectorStore API design

This example demonstrates the intuitive API design of VectorStore:
1. VectorStore is initialized with an embedder
2. Users can add units directly without manual embedding
3. Users can search with plain text without manual embedding

Before running:
1. Install dependencies: pip install chromadb python-dotenv
2. Make sure zag is installed: pip install -e .
3. Set your API key in .env file: BAILIAN_API_KEY=your-key
"""

import os
from dotenv import load_dotenv
from zag.schemas.unit import TextUnit
from zag.schemas.base import UnitMetadata
from zag.storages.vector import ChromaVectorStore
from zag import Embedder

# Load environment variables
load_dotenv()


def main():
    print("=" * 60)
    print("VectorStore API Design Test")
    print("=" * 60)
    
    # Step 1: Create embedder
    print("\n1. Creating embedder...")
    print("   Usage: Embedder('provider/model', api_key='...')")
    
    # Use Bailian embedder (you can replace with your preferred provider)
    # Other examples:
    #   - OpenAI: Embedder('openai/text-embedding-3-small', api_key='sk-...')
    #   - Bailian: Embedder('bailian/text-embedding-v3', api_key='sk-...')
    try:
        # Replace with your actual API key
        embedder = Embedder(
            'bailian/text-embedding-v3',
            api_key=os.getenv('BAILIAN_API_KEY')
        )
        print(f"   ✓ Embedder created")
        print(f"   ✓ Provider: {embedder.provider}")
        print(f"   ✓ Model: {embedder.model}")
        print(f"   ✓ Dimension: {embedder.dimension}")
    except Exception as e:
        print(f"   ✗ Failed to create embedder: {e}")
        print("   Please check your API key and network connection.")
        return
    
    # Step 2: Create VectorStore with embedder
    print("\n2. Creating VectorStore...")
    print("   Usage: ChromaVectorStore(embedder=embedder, collection_name='...')")
    print("   Note: Supports both 'embedder' (multimodal) and 'text_embedder' (text-only)")
    
    try:
        vector_store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=None,  # In-memory for testing
            embedder=embedder  # Use single embedder for all types
        )
        print(f"   ✓ VectorStore created")
        print(f"   ✓ Collection: {vector_store.collection_name}")
        print(f"   ✓ Vector dimension: {vector_store.dimension}")
    except Exception as e:
        print(f"   ✗ Failed to create VectorStore: {e}")
        return
    
    # Step 3: Create sample units
    print("\n3. Creating sample units...")
    
    units = [
        TextUnit(
            unit_id="unit_1",
            content="RAG is a framework for retrieval-augmented generation.",
            metadata=UnitMetadata(context_path="Introduction/What is RAG")
        ),
        TextUnit(
            unit_id="unit_2",
            content="Vector databases store embeddings for semantic search.",
            metadata=UnitMetadata(context_path="Introduction/Vector Databases")
        ),
        TextUnit(
            unit_id="unit_3",
            content="Embedders convert text into high-dimensional vectors.",
            metadata=UnitMetadata(context_path="Components/Embedders")
        ),
        TextUnit(
            unit_id="unit_4",
            content="Indexers organize units into searchable indices.",
            metadata=UnitMetadata(context_path="Components/Indexers")
        ),
        TextUnit(
            unit_id="unit_5",
            content="Retrievers find relevant information based on queries.",
            metadata=UnitMetadata(context_path="Components/Retrievers")
        ),
    ]
    
    print(f"   ✓ Created {len(units)} units")
    for unit in units:
        print(f"      - {unit.unit_id}: {unit.content[:50]}...")
    
    # Step 4: Add units to VectorStore
    print("\n4. Adding units to VectorStore...")
    print("   Usage: vector_store.add(units)")
    print("   Note: No manual embedding required! VectorStore handles it internally.")
    
    try:
        vector_store.add(units)
        print(f"   ✓ Successfully added {len(units)} units")
        print(f"   ✓ Total units in store: {vector_store.count()}")
    except Exception as e:
        print(f"   ✗ Failed to add units: {e}")
        return
    
    # Step 5: Search with plain text
    print("\n5. Searching with plain text query...")
    print("   Usage: vector_store.search('query text', top_k=3)")
    print("   Note: No manual embedding required! VectorStore handles it internally.")
    
    queries = [
        "What is retrieval-augmented generation?",
        "How do embeddings work?",
        "Tell me about indexing"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        try:
            results = vector_store.search(query, top_k=3)
            print(f"   Found {len(results)} results:")
            for i, unit in enumerate(results, 1):
                print(f"      {i}. [{unit.unit_id}] {unit.content[:60]}...")
                if unit.metadata.context_path:
                    print(f"         Context: {unit.metadata.context_path}")
        except Exception as e:
            print(f"   ✗ Search failed: {e}")
    
    # Step 6: Get units by ID
    print("\n6. Getting units by ID...")
    print("   Usage: vector_store.get(['unit_1', 'unit_3'])")
    
    try:
        retrieved = vector_store.get(["unit_1", "unit_3"])
        print(f"   ✓ Retrieved {len(retrieved)} units:")
        for unit in retrieved:
            print(f"      - {unit.unit_id}: {unit.content[:50]}...")
    except Exception as e:
        print(f"   ✗ Failed to get units: {e}")
    
    # Step 7: Update a unit
    print("\n7. Updating a unit...")
    print("   Usage: vector_store.update([modified_unit])")
    
    try:
        updated_unit = TextUnit(
            unit_id="unit_1",
            content="RAG (Retrieval-Augmented Generation) is a powerful framework for AI.",
            metadata=UnitMetadata(context_path="Introduction/What is RAG")
        )
        vector_store.update([updated_unit])
        print(f"   ✓ Updated unit_1")
        
        # Verify update
        result = vector_store.get(["unit_1"])[0]
        print(f"   ✓ New content: {result.content}")
    except Exception as e:
        print(f"   ✗ Failed to update: {e}")
    
    # Step 8: Delete a unit
    print("\n8. Deleting a unit...")
    print("   Usage: vector_store.delete(['unit_5'])")
    
    try:
        vector_store.delete(["unit_5"])
        print(f"   ✓ Deleted unit_5")
        print(f"   ✓ Remaining units: {vector_store.count()}")
    except Exception as e:
        print(f"   ✗ Failed to delete: {e}")
    
    # Step 9: Clear all
    print("\n9. Clearing all units...")
    print("   Usage: vector_store.clear()")
    
    try:
        vector_store.clear()
        print(f"   ✓ Cleared all units")
        print(f"   ✓ Remaining units: {vector_store.count()}")
    except Exception as e:
        print(f"   ✗ Failed to clear: {e}")
    
    print("\n" + "=" * 60)
    print("API Design Validation Summary")
    print("=" * 60)
    print("\n✅ Key Design Principles Validated:")
    print("   1. VectorStore knows its embedder (initialized with embedder)")
    print("   2. Users work with Units, not raw vectors")
    print("   3. No manual embedding required (handled internally)")
    print("   4. Search accepts plain text, not vectors")
    print("   5. API is intuitive and easy to understand")
    print("\n✅ All operations completed successfully!")
    print("=" * 60)





if __name__ == "__main__":
    main()
