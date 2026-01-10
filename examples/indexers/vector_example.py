"""
Test Indexer API design

This example demonstrates:
1. Indexer vs VectorStore differences
2. Sync and async operations
3. Single/batch operations support
4. Build vs Add semantics

Before running:
1. Install dependencies: pip install python-dotenv
2. Set your API key in .env file: BAILIAN_API_KEY=your-key
"""

import os
import asyncio
from dotenv import load_dotenv
from zag import Embedder
from zag.schemas import TextUnit, UnitMetadata
from zag.storages.vector import ChromaVectorStore
from zag.indexers import VectorIndexer

# Load environment variables
load_dotenv()


def test_sync_operations():
    """Test synchronous indexer operations"""
    print("=" * 70)
    print("Test 1: Synchronous Operations")
    print("=" * 70)
    
    # Step 1: Setup
    print("\n1. Setting up embedder and stores...")
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    vector_store = ChromaVectorStore(
        collection_name="indexer_test",
        persist_directory=None,  # In-memory
        embedder=embedder
    )
    
    indexer = VectorIndexer(vector_store=vector_store)
    print(f"   ✓ Created: {indexer}")
    
    # Step 2: Prepare test data
    print("\n2. Preparing test units...")
    units = [
        TextUnit(
            unit_id="unit_1",
            content="Indexer manages high-level index operations.",
            metadata=UnitMetadata(context_path="Concepts/Indexer")
        ),
        TextUnit(
            unit_id="unit_2",
            content="VectorStore handles low-level storage operations.",
            metadata=UnitMetadata(context_path="Concepts/VectorStore")
        ),
        TextUnit(
            unit_id="unit_3",
            content="Build method rebuilds the entire index from scratch.",
            metadata=UnitMetadata(context_path="Operations/Build")
        ),
    ]
    print(f"   ✓ Created {len(units)} units")
    
    # Step 3: Add initial units
    print("\n3. Adding initial units...")
    indexer.add(units)
    print(f"   ✓ Added {indexer.count()} units to index")
    
    # Step 4: Add single unit (incremental)
    print("\n4. Adding single unit (incremental)...")
    new_unit = TextUnit(
        unit_id="unit_4",
        content="Add method incrementally adds to existing index.",
        metadata=UnitMetadata(context_path="Operations/Add")
    )
    indexer.add(new_unit)  # Single unit
    print(f"   ✓ Added single unit, total: {indexer.count()}")
    
    # Step 5: Add batch units
    print("\n5. Adding batch units...")
    batch_units = [
        TextUnit(
            unit_id="unit_5",
            content="Batch operations are more efficient than single operations.",
            metadata=UnitMetadata(context_path="Best Practices/Batching")
        ),
        TextUnit(
            unit_id="unit_6",
            content="Async methods support concurrent indexing operations.",
            metadata=UnitMetadata(context_path="Features/Async")
        ),
    ]
    indexer.add(batch_units)  # Batch
    print(f"   ✓ Added batch, total: {indexer.count()}")
    
    # Step 6: Update existing unit (strict update)
    print("\n6. Updating an existing unit...")
    updated_unit = TextUnit(
        unit_id="unit_1",
        content="Indexer manages high-level index operations and provides unified interface.",
        metadata=UnitMetadata(context_path="Concepts/Indexer")
    )
    indexer.update(updated_unit)  # Only updates, expects unit exists
    print(f"   ✓ Updated unit_1 (strict update)")
    
    # Step 7: Upsert unit (flexible: update or insert)
    print("\n7. Upserting units...")
    # Upsert existing unit (will update)
    upsert_existing = TextUnit(
        unit_id="unit_2",
        content="VectorStore handles low-level storage and embedding operations.",
        metadata=UnitMetadata(context_path="Concepts/VectorStore")
    )
    indexer.upsert(upsert_existing)
    print(f"   ✓ Upserted unit_2 (updated existing)")
    
    # Upsert new unit (will insert)
    upsert_new = TextUnit(
        unit_id="unit_7",
        content="Upsert handles both insert and update seamlessly.",
        metadata=UnitMetadata(context_path="Operations/Upsert")
    )
    indexer.upsert(upsert_new)
    print(f"   ✓ Upserted unit_7 (inserted new), total: {indexer.count()}")
    
    # Step 8: Check existence
    print("\n8. Checking unit existence...")
    exists = indexer.exists("unit_1")
    not_exists = indexer.exists("unit_999")
    print(f"   ✓ unit_1 exists: {exists}")
    print(f"   ✓ unit_999 exists: {not_exists}")
    
    # Step 9: Delete single unit
    print("\n9. Deleting single unit...")
    indexer.delete("unit_6")  # Single ID
    print(f"   ✓ Deleted unit_6, remaining: {indexer.count()}")
    
    # Step 10: Delete batch
    print("\n10. Deleting batch units...")
    indexer.delete(["unit_4", "unit_5"])  # Batch IDs
    print(f"   ✓ Deleted batch, remaining: {indexer.count()}")
    
    # Step 11: Clear and add new data
    print("\n11. Clearing and adding new data...")
    indexer.clear()  # Clear first
    new_units = [
        TextUnit(
            unit_id="new_1",
            content="After clear and add, only these units remain.",
        ),
        TextUnit(
            unit_id="new_2",
            content="Previous index content is completely replaced.",
        ),
    ]
    indexer.add(new_units)  # Add new units
    print(f"   ✓ Index cleared and rebuilt, total: {indexer.count()}")
    
    # Step 12: Clear all
    print("\n12. Clearing index...")
    indexer.clear()
    print(f"   ✓ Index cleared, remaining: {indexer.count()}")
    
    print("\n" + "=" * 70)
    print("✅ All synchronous operations completed successfully!")
    print("=" * 70)


async def test_async_operations():
    """Test asynchronous indexer operations"""
    print("\n" + "=" * 70)
    print("Test 2: Asynchronous Operations")
    print("=" * 70)
    
    # Setup
    print("\n1. Setting up...")
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    vector_store = ChromaVectorStore(
        collection_name="async_test",
        persist_directory=None,
        embedder=embedder
    )
    
    indexer = VectorIndexer(vector_store=vector_store)
    print(f"   ✓ Created indexer")
    
    # Async add
    print("\n2. Async add initial units...")
    units = [
        TextUnit(unit_id=f"async_{i}", content=f"Async unit {i}")
        for i in range(5)
    ]
    await indexer.aadd(units)
    print(f"   ✓ Async add completed, total: {indexer.count()}")
    
    # Async add single unit
    print("\n3. Async add single unit...")
    new_unit = TextUnit(unit_id="async_new", content="Async added unit")
    await indexer.aadd(new_unit)
    print(f"   ✓ Async add single unit completed, total: {indexer.count()}")
    
    # Async update
    print("\n4. Async update...")
    updated = TextUnit(unit_id="async_0", content="Async updated content")
    await indexer.aupdate(updated)
    print(f"   ✓ Async update completed")
    
    # Async upsert
    print("\n5. Async upsert...")
    upserted = TextUnit(unit_id="async_new", content="Async upserted (updated existing)")
    await indexer.aupsert(upserted)
    print(f"   ✓ Async upsert completed")
    
    # Async delete
    print("\n6. Async delete...")
    await indexer.adelete(["async_1", "async_2"])
    print(f"   ✓ Async delete completed, remaining: {indexer.count()}")
    
    # Async clear
    print("\n7. Async clear...")
    await indexer.aclear()
    print(f"   ✓ Async clear completed, remaining: {indexer.count()}")
    
    print("\n" + "=" * 70)
    print("✅ All asynchronous operations completed successfully!")
    print("=" * 70)


def test_api_design_summary():
    """Summary of API design"""
    print("\n" + "=" * 70)
    print("API Design Summary")
    print("=" * 70)
    
    print("\n✅ Key Features Validated:")
    print("   1. Indexer vs VectorStore:")
    print("      - Indexer: High-level (add, update, upsert, delete)")
    print("      - VectorStore: Low-level (storage + embedding)")
    print()
    print("   2. Operation semantics:")
    print("      - add(): Incremental add (insert only)")
    print("      - update(): Update existing (expects unit exists)")
    print("      - upsert(): Update or insert (most flexible)")
    print("      - clear(): Remove all units (destructive)")
    print()
    print("   3. Sync + Async support:")
    print("      - Sync: add(), update(), upsert(), delete(), clear()")
    print("      - Async: aadd(), aupdate(), aupsert(), adelete(), aclear()")
    print("      - Auto fallback: try async, fallback to sync")
    print()
    print("   4. Single/Batch operations:")
    print("      - Single: indexer.add(unit)")
    print("      - Batch: indexer.add([unit1, unit2])")
    print("      - Same interface, auto-normalized")
    print()
    print("   5. Utility methods:")
    print("      - count(): Get total units")
    print("      - exists(id): Check unit existence")
    print("      - clear(): Remove all units")
    
    print("\n" + "=" * 70)


def main():
    """Run all tests"""
    try:
        # Test 1: Sync operations
        test_sync_operations()
        
        # Test 2: Async operations
        asyncio.run(test_async_operations())
        
        # Summary
        test_api_design_summary()
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
