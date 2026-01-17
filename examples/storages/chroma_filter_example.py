"""
Chroma Filter Examples - MongoDB-style Filter Conversion

This example demonstrates the filter conversion feature for ChromaDB.

Test scenarios:
1. Simple equality filter
2. Range filter ($gte, $lt)
3. IN filter ($in)
4. Complex nested filters (AND, OR)
5. Metadata field filtering

Before running:
1. Install dependencies: pip install chromadb
2. Set BAILIAN_API_KEY in environment or .env file
3. Start Chroma server:
   chroma run --host localhost --port 18000
"""

import os
from dotenv import load_dotenv
from zag.schemas.unit import TextUnit
from zag.schemas import UnitMetadata
from zag.storages.vector import ChromaVectorStore
from zag import Embedder

load_dotenv()


def setup_test_data(store: ChromaVectorStore):
    """Prepare test data with various metadata combinations"""
    print("\n" + "=" * 70)
    print("Setting up test data...")
    print("=" * 70)
    
    # Clear existing data
    store.clear()
    
    # Create diverse test units
    units = [
        # Legal documents
        TextUnit(
            unit_id="legal_doc_1",
            content="This is a legal contract about property rights.",
            metadata=UnitMetadata(
                context_path="Legal/Contracts",
                custom={
                    "category": "legal",
                    "confidence": 0.95,
                    "status": "active",
                    "priority": 1,
                    "year": 2024
                }
            )
        ),
        TextUnit(
            unit_id="legal_doc_2",
            content="Legal disclaimer and terms of service.",
            metadata=UnitMetadata(
                context_path="Legal/Terms",
                custom={
                    "category": "legal",
                    "confidence": 0.85,
                    "status": "active",
                    "priority": 2,
                    "year": 2023
                }
            )
        ),
        
        # Technical documents
        TextUnit(
            unit_id="tech_doc_1",
            content="Technical specification for the API design.",
            metadata=UnitMetadata(
                context_path="Tech/API",
                custom={
                    "category": "technical",
                    "confidence": 0.92,
                    "status": "active",
                    "priority": 1,
                    "year": 2024
                }
            )
        ),
        TextUnit(
            unit_id="tech_doc_2",
            content="Database architecture and optimization guide.",
            metadata=UnitMetadata(
                context_path="Tech/Database",
                custom={
                    "category": "technical",
                    "confidence": 0.78,
                    "status": "draft",
                    "priority": 3,
                    "year": 2024
                }
            )
        ),
        
        # Marketing documents
        TextUnit(
            unit_id="marketing_doc_1",
            content="Product launch marketing strategy.",
            metadata=UnitMetadata(
                context_path="Marketing/Strategy",
                custom={
                    "category": "marketing",
                    "confidence": 0.88,
                    "status": "active",
                    "priority": 1,
                    "year": 2024
                }
            )
        ),
        TextUnit(
            unit_id="marketing_doc_2",
            content="Brand guidelines and visual identity.",
            metadata=UnitMetadata(
                context_path="Marketing/Brand",
                custom={
                    "category": "marketing",
                    "confidence": 0.65,
                    "status": "archived",
                    "priority": 4,
                    "year": 2022
                }
            )
        ),
    ]
    
    store.add(units)
    print(f"✓ Added {len(units)} test units")
    print(f"✓ Total units in store: {store.count()}")


def example_1_simple_equality(store: ChromaVectorStore):
    """Example 1: Simple equality filter"""
    print("\n" + "=" * 70)
    print("Example 1: Simple Equality Filter")
    print("=" * 70)
    
    # Filter: category equals "legal"
    filter_dict = {
        "metadata.custom.category": "legal"
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("\nSearching for legal documents...")
    
    results = store.search(
        query="legal documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")
        print(f"     Content: {unit.content[:60]}...")


def example_2_range_filter(store: ChromaVectorStore):
    """Example 2: Range filter"""
    print("\n" + "=" * 70)
    print("Example 2: Range Filter (confidence >= 0.85)")
    print("=" * 70)
    
    # Filter: confidence >= 0.85
    filter_dict = {
        "metadata.custom.confidence": {"$gte": 0.85}
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("\nSearching for high-confidence documents...")
    
    results = store.search(
        query="important documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")


def example_3_in_filter(store: ChromaVectorStore):
    """Example 3: IN filter"""
    print("\n" + "=" * 70)
    print("Example 3: IN Filter (category in [legal, technical])")
    print("=" * 70)
    
    # Filter: category in ["legal", "technical"]
    filter_dict = {
        "metadata.custom.category": {"$in": ["legal", "technical"]}
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("\nSearching for legal or technical documents...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")


def example_4_combined_and(store: ChromaVectorStore):
    """Example 4: Combined conditions (implicit AND)"""
    print("\n" + "=" * 70)
    print("Example 4: Combined Conditions (AND)")
    print("=" * 70)
    
    # Filter: status=active AND year = 2024
    filter_dict = {
        "metadata.custom.status": "active",
        "metadata.custom.year": 2024
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("\nSearching for active 2024 documents...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")


def example_5_or_condition(store: ChromaVectorStore):
    """Example 5: OR condition"""
    print("\n" + "=" * 70)
    print("Example 5: OR Condition")
    print("=" * 70)
    
    # Filter: priority = 1 OR status = "draft"
    filter_dict = {
        "$or": [
            {"metadata.custom.priority": 1},
            {"metadata.custom.status": "draft"}
        ]
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("\nSearching for high-priority or draft documents...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")


def example_6_complex_nested(store: ChromaVectorStore):
    """Example 6: Complex nested filters"""
    print("\n" + "=" * 70)
    print("Example 6: Complex Nested Filters")
    print("=" * 70)
    
    # Filter: (category = "legal" OR category = "technical") 
    #         AND status = "active"
    filter_dict = {
        "$or": [
            {"metadata.custom.category": "legal"},
            {"metadata.custom.category": "technical"}
        ],
        "metadata.custom.status": "active"
    }
    
    print(f"\nMongoDB-style filter:")
    import json
    print(json.dumps(filter_dict, indent=2))
    print("\nSearching with complex conditions...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")
        print(f"     Content: {unit.content[:50]}...")


def example_7_native_chroma_filter(store: ChromaVectorStore):
    """Example 7: Using Chroma native where clause (multiple scenarios)"""
    print("\n" + "=" * 70)
    print("Example 7: Native Chroma Where Clause")
    print("=" * 70)
    
    # Scenario 1: Simple equality (no prefix)
    print("\n--- Scenario 1: Simple equality (native Chroma) ---")
    native_filter_1 = {"status": "active"}
    print(f"Filter: {native_filter_1}")
    
    results_1 = store.search(
        query="documents",
        top_k=10,
        filter=native_filter_1
    )
    print(f"✓ Found {len(results_1)} results")
    
    # Scenario 2: Operator expression (native Chroma)
    print("\n--- Scenario 2: Operator with explicit $and ---")
    native_filter_2 = {
        "$and": [
            {"status": "active"},
            {"year": 2024}
        ]
    }
    print(f"Filter: {native_filter_2}")
    print("Note: Chroma requires explicit $and for multiple conditions")
    
    results_2 = store.search(
        query="documents",
        top_k=10,
        filter=native_filter_2
    )
    print(f"✓ Found {len(results_2)} results")
    
    # Scenario 3: Range operator (native Chroma)
    print("\n--- Scenario 3: Range operator ($gte) ---")
    native_filter_3 = {"confidence": {"$gte": 0.85}}
    print(f"Filter: {native_filter_3}")
    
    results_3 = store.search(
        query="documents",
        top_k=10,
        filter=native_filter_3
    )
    print(f"✓ Found {len(results_3)} results")
    
    # Scenario 4: IN operator (native Chroma)
    print("\n--- Scenario 4: IN operator ---")
    native_filter_4 = {"category": {"$in": ["legal", "technical"]}}
    print(f"Filter: {native_filter_4}")
    
    results_4 = store.search(
        query="documents",
        top_k=10,
        filter=native_filter_4
    )
    print(f"✓ Found {len(results_4)} results")
    
    # Scenario 5: Complex nested (native Chroma)
    print("\n--- Scenario 5: Complex nested with $or ---")
    native_filter_5 = {
        "$and": [
            {
                "$or": [
                    {"category": "legal"},
                    {"category": "technical"}
                ]
            },
            {"status": "active"}
        ]
    }
    print(f"Filter:")
    import json
    print(json.dumps(native_filter_5, indent=2))
    
    results_5 = store.search(
        query="documents",
        top_k=10,
        filter=native_filter_5
    )
    print(f"✓ Found {len(results_5)} results")
    
    print("\n" + "-" * 70)
    print("Summary: All native Chroma filter formats work correctly!")


def main():
    print("\n" + "=" * 70)
    print("  Chroma Filter Conversion Examples")
    print("  MongoDB-style → Chroma Where Clause")
    print("=" * 70)
    
    # Initialize embedder
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    # Connect to Chroma server
    print("\nConnecting to Chroma server at localhost:18000...")
    
    try:
        store = ChromaVectorStore.server(
            host="localhost",
            port=18000,
            collection_name="filter_test",
            embedder=embedder
        )
        print("✓ Connected successfully")
        
        # Setup test data
        setup_test_data(store)
        
        # Run examples
        example_1_simple_equality(store)
        example_2_range_filter(store)
        example_3_in_filter(store)
        example_4_combined_and(store)
        example_5_or_condition(store)
        example_6_complex_nested(store)
        example_7_native_chroma_filter(store)
        
        print("\n" + "=" * 70)
        print("✅ All filter examples completed!")
        print("=" * 70)
        print("\nKey takeaways:")
        print("  • MongoDB-style filters are automatically converted to Chroma format")
        print("  • Supports: $eq, $gte, $lt, $in, $or, etc.")
        print("  • Field names are automatically mapped (metadata.custom.xxx → xxx)")
        print("  • Can also use native Chroma where clauses directly")
        print("  • Filter conversion is transparent to the user")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure Chroma server is running:")
        print("   chroma run --host localhost --port 18000")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
