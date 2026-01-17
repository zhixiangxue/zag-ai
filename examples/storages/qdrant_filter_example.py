"""
Qdrant Filter Examples - MongoDB-style Filter Conversion

This example demonstrates the filter conversion feature that translates
MongoDB-style filter syntax to Qdrant's native Filter objects.

Test scenarios:
1. Simple equality filter
2. Range filter ($gte, $lt)
3. IN filter ($in)
4. Complex nested filters (AND, OR)
5. Metadata nested field filtering

Before running:
1. Install dependencies: pip install qdrant-client
2. Set BAILIAN_API_KEY in environment or .env file
3. Start Qdrant server:
   qdrant.exe --http-port 16333 --grpc-port 16334
"""

import os
from dotenv import load_dotenv
from zag.schemas.unit import TextUnit
from zag.schemas import UnitMetadata
from zag.storages.vector import QdrantVectorStore
from zag import Embedder

load_dotenv()


def setup_test_data(store: QdrantVectorStore):
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
                    "tags": ["property", "contract"],
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
                    "tags": ["terms", "service"],
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
                    "tags": ["api", "design"],
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
                    "tags": ["database", "optimization"],
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
                    "tags": ["strategy", "launch"],
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
                    "tags": ["brand", "guidelines"],
                    "year": 2022
                }
            )
        ),
    ]
    
    store.add(units)
    print(f"✓ Added {len(units)} test units")
    print(f"✓ Total units in store: {store.count()}")


def example_1_simple_equality(store: QdrantVectorStore):
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
        category = unit.metadata.custom.get("category", "N/A")
        confidence = unit.metadata.custom.get("confidence", 0)
        print(f"  {i}. [{unit.unit_id}] category={category}, confidence={confidence:.2f}")
        print(f"     Content: {unit.content[:60]}...")


def example_2_range_filter(store: QdrantVectorStore):
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
        category = unit.metadata.custom.get("category", "N/A")
        confidence = unit.metadata.custom.get("confidence", 0)
        status = unit.metadata.custom.get("status", "N/A")
        print(f"  {i}. [{unit.unit_id}] category={category}, confidence={confidence:.2f}, status={status}")


def example_3_in_filter(store: QdrantVectorStore):
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
        category = unit.metadata.custom.get("category", "N/A")
        priority = unit.metadata.custom.get("priority", 0)
        print(f"  {i}. [{unit.unit_id}] category={category}, priority={priority}")


def example_4_combined_and(store: QdrantVectorStore):
    """Example 4: Combined conditions (implicit AND)"""
    print("\n" + "=" * 70)
    print("Example 4: Combined Conditions (AND)")
    print("=" * 70)
    
    # Filter: status=active AND confidence >= 0.85 AND year = 2024
    filter_dict = {
        "metadata.custom.status": "active",
        "metadata.custom.confidence": {"$gte": 0.85},
        "metadata.custom.year": 2024
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("\nSearching for active 2024 high-confidence documents...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        category = unit.metadata.custom.get("category", "N/A")
        confidence = unit.metadata.custom.get("confidence", 0)
        status = unit.metadata.custom.get("status", "N/A")
        year = unit.metadata.custom.get("year", "N/A")
        print(f"  {i}. [{unit.unit_id}] category={category}, confidence={confidence:.2f}, status={status}, year={year}")


def example_5_or_condition(store: QdrantVectorStore):
    """Example 5: OR condition"""
    print("\n" + "=" * 70)
    print("Example 5: OR Condition")
    print("=" * 70)
    
    # Filter: priority = 1 OR confidence >= 0.9
    filter_dict = {
        "$or": [
            {"metadata.custom.priority": 1},
            {"metadata.custom.confidence": {"$gte": 0.9}}
        ]
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("\nSearching for high-priority or high-confidence documents...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        category = unit.metadata.custom.get("category", "N/A")
        priority = unit.metadata.custom.get("priority", 0)
        confidence = unit.metadata.custom.get("confidence", 0)
        print(f"  {i}. [{unit.unit_id}] category={category}, priority={priority}, confidence={confidence:.2f}")


def example_6_complex_nested(store: QdrantVectorStore):
    """Example 6: Complex nested filters"""
    print("\n" + "=" * 70)
    print("Example 6: Complex Nested Filters")
    print("=" * 70)
    
    # Filter: (category = "legal" OR category = "technical") 
    #         AND status = "active" 
    #         AND confidence >= 0.85
    #         AND year >= 2023
    filter_dict = {
        "$or": [
            {"metadata.custom.category": "legal"},
            {"metadata.custom.category": "technical"}
        ],
        "metadata.custom.status": "active",
        "metadata.custom.confidence": {"$gte": 0.85},
        "metadata.custom.year": {"$gte": 2023}
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
        category = unit.metadata.custom.get("category", "N/A")
        status = unit.metadata.custom.get("status", "N/A")
        confidence = unit.metadata.custom.get("confidence", 0)
        year = unit.metadata.custom.get("year", "N/A")
        print(f"  {i}. [{unit.unit_id}]")
        print(f"     category={category}, status={status}, confidence={confidence:.2f}, year={year}")
        print(f"     Content: {unit.content[:50]}...")


def example_7_range_combined(store: QdrantVectorStore):
    """Example 7: Range with multiple bounds"""
    print("\n" + "=" * 70)
    print("Example 7: Range Filter (0.8 <= confidence < 0.9)")
    print("=" * 70)
    
    # Filter: 0.8 <= confidence < 0.9
    filter_dict = {
        "metadata.custom.confidence": {
            "$gte": 0.8,
            "$lt": 0.9
        }
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("\nSearching for medium-confidence documents...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        category = unit.metadata.custom.get("category", "N/A")
        confidence = unit.metadata.custom.get("confidence", 0)
        print(f"  {i}. [{unit.unit_id}] category={category}, confidence={confidence:.2f}")


def example_8_native_filter(store: QdrantVectorStore):
    """Example 8: Using Qdrant native Filter object"""
    print("\n" + "=" * 70)
    print("Example 8: Native Qdrant Filter Object")
    print("=" * 70)
    
    # Import Qdrant filter classes
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
    
    # Build native Qdrant Filter
    # Filter: status = "active" AND confidence >= 0.85
    native_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.custom.status",
                match=MatchValue(value="active")
            ),
            FieldCondition(
                key="metadata.custom.confidence",
                range=Range(gte=0.85)
            )
        ]
    )
    
    print("\nNative Qdrant Filter object:")
    print("  Filter(must=[")
    print("    FieldCondition(key='metadata.custom.status', match=MatchValue(value='active')),")
    print("    FieldCondition(key='metadata.custom.confidence', range=Range(gte=0.85))")
    print("  ])")
    print("\nSearching with native Filter object...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=native_filter  # Pass native Filter directly
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        category = unit.metadata.custom.get("category", "N/A")
        status = unit.metadata.custom.get("status", "N/A")
        confidence = unit.metadata.custom.get("confidence", 0)
        print(f"  {i}. [{unit.unit_id}] category={category}, status={status}, confidence={confidence:.2f}")


def main():
    print("\n" + "=" * 70)
    print("  Qdrant Filter Conversion Examples")
    print("  MongoDB-style → Qdrant Filter Objects")
    print("=" * 70)
    
    # Initialize embedder
    embedder = Embedder(
        'bailian/text-embedding-v3',
        api_key=os.getenv('BAILIAN_API_KEY')
    )
    
    # Connect to Qdrant server
    print("\nConnecting to Qdrant server at localhost:16333...")
    
    try:
        store = QdrantVectorStore.server(
            host="localhost",
            port=16333,
            grpc_port=16334,
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
        example_7_range_combined(store)
        example_8_native_filter(store)
        
        print("\n" + "=" * 70)
        print("✅ All filter examples completed!")
        print("=" * 70)
        print("\nKey takeaways:")
        print("  • MongoDB-style filters are automatically converted to Qdrant format")
        print("  • Supports: $eq, $gte, $lt, $in, $or, etc.")
        print("  • Nested metadata fields work with dot notation")
        print("  • Can also use native Qdrant Filter objects directly")
        print("  • Filter conversion is transparent to the user")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure Qdrant server is running:")
        print("   qdrant.exe --http-port 16333 --grpc-port 16334")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
