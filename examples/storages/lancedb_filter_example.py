"""
LanceDB Filter Examples - MongoDB-style Filter Conversion

This example demonstrates the filter conversion feature for LanceDB.
MongoDB-style filters are automatically converted to SQL WHERE clauses.

Test scenarios:
1. Simple equality filter
2. Range filter ($gte, $lt)
3. IN filter ($in)
4. Complex nested filters (AND, OR)
5. Metadata field filtering

Installation:
    pip install lancedb
"""

from zag.schemas.unit import TextUnit
from zag.schemas import UnitMetadata
from zag.storages.vector import LanceDBVectorStore
from zag import Embedder


def setup_test_data(store: LanceDBVectorStore):
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


def example_1_simple_equality(store: LanceDBVectorStore):
    """Example 1: Simple equality filter"""
    print("\n" + "=" * 70)
    print("Example 1: Simple Equality Filter")
    print("=" * 70)
    
    # Filter: category equals "legal"
    filter_dict = {
        "metadata.custom.category": "legal"
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("Converted SQL: category = 'legal'")
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


def example_2_range_filter(store: LanceDBVectorStore):
    """Example 2: Range filter"""
    print("\n" + "=" * 70)
    print("Example 2: Range Filter (confidence >= 0.85)")
    print("=" * 70)
    
    # Filter: confidence >= 0.85
    filter_dict = {
        "metadata.custom.confidence": {"$gte": 0.85}
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("Converted SQL: confidence >= 0.85")
    print("\nSearching for high-confidence documents...")
    
    results = store.search(
        query="important documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")


def example_3_in_filter(store: LanceDBVectorStore):
    """Example 3: IN filter"""
    print("\n" + "=" * 70)
    print("Example 3: IN Filter (category in [legal, technical])")
    print("=" * 70)
    
    # Filter: category in ["legal", "technical"]
    filter_dict = {
        "metadata.custom.category": {"$in": ["legal", "technical"]}
    }
    
    print(f"\nMongoDB-style filter: {filter_dict}")
    print("Converted SQL: category IN ('legal', 'technical')")
    print("\nSearching for legal or technical documents...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")


def example_4_combined_and(store: LanceDBVectorStore):
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
    print("Converted SQL: status = 'active' AND year = 2024")
    print("\nSearching for active 2024 documents...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")


def example_5_or_condition(store: LanceDBVectorStore):
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
    print("Converted SQL: (priority = 1 OR status = 'draft')")
    print("\nSearching for high-priority or draft documents...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=filter_dict
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")


def example_6_complex_nested(store: LanceDBVectorStore):
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
    print("\nConverted SQL: (category = 'legal' OR category = 'technical') AND status = 'active'")
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


def example_7_native_sql_filter(store: LanceDBVectorStore):
    """Example 7: Using native SQL WHERE clause"""
    print("\n" + "=" * 70)
    print("Example 7: Native SQL WHERE Clause")
    print("=" * 70)
    
    # Native SQL WHERE clause string
    # Filter: status = "active" AND year = 2024 AND confidence >= 0.85
    sql_filter = "status = 'active' AND year = 2024 AND confidence >= 0.85"
    
    print(f"\nNative SQL WHERE clause:")
    print(f"  {sql_filter}")
    print("\nSearching with native SQL filter...")
    
    results = store.search(
        query="documents",
        top_k=10,
        filter=sql_filter  # Pass SQL string directly
    )
    
    print(f"\n✓ Found {len(results)} results:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [{unit.unit_id}]")


def main():
    print("\n" + "=" * 70)
    print("  LanceDB Filter Conversion Examples")
    print("  MongoDB-style → SQL WHERE Clause")
    print("=" * 70)
    
    # Initialize embedder
    print("\nInitializing Ollama embedder...")
    embedder = Embedder("ollama/jina/jina-embeddings-v2-base-en:latest")
    print(f"✓ Embedder dimension: {embedder.dimension}")
    
    # Create local LanceDB store
    print("\nCreating local LanceDB store...")
    
    try:
        store = LanceDBVectorStore.local(
            path="./tmp/lancedb_filter_test",
            table_name="filter_test",
            embedder=embedder
        )
        print("✓ Created successfully")
        
        # Setup test data
        setup_test_data(store)
        
        # Run examples
        example_1_simple_equality(store)
        example_2_range_filter(store)
        example_3_in_filter(store)
        example_4_combined_and(store)
        example_5_or_condition(store)
        example_6_complex_nested(store)
        example_7_native_sql_filter(store)
        
        print("\n" + "=" * 70)
        print("✅ All filter examples completed!")
        print("=" * 70)
        print("\nKey takeaways:")
        print("  • MongoDB-style filters are automatically converted to SQL WHERE")
        print("  • Supports: $eq, $gte, $lt, $in, $or, etc.")
        print("  • Field names are automatically mapped (metadata.custom.xxx → xxx)")
        print("  • Can also use native SQL WHERE strings directly")
        print("  • Filter conversion is transparent to the user")
        print("  • No server needed - runs as embedded database")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
