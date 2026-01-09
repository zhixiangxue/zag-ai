"""
Simple retriever demo without requiring API key

This demo uses mock data to demonstrate the retriever architecture
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from zag.retrievers import BaseRetriever, QueryFusionRetriever, FusionMode
from zag.schemas.unit import TextUnit
from typing import Any, Optional


class MockRetriever(BaseRetriever):
    """Mock retriever for testing without actual vector store"""
    
    def __init__(self, name: str, units: list[TextUnit]):
        self.name = name
        self.units = units
    
    def retrieve(
        self, 
        query: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[TextUnit]:
        """Return all units with mock scores"""
        # Assign mock scores (simulating similarity)
        import random
        results = []
        for unit in self.units[:top_k]:
            unit_copy = unit.model_copy()
            unit_copy.score = random.uniform(0.7, 0.95)
            results.append(unit_copy)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results


def main():
    print("\n" + "=" * 70)
    print("Mock Retriever Demo - Testing Fusion Logic")
    print("=" * 70)
    print()
    
    # Create mock data for two retrievers
    units1 = [
        TextUnit(unit_id="doc1_chunk1", content="Machine learning is awesome"),
        TextUnit(unit_id="doc1_chunk2", content="Deep learning uses neural networks"),
        TextUnit(unit_id="doc1_chunk3", content="AI is transforming industries"),
    ]
    
    units2 = [
        TextUnit(unit_id="doc2_chunk1", content="RAG combines retrieval and generation"),
        TextUnit(unit_id="doc2_chunk2", content="Vector databases enable semantic search"),
        TextUnit(unit_id="doc1_chunk1", content="Machine learning is awesome"),  # Duplicate
    ]
    
    # Create mock retrievers
    retriever1 = MockRetriever("Retriever-1", units1)
    retriever2 = MockRetriever("Retriever-2", units2)
    
    print("Test 1: SIMPLE Fusion Mode")
    print("-" * 70)
    fusion_simple = QueryFusionRetriever(
        retrievers=[retriever1, retriever2],
        mode=FusionMode.SIMPLE,
        top_k=5
    )
    results = fusion_simple.retrieve("test query")
    print(f"Retrieved {len(results)} unique units:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [Score: {unit.score:.4f}] {unit.content}")
    print()
    
    print("Test 2: RECIPROCAL_RANK Fusion Mode")
    print("-" * 70)
    fusion_rrf = QueryFusionRetriever(
        retrievers=[retriever1, retriever2],
        mode=FusionMode.RECIPROCAL_RANK,
        top_k=5
    )
    results = fusion_rrf.retrieve("test query")
    print(f"Retrieved {len(results)} units with RRF scores:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [RRF Score: {unit.score:.4f}] {unit.content}")
    print()
    
    print("Test 3: RELATIVE_SCORE Fusion Mode")
    print("-" * 70)
    fusion_relative = QueryFusionRetriever(
        retrievers=[retriever1, retriever2],
        mode=FusionMode.RELATIVE_SCORE,
        top_k=5,
        retriever_weights=[0.7, 0.3]
    )
    results = fusion_relative.retrieve("test query")
    print(f"Retrieved {len(results)} units with weighted scores:")
    for i, unit in enumerate(results, 1):
        print(f"  {i}. [Weighted Score: {unit.score:.4f}] {unit.content}")
    print()
    
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
