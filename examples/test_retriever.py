"""
Demo script to test retriever implementation

Tests:
1. VectorRetriever - basic layer retriever
2. QueryFusionRetriever - composite layer retriever with different fusion modes

Before running:
1. Install dependencies: pip install python-dotenv
2. Set your API key in .env file: BAILIAN_API_KEY=your-key
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from zag.embedders import Embedder
from zag.storages.vector.chroma import ChromaVectorStore
from zag.retrievers import VectorRetriever, QueryFusionRetriever, FusionMode
from zag.schemas.unit import TextUnit

# Load environment variables
load_dotenv()


def setup_vector_stores():
    """Create two vector stores with sample data for testing"""
    
    # Create embedder (you'll need to provide your API key)
    embedder = Embedder(
        "bailian/text-embedding-v3",
        api_key=os.getenv("BAILIAN_API_KEY")
    )
    
    # Create two separate vector stores (simulating different data sources)
    print("Creating vector stores...")
    store1 = ChromaVectorStore(
        embedder=embedder,
        collection_name="test_collection_1",
        persist_directory="./chroma_db_1"
    )
    
    store2 = ChromaVectorStore(
        embedder=embedder,
        collection_name="test_collection_2",
        persist_directory="./chroma_db_2"
    )
    
    # Clear existing data
    store1.clear()
    store2.clear()
    
    # Add sample data to store1 (ML/AI focused)
    print("Adding data to store 1 (ML/AI focused)...")
    units1 = [
        TextUnit(
            unit_id="ml_1",
            content="Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        ),
        TextUnit(
            unit_id="ml_2",
            content="Deep learning uses neural networks with multiple layers to process complex patterns in data."
        ),
        TextUnit(
            unit_id="ml_3",
            content="Supervised learning involves training models on labeled data to make predictions."
        ),
        TextUnit(
            unit_id="ml_4",
            content="Natural language processing (NLP) helps computers understand and generate human language."
        ),
    ]
    store1.add(units1)
    
    # Add sample data to store2 (RAG/Vector DB focused)
    print("Adding data to store 2 (RAG/Vector DB focused)...")
    units2 = [
        TextUnit(
            unit_id="rag_1",
            content="RAG (Retrieval-Augmented Generation) combines information retrieval with language generation."
        ),
        TextUnit(
            unit_id="rag_2",
            content="Vector databases store embeddings and enable efficient similarity search for semantic retrieval."
        ),
        TextUnit(
            unit_id="rag_3",
            content="ChromaDB is an open-source vector database designed for AI applications."
        ),
        TextUnit(
            unit_id="rag_4",
            content="Embedding models convert text into numerical vectors that capture semantic meaning."
        ),
    ]
    store2.add(units2)
    
    print(f"Store 1 count: {store1.count()}")
    print(f"Store 2 count: {store2.count()}")
    print()
    
    return store1, store2


def test_vector_retriever(store):
    """Test basic VectorRetriever"""
    
    print("=" * 70)
    print("TEST 1: VectorRetriever (Basic Layer)")
    print("=" * 70)
    
    retriever = VectorRetriever(vector_store=store, top_k=3)
    
    query = "What is machine learning?"
    print(f"Query: {query}\n")
    
    results = retriever.retrieve(query)
    
    print(f"Retrieved {len(results)} units:")
    for i, unit in enumerate(results, 1):
        print(f"{i}. [Score: {unit.score:.4f}] {unit.content[:80]}...")
    print()


def test_fusion_retriever_simple(store1, store2):
    """Test FusionRetriever with SIMPLE mode"""
    
    print("=" * 70)
    print("TEST 2: QueryFusionRetriever - SIMPLE Mode")
    print("=" * 70)
    
    # Create two vector retrievers
    retriever1 = VectorRetriever(vector_store=store1, top_k=3)
    retriever2 = VectorRetriever(vector_store=store2, top_k=3)
    
    # Combine with simple fusion
    fusion_retriever = QueryFusionRetriever(
        retrievers=[retriever1, retriever2],
        mode=FusionMode.SIMPLE,
        top_k=5
    )
    
    query = "How does semantic search work?"
    print(f"Query: {query}\n")
    
    results = fusion_retriever.retrieve(query)
    
    print(f"Fused results ({len(results)} units):")
    for i, unit in enumerate(results, 1):
        print(f"{i}. [Score: {unit.score:.4f}] [ID: {unit.unit_id}] {unit.content[:60]}...")
    print()


def test_fusion_retriever_rrf(store1, store2):
    """Test FusionRetriever with RECIPROCAL_RANK mode"""
    
    print("=" * 70)
    print("TEST 3: QueryFusionRetriever - RECIPROCAL_RANK Mode")
    print("=" * 70)
    
    # Create two vector retrievers
    retriever1 = VectorRetriever(vector_store=store1, top_k=3)
    retriever2 = VectorRetriever(vector_store=store2, top_k=3)
    
    # Combine with RRF fusion
    fusion_retriever = QueryFusionRetriever(
        retrievers=[retriever1, retriever2],
        mode=FusionMode.RECIPROCAL_RANK,
        top_k=5
    )
    
    query = "What is deep learning and how is it used?"
    print(f"Query: {query}\n")
    
    results = fusion_retriever.retrieve(query)
    
    print(f"Fused results ({len(results)} units):")
    for i, unit in enumerate(results, 1):
        print(f"{i}. [RRF Score: {unit.score:.4f}] [ID: {unit.unit_id}] {unit.content[:60]}...")
    print()


def test_fusion_retriever_relative(store1, store2):
    """Test FusionRetriever with RELATIVE_SCORE mode"""
    
    print("=" * 70)
    print("TEST 4: QueryFusionRetriever - RELATIVE_SCORE Mode")
    print("=" * 70)
    
    # Create two vector retrievers
    retriever1 = VectorRetriever(vector_store=store1, top_k=3)
    retriever2 = VectorRetriever(vector_store=store2, top_k=3)
    
    # Combine with relative score fusion (with custom weights)
    fusion_retriever = QueryFusionRetriever(
        retrievers=[retriever1, retriever2],
        mode=FusionMode.RELATIVE_SCORE,
        top_k=5,
        retriever_weights=[0.6, 0.4]  # Give more weight to store1
    )
    
    query = "Explain vector embeddings"
    print(f"Query: {query}\n")
    print("Weights: Store1=0.6, Store2=0.4\n")
    
    results = fusion_retriever.retrieve(query)
    
    print(f"Fused results ({len(results)} units):")
    for i, unit in enumerate(results, 1):
        print(f"{i}. [Weighted Score: {unit.score:.4f}] [ID: {unit.unit_id}] {unit.content[:60]}...")
    print()


def main():
    """Main test function"""
    
    print("\n" + "=" * 70)
    print("Retriever Implementation Demo")
    print("=" * 70)
    print()
    
    # Setup
    print("Setting up test environment...")
    store1, store2 = setup_vector_stores()
    
    # Test 1: Basic VectorRetriever
    test_vector_retriever(store1)
    
    # Test 2: Fusion with SIMPLE mode
    test_fusion_retriever_simple(store1, store2)
    
    # Test 3: Fusion with RECIPROCAL_RANK mode
    test_fusion_retriever_rrf(store1, store2)
    
    # Test 4: Fusion with RELATIVE_SCORE mode
    test_fusion_retriever_relative(store1, store2)
    
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
