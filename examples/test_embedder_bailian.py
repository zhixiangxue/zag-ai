"""
Test Bailian Embedder

This example demonstrates how to use the unified Embedder interface
with Bailian (DashScope) provider.

Before running:
1. Install dependencies: pip install openai python-dotenv
2. Set your API key in .env file: BAILIAN_API_KEY=your-key
"""

import os
from dotenv import load_dotenv
from zag import Embedder

# Load environment variables
load_dotenv()

# ============ Configuration ============
API_KEY = os.getenv("BAILIAN_API_KEY")

# Model to use
MODEL = "text-embedding-v3"

# Optional: Custom dimension (if supported by the model)
DIMENSION = None  # Set to e.g., 1024 if you want to specify


def test_simple_uri():
    """Test embedder with simple URI format"""
    print("\n" + "=" * 60)
    print("Test 1: Simple URI Format")
    print("=" * 60)
    
    # Create embedder using simple URI format
    embedder = Embedder(
        f"bailian/{MODEL}",
        api_key=API_KEY,
        dimensions=DIMENSION
    )
    
    print(f"Embedder: {embedder}")
    print(f"Provider: {embedder.provider}")
    print(f"Model: {embedder.model}")
    
    # Test single text embedding
    text = "Hello, this is a test."
    print(f"\nEmbedding text: '{text}'")
    vector = embedder.embed(text)
    print(f"Vector dimension: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    
    # Verify dimension property
    print(f"\nEmbedder dimension: {embedder.dimension}")


def test_full_uri():
    """Test embedder with full URI format"""
    print("\n" + "=" * 60)
    print("Test 2: Full URI Format")
    print("=" * 60)
    
    # Create embedder using full URI format with all parameters
    uri = f"bailian@https://dashscope.aliyuncs.com/compatible-mode/v1:{MODEL}"
    if DIMENSION:
        uri += f"?dimensions={DIMENSION}"
    
    embedder = Embedder(uri, api_key=API_KEY)
    
    print(f"URI: {uri}")
    print(f"Embedder: {embedder}")
    
    # Test embedding
    text = "Another test sentence."
    vector = embedder.embed(text)
    print(f"\nEmbedded: '{text}'")
    print(f"Vector dimension: {len(vector)}")


def test_batch_embedding():
    """Test batch embedding"""
    print("\n" + "=" * 60)
    print("Test 3: Batch Embedding")
    print("=" * 60)
    
    embedder = Embedder(
        f"bailian/{MODEL}",
        api_key=API_KEY,
        dimensions=DIMENSION
    )
    
    # Prepare multiple texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Embedding models convert text into dense vectors.",
        "Natural language processing enables computers to understand human language.",
    ]
    
    print(f"Embedding {len(texts)} texts...")
    vectors = embedder.embed_batch(texts)
    
    print(f"Result: {len(vectors)} vectors")
    print(f"Each vector dimension: {len(vectors[0])}")
    
    # Show first few values of each vector
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        print(f"\nText {i+1}: '{text[:50]}...'")
        print(f"Vector preview: {vec[:3]}")


def test_parameter_format():
    """Test embedder with parameter format (non-URI)"""
    print("\n" + "=" * 60)
    print("Test 4: Parameter Format")
    print("=" * 60)
    
    # Create embedder using traditional parameters
    embedder = Embedder(
        provider="bailian",
        model=MODEL,
        api_key=API_KEY,
        dimensions=DIMENSION,
        timeout=60,  # Custom timeout
    )
    
    print(f"Embedder: {embedder}")
    
    text = "Testing with parameter format."
    vector = embedder.embed(text)
    print(f"\nEmbedded: '{text}'")
    print(f"Vector dimension: {len(vector)}")


def test_similarity():
    """Test semantic similarity using embeddings"""
    print("\n" + "=" * 60)
    print("Test 5: Semantic Similarity")
    print("=" * 60)
    
    embedder = Embedder(f"bailian/{MODEL}", api_key=API_KEY)
    
    # Prepare test texts
    texts = [
        "I love programming in Python.",
        "Python is my favorite programming language.",
        "I enjoy eating pizza for dinner.",
    ]
    
    print("Embedding texts...")
    vectors = embedder.embed_batch(texts)
    
    # Calculate cosine similarity
    import math
    
    def cosine_similarity(v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot_product / (norm1 * norm2)
    
    print("\nSimilarity scores:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"  '{texts[i][:40]}...' <-> '{texts[j][:40]}...': {sim:.4f}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Bailian Embedder Tests")
    print("=" * 60)
    
    if API_KEY == "your-api-key-here":
        print("\n⚠️  WARNING: Please set your API_KEY before running!")
        print("Edit this file and replace 'your-api-key-here' with your actual API key.")
        return
    
    try:
        # Run all test functions
        test_simple_uri()
        test_full_uri()
        test_batch_embedding()
        test_parameter_format()
        test_similarity()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Error occurred: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
