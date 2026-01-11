"""
Test Ollama Embedder

This example demonstrates how to use the unified Embedder interface
with Ollama provider for local embedding generation.

Before running:
1. Install dependencies: pip install httpx
2. Install Ollama: https://ollama.ai/download
3. Pull an embedding model: ollama pull nomic-embed-text
4. Ensure Ollama is running: ollama serve
"""

from zag import Embedder

# ============ Configuration ============
# Model to use (must be installed in Ollama)
MODEL = "jina/jina-embeddings-v2-base-en:latest"

# Base URL (default is http://localhost:11434)
BASE_URL = "http://localhost:11434"


def test_simple_uri():
    """Test embedder with simple URI format"""
    print("\n" + "=" * 60)
    print("Test 1: Simple URI Format")
    print("=" * 60)
    
    # Create embedder using simple URI format
    embedder = Embedder(f"ollama/{MODEL}")
    
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
    
    # Create embedder using full URI format with custom base URL
    uri = f"ollama@{BASE_URL}:{MODEL}"
    
    embedder = Embedder(uri)
    
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
    
    embedder = Embedder(f"ollama/{MODEL}")
    
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
        provider="ollama",
        model=MODEL,
        base_url=BASE_URL,
        timeout=90,  # Custom timeout (longer for local inference)
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
    
    embedder = Embedder(f"ollama/{MODEL}")
    
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


def test_different_models():
    """Test with different Ollama embedding models"""
    print("\n" + "=" * 60)
    print("Test 6: Different Models Comparison")
    print("=" * 60)
    
    # List of models to test (comment out if not installed)
    models = [
        "jina/jina-embeddings-v2-base-en:latest",
        # "nomic-embed-text",
        # "mxbai-embed-large",
        # "all-minilm",
    ]
    
    text = "This is a test sentence for comparison."
    
    for model in models:
        try:
            print(f"\n--- Model: {model} ---")
            embedder = Embedder(f"ollama/{model}")
            vector = embedder.embed(text)
            print(f"Dimension: {len(vector)}")
            print(f"First 3 values: {vector[:3]}")
        except Exception as e:
            print(f"Error with {model}: {e}")
            print(f"Make sure to run: ollama pull {model}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Ollama Embedder Tests")
    print("=" * 60)
    print("\nPrerequisites:")
    print("1. Ollama is installed and running")
    print("2. Model is pulled: ollama pull jina/jina-embeddings-v2-base-en:latest")
    print("=" * 60)
    
    try:
        # Run all test functions
        test_simple_uri()
        test_full_uri()
        test_batch_embedding()
        test_parameter_format()
        test_similarity()
        test_different_models()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Error occurred: {e}")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Check if Ollama is running: ollama list")
        print("2. Check if model is installed: ollama list")
        print("3. Try pulling the model: ollama pull jina/jina-embeddings-v2-base-en:latest")
        print("4. Check Ollama logs for errors")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
