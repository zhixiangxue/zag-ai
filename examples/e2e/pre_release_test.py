#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-Release E2E Test - Comprehensive RAG Pipeline Validation

This test validates the complete ZagPy RAG pipeline before PyPI release.

== PREREQUISITES ==
1. Ollama installed and running:
   - Download: https://ollama.ai/download
   - Windows: Run installer, service starts automatically
   - Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh

2. Pull the embedding model:
   ollama pull nomic-embed-text

3. Verify Ollama is running:
   ollama list  # Should show nomic-embed-text

== WHAT THIS TEST DOES ==
- ✓ Reads PDF documents (DoclingReader)
- ✓ Splits text into chunks (RecursiveMergingSplitter)
- ✓ Creates embeddings (Ollama + nomic-embed-text)
- ✓ Indexes in vector database (ChromaDB local)
- ✓ Performs semantic retrieval (VectorRetriever)
- ✓ Applies postprocessing (SimilarityFilter)

== NO EXTERNAL SERVICES NEEDED ==
- No Meilisearch
- No Qdrant server
- No cloud APIs
- All data stored locally in tmp/

Usage:
    python examples/e2e/pre_release_test.py
"""
import sys
import time
from pathlib import Path

# Ensure we can import zag from installed package
try:
    import zag
    print(f"✓ ZagPy version: {zag.__version__}")
except ImportError as e:
    print(f"✗ Failed to import zag: {e}")
    print("  Make sure zagpy is installed: pip install zagpy")
    sys.exit(1)

from zag.readers import DoclingReader
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from zag.splitters import MarkdownHeaderSplitter, RecursiveMergingSplitter
from zag.parsers import TableParser
from zag.embedders import Embedder
from zag.storages.vector import ChromaVectorStore
from zag.indexers import VectorIndexer
from zag.retrievers import VectorRetriever
from zag.postprocessors import SimilarityFilter, ChainPostprocessor


# Configuration
TEST_FILE = Path(__file__).parent.parent / "files" / "mortgage_products.pdf"
CHROMA_PERSIST_DIR = Path(__file__).parent.parent.parent / "tmp" / "pre_release_chroma"
EMBEDDING_MODEL = "ollama/jina/jina-embeddings-v2-base-en"  # Use ollama with local model


def print_section(title: str, char: str = "="):
    """Print section header"""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}\n")


def assert_test(condition: bool, message: str):
    """Assert with clear error message"""
    if not condition:
        print(f"\n✗ ASSERTION FAILED: {message}")
        sys.exit(1)
    print(f"✓ {message}")


def test_step1_read_document():
    """Step 1: Read document"""
    print_section("Step 1: Read Document", "-")
    
    # Check test file exists
    assert_test(TEST_FILE.exists(), f"Test file exists: {TEST_FILE.name}")
    
    # Read document with DoclingReader (more stable for PDFs)
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CPU  # Use CPU to avoid platform issues
    )
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    doc = reader.read(str(TEST_FILE))
    
    assert_test(doc is not None, "Document loaded successfully")
    assert_test(len(doc.content) > 0, f"Document has content ({len(doc.content)} chars)")
    assert_test(doc.metadata.file_name == TEST_FILE.name, "Metadata contains correct filename")
    
    print(f"  Content preview: {doc.content[:100]}...")
    return doc


def test_step2_split_document(doc):
    """Step 2: Split document into chunks"""
    print_section("Step 2: Split Document", "-")
    
    base_splitter = MarkdownHeaderSplitter()
    merger = RecursiveMergingSplitter(
        base_splitter=base_splitter,
        target_token_size=500
    )
    units = doc.split(merger)
    
    assert_test(len(units) > 0, f"Document split into {len(units)} units")
    assert_test(all(u.content for u in units), "All units have content")
    assert_test(all(u.unit_id for u in units), "All units have unique IDs")
    
    print(f"  Sample unit ID: {units[0].unit_id}")
    print(f"  Sample content: {units[0].content[:80]}...")
    return units


def test_step3_parse_tables(units):
    """Step 3: Parse tables in content (validation only)"""
    print_section("Step 3: Validate Content", "-")
    
    # Just validate units have content, skip table parsing for now
    # (Table parsing is optional feature, not core requirement)
    print(f"  Validated {len(units)} units with content")
    assert_test(all(u.content for u in units), "All units have valid content")
    assert_test(all(u.unit_type for u in units), "All units have type")
    
    # Check if any unit has markdown tables (string pattern)
    units_with_tables = sum(1 for u in units if '|' in u.content and '---' in u.content)
    if units_with_tables > 0:
        print(f"  Found {units_with_tables} units containing markdown table patterns")
    
    return units


def test_step4_create_embeddings(units):
    """Step 4: Create embeddings and build index"""
    print_section("Step 4: Create Embeddings & Index", "-")
    
    print(f"  Using embedding model: {EMBEDDING_MODEL}")
    print(f"  Note: Requires Ollama running locally with jina-embeddings-v2-base-en model")
    print(f"  Install: ollama pull jina/jina-embeddings-v2-base-en")
    print()
    
    # Create embedder (requires Ollama)
    try:
        embedder = Embedder(EMBEDDING_MODEL)
    except Exception as e:
        print(f"\n⚠️  Failed to create embedder: {e}")
        print(f"\n" + "=" * 70)
        print(f"❌ PREREQUISITE MISSING: Ollama")
        print(f"=" * 70)
        print(f"\n💡 This test requires Ollama with jina-embeddings-v2-base-en model.")
        print(f"\nQuick setup (takes 2 minutes):")
        print(f"\n  1. Install Ollama:")
        print(f"     Windows: https://ollama.ai/download/windows")
        print(f"     Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print(f"     Mac: brew install ollama")
        print(f"\n  2. Pull embedding model:")
        print(f"     ollama pull jina/jina-embeddings-v2-base-en")
        print(f"\n  3. Verify it's running:")
        print(f"     ollama list")
        print(f"\n  4. Re-run this test")
        print(f"\n" + "=" * 70)
        raise
    
    # Test embedding creation
    test_text = "This is a test sentence for embedding."
    embedding = embedder.embed(test_text)
    
    assert_test(embedding is not None, "Embedder created successfully")
    assert_test(len(embedding) > 0, f"Embedding created (dim={len(embedding)})")
    
    # Create vector store (local, no server needed)
    vector_store = ChromaVectorStore.local(
        path=str(CHROMA_PERSIST_DIR),
        collection_name="pre_release_test",
        embedder=embedder
    )
    
    print(f"  ChromaDB path: {CHROMA_PERSIST_DIR}")
    
    # Create indexer
    indexer = VectorIndexer(vector_store=vector_store)
    
    # Clear existing data (ensure clean test)
    indexer.clear()
    print(f"  Cleared existing data")
    
    # Add units to index
    print(f"  Indexing {len(units)} units...")
    start = time.time()
    indexer.add(units)
    duration = time.time() - start
    
    count = indexer.count()
    assert_test(count == len(units), f"All {len(units)} units indexed (took {duration:.1f}s)")
    
    return indexer


def test_step5_retrieve(indexer):
    """Step 5: Test retrieval"""
    print_section("Step 5: Test Retrieval", "-")
    
    # Create retriever
    retriever = VectorRetriever(vector_store=indexer.vector_store, top_k=5)
    
    # Test queries
    test_queries = [
        "What are the interest rates for mortgages?",
        "What is required for loan eligibility?",
        "Tell me about closing costs"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = retriever.retrieve(query, top_k=3)
        
        assert_test(len(results) > 0, f"Retrieved {len(results)} results")
        assert_test(all(hasattr(r, 'score') for r in results), "All results have scores")
        assert_test(all(r.content for r in results), "All results have content")
        
        # Check scores are descending
        scores = [r.score for r in results if r.score]
        if len(scores) > 1:
            assert_test(
                all(scores[i] >= scores[i+1] for i in range(len(scores)-1)),
                "Results sorted by score (descending)"
            )
        
        # Print top result
        if results:
            top_result = results[0]
            print(f"    Top result (score={top_result.score:.3f}): {top_result.content[:60]}...")
    
    return retriever


def test_step6_postprocessing(retriever):
    """Step 6: Test postprocessing"""
    print_section("Step 6: Test Postprocessing", "-")
    
    query = "What are the mortgage interest rates?"
    
    # Retrieve without postprocessing
    raw_results = retriever.retrieve(query, top_k=10)
    print(f"  Raw results: {len(raw_results)} units")
    
    # Apply postprocessing
    postprocessor = ChainPostprocessor([
        SimilarityFilter(threshold=0.3)
    ])
    
    filtered_results = postprocessor.process(query, raw_results)
    print(f"  After filtering: {len(filtered_results)} units")
    
    assert_test(len(filtered_results) > 0, "Postprocessing returns results")
    assert_test(len(filtered_results) <= len(raw_results), "Filtering reduces or maintains result count")
    
    # Check filtered results meet threshold
    for result in filtered_results:
        if hasattr(result, 'score') and result.score:
            assert_test(result.score >= 0.3, f"Filtered result meets threshold (score={result.score:.3f})")
    
    return filtered_results


def main():
    """Main test runner"""
    print("\n" + "=" * 70)
    print("  🧪 ZagPy Pre-Release E2E Test")
    print("=" * 70)
    print("\nValidating complete RAG pipeline before PyPI release:")
    print("  ✓ Local test files (no network required)")
    print("  ✓ Local embedding via Ollama (no cloud API)")
    print("  ✓ Local vector storage ChromaDB (no server)")
    print("\n⚠️  Prerequisites: Ollama with jina-embeddings-v2-base-en model")
    print("   Quick check: ollama list | grep jina-embeddings-v2-base-en")
    print()
    
    start_time = time.time()
    
    try:
        # Execute pipeline
        doc = test_step1_read_document()
        units = test_step2_split_document(doc)
        units = test_step3_parse_tables(units)
        indexer = test_step4_create_embeddings(units)
        retriever = test_step5_retrieve(indexer)
        results = test_step6_postprocessing(retriever)
        
        total_time = time.time() - start_time
        
        # Summary
        print_section("Test Summary")
        print(f"✅ All tests passed in {total_time:.1f}s")
        print()
        print("Pipeline validated:")
        print(f"  1. ✅ Document reading: DoclingReader (PDF)")
        print(f"  2. ✅ Document splitting: RecursiveMergingSplitter")
        print(f"  3. ✅ Content validation: TextUnit structure")
        print(f"  4. ✅ Embedding & indexing: ChromaDB local + sentence-transformers")
        print(f"  5. ✅ Retrieval: VectorRetriever")
        print(f"  6. ✅ Postprocessing: SimilarityFilter")
        print()
        print("💡 ZagPy is ready for release!")
        print()
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ Test Failed")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Please fix the errors before releasing to PyPI.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
