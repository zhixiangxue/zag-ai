#!/usr/bin/env python3
"""
E2E RAG Pipeline Test - Complete workflow demonstration

This example demonstrates a complete RAG pipeline:
1. Document Reading (PDF)
2. Document Splitting (Markdown header-based + recursive merging)
3. Table Processing (parse + summarize)
4. Metadata Extraction (keywords)
5. Indexing (Vector + FullText)
6. Retrieval (Fusion retrieval with multiple strategies)
7. Postprocessing (filter + deduplicate + context augmentation)

Before running:
1. Start Meilisearch: ./meilisearch (download from https://github.com/meilisearch/meilisearch/releases)
2. Set environment variables in .env:
   BAILIAN_API_KEY=your-api-key
3. Prepare test PDF: tmp/Thunderbird Product Overview 2025 - No Doc.pdf
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from zag.readers.docling import DoclingReader
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from zag.splitters import MarkdownHeaderSplitter, RecursiveMergingSplitter
from zag.parsers import TableParser
from zag.extractors import TableExtractor, KeywordExtractor
from zag.embedders import Embedder
from zag.storages.vector import ChromaVectorStore
from zag.indexers import VectorIndexer, FullTextIndexer
from zag.retrievers import VectorRetriever, FullTextRetriever, QueryFusionRetriever, FusionMode
from zag.postprocessors import (
    SimilarityFilter,
    Deduplicator,
    ContextAugmentor,
    ChainPostprocessor,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# Load environment
load_dotenv()

# Configuration
API_KEY = os.getenv("BAILIAN_API_KEY")
EMBEDDING_MODEL = "text-embedding-v3"
LLM_MODEL = "qwen-plus"
EMBEDDING_URI = f"bailian/{EMBEDDING_MODEL}"
LLM_URI = f"bailian/{LLM_MODEL}"
MEILISEARCH_URL = "http://127.0.0.1:7700"
PDF_PATH = project_root / "examples" / "files" / "mortgage_products.pdf"
CHROMA_PERSIST_DIR = project_root / "tmp" / "chroma_db"


def print_section(title: str, char: str = "="):
    """Print section header"""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}\n")


def print_retrieval_results(query: str, results: list, strategy: str):
    """Print retrieval results using rich"""
    if not results:
        console.print(f"[yellow]No results found for: {query}[/yellow]")
        return
    
    # Create table
    table = Table(title=f"{strategy} Results for: '{query}'", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Score", style="green", width=10)
    table.add_column("Source", style="yellow", width=10)
    table.add_column("Unit ID", style="blue", width=25)
    table.add_column("Content Preview", style="white", width=50)
    
    for i, unit in enumerate(results[:3], 1):
        score = f"{unit.score:.4f}" if hasattr(unit, 'score') and unit.score else "N/A"
        source = unit.source.value if hasattr(unit, 'source') and unit.source else "N/A"
        unit_id = unit.unit_id[:22] + "..." if len(unit.unit_id) > 25 else unit.unit_id
        content_preview = unit.content[:47] + "..." if len(unit.content) > 50 else unit.content
        content_preview = content_preview.replace("\n", " ")
        
        table.add_row(str(i), score, source, unit_id, content_preview)
    
    console.print(table)


def check_prerequisites():
    """Check if all prerequisites are met"""
    print_section("🔍 Checking Prerequisites")
    
    issues = []
    
    # Check API key
    if not API_KEY:
        issues.append("❌ BAILIAN_API_KEY not found in .env file")
    else:
        print(f"✅ API Key found: {API_KEY[:10]}...")
    
    # Check PDF file
    if not PDF_PATH.exists():
        issues.append(f"❌ PDF file not found: {PDF_PATH}")
    else:
        print(f"✅ PDF file exists: {PDF_PATH.name}")
    
    # Check Meilisearch
    try:
        import meilisearch
        client = meilisearch.Client(MEILISEARCH_URL)
        health = client.health()
        if health.get("status") == "available":
            print(f"✅ Meilisearch is running: {MEILISEARCH_URL}")
        else:
            issues.append("❌ Meilisearch is not available")
    except Exception as e:
        issues.append(f"❌ Cannot connect to Meilisearch: {e}")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    
    print("\n✅ All prerequisites met!")
    return True


async def step1_read_document():
    """Step 1: Read PDF document"""
    print_section("📄 Step 1: Read Document", "-")
    
    print(f"Reading PDF: {PDF_PATH.name}")
    print("Using DoclingReader with default pipeline (CPU)...")
    
    # Configure DoclingReader (use CPU to avoid platform issues)
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CPU
    )
    
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    doc = reader.read(str(PDF_PATH))
    
    print(f"✅ Document loaded:")
    print(f"   - Type: {type(doc).__name__}")
    print(f"   - Content length: {len(doc.content):,} characters")
    print(f"   - File size: {doc.metadata.file_size:,} bytes")
    print(f"   - Pages: {len(doc.pages)}")
    if doc.metadata.custom:
        print(f"   - Text items: {doc.metadata.custom.get('text_items_count', 0)}")
        print(f"   - Table items: {doc.metadata.custom.get('table_items_count', 0)}")
        print(f"   - Picture items: {doc.metadata.custom.get('picture_items_count', 0)}")
    
    # Save markdown content to file
    markdown_path = project_root / "tmp" / "document_content.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(doc.content)
    print(f"\nMarkdown content saved to: {markdown_path}")
    
    return doc


async def step2_split_document(doc):
    """Step 2: Split document into chunks"""
    print_section("🔪 Step 2: Split Document", "-")
    
    print("Splitting with RecursiveMergingSplitter (target: 800 tokens)...")
    base_splitter = MarkdownHeaderSplitter()
    merger = RecursiveMergingSplitter(
        base_splitter=base_splitter,
        target_token_size=800
    )
    units = doc.split(merger)
    
    # Calculate token stats
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = [len(tokenizer.encode(u.content)) for u in units]
    
    print(f"✅ Split complete:")
    print(f"   - Total units: {len(units)}")
    print(f"   - Token range: {min(tokens)}-{max(tokens)} (avg: {sum(tokens)//len(tokens)})")
    
    return units


async def step3_process_tables(units):
    """Step 3: Process tables (parse + summarize)"""
    print_section("📊 Step 3: Process Tables", "-")
    
    extractor = TableExtractor(
        llm_uri=LLM_URI,
        api_key=API_KEY
    )
    
    # Extract from all units directly
    # TableExtractor will handle TextUnit with/without tables
    results = await extractor.aextract(units)
    
    # Update embedding_content for all units
    for unit, metadata in zip(units, results):
        if metadata.get("embedding_content"):
            unit.embedding_content = metadata["embedding_content"]
    
    print(f"✅ Processed {len(units)} units")
    return units


async def step4_extract_metadata(units):
    """Step 4: Extract metadata (keywords)"""
    print_section("🏷️  Step 4: Extract Metadata", "-")
    
    print("Extracting keywords for all units...")
    extractor = KeywordExtractor(
        llm_uri=LLM_URI,
        api_key=API_KEY,
        num_keywords=5
    )
    
    # Extract from all units
    results = await extractor.aextract(units)
    
    # Update metadata for all units
    for unit, metadata in zip(units, results):
        unit.metadata.custom.update(metadata)
    
    print(f"✅ Extracted keywords for {len(units)} units")
    print("\nSample keywords (first 3 units):")
    for i, unit in enumerate(units[:3], 1):
        keywords = unit.metadata.custom.get("excerpt_keywords", [])
        print(f"   {i}. {keywords}")
    
    return units


async def step5_build_indices(units):
    """Step 5: Build indices (Vector + FullText)"""
    print_section("📚 Step 5: Build Indices", "-")
    
    # Save units to JSON for inspection
    import json
    units_json_path = project_root / "tmp" / "units_debug.json"
    units_data = [unit.model_dump(mode='json') for unit in units]
    
    with open(units_json_path, 'w', encoding='utf-8') as f:
        json.dump(units_data, f, ensure_ascii=False, indent=2)
    
    print(f"Units saved to: {units_json_path}")
    print(f"Total units: {len(units)}\n")
    
    # 5.1 Vector Index
    print("Building vector index...")
    embedder = Embedder(
        EMBEDDING_URI,
        api_key=API_KEY
    )
    
    vector_store = ChromaVectorStore.local(
        path=str(CHROMA_PERSIST_DIR),
        collection_name="e2e_rag_test",
        embedder=embedder
    )
    print(f"   Persist directory: {CHROMA_PERSIST_DIR}")
    
    vector_indexer = VectorIndexer(vector_store=vector_store)
    # Clear existing data
    await vector_indexer.aclear()
    await vector_indexer.aadd(units)
    print(f"   ✅ Vector index built: {vector_indexer.count()} units")
    
    # 5.2 FullText Index
    print("\nBuilding fulltext index...")
    fulltext_indexer = FullTextIndexer(
        url=MEILISEARCH_URL,
        index_name="e2e_rag_test",
        primary_key="unit_id"
    )
    
    # Clear existing data
    fulltext_indexer.clear()
    fulltext_indexer.configure_settings(
        searchable_attributes=["content", "context_path"],
        filterable_attributes=["unit_type", "source_doc_id"],
        sortable_attributes=["created_at"],
    )
    fulltext_indexer.add(units)
    print(f"   ✅ Fulltext index built: {fulltext_indexer.count()} units")
    
    return vector_indexer, fulltext_indexer


async def step6_test_retrieval(vector_indexer, fulltext_indexer):
    """Step 6: Test different retrieval strategies"""
    print_section("🔍 Step 6: Test Retrieval Strategies", "-")
    
    # Prepare test queries: (natural language for vector, keywords for fulltext)
    test_cases = [
        {
            "vector_query": "What is the interest rate for 30-Year Fixed Rate Mortgage?",
            "fulltext_query": "interest rate mortgage",
        },
        {
            "vector_query": "What are the LTV requirements?",
            "fulltext_query": "LTV requirements",
        },
        {
            "vector_query": "Tell me about the loan terms available",
            "fulltext_query": "loan terms",
        },
    ]
    
    # Create retrievers
    vector_retriever = VectorRetriever(vector_store=vector_indexer.vector_store, top_k=5)
    fulltext_retriever = FullTextRetriever(url=MEILISEARCH_URL, index_name="e2e_rag_test", top_k=5)
    
    results_summary = {}
    
    for test_case in test_cases:
        vector_query = test_case["vector_query"]
        fulltext_query = test_case["fulltext_query"]
        
        console.print(f"\n[bold cyan]Test Case:[/bold cyan]")
        console.print(f"  Vector query: '{vector_query}'")
        console.print(f"  FullText query: '{fulltext_query}'")
        console.print("─" * 70)
        
        # Strategy 1: Vector only
        start = time.time()
        vector_results = vector_retriever.retrieve(vector_query, top_k=3)
        vector_time = time.time() - start
        print(f"  Strategy 1 - Vector only: {len(vector_results)} results ({vector_time*1000:.0f}ms)")
        if vector_results:
            print_retrieval_results(vector_query, vector_results, "Vector Search")
        
        # Strategy 2: FullText only (use keyword query)
        start = time.time()
        fulltext_results = fulltext_retriever.retrieve(fulltext_query, top_k=3)
        fulltext_time = time.time() - start
        print(f"  Strategy 2 - FullText only: {len(fulltext_results)} results ({fulltext_time*1000:.0f}ms)")
        if fulltext_results:
            print_retrieval_results(fulltext_query, fulltext_results, "FullText Search")
        
        # Strategy 3: Fusion (SIMPLE) - use vector query for both
        fusion_simple = QueryFusionRetriever(
            retrievers=[vector_retriever, fulltext_retriever],
            mode=FusionMode.SIMPLE,
            top_k=3
        )
        start = time.time()
        fusion_results = fusion_simple.retrieve(vector_query)
        fusion_time = time.time() - start
        print(f"  Strategy 3 - Fusion (SIMPLE): {len(fusion_results)} results ({fusion_time*1000:.0f}ms)")
        
        # Strategy 4: Fusion (RRF) - use vector query for both
        fusion_rrf = QueryFusionRetriever(
            retrievers=[vector_retriever, fulltext_retriever],
            mode=FusionMode.RECIPROCAL_RANK,
            top_k=3
        )
        start = time.time()
        rrf_results = fusion_rrf.retrieve(vector_query)
        rrf_time = time.time() - start
        print(f"  Strategy 4 - Fusion (RRF): {len(rrf_results)} results ({rrf_time*1000:.0f}ms)")
        if rrf_results:
            print_retrieval_results(vector_query, rrf_results, "Fusion (RRF)")
    
    return vector_retriever, fulltext_retriever


async def step7_test_postprocessing(vector_retriever, fulltext_retriever):
    """Step 7: Test postprocessing pipeline"""
    print_section("🔄 Step 7: Test Postprocessing", "-")
    
    query = "What are the interest rates for different mortgage products?"
    
    # Use Fusion (RRF) to retrieve
    print(f"Query: '{query}'")
    print("Retrieving with Fusion (RRF)...")
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, fulltext_retriever],
        mode=FusionMode.RECIPROCAL_RANK,
        top_k=10
    )
    raw_results = fusion_retriever.retrieve(query)
    print(f"   Raw results from fusion: {len(raw_results)} units")
    
    # Create postprocessing chain
    postprocessor = ChainPostprocessor([
        SimilarityFilter(threshold=0.6),
        Deduplicator(strategy="exact"),
        ContextAugmentor(window_size=1),
    ])
    
    print("\nApplying postprocessing chain:")
    print("   1. SimilarityFilter(threshold=0.6)")
    print("   2. Deduplicator(strategy='exact')")
    print("   3. ContextAugmentor(window_size=1)")
    
    processed_results = postprocessor.process(query, raw_results)
    print(f"\n   ✅ Processed results: {len(processed_results)} units")
    
    # Show results with rich
    if processed_results:
        print_retrieval_results(query, processed_results, "After Postprocessing")
    else:
        console.print("[yellow]No results after postprocessing (possibly filtered out)[/yellow]")
    
    return processed_results


async def main():
    """Main E2E workflow"""
    print("\n" + "=" * 70)
    print("  🚀 E2E RAG Pipeline Test")
    print("=" * 70)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    try:
        # Execute pipeline
        start_time = time.time()
        
        doc = await step1_read_document()
        units = await step2_split_document(doc)
        units = await step3_process_tables(units)
        units = await step4_extract_metadata(units)
        vector_indexer, fulltext_indexer = await step5_build_indices(units)
        vector_retriever, fulltext_retriever = await step6_test_retrieval(vector_indexer, fulltext_indexer)
        final_results = await step7_test_postprocessing(vector_retriever, fulltext_retriever)
        
        total_time = time.time() - start_time
        
        # Summary
        print_section("📊 Pipeline Summary")
        print(f"✅ E2E pipeline completed in {total_time:.2f}s")
        print(f"\nPipeline stages:")
        print(f"   1. ✅ Document reading: PDF → Markdown")
        print(f"   2. ✅ Document splitting: Header-based + Recursive merging")
        print(f"   3. ✅ Table processing: Parse + Summarize")
        print(f"   4. ✅ Metadata extraction: Keywords")
        print(f"   5. ✅ Indexing: Vector + FullText")
        print(f"   6. ✅ Retrieval: Multiple strategies tested")
        print(f"   7. ✅ Postprocessing: Filter + Deduplicate + Augment")
        
        print(f"\n💡 Key insights:")
        print(f"   - Total units indexed: {len(units)}")
        print(f"   - Fusion retrieval combines best of both worlds")
        print(f"   - Postprocessing improves result quality")
        print(f"   - Complete RAG pipeline is ready for production")
        
        print("\n" + "=" * 70)
        print("✅ Test completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
