#!/usr/bin/env python3
"""
E2E RAG Pipeline Test - Complete workflow demonstration

This example demonstrates a complete RAG pipeline:
1. Document Reading (PDF)
2. Document Splitting (Markdown header-based + recursive merging)
3. Table Processing (TextUnit + TableUnit parallel processing)
4. Metadata Extraction (keywords)
5. Indexing (Vector + FullText)
6. Retrieval (Fusion retrieval with multiple strategies)
7. Postprocessing (filter + deduplicate + context augmentation)

Before running:
1. Start services:
   bash rag-service/playground/start_services.sh
   (This will start Qdrant and Meilisearch)

2. Set environment variables in .env:
   BAILIAN_API_KEY=your-api-key

3. Prepare test PDF: examples/files/mortgage_products.pdf

Note: This test uses collection 'e2e_rag_test' (NOT 'mortgage_guidelines')
"""

from rich import print as rich_print
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from zag.postprocessors import (
    SimilarityFilter,
    Deduplicator,
    ContextAugmentor,
)
from zag.retrievers import VectorRetriever, FullTextRetriever, QueryFusionRetriever, FusionMode
from zag.indexers import VectorIndexer, FullTextIndexer
from zag.storages.vector import QdrantVectorStore
from zag.embedders import Embedder
from zag.extractors import KeywordExtractor, TableEnricher, TableSummarizer
from zag.parsers import TableParser
from zag.splitters import MarkdownHeaderSplitter, RecursiveMergingSplitter
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from zag.readers.docling import DoclingReader
import os
import sys
import time
import asyncio
import tempfile
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Set HuggingFace mirror BEFORE any imports that might use it
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Force offline mode if models are already cached
os.environ['HF_HUB_OFFLINE'] = '1'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


console = Console()

# Load environment
load_dotenv()

# Generate unique run ID and create run directory
RUN_ID = uuid.uuid4().hex[:8]
RUN_DIR = Path(tempfile.gettempdir()) / f"rag_example_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
API_KEY = os.getenv("BAILIAN_API_KEY")
EMBEDDING_MODEL = "text-embedding-v3"
LLM_MODEL = "qwen-plus"
EMBEDDING_URI = f"bailian/{EMBEDDING_MODEL}"
LLM_URI = f"bailian/{LLM_MODEL}"
MEILISEARCH_URL = "http://127.0.0.1:7700"
QDRANT_HOST = "localhost"
QDRANT_PORT = 16333  # Custom port from qdrant_conf.yaml
QDRANT_GRPC_PORT = 16334  # Custom gRPC port from qdrant_conf.yaml
COLLECTION_NAME = "e2e_rag_test"  # ⚠️ Do NOT use 'mortgage_guidelines' - it exists in production!


def normalize_path(path_str: str) -> Path:
    """
    Normalize file path by removing quotes and handling drag & drop paths.
    
    Handles:
    - Single quotes: '/path/to/file'
    - Double quotes: "/path/to/file"
    - Escaped spaces: /path/to/my\ file.pdf
    """
    path_str = path_str.strip()
    
    # Remove surrounding quotes
    if (path_str.startswith("'") and path_str.endswith("'")) or \
       (path_str.startswith('"') and path_str.endswith('"')):
        path_str = path_str[1:-1]
    
    # Handle escaped spaces (from drag & drop)
    path_str = path_str.replace("\\ ", " ")
    
    return Path(path_str)


def get_pdf_path() -> Path:
    """Get PDF file path from user input"""
    print("\n" + "=" * 70)
    print("  📄 PDF File Selection")
    print("=" * 70)
    print("\nEnter PDF file path (or drag & drop file):")
    print("Example: /path/to/document.pdf")
    print()
    
    path_input = input("PDF Path: ").strip()
    
    if not path_input:
        print("❌ No file path provided")
        sys.exit(1)
    
    pdf_path = normalize_path(path_input)
    
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    if pdf_path.suffix.lower() != '.pdf':
        print(f"❌ Not a PDF file: {pdf_path}")
        sys.exit(1)
    
    print(f"✅ PDF file selected: {pdf_path.name}")
    return pdf_path


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
    table = Table(title=f"{strategy} Results for: '{query}'",
                  show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Score", style="green", width=10)
    table.add_column("Type", style="yellow", width=8)
    table.add_column("Unit ID", style="blue", width=20)
    table.add_column("Caption/Content Preview", style="white", width=60)

    for i, unit in enumerate(results[:5], 1):  # Show top 5
        score = f"{unit.score:.4f}" if hasattr(
            unit, 'score') and unit.score else "N/A"
        
        # Get unit type
        unit_type = unit.unit_type.value if hasattr(unit, 'unit_type') else "unknown"
        
        unit_id = unit.unit_id[:17] + \
            "..." if len(unit.unit_id) > 20 else unit.unit_id
        
        # For TableUnit, show caption; for others, show content preview
        from zag.schemas.unit import TableUnit
        if isinstance(unit, TableUnit) and hasattr(unit, 'caption') and unit.caption:
            preview = f"[Caption] {unit.caption}"
            if len(preview) > 60:
                preview = preview[:57] + "..."
        else:
            content_preview = unit.content[:57] + \
                "..." if len(unit.content) > 60 else unit.content
            preview = content_preview.replace("\n", " ")

        table.add_row(str(i), score, unit_type, unit_id, preview)

    console.print(table)


def check_prerequisites(pdf_path: Path):
    """Check if all prerequisites are met"""
    print_section("🔍 Checking Prerequisites")

    issues = []

    # Check API key
    if not API_KEY:
        issues.append("❌ BAILIAN_API_KEY not found in .env file")
    else:
        print(f"✅ API Key found: {API_KEY[:10]}...")

    # Check PDF file
    if not pdf_path.exists():
        issues.append(f"❌ PDF file not found: {pdf_path}")
    else:
        print(f"✅ PDF file exists: {pdf_path.name}")

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Try to get collections (will fail if Qdrant is not running)
        client.get_collections()
        print(f"✅ Qdrant is running: {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        issues.append(f"❌ Cannot connect to Qdrant: {e}")

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


async def step1_read_document(pdf_path: Path):
    """Step 1: Read PDF document"""
    print_section("📄 Step 1: Read Document", "-")

    print(f"Reading PDF: {pdf_path.name}")
    print("Using DoclingReader with default pipeline (CPU)...")

    # Configure DoclingReader (use CPU to avoid platform issues)
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA
    )

    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    doc = reader.read(str(pdf_path))

    print(f"✅ Document loaded:")
    print(f"   - Type: {type(doc).__name__}")
    print(f"   - Content length: {len(doc.content):,} characters")
    print(f"   - File size: {doc.metadata.file_size:,} bytes")
    print(f"   - Pages: {len(doc.pages)}")
    if doc.metadata.custom:
        print(
            f"   - Text items: {doc.metadata.custom.get('text_items_count', 0)}")
        print(
            f"   - Table items: {doc.metadata.custom.get('table_items_count', 0)}")
        print(
            f"   - Picture items: {doc.metadata.custom.get('picture_items_count', 0)}")

    # Save markdown content to file
    markdown_path = RUN_DIR / "document_content.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(doc.content)
    print(f"\nMarkdown content saved to: {markdown_path}")

    return doc


async def step2_split_document(doc):
    """Step 2: Split document into chunks"""
    print_section("🔪 Step 2: Split Document", "-")

    print("Splitting with MarkdownHeaderSplitter | RecursiveMergingSplitter (target: 800 tokens)...")
    # Create pipeline: first split by headers, then merge small chunks
    pipeline = MarkdownHeaderSplitter() | RecursiveMergingSplitter(target_token_size=800)
    units = doc.split(pipeline)

    # Calculate token stats
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = [len(tokenizer.encode(u.content)) for u in units]

    print(f"✅ Split complete:")
    print(f"   - Total units: {len(units)}")
    print(
        f"   - Token range: {min(tokens)}-{max(tokens)} (avg: {sum(tokens)//len(tokens)})")

    return units


async def step3_process_tables(units, pdf_path):
    """Step 3: Process tables (TextUnit + TableUnit parallel processing)"""
    print_section("📊 Step 3: Process Tables (TextUnit + TableUnit)", "-")

    # 3.1 Process TextUnit embedding_content (replace tables with natural language)
    print("Step 3.1: Processing TextUnit embedding_content...")
    extractor = TableSummarizer(
        llm_uri=LLM_URI,
        api_key=API_KEY
    )
    results = await extractor.aextract(units)
    
    # Update embedding_content for TextUnits
    for unit, metadata in zip(units, results):
        if metadata.get("embedding_content"):
            unit.embedding_content = metadata["embedding_content"]
    
    print(f"   ✅ Processed {len(units)} TextUnits")
    
    # 3.2 Extract tables directly using CamelotReader
    print("\nStep 3.2: Extracting TableUnits using CamelotReader...")
    from zag.readers import CamelotReader
    
    # Use Camelot to extract tables directly from PDF
    camelot_reader = CamelotReader(flavor="lattice")  # Use lattice mode for bordered tables
    camelot_doc = camelot_reader.read(str(pdf_path))
    table_units = list(camelot_doc.units)  # Get TableUnits directly
    
    print(f"   ✅ Extracted {len(table_units)} TableUnits using CamelotReader")
    
    if table_units:
        # Show extracted table info
        print("\n   Extracted tables:")
        for i, table_unit in enumerate(table_units[:3], 1):  # Show first 3
            print(f"     {i}. unit_id: {table_unit.unit_id[:20]}...")
            print(f"        page: {table_unit.metadata.page_numbers}")
            if table_unit.df is not None:
                print(f"        df shape: {table_unit.df.shape}")
                print(f"        columns: {list(table_unit.df.columns)}")
        
        # 3.3 Enrich TableUnits with caption and embedding_content
        print("\nStep 3.3: Enriching TableUnits with LLM...")
        enricher = TableEnricher(
            llm_uri=LLM_URI,
            api_key=API_KEY,
            normalize_table=True  # Enable table structure normalization for complex tables
        )
        await enricher.aextract(table_units)
        print(f"   ✅ Enriched {len(table_units)} TableUnits")
        
        # Show enriched table info
        print("\n   Enriched tables:")
        for i, table_unit in enumerate(table_units[:3], 1):  # Show first 3
            print(f"     {i}. caption: {table_unit.caption}")
            if table_unit.embedding_content:
                print(f"        embedding_content: {table_unit.embedding_content[:80]}...")
    else:
        print("   ⚠️  No tables found in document")

    # Return all units (TextUnit + TableUnit)
    all_units = units + table_units
    print(f"\n📦 Total units: {len(units)} TextUnit + {len(table_units)} TableUnit = {len(all_units)}")
    return all_units


async def step4_extract_metadata(all_units):
    """Step 4: Extract metadata (keywords) for ALL units"""
    print_section("🏷️  Step 4: Extract Metadata", "-")

    print(f"Extracting keywords for {len(all_units)} units (TextUnit + TableUnit)...")
    extractor = KeywordExtractor(
        llm_uri=LLM_URI,
        api_key=API_KEY,
        num_keywords=5
    )

    # Extract from all units
    results = await extractor.aextract(all_units)

    # Update metadata for all units
    for unit, metadata in zip(all_units, results):
        unit.metadata.custom.update(metadata)

    print(f"✅ Extracted keywords for {len(all_units)} units")
    print("\nSample keywords (first 3 units):")
    for i, unit in enumerate(all_units[:3], 1):
        keywords = unit.metadata.custom.get("excerpt_keywords", [])
        unit_type = unit.unit_type.value if hasattr(unit, 'unit_type') else 'unknown'
        print(f"   {i}. [{unit_type}] {keywords}")

    return all_units


async def step5_build_indices(units):
    """Step 5: Build indices (Vector + FullText)"""
    print_section("📚 Step 5: Build Indices", "-")

    # Save units to JSON for inspection
    import json
    units_json_path = RUN_DIR / "units_debug.json"
    units_data = [unit.model_dump(mode='json') for unit in units]

    with open(units_json_path, 'w', encoding='utf-8') as f:
        json.dump(units_data, f, ensure_ascii=False, indent=2)

    print(f"Units saved to: {units_json_path}")
    print(f"Total units: {len(units)}\n")

    # 5.1 Vector Index (Qdrant)
    print("Building vector index with Qdrant...")
    embedder = Embedder(
        EMBEDDING_URI,
        api_key=API_KEY
    )

    vector_store = QdrantVectorStore.server(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=True,
        collection_name=COLLECTION_NAME,
        embedder=embedder
    )
    print(f"   Qdrant: {QDRANT_HOST}:{QDRANT_PORT} (gRPC: {QDRANT_GRPC_PORT})")
    print(f"   Collection: {COLLECTION_NAME}")

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
    """Step 6: Test different retrieval strategies (including table-specific queries)"""
    print_section("🔍 Step 6: Test Retrieval Strategies", "-")

    # Prepare test queries: including table-specific queries
    test_cases = [
        {
            "vector_query": "What is the interest rate for 30-Year Fixed Rate Mortgage?",
            "fulltext_query": "interest rate mortgage",
            "description": "Query for specific table data (interest rates)"
        },
        {
            "vector_query": "What are the LTV requirements?",
            "fulltext_query": "LTV requirements",
            "description": "Query for LTV table"
        },
        {
            "vector_query": "Show me comparison of different mortgage products",
            "fulltext_query": "mortgage products comparison",
            "description": "Query for product comparison table"
        },
        {
            "vector_query": "What are the APR rates for fixed mortgages?",
            "fulltext_query": "APR fixed mortgage",
            "description": "Query for APR data in tables"
        },
    ]

    # Create retrievers
    vector_retriever = VectorRetriever(
        vector_store=vector_indexer.vector_store, top_k=5)
    fulltext_retriever = FullTextRetriever(
        url=MEILISEARCH_URL, index_name="e2e_rag_test", top_k=5)

    results_summary = {}

    for idx, test_case in enumerate(test_cases, 1):
        vector_query = test_case["vector_query"]
        fulltext_query = test_case["fulltext_query"]
        description = test_case.get("description", "")

        console.print(f"\n[bold cyan]Test Case {idx}:[/bold cyan] {description}")
        console.print(f"  Vector query: '{vector_query}'")
        console.print(f"  FullText query: '{fulltext_query}'")
        console.print("─" * 70)

        # Strategy 1: Vector only
        start = time.time()
        vector_results = vector_retriever.retrieve(vector_query, top_k=3)
        vector_time = time.time() - start
        print(
            f"  Strategy 1 - Vector only: {len(vector_results)} results ({vector_time*1000:.0f}ms)")
        if vector_results:
            print_retrieval_results(
                vector_query, vector_results, "Vector Search")

        # Strategy 2: FullText only (use keyword query)
        start = time.time()
        fulltext_results = fulltext_retriever.retrieve(fulltext_query, top_k=3)
        fulltext_time = time.time() - start
        print(
            f"  Strategy 2 - FullText only: {len(fulltext_results)} results ({fulltext_time*1000:.0f}ms)")
        if fulltext_results:
            print_retrieval_results(
                fulltext_query, fulltext_results, "FullText Search")

        # Strategy 3: Fusion (SIMPLE) - use vector query for both
        fusion_simple = QueryFusionRetriever(
            retrievers=[vector_retriever, fulltext_retriever],
            mode=FusionMode.SIMPLE,
            top_k=3
        )
        start = time.time()
        fusion_results = fusion_simple.retrieve(vector_query)
        fusion_time = time.time() - start
        print(
            f"  Strategy 3 - Fusion (SIMPLE): {len(fusion_results)} results ({fusion_time*1000:.0f}ms)")

        # Strategy 4: Fusion (RRF) - use vector query for both
        fusion_rrf = QueryFusionRetriever(
            retrievers=[vector_retriever, fulltext_retriever],
            mode=FusionMode.RECIPROCAL_RANK,
            top_k=3
        )
        start = time.time()
        rrf_results = fusion_rrf.retrieve(vector_query)
        rrf_time = time.time() - start
        print(
            f"  Strategy 4 - Fusion (RRF): {len(rrf_results)} results ({rrf_time*1000:.0f}ms)")
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
    postprocessor = SimilarityFilter(threshold=0.6) | Deduplicator(
        strategy="exact") | ContextAugmentor(window_size=1)

    print("\nApplying postprocessing chain:")
    print("   1. SimilarityFilter(threshold=0.6)")
    print("   2. Deduplicator(strategy='exact')")
    print("   3. ContextAugmentor(window_size=1)")

    processed_results = postprocessor.process(query, raw_results)
    print(f"\n   ✅ Processed results: {len(processed_results)} units")

    # Show results with rich
    if processed_results:
        print_retrieval_results(
            query, processed_results, "After Postprocessing")
    else:
        console.print(
            "[yellow]No results after postprocessing (possibly filtered out)[/yellow]")

    return processed_results


async def test_table_retrieval_focused(vector_store):
    """Focused test: Table-specific retrieval (TableUnit)"""
    print_section("📋 Focused Test: Table Retrieval (TableUnit)", "=")

    # Create vector retriever
    vector_retriever = VectorRetriever(
        vector_store=vector_store, top_k=10)

    # Table-specific test queries
    table_queries = [
        {
            "query": "What is the interest rate for 30-Year Fixed Rate Mortgage?",
            "expected": "Should find table with rate and APR data"
        },
        {
            "query": "Show me the APR for 15-Year Fixed mortgage",
            "expected": "Should find table with APR column"
        },
        {
            "query": "What are the mortgage product rates?",
            "expected": "Should find product comparison table"
        },
        {
            "query": "Compare different mortgage products",
            "expected": "Should find product table with multiple rows"
        },
    ]

    console.print(f"\n[bold yellow]Testing {len(table_queries)} table-specific queries...[/bold yellow]\n")

    for idx, test_case in enumerate(table_queries, 1):
        query = test_case["query"]
        expected = test_case["expected"]

        console.print(f"[bold cyan]═══ Query {idx} ═══[/bold cyan]")
        console.print(f"[white]{query}[/white]")
        console.print(f"[dim italic]Expected: {expected}[/dim italic]")
        console.print("─" * 70)

        # Retrieve
        start = time.time()
        results = vector_retriever.retrieve(query, top_k=5)
        elapsed = time.time() - start

        console.print(f"  ⏱️  Retrieved [yellow]{len(results)}[/yellow] results in [yellow]{elapsed*1000:.0f}ms[/yellow]\n")

        # Analyze results
        from zag.schemas.unit import TableUnit
        table_units = [u for u in results if isinstance(u, TableUnit)]
        text_units = [u for u in results if not isinstance(u, TableUnit)]
        
        console.print(f"\n  📊 TableUnits: [yellow]{len(table_units)}[/yellow]  |  📄 TextUnits: [yellow]{len(text_units)}[/yellow]\n")
        
        # Use LLM to analyze relevance for TableUnits
        if table_units:
            console.print(f"[bold green]✅ Found {len(table_units)} TableUnit(s) - Analyzing with LLM...[/bold green]\n")
            
            # Define structured output schema
            from pydantic import BaseModel, Field
            
            class TableRelevanceAnalysis(BaseModel):
                is_relevant: bool = Field(description="Whether this table is relevant to the query")
                confidence: str = Field(description="Confidence level: high/medium/low")
                reason: str = Field(description="Why this table is relevant or not")
                key_evidence: list[str] = Field(description="Key data points from the table that answer the query (e.g., 'Interest Rate: 6.125%')")
            
            # Analyze each TableUnit with LLM
            import chak
            for i, table_unit in enumerate(table_units[:3], 1):  # Top 3 TableUnits
                # Prepare table data for LLM
                table_data = ""
                if hasattr(table_unit, 'df') and table_unit.df is not None:
                    # Convert DataFrame to readable format
                    table_data = table_unit.df.to_string(index=False)
                
                analysis_prompt = f"""Analyze whether this table is relevant to the user's query.

User Query: {query}

Table Caption: {table_unit.caption if table_unit.caption else 'N/A'}

Table Data:
{table_data}

Please analyze:
1. Is this table relevant to answering the query?
2. What is your confidence level (high/medium/low)?
3. Why is it relevant or not?
4. If relevant, extract the KEY DATA POINTS that directly answer the query (e.g., specific values from cells).

IMPORTANT: For key_evidence, extract actual cell values in format "Column: Value" (e.g., "Interest Rate: 6.125%", "APR: 6.275%").
"""
                
                try:
                    conv = chak.Conversation(LLM_URI, api_key=API_KEY)
                    analysis = await conv.asend(analysis_prompt, returns=TableRelevanceAnalysis)
                    
                    # Display analysis with Rich
                    relevance_color = "green" if analysis.is_relevant else "red"
                    relevance_icon = "✅" if analysis.is_relevant else "❌"
                    
                    panel_lines = []
                    panel_lines.append(f"[bold cyan]📊 TableUnit {i}: {table_unit.caption if table_unit.caption else 'Untitled'}[/bold cyan]")
                    panel_lines.append(f"")
                    panel_lines.append(f"[bold {relevance_color}]{relevance_icon} Relevant: {analysis.is_relevant} (Confidence: {analysis.confidence})[/bold {relevance_color}]")
                    panel_lines.append(f"[yellow]Reason:[/yellow] {analysis.reason}")
                    
                    if analysis.key_evidence:
                        panel_lines.append(f"")
                        panel_lines.append(f"[bold yellow]🔑 Key Evidence:[/bold yellow]")
                        for evidence in analysis.key_evidence:
                            panel_lines.append(f"  • [green]{evidence}[/green]")
                    
                    panel = Panel(
                        "\n".join(panel_lines),
                        border_style=relevance_color,
                        padding=(1, 2),
                        expand=False
                    )
                    console.print(panel)
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed to analyze TableUnit {i}: {e}[/red]")
        else:
            console.print(f"[bold yellow]⚠️  WARNING: No TableUnits found in top 5 results[/bold yellow]")

        console.print("\n")


async def main():
    """Main E2E workflow"""
    print("\n" + "=" * 70)
    print("  🚀 E2E RAG Pipeline Test")
    print("=" * 70)
    print(f"\n📁 Run ID: {RUN_ID}")
    print(f"📂 Working directory: {RUN_DIR}")
    print(f"   All output files will be saved here.\n")

    # Get PDF file path from user
    pdf_path = get_pdf_path()

    # Check prerequisites
    if not check_prerequisites(pdf_path):
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)

    try:
        # Execute pipeline
        start_time = time.time()

        doc = await step1_read_document(pdf_path)
        units = await step2_split_document(doc)

        rich_print(units[0])

        all_units = await step3_process_tables(units, pdf_path)
        all_units = await step4_extract_metadata(all_units)
        vector_indexer, fulltext_indexer = await step5_build_indices(all_units)
        
        # Focused table retrieval test
        await test_table_retrieval_focused(vector_indexer.vector_store)
        
        # Original retrieval tests
        vector_retriever, fulltext_retriever = await step6_test_retrieval(vector_indexer, fulltext_indexer)
        final_results = await step7_test_postprocessing(vector_retriever, fulltext_retriever)

        total_time = time.time() - start_time

        # Summary
        print_section("📊 Pipeline Summary")
        print(f"✅ E2E pipeline completed in {total_time:.2f}s")
        print("✅ Test completed successfully!")
        print(f"📂 All outputs saved to: {RUN_DIR}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Quick test: just connect to existing Qdrant and run table retrieval test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-table-only":
        async def test_only():
            print("\n" + "=" * 70)
            print("  🎯 Quick Test: TableUnit Retrieval Only")
            print("=" * 70 + "\n")
            
            # Connect to existing Qdrant collection
            embedder = Embedder(uri=EMBEDDING_URI, api_key=API_KEY)
            vector_store = QdrantVectorStore.server(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                grpc_port=QDRANT_GRPC_PORT,
                prefer_grpc=True,
                collection_name=COLLECTION_NAME,
                embedder=embedder
            )
            
            # Run table retrieval test
            await test_table_retrieval_focused(vector_store)
            
            print("\n" + "=" * 70)
            print("✅ Test completed!")
            print("=" * 70 + "\n")
        
        asyncio.run(test_only())
    else:
        asyncio.run(main())
