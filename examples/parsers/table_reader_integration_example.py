"""Example: Run MinerUReader with TableParser and TableEnricher.

This script demonstrates the complete table processing pipeline:
1. Read PDF using MinerUReader
2. Parse tables using TableParser
3. Enrich tables using TableEnricher (judge critical + generate caption/embedding)

Requirements:
- MinerU must be installed (already included in optional deps)
- MinerU GPU backends require a CUDA-capable GPU
- .env file with OPENAI_API_KEY must be present in project root
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from rich import print

from zag.readers.mineru import MinerUReader
from zag.parsers import TableParser
from zag.extractors import TableEnricher, TableEnrichMode
from zag.schemas import UnitMetadata
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore

# Load environment variables from .env file
load_dotenv()


def _get_pdf_path() -> Path:
    """Ask user for a PDF path (supports drag & drop in console).

    The input may contain surrounding quotes if the file is dragged into
    the console. This helper strips single/double quotes and unescapes
    "\\ " into spaces.
    """

    print("Enter PDF file path (drag & drop supported): ", end="")
    raw = input().strip()

    if (raw.startswith("'") and raw.endswith("'")) or (
        raw.startswith('"') and raw.endswith('"')
    ):
        raw = raw[1:-1]

    raw = raw.replace("\\ ", " ")

    return Path(raw)


async def run_complete_pipeline() -> None:
    """Run complete table processing pipeline: MinerU → Parser → Enricher."""

    # Step 1: Get PDF path
    # TODO: Uncomment below for interactive input
    # pdf_path = _get_pdf_path()
    
    # Hardcoded for testing (remove this when done)
    pdf_path = Path(r"c:/Users/xue/PycharmProjects/zag-ai/playground/emet_nonqm_ratesheet_page2.pdf")
    
    print(f"\n{'='*80}")
    print(f"[bold cyan]Processing PDF:[/bold cyan] {pdf_path}")
    print(f"{'='*80}\n")

    # Step 2: Read PDF with MinerU
    print("[bold yellow]Step 1: Reading PDF with MinerUReader...[/bold yellow]")
    try:
        reader = MinerUReader()  # Use default hybrid-auto-engine backend
        doc = reader.read(str(pdf_path))
        print(f"[green]✓[/green] PDF read successfully")
        print(f"  - Content length: {len(doc.content)} characters")
        print(f"  - Pages: {len(doc.pages)}")
        print()
    except Exception as exc:
        print(f"[red]✗ MinerUReader failed: {exc}[/red]")
        return

    # Step 3: Parse tables with TableParser
    print("[bold yellow]Step 2: Parsing tables with TableParser...[/bold yellow]")
    parser = TableParser()
    unit_metadata = UnitMetadata(document=doc.metadata.model_dump())
    table_units = parser.parse(
        text=doc.content,
        metadata=unit_metadata,
        doc_id=doc.doc_id,
    )
    print(f"[green]✓[/green] Found {len(table_units)} tables")
    
    if not table_units:
        print("[yellow]No tables found in document[/yellow]")
        return
    
    # Print parsed table info
    print("\n[bold]Parsed TableUnits:[/bold]")
    for idx, unit in enumerate(table_units, 1):
        meta_table = (unit.metadata.custom or {}).get("table", {})
        print(f"\n  Table {idx}:")
        print(f"    - Format: {meta_table.get('source_format', '?')}")
        print(f"    - Shape: {len(unit.df)} rows × {len(unit.df.columns)} columns")
        print(f"    - Complex: {meta_table.get('is_complex', False)}")
        print(f"    - Content preview: {unit.content[:100]}...")
    print()

    # Step 4: Enrich tables with TableEnricher
    print("[bold yellow]Step 3: Enriching tables with TableEnricher...[/bold yellow]")
    print("  Using mode: CRITICAL_ONLY (judge all + enrich only critical)")
    
    # Get API key from environment (loaded from .env)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[red]✗ OPENAI_API_KEY not found in .env file[/red]")
        print("  Please add OPENAI_API_KEY=your-key to .env file in project root")
        return
    
    enricher = TableEnricher(
        llm_uri="openai/gpt-4o",
        api_key=api_key
    )
    
    enriched_tables = await enricher.aextract(
        table_units,
        mode=TableEnrichMode.CRITICAL_ONLY
    )
    
    print(f"[green]✓[/green] Enrichment complete")
    print()

    # Step 5: Print enrichment results
    print(f"\n{'='*80}")
    print("[bold green]Enrichment Results:[/bold green]")
    print(f"{'='*80}\n")
    
    for idx, unit in enumerate(enriched_tables, 1):
        meta_table = (unit.metadata.custom or {}).get("table", {})
        is_critical = meta_table.get("is_data_critical", False)
        reason = meta_table.get("criticality_reason", "N/A")
        
        print(f"[bold]Table {idx}:[/bold]")
        print(f"  [cyan]is_data_critical:[/cyan] {is_critical}")
        print(f"  [cyan]criticality_reason:[/cyan] {reason}")
        
        if is_critical:
            print(f"  [cyan]caption:[/cyan] {unit.caption or 'N/A'}")
            print(f"  [cyan]embedding_content:[/cyan]")
            if unit.embedding_content:
                # Print first 200 chars of embedding_content
                preview = unit.embedding_content[:2000]
                print(f"    {preview}...")
            else:
                print("    N/A")
        else:
            print(f"  [dim](Not critical, skipped enrichment)[/dim]")
        
        print()

    # Summary
    critical_count = sum(
        1 for u in enriched_tables 
        if u.metadata.custom.get("table", {}).get("is_data_critical", False)
    )
    print(f"{'='*80}")
    print(f"[bold]Summary:[/bold]")
    print(f"  Total tables: {len(enriched_tables)}")
    print(f"  Critical tables: {critical_count}")
    print(f"  Non-critical tables: {len(enriched_tables) - critical_count}")
    print(f"{'='*80}\n")
    
    # Step 6: Quick retrieval validation (only for critical tables)
    critical_tables = [
        u for u in enriched_tables 
        if u.metadata.custom.get("table", {}).get("is_data_critical", False)
    ]
    
    if critical_tables and len(critical_tables) > 0:
        print("[bold yellow]Step 4: Quick Retrieval Validation...[/bold yellow]")
        print("  Testing semantic search quality with in-memory vector store")
        print()
        
        try:
            # Create in-memory vector store (Qdrant)
            embedder = Embedder(uri="openai/text-embedding-3-small", api_key=api_key)
            store = QdrantVectorStore.in_memory(
                collection_name="test_tables",
                embedder=embedder
            )
            
            # Index critical tables
            print(f"  Indexing {len(critical_tables)} critical tables...")
            store.add(critical_tables)  # Use add() not add_units()
            print(f"  [green]✓[/green] Indexed successfully")
            print()
            
            # Test queries (针对复杂 rate sheet 的具体问题)
            test_queries = [
                "What is the maximum loan amount for owner occupied purchase with 80% LTV?",
                "What are the FICO requirements for cash-out refinance with investment property at 75% LTV?",
                "What property types are accepted and what are the documentation requirements?",
            ]
            
            print("  [bold]Testing sample queries:[/bold]")
            for query in test_queries:
                results = store.search(query, top_k=2)
                print(f"\n  Query: [cyan]\"{query}\"[/cyan]")
                print(f"  Top result (score: {results[0].score:.3f}):")
                
                # Show table caption
                table = results[0]  # results[0] is already the TableUnit
                caption = table.caption or "N/A"
                print(f"    Caption: {caption}")
                
                # Show embedding_content preview
                if table.embedding_content:
                    preview = table.embedding_content[:150]
                    print(f"    Preview: {preview}...")
            
            print(f"\n  [green]✓[/green] Retrieval validation complete")
            print()
            
        except Exception as exc:
            print(f"  [yellow]⚠[/yellow] Retrieval validation skipped: {exc}")
            print()


def main() -> None:
    """Run complete table processing pipeline."""

    asyncio.run(run_complete_pipeline())


if __name__ == "__main__":  # pragma: no cover - manual example
    main()
