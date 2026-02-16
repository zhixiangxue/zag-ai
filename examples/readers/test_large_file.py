#!/usr/bin/env python3
"""
Test script for large file processor refactoring.

Usage:
    python playground/test_large_file_processor.py

Interactive prompts:
    - PDF file path
    - Pages per chunk (default 100)
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env file first (for API keys)
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from zag.schemas.pdf import PDF
from zag.readers.docling import DoclingReader
from zag.utils.hash import calculate_file_hash
from zag.postprocessors.correctors import HeadingCorrector
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

console = Console()


def calculate_page_ranges(total_pages: int, pages_per_part: int) -> list:
    """Calculate page ranges for large file processing."""
    ranges = []
    start = 1
    while start <= total_pages:
        end = min(start + pages_per_part - 1, total_pages)
        ranges.append((start, end))
        start = end + 1
    return ranges


async def process_large_file(pdf_path: Path, pages_per_part: int = 100):
    """
    Process a large PDF file by reading in chunks and merging.
    
    Args:
        pdf_path: Path to the PDF file
        pages_per_part: Number of pages per chunk
    """
    console.print(f"\n[bold cyan]🚀 Large File Processor Test[/bold cyan]")
    console.print(f"   File: {pdf_path}")
    console.print(f"   Chunk size: {pages_per_part} pages\n")
    
    # Get total pages
    from pypdf import PdfReader
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        console.print(f"📖 Total pages: {total_pages}")
    except Exception as e:
        raise Exception(f"Failed to read PDF: {e}")
    
    # Calculate page ranges
    page_ranges = calculate_page_ranges(total_pages, pages_per_part)
    console.print(f"✂️  Will process in {len(page_ranges)} chunks: {page_ranges[:3]}{'...' if len(page_ranges) > 3 else ''}")
    
    # Configure reader
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA
    )
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    
    # Setup heading corrector
    import os
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    
    corrector = HeadingCorrector(
        llm_uri=f"{llm_provider}/{llm_model}",
        api_key=llm_api_key,
        llm_correction=True
    )
    
    # Read and merge
    merged_doc: PDF = None
    source_hash = calculate_file_hash(pdf_path)
    
    for idx, (start, end) in enumerate(page_ranges, 1):
        console.print(f"\n[cyan]--- Chunk {idx}/{len(page_ranges)}: pages {start}-{end} ---[/cyan]")
        
        # Read this page range
        console.print(f"  📄 Reading...")
        doc = reader.read(str(pdf_path), page_range=(start, end))
        console.print(f"     ✅ {len(doc.content):,} chars, {len(doc.pages)} pages")
        
        # Apply heading correction
        console.print(f"  🔧 Heading correction...")
        doc = await corrector.acorrect_document(doc)
        console.print(f"     ✅ Done")
        
        # Merge
        if merged_doc is None:
            merged_doc = doc
        else:
            merged_doc = merged_doc + doc
            console.print(f"  🔗 Merged: {len(merged_doc.pages)} pages total")
    
    # Update doc_id
    merged_doc.doc_id = source_hash
    merged_doc.metadata.md5 = source_hash
    
    console.print(f"\n[green]✅ Processing complete![/green]")
    console.print(f"   Total pages: {len(merged_doc.pages)}")
    console.print(f"   Total content: {len(merged_doc.content):,} characters")
    console.print(f"   doc_id: {merged_doc.doc_id}")
    
    # Verify page numbers
    if merged_doc.pages:
        page_nums = sorted(set(p.page_number for p in merged_doc.pages))
        console.print(f"   Page numbers: {page_nums[0]} - {page_nums[-1]} ({len(page_nums)} unique)")
    
    # Output to markdown file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = pdf_path.parent / f"{pdf_path.stem}_merged_{timestamp}.md"
    
    console.print(f"\n💾 Writing output to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# {pdf_path.stem}\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Source**: {pdf_path.name}\n\n")
        f.write(f"**Pages**: {len(merged_doc.pages)}\n\n")
        f.write(f"**doc_id**: {merged_doc.doc_id}\n\n")
        f.write("---\n\n")
        f.write(merged_doc.content)
    
    console.print(f"[green]✅ Output saved: {output_file}[/green]")
    
    # Preview content
    console.print(f"\n[dim]Content preview (first 5 lines):[/dim]")
    lines = merged_doc.content.split('\n')[:5]
    for line in lines:
        console.print(f"   {line[:100]}{'...' if len(line) > 100 else ''}")
    
    # Ask to dump archive
    dump_choice = input("\n💾 Dump to archive for sharing? (y/n, default n): ").strip().lower()
    if dump_choice == 'y':
        archive_dir = pdf_path.parent / "archive"
        console.print(f"\n📦 Dumping to: {archive_dir}")
        archive_path = merged_doc.dump(archive_dir)
        console.print(f"[green]✅ Archive created: {archive_path}[/green]")
        console.print(f"[dim]   Share this folder with colleagues[/dim]")
        
        # Demo: load back and verify
        load_demo = input("\n🔄 Demo: Load back from archive? (y/n, default n): ").strip().lower()
        if load_demo == 'y':
            console.print(f"\n[cyan]📖 Loading from archive...[/cyan]")
            loaded_pdf = PDF.load(archive_path)
            console.print(f"[green]✅ Loaded successfully![/green]")
            console.print(f"   doc_id: {loaded_pdf.doc_id}")
            console.print(f"   Pages: {len(loaded_pdf.pages)}")
            console.print(f"   Content length: {len(loaded_pdf.content):,} chars")
            
            # Verify page numbers
            if loaded_pdf.pages:
                page_nums = sorted(set(p.page_number for p in loaded_pdf.pages))
                console.print(f"   Page numbers: {page_nums[0]} - {page_nums[-1]} ({len(page_nums)} unique)")
            
            # Verify content match
            content_match = loaded_pdf.content == merged_doc.content
            console.print(f"\n📊 Verification:")
            console.print(f"   Content match: {'✅ Yes' if content_match else '❌ No'}")
            console.print(f"   Page count match: {'✅ Yes' if len(loaded_pdf.pages) == len(merged_doc.pages) else '❌ No'}")
            
            # Preview loaded content
            console.print(f"\n[dim]Loaded content preview (first 3 lines):[/dim]")
            lines = loaded_pdf.content.split('\n')[:3]
            for line in lines:
                console.print(f"   {line[:80]}{'...' if len(line) > 80 else ''}")
    
    return merged_doc


async def main():
    """Main entry point."""
    console.print("\n[bold cyan]🧪 Large File Processor Test[/bold cyan]")
    
    # Interactive input
    pdf_path_str = input("\n📂 Enter PDF file path: ").strip().strip('"')
    if not pdf_path_str:
        console.print("[red]No file provided, exiting.[/red]")
        return
    
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        console.print(f"[red]File not found: {pdf_path}[/red]")
        return
    
    pages_per_part_str = input("📄 Pages per chunk (default 100): ").strip()
    pages_per_part = int(pages_per_part_str) if pages_per_part_str else 100
    
    try:
        await process_large_file(pdf_path, pages_per_part)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
