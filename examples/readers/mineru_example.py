"""Test MinerUReader with different configurations

This script tests MinerUReader with various backend configurations:
1. Pipeline backend (CPU-friendly, 82+ accuracy)
2. Hybrid-auto-engine backend (GPU, 90+ accuracy, default)
3. VLM-auto-engine backend (GPU, 90+ accuracy, pure VLM)
4. Different language support
5. Different page ranges
6. Formula and table parsing options

Test file: Thunderbird Product Overview 2025 - No Doc.pdf

Requirements:
    - mineru installed: pip install mineru[all]
    - For GPU backends: CUDA-capable GPU with 8GB+ VRAM
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from rich.table import Table
from rich.panel import Panel
from rich import box

# Load environment variables from .env file
load_dotenv()

from zag.readers.mineru import MinerUReader


def print_separator(title: str):
    """Print a visual separator"""
    print(f"\n[bold cyan on dark_blue]{title}[/bold cyan on dark_blue]")
    print("[cyan]" + "=" * 80 + "[/cyan]")


def print_document_info(doc, config_name: str):
    """Print document information"""
    print_separator(config_name)
    
    print(f"Document Type: {type(doc).__name__}")
    print(f"Content Length: {len(doc.content)} characters")
    print(f"Number of Pages: {len(doc.pages)}")
    
    # Metadata
    print("\n--- Metadata ---")
    print(f"Source: {doc.metadata.source}")
    print(f"File Type: {doc.metadata.file_type}")
    print(f"File Name: {doc.metadata.file_name}")
    print(f"File Size: {doc.metadata.file_size} bytes")
    print(f"Reader: {doc.metadata.reader_name}")
    
    # Custom metadata from MinerU
    if doc.metadata.custom:
        print("\n--- MinerU Custom Metadata ---")
        for key, value in doc.metadata.custom.items():
            print(f"{key}: {value}")
    
    # Page structure info
    print("\n--- Page Structure ---")
    for i, page in enumerate(doc.pages[:3], 1):  # Show first 3 pages
        print(f"\nPage {page.page_number}:")
        print(f"  Text items: {page.metadata.get('text_count', 0)}")
        print(f"  Tables: {page.metadata.get('table_count', 0)}")
        print(f"  Images: {page.metadata.get('image_count', 0)}")
        
        # Show first few text items
        if page.content.get('texts'):
            print(f"  First text item type: {page.content['texts'][0].get('type', 'N/A')}")
    
    # Show content preview (first 500 chars)
    print("\n--- Content Preview (first 500 chars) ---")
    print(doc.content[:500])
    print("...")


def save_document_content(doc, config_name: str, output_dir: Path):
    """Save document content to file"""
    # Create safe filename
    safe_name = config_name.replace(" ", "_").replace(":", "").replace("/", "_")
    
    # Save markdown content
    md_file = output_dir / f"{safe_name}.md"
    with md_file.open("w", encoding="utf-8") as f:
        f.write(f"# {config_name}\n\n")
        f.write(doc.content)
    
    # Save metadata as JSON
    import json
    json_file = output_dir / f"{safe_name}_metadata.json"
    metadata_dict = {
        "source": doc.metadata.source,
        "file_type": doc.metadata.file_type,
        "file_name": doc.metadata.file_name,
        "file_size": doc.metadata.file_size,
        "content_length": doc.metadata.content_length,
        "reader_name": doc.metadata.reader_name,
        "custom": doc.metadata.custom,
        "page_count": len(doc.pages),
        "pages_summary": [
            {
                "page_number": page.page_number,
                "text_count": page.metadata.get('text_count', 0),
                "table_count": page.metadata.get('table_count', 0),
                "image_count": page.metadata.get('image_count', 0),
            }
            for page in doc.pages
        ]
    }
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved to: {md_file}")
    print(f"  Metadata: {json_file}")


def test_pipeline_backend(pdf_path: str, output_dir: Path):
    """Test 1: Pipeline backend (CPU-friendly)"""
    print("\n[yellow]Running Test 1: Pipeline Backend (CPU-friendly)...[/yellow]")
    try:
        start_time = time.time()
        
        reader = MinerUReader(backend="pipeline")
        doc = reader.read(pdf_path)
        
        elapsed_time = time.time() - start_time
        print_document_info(doc, f"TEST 1: Pipeline Backend (⏱️ {elapsed_time:.2f}s)")
        save_document_content(doc, "TEST_1_Pipeline", output_dir)
        return doc, elapsed_time
    except Exception as e:
        print_separator("TEST 1: Pipeline Backend - ERROR")
        print(f"[red]Error: {e}[/red]")
        return None, 0


def test_hybrid_backend(pdf_path: str, output_dir: Path):
    """Test 2: Hybrid-auto-engine backend (default, GPU)"""
    print("\n[yellow]Running Test 2: Hybrid Backend (GPU)...[/yellow]")
    try:
        start_time = time.time()
        
        reader = MinerUReader(backend="hybrid-auto-engine")
        doc = reader.read(pdf_path)
        
        elapsed_time = time.time() - start_time
        print_document_info(doc, f"TEST 2: Hybrid Backend (⏱️ {elapsed_time:.2f}s)")
        save_document_content(doc, "TEST_2_Hybrid", output_dir)
        return doc, elapsed_time
    except Exception as e:
        print_separator("TEST 2: Hybrid Backend - ERROR")
        print(f"[red]Error (GPU required): {e}[/red]")
        return None, 0


def test_vlm_backend(pdf_path: str, output_dir: Path):
    """Test 3: VLM-auto-engine backend (pure VLM, GPU)"""
    print("\n[yellow]Running Test 3: VLM Backend (GPU)...[/yellow]")
    try:
        start_time = time.time()
        
        reader = MinerUReader(backend="vlm-auto-engine")
        doc = reader.read(pdf_path)
        
        elapsed_time = time.time() - start_time
        print_document_info(doc, f"TEST 3: VLM Backend (⏱️ {elapsed_time:.2f}s)")
        save_document_content(doc, "TEST_3_VLM", output_dir)
        return doc, elapsed_time
    except Exception as e:
        print_separator("TEST 3: VLM Backend - ERROR")
        print(f"[red]Error (GPU required): {e}[/red]")
        return None, 0


def test_english_language(pdf_path: str, output_dir: Path):
    """Test 4: Pipeline with English OCR"""
    print("\n[yellow]Running Test 4: English Language OCR...[/yellow]")
    try:
        start_time = time.time()
        
        reader = MinerUReader(backend="pipeline", lang="en")
        doc = reader.read(pdf_path)
        
        elapsed_time = time.time() - start_time
        print_document_info(doc, f"TEST 4: English Language (⏱️ {elapsed_time:.2f}s)")
        save_document_content(doc, "TEST_4_English", output_dir)
        return doc, elapsed_time
    except Exception as e:
        print_separator("TEST 4: English Language - ERROR")
        print(f"[red]Error: {e}[/red]")
        return None, 0


def test_page_range(pdf_path: str, output_dir: Path):
    """Test 5: Parse first 2 pages only"""
    print("\n[yellow]Running Test 5: Page Range (first 2 pages)...[/yellow]")
    try:
        start_time = time.time()
        
        reader = MinerUReader(
            backend="pipeline",
            start_page_id=0,
            end_page_id=2
        )
        doc = reader.read(pdf_path)
        
        elapsed_time = time.time() - start_time
        print_document_info(doc, f"TEST 5: Page Range (⏱️ {elapsed_time:.2f}s)")
        save_document_content(doc, "TEST_5_PageRange", output_dir)
        return doc, elapsed_time
    except Exception as e:
        print_separator("TEST 5: Page Range - ERROR")
        print(f"[red]Error: {e}[/red]")
        return None, 0


def test_no_formula(pdf_path: str, output_dir: Path):
    """Test 6: Disable formula recognition"""
    print("\n[yellow]Running Test 6: Without Formula Recognition...[/yellow]")
    try:
        start_time = time.time()
        
        reader = MinerUReader(
            backend="pipeline",
            formula_enable=False
        )
        doc = reader.read(pdf_path)
        
        elapsed_time = time.time() - start_time
        print_document_info(doc, f"TEST 6: No Formula (⏱️ {elapsed_time:.2f}s)")
        save_document_content(doc, "TEST_6_NoFormula", output_dir)
        return doc, elapsed_time
    except Exception as e:
        print_separator("TEST 6: No Formula - ERROR")
        print(f"[red]Error: {e}[/red]")
        return None, 0


def compare_results(results: dict):
    """Compare results from different configurations"""
    print("\n")
    
    # Create a rich table
    table = Table(title="📊 Test Results Summary", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Test Name", style="cyan", width=25)
    table.add_column("Time (s)", justify="right", style="yellow", width=12)
    table.add_column("Content", justify="right", style="green", width=12)
    table.add_column("Pages", justify="right", style="blue", width=8)
    table.add_column("Texts", justify="right", style="white", width=8)
    table.add_column("Tables", justify="right", style="white", width=8)
    table.add_column("Images", justify="right", style="white", width=10)
    
    for name, (doc, elapsed_time) in results.items():
        if doc is None:
            table.add_row(
                name,
                "[red]FAILED[/red]",
                "-",
                "-",
                "-",
                "-",
                "-"
            )
            continue
        
        # Calculate totals
        total_texts = sum(page.metadata.get('text_count', 0) for page in doc.pages)
        total_tables = sum(page.metadata.get('table_count', 0) for page in doc.pages)
        total_images = sum(page.metadata.get('image_count', 0) for page in doc.pages)
        
        table.add_row(
            name,
            f"{elapsed_time:.2f}s" if elapsed_time > 0 else "N/A",
            f"{len(doc.content):,} chars",
            str(len(doc.pages)),
            str(total_texts),
            str(total_tables),
            str(total_images)
        )
    
    print(table)


def check_gpu_available():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    # PDF file path - use test file from files directory
    import os
    pdf_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'files',
        'thunderbird_overview.pdf'
    )
    
    # Create output directory (under output/ to be ignored by git)
    output_dir = Path("output") / "mineru"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return
    
    # Check GPU availability
    gpu_available = check_gpu_available()
    
    # Print header
    print(Panel.fit(
        "[bold green]MinerUReader Test Suite[/bold green]\n" +
        f"PDF: {pdf_path}\n" +
        f"Output: {output_dir.absolute()}\n" +
        f"GPU: {'[green]✓ CUDA Available[/green]' if gpu_available else '[yellow]⚠ GPU Not Available (Some tests will be skipped)[/yellow]'}\n" +
        f"Note: MinerU requires mineru[all] package installed",
        title="⚡ Test Configuration",
        border_style="blue"
    ))
    
    # Run tests
    results = {}
    
    # Test 1: Pipeline (always available)
    try:
        results["Pipeline"] = test_pipeline_backend(pdf_path, output_dir)
    except Exception as e:
        print(f"[red]Error in pipeline test: {e}[/red]")
        results["Pipeline"] = (None, 0)
    
    # Test 2: Hybrid (GPU required)
    if gpu_available:
        try:
            results["Hybrid"] = test_hybrid_backend(pdf_path, output_dir)
        except Exception as e:
            print(f"[red]Error in hybrid test: {e}[/red]")
            results["Hybrid"] = (None, 0)
    else:
        print("\n[yellow]Test 2: Hybrid Backend - SKIPPED (GPU required)[/yellow]")
        results["Hybrid"] = (None, 0)
    
    # Test 3: VLM (GPU required)
    if gpu_available:
        try:
            results["VLM"] = test_vlm_backend(pdf_path, output_dir)
        except Exception as e:
            print(f"[red]Error in VLM test: {e}[/red]")
            results["VLM"] = (None, 0)
    else:
        print("\n[yellow]Test 3: VLM Backend - SKIPPED (GPU required)[/yellow]")
        results["VLM"] = (None, 0)
    
    # Test 4: English language
    try:
        results["English"] = test_english_language(pdf_path, output_dir)
    except Exception as e:
        print(f"[red]Error in English test: {e}[/red]")
        results["English"] = (None, 0)
    
    # Test 5: Page range
    try:
        results["Page Range"] = test_page_range(pdf_path, output_dir)
    except Exception as e:
        print(f"[red]Error in page range test: {e}[/red]")
        results["Page Range"] = (None, 0)
    
    # Test 6: No formula
    try:
        results["No Formula"] = test_no_formula(pdf_path, output_dir)
    except Exception as e:
        print(f"[red]Error in no formula test: {e}[/red]")
        results["No Formula"] = (None, 0)
    
    # Compare results
    compare_results(results)
    
    print("\n")
    print(Panel("[bold green]✅ All Tests Completed![/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
