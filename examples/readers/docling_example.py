"""Test DoclingReader with different configurations

This script tests DoclingReader with various pipeline configurations:
1. Default standard PDF pipeline
2. Standard PDF pipeline with OCR enabled
3. Standard PDF pipeline without OCR
4. Standard PDF pipeline with GPU acceleration (CUDA/AUTO)
5. VLM pipeline with Alibaba Qwen-VL (requires BAILIAN_API_KEY environment variable)

Test file: Thunderbird Product Overview 2025 - No Doc.pdf

Environment Variables:
    BAILIAN_API_KEY: API key for Alibaba Bailian Qwen-VL model (optional, for VLM test)
"""

import os
import time
import ssl
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from rich.table import Table
from rich.panel import Panel
from rich import box

# Disable SSL verification for HuggingFace downloads (workaround for SSL issues)
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables from .env file
load_dotenv()

from zag.readers.docling import DoclingReader
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions


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
    
    # Custom metadata from Docling
    if doc.metadata.custom:
        print("\n--- Docling Custom Metadata ---")
        for key, value in doc.metadata.custom.items():
            print(f"{key}: {value}")
    
    # Page structure info
    print("\n--- Page Structure ---")
    for i, page in enumerate(doc.pages[:3], 1):  # Show first 3 pages
        print(f"\nPage {page.page_number}:")
        print(f"  Content length: {len(page.content)} characters")
        print(f"  Content preview: {page.content[:100]}...")
    
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
                "content_length": len(page.content),
            }
            for page in doc.pages
        ]
    }
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved to: {md_file}")
    print(f"  Metadata: {json_file}")


def test_default_pipeline(pdf_path: str, output_dir: Path):
    """Test 1: Default standard PDF pipeline"""
    print("\n[yellow]Running Test 1: Default Standard PDF Pipeline...[/yellow]")
    start_time = time.time()
    
    # Use AUTO to automatically detect best device (CUDA/CPU)
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.AUTO  # Auto-detect CUDA or fallback to CPU
    )
    
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    doc = reader.read(pdf_path)
    
    elapsed_time = time.time() - start_time
    print_document_info(doc, f"TEST 1: Default Standard PDF Pipeline (⏱️ {elapsed_time:.2f}s)")
    save_document_content(doc, "TEST_1_Default_Pipeline", output_dir)
    return doc, elapsed_time


def test_with_ocr(pdf_path: str, output_dir: Path):
    """Test 2: Standard PDF pipeline with OCR enabled"""
    print("\n[yellow]Running Test 2: Standard PDF Pipeline with OCR...[/yellow]")
    start_time = time.time()
    
    pdf_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
    )
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    doc = reader.read(pdf_path)
    
    elapsed_time = time.time() - start_time
    print_document_info(doc, f"TEST 2: Standard PDF Pipeline with OCR (⏱️ {elapsed_time:.2f}s)")
    save_document_content(doc, "TEST_2_With_OCR", output_dir)
    return doc, elapsed_time


def test_without_ocr(pdf_path: str, output_dir: Path):
    """Test 3: Standard PDF pipeline without OCR"""
    print("\n[yellow]Running Test 3: Standard PDF Pipeline without OCR...[/yellow]")
    start_time = time.time()
    
    pdf_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
    )
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    doc = reader.read(pdf_path)
    
    elapsed_time = time.time() - start_time
    print_document_info(doc, f"TEST 3: Standard PDF Pipeline without OCR (⏱️ {elapsed_time:.2f}s)")
    save_document_content(doc, "TEST_3_Without_OCR", output_dir)
    return doc, elapsed_time


def test_with_gpu_auto(pdf_path: str, output_dir: Path):
    """Test 4: Standard PDF pipeline with GPU acceleration (AUTO)"""
    print("\n[yellow]Running Test 4: GPU Acceleration (AUTO)...[/yellow]")
    try:
        start_time = time.time()
        
        pdf_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
        )
        pdf_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.AUTO  # Auto-detect best device
        )
        reader = DoclingReader(pdf_pipeline_options=pdf_options)
        doc = reader.read(pdf_path)
        
        elapsed_time = time.time() - start_time
        print_document_info(doc, f"TEST 4: GPU (AUTO) (⏱️ {elapsed_time:.2f}s)")
        save_document_content(doc, "TEST_4_GPU_AUTO", output_dir)
        return doc, elapsed_time
    except Exception as e:
        print_separator("TEST 4: GPU Acceleration (AUTO) - ERROR")
        print(f"[red]Error: {e}[/red]")
        return None, 0


def test_with_gpu_cuda(pdf_path: str, output_dir: Path):
    """Test 5: Standard PDF pipeline with GPU acceleration (CUDA)"""
    print("\n[yellow]Running Test 5: GPU Acceleration (CUDA)...[/yellow]")
    try:
        start_time = time.time()
        
        pdf_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
        )
        pdf_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA  # Explicit CUDA
        )
        reader = DoclingReader(pdf_pipeline_options=pdf_options)
        doc = reader.read(pdf_path)
        
        elapsed_time = time.time() - start_time
        print_document_info(doc, f"TEST 5: GPU (CUDA) (⏱️ {elapsed_time:.2f}s)")
        save_document_content(doc, "TEST_5_GPU_CUDA", output_dir)
        return doc, elapsed_time
    except Exception as e:
        print_separator("TEST 5: GPU Acceleration (CUDA) - ERROR")
        print(f"[red]Error (expected if no CUDA GPU available): {e}[/red]")
        return None, 0


def test_vlm_pipeline_qwen(pdf_path: str, output_dir: Path, api_key: str = None):
    """Test 6: VLM pipeline with Alibaba Qwen-VL (requires API key)"""
    print("\n[yellow]Running Test 6: VLM Pipeline with Qwen-VL...[/yellow]")
    if not api_key:
        print_separator("TEST 6: VLM Pipeline with Qwen-VL (SKIPPED)")
        print("[yellow]Qwen-VL requires API key. Set BAILIAN_API_KEY environment variable.[/yellow]")
        print("Get your API key from: https://help.aliyun.com/zh/model-studio/")
        return None, 0
    
    try:
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
        
        start_time = time.time()
        
        vlm_options = VlmPipelineOptions(
            enable_remote_services=True,
            vlm_options=ApiVlmOptions(
                url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                params=dict(
                    model="qwen-vl-max-latest",  # Best quality model
                    max_tokens=4096,
                ),
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
                prompt="Convert this page to markdown. Extract all text, tables, and layout information.",
                timeout=90,
                response_format=ResponseFormat.MARKDOWN,
            )
        )
        reader = DoclingReader(vlm_pipeline_options=vlm_options)
        doc = reader.read(pdf_path)
        
        elapsed_time = time.time() - start_time
        print_document_info(doc, f"TEST 6: VLM Pipeline with Qwen-VL (⏱️ {elapsed_time:.2f}s)")
        save_document_content(doc, "TEST_6_VLM_Qwen", output_dir)
        return doc, elapsed_time
    except ImportError as e:
        print_separator("TEST 6: VLM Pipeline with Qwen-VL (SKIPPED)")
        print(f"[yellow]VLM pipeline requires extra dependencies: {e}[/yellow]")
        print("To enable VLM pipeline, install: pip install docling[vlm]")
        return None, 0
    except Exception as e:
        print_separator("TEST 6: VLM Pipeline with Qwen-VL (ERROR)")
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
    table.add_column("Pictures", justify="right", style="white", width=10)
    
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
        
        table.add_row(
            name,
            f"{elapsed_time:.2f}s" if elapsed_time > 0 else "N/A",
            f"{len(doc.content):,} chars",
            str(len(doc.pages)),
            str(doc.metadata.custom.get('text_items_count', 0)),
            str(doc.metadata.custom.get('table_items_count', 0)),
            str(doc.metadata.custom.get('picture_items_count', 0))
        )
    
    print(table)


def main():
    # PDF file path - use test file from files directory
    import os
    pdf_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'files',
        'mortgage_products.pdf'
    )

    # Create output directory (under output/ to be ignored by git)
    output_dir = Path("output") / "docling"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return
    
    # Get Bailian API key from environment variable
    bailian_api_key = os.getenv("BAILIAN_API_KEY")
    
    # Detect GPU availability
    import torch
    gpu_status = "[yellow]Using CPU[/yellow]"
    if torch.cuda.is_available():
        gpu_status = "[green]✓ CUDA Available[/green]"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_status = "[cyan]✓ MPS Available (Apple GPU)[/cyan]"
    
    # Print header
    print(Panel.fit(
        "[bold green]DoclingReader Test Suite[/bold green]\n" +
        f"PDF: {pdf_path}\n" +
        f"Output: {output_dir.absolute()}\n" +
        f"GPU: {gpu_status}\n" +
        f"API Key: {'[green]✓ Found[/green]' if bailian_api_key else '[yellow]Not Found (VLM test will be skipped)[/yellow]'}",
        title="⚡ Test Configuration",
        border_style="blue"
    ))
    
    # Run tests
    results = {}
    
    try:
        results["Default"] = test_default_pipeline(pdf_path, output_dir)
    except Exception as e:
        print(f"Error in default pipeline test: {e}")
    
    # try:
    #     results["With OCR"] = test_with_ocr(pdf_path, output_dir)
    # except Exception as e:
    #     print(f"Error in OCR enabled test: {e}")
    
    # try:
    #     results["Without OCR"] = test_without_ocr(pdf_path, output_dir)
    # except Exception as e:
    #     print(f"Error in OCR disabled test: {e}")
    
    # try:
    #     results["GPU AUTO"] = test_with_gpu_auto(pdf_path, output_dir)
    # except Exception as e:
    #     print(f"Error in GPU AUTO test: {e}")
    
    # try:
    #     results["GPU CUDA"] = test_with_gpu_cuda(pdf_path, output_dir)
    # except Exception as e:
    #     print(f"Error in GPU CUDA test: {e}")
    
    # try:
    #     results["VLM Qwen"] = test_vlm_pipeline_qwen(pdf_path, output_dir, bailian_api_key)
    # except Exception as e:
    #     print(f"Error in VLM Qwen test: {e}")
    
    # Compare results
    compare_results(results)
    
    print("\n")
    print(Panel("[bold green]✅ All Tests Completed![/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
