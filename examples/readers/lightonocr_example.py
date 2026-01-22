"""Example of using LightOnOCR reader for advanced OCR."""

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table as RichTable

console = Console()


def normalize_path(path_str: str) -> Path:
    """
    Normalize file path by removing quotes and handling drag & drop paths.
    
    Handles:
    - Single quotes: '/path/to/file'
    - Double quotes: "/path/to/file"
    - Escaped spaces: /path/to/my\\ file.pdf
    """
    path_str = path_str.strip()
    
    # Remove surrounding quotes
    if (path_str.startswith("'") and path_str.endswith("'")) or \
       (path_str.startswith('"') and path_str.endswith('"')):
        path_str = path_str[1:-1]
    
    # Handle escaped spaces (from drag & drop)
    path_str = path_str.replace("\\ ", " ")
    
    return Path(path_str)


def test_read_pdf(file_path: str, device: str = "auto", dpi: int = 200):
    """Test reading PDF with LightOnOCR."""
    from zag.readers import LightOnOCRReader
    import tempfile
    
    file_path = normalize_path(file_path)
    
    if not file_path.exists():
        console.print(f"[red]✗ File not found: {file_path}[/red]")
        return
    
    console.print(Panel.fit(
        f"[cyan]File:[/cyan] {file_path.name}\n"
        f"[cyan]Device:[/cyan] {device}\n"
        f"[cyan]DPI:[/cyan] {dpi}",
        title="LightOnOCR Reader",
        border_style="cyan"
    ))
    
    # Create reader
    console.print("\n[yellow]Initializing LightOnOCR...[/yellow]")
    reader = LightOnOCRReader(device=device, dpi=dpi)
    
    # Read document
    doc = reader.read(str(file_path))
    
    # Print document info
    console.print("\n[bold green]✓ Document processed successfully![/bold green]")
    console.print(f"\n[bold]Document Info:[/bold]")
    console.print(f"  File: {doc.metadata.file_name}")
    console.print(f"  Pages: {len(doc.pages)}")
    console.print(f"  Total Pages: {doc.metadata.custom.get('total_pages', 'N/A')}")
    console.print(f"  OCR Model: {doc.metadata.custom.get('ocr_model', 'N/A')}")
    console.print(f"  Device: {doc.metadata.custom.get('ocr_device', 'N/A')}")
    console.print(f"  DPI: {doc.metadata.custom.get('ocr_dpi', 'N/A')}")
    
    if doc.metadata.custom.get('title'):
        console.print(f"  Title: {doc.metadata.custom.get('title')}")
    
    if doc.metadata.custom.get('author'):
        console.print(f"  Author: {doc.metadata.custom.get('author')}")
    
    # Show first page preview
    if doc.pages:
        console.print(f"\n[bold]First Page Preview:[/bold]")
        preview = doc.pages[0].content[:800]
        console.print(preview)
        if len(doc.pages[0].content) > 800:
            console.print("[dim]...(truncated)[/dim]")
    
    # Save to temp directory
    temp_dir = Path(tempfile.gettempdir())
    output_file = temp_dir / f"{file_path.stem}_lightonocr.md"
    
    with output_file.open('w', encoding='utf-8') as f:
        # Write metadata header
        title = doc.metadata.custom.get('title') or doc.metadata.file_name
        f.write(f"# {title}\n\n")
        if doc.metadata.custom.get('author'):
            f.write(f"**Author:** {doc.metadata.custom.get('author')}\n")
        f.write(f"**Pages:** {len(doc.pages)}\n")
        f.write(f"**OCR Model:** {doc.metadata.custom.get('ocr_model')}\n")
        f.write(f"**Device:** {doc.metadata.custom.get('ocr_device')}\n")
        f.write(f"**DPI:** {doc.metadata.custom.get('ocr_dpi')}\n")
        f.write(f"**Source:** {doc.metadata.source}\n\n")
        f.write("---\n\n")
        
        # Write content
        f.write(doc.content)
    
    console.print(f"\n[bold green]✓ Content saved to:[/bold green] {output_file}")
    console.print("\n[dim]You can now open this file to check the OCR quality[/dim]")


if __name__ == "__main__":
    console.print("[bold cyan]LightOnOCR Reader Example[/bold cyan]")
    console.print("[dim]State-of-the-art OCR for documents, tables, and forms[/dim]\n")
    
    # Display device selection table
    device_table = RichTable(title="Device Options", show_header=True)
    device_table.add_column("Device", style="cyan", width=12)
    device_table.add_column("Description", style="green")
    device_table.add_column("Speed", style="yellow")
    
    device_table.add_row(
        "auto",
        "Auto-detect (CUDA > MPS > CPU)",
        "Depends on hardware"
    )
    device_table.add_row(
        "cuda",
        "NVIDIA GPU (fastest)",
        "⚡⚡⚡"
    )
    device_table.add_row(
        "mps",
        "Apple Silicon GPU",
        "⚡⚡"
    )
    device_table.add_row(
        "cpu",
        "CPU only (slowest but compatible)",
        "⚡"
    )
    
    console.print(device_table)
    console.print()
    
    # Get file path from user
    console.print("[yellow]Enter PDF file path (or drag & drop):[/yellow] ", end="")
    file_path = input().strip()
    
    if not file_path:
        console.print("[red]No file specified[/red]")
        exit(1)
    
    # Ask for device
    console.print("\n[yellow]Select device (default: auto):[/yellow]")
    console.print("  1. auto (recommended)")
    console.print("  2. cuda (NVIDIA GPU)")
    console.print("  3. mps (Apple Silicon)")
    console.print("  4. cpu (slowest)")
    console.print("[yellow]Choice [1/2/3/4]:[/yellow] ", end="")
    device_choice = input().strip() or "1"
    
    device_map = {
        "1": "auto",
        "2": "cuda",
        "3": "mps",
        "4": "cpu",
    }
    device = device_map.get(device_choice, "auto")
    
    # Ask for DPI
    console.print("\n[yellow]Select DPI (default: 200):[/yellow]")
    console.print("  Higher DPI = better quality but slower")
    console.print("  Recommended: 150-300")
    console.print("[yellow]DPI:[/yellow] ", end="")
    dpi_input = input().strip()
    dpi = int(dpi_input) if dpi_input.isdigit() else 200
    
    try:
        test_read_pdf(file_path, device, dpi)
    
    except ImportError as e:
        console.print(f"\n[red]✗ Missing dependency: {e}[/red]")
        console.print("\n[yellow]Install required packages:[/yellow]")
        console.print("  pip install transformers torch pillow PyMuPDF")
        console.print("\n[dim]Note:[/dim]")
        console.print("  - Model will be downloaded on first use (~2GB)")
        console.print("  - Set HF_HOME to control cache location")
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
