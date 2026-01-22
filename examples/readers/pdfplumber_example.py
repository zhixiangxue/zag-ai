"""Example of using PDFPlumber reader for complex table extraction."""

from pathlib import Path
from rich.console import Console

console = Console()


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


def test_read_pdf(file_path: str):
    """Test reading PDF with complex tables using pdfplumber."""
    from zag.readers import PDFPlumberReader
    import tempfile
    
    file_path = normalize_path(file_path)
    
    if not file_path.exists():
        console.print(f"[red]✗ File not found: {file_path}[/red]")
        return
    
    console.print(f"[cyan]Reading PDF: {file_path.name}[/cyan]")
    
    # Create reader
    reader = PDFPlumberReader()
    
    # Read document
    console.print("[yellow]Extracting tables...[/yellow]")
    doc = reader.read(str(file_path))
    
    # Debug: Check if tables were found
    total_tables = 0
    for page in doc.pages:
        # Count tables in page content
        table_count = page.content.count("**Table")
        total_tables += table_count
    
    console.print(f"[yellow]Tables detected: {total_tables}[/yellow]")
    
    # Print document info
    console.print("\n[bold green]✓ Document extracted successfully![/bold green]")
    console.print(f"\n[bold]Document Info:[/bold]")
    console.print(f"  File: {doc.metadata.file_name}")
    console.print(f"  Pages: {len(doc.pages)}")
    console.print(f"  Total Pages: {doc.metadata.custom.get('total_pages', 'N/A')}")
    
    if doc.metadata.custom.get('title'):
        console.print(f"  Title: {doc.metadata.custom.get('title')}")
    
    if doc.metadata.custom.get('author'):
        console.print(f"  Author: {doc.metadata.custom.get('author')}")
    
    # Show first page preview
    if doc.pages:
        console.print(f"\n[bold]First Page Preview:[/bold]")
        preview = doc.pages[0].content[:500]
        console.print(preview)
        if len(doc.pages[0].content) > 500:
            console.print("[dim]...(truncated)[/dim]")
    
    # Save to temp directory
    temp_dir = Path(tempfile.gettempdir())
    output_file = temp_dir / f"{file_path.stem}_pdfplumber.md"
    
    with output_file.open('w', encoding='utf-8') as f:
        # Write metadata header
        title = doc.metadata.custom.get('title') or doc.metadata.file_name
        f.write(f"# {title}\n\n")
        if doc.metadata.custom.get('author'):
            f.write(f"**Author:** {doc.metadata.custom.get('author')}\n")
        f.write(f"**Pages:** {len(doc.pages)}\n")
        f.write(f"**Source:** {doc.metadata.source}\n\n")
        f.write("---\n\n")
        
        # Write content
        f.write(doc.content)
    
    console.print(f"\n[bold green]✓ Content saved to:[/bold green] {output_file}")
    console.print("\n[dim]You can now open this file to check the table extraction quality[/dim]")


def test_with_custom_settings(file_path: str):
    """Test with custom table extraction settings."""
    from zag.readers import PDFPlumberReader
    
    file_path = normalize_path(file_path)
    
    console.print(f"[cyan]Reading PDF with custom settings: {file_path.name}[/cyan]")
    
    # Custom settings for complex tables
    table_settings = {
        "vertical_strategy": "lines",  # Use lines to detect columns
        "horizontal_strategy": "lines",  # Use lines to detect rows
        "snap_tolerance": 3,  # Snap tolerance for line alignment
        "join_tolerance": 3,  # Join tolerance for joining lines
        "edge_min_length": 3,  # Minimum length for edges
        "intersection_tolerance": 3,  # Tolerance for line intersections
    }
    
    reader = PDFPlumberReader(table_settings=table_settings)
    doc = reader.read(str(file_path))
    
    console.print(f"[green]✓ Extracted {len(doc.pages)} pages[/green]")


if __name__ == "__main__":
    console.print("[bold cyan]PDFPlumber Reader Example[/bold cyan]")
    console.print("[dim]Best for: Native PDF with text and complex tables[/dim]\n")
    
    # Get file path from user
    console.print("[yellow]Enter PDF file path (or drag & drop):[/yellow] ", end="")
    file_path = input().strip()
    
    if not file_path:
        console.print("[red]No file specified[/red]")
        exit(1)
    
    try:
        test_read_pdf(file_path)
        
        # Ask if user wants to try custom settings
        console.print("\n[yellow]Try with custom table settings? [y/N]:[/yellow] ", end="")
        try_custom = input().strip().lower()
        
        if try_custom == 'y':
            test_with_custom_settings(file_path)
    
    except ImportError as e:
        console.print(f"\n[red]✗ Missing dependency: {e}[/red]")
        console.print("\n[yellow]Install pdfplumber:[/yellow]")
        console.print("  pip install pdfplumber")
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
