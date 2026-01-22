"""Example of using Camelot reader for advanced table extraction."""

from pathlib import Path
from rich.console import Console
from rich.table import Table as RichTable

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


def test_read_pdf(file_path: str, flavor: str = "lattice", flatten: bool = False):
    """Test reading PDF with Camelot."""
    from zag.readers import CamelotReader
    import tempfile
    
    file_path = normalize_path(file_path)
    
    if not file_path.exists():
        console.print(f"[red]✗ File not found: {file_path}[/red]")
        return
    
    console.print(f"[cyan]Reading PDF: {file_path.name}[/cyan]")
    console.print(f"[cyan]Extraction mode: {flavor}[/cyan]")
    if flatten:
        console.print(f"[cyan]Flatten complex tables: Yes[/cyan]")
    console.print()
    
    # Build kwargs for complex table flattening
    kwargs = {}
    if flatten and flavor == "lattice":
        # Flatten merged cells by copying text in both directions
        kwargs['copy_text'] = ['h', 'v']
        kwargs['split_text'] = True
        console.print("[yellow]Using flatten mode (Lattice):[/yellow]")
        console.print("  - copy_text=['h', 'v']: Copy merged cell text")
        console.print("  - split_text=True: Split spanning text")
        console.print()
    elif flatten and flavor == "stream":
        # For stream mode, increase tolerance MODERATELY to better handle complex tables
        # DO NOT use split_text - it will break words apart!
        # DO NOT set tolerance too high - it will merge columns incorrectly!
        kwargs['row_tol'] = 5       # Moderate row tolerance (default: 2)
        kwargs['column_tol'] = 2    # Moderate column tolerance (default: 0)
        kwargs['edge_tol'] = 75     # Moderate edge tolerance (default: 50)
        console.print("[yellow]Using flatten mode (Stream):[/yellow]")
        console.print("  - row_tol=5: Moderately combine text vertically")
        console.print("  - column_tol=2: Moderately combine text horizontally")
        console.print("  - edge_tol=75: Moderately extend text edges")
        console.print("  [dim](Not using split_text - it breaks words)[/dim]")
        console.print("  [dim]If columns merge incorrectly, try lower values or Lattice mode[/dim]")
        console.print()
    
    # Create reader
    reader = CamelotReader(flavor=flavor, **kwargs)
    
    # Read document
    console.print("[yellow]Extracting tables...[/yellow]")
    doc = reader.read(str(file_path))
    
    # Print document info
    console.print("\n[bold green]✓ Document extracted successfully![/bold green]")
    console.print(f"\n[bold]Document Info:[/bold]")
    console.print(f"  File: {doc.metadata.file_name}")
    console.print(f"  Pages: {len(doc.pages)}")
    console.print(f"  Total Pages: {doc.metadata.custom.get('total_pages', 'N/A')}")
    console.print(f"  Extraction Mode: {doc.metadata.custom.get('extraction_flavor', 'N/A')}")
    
    if doc.metadata.custom.get('title'):
        console.print(f"  Title: {doc.metadata.custom.get('title')}")
    
    if doc.metadata.custom.get('author'):
        console.print(f"  Author: {doc.metadata.custom.get('author')}")
    
    # Show table statistics
    console.print(f"\n[bold]Table Statistics:[/bold]")
    total_tables = 0
    for page in doc.pages:
        table_count = page.metadata.get('table_count', 0)
        if table_count > 0:
            console.print(f"  Page {page.page_number}: {table_count} table(s)")
            total_tables += table_count
    
    if total_tables == 0:
        console.print("  [yellow]No tables detected[/yellow]")
    else:
        console.print(f"  [green]Total: {total_tables} table(s)[/green]")
    
    # Show first page preview
    if doc.pages:
        console.print(f"\n[bold]First Page Preview:[/bold]")
        preview = doc.pages[0].content[:800]
        console.print(preview)
        if len(doc.pages[0].content) > 800:
            console.print("[dim]...(truncated)[/dim]")
    
    # Save to temp directory
    temp_dir = Path(tempfile.gettempdir())
    output_file = temp_dir / f"{file_path.stem}_camelot_{flavor}.md"
    
    with output_file.open('w', encoding='utf-8') as f:
        # Write metadata header
        title = doc.metadata.custom.get('title') or doc.metadata.file_name
        f.write(f"# {title}\n\n")
        if doc.metadata.custom.get('author'):
            f.write(f"**Author:** {doc.metadata.custom.get('author')}\n")
        f.write(f"**Pages:** {len(doc.pages)}\n")
        f.write(f"**Extraction Mode:** {flavor}\n")
        if flatten:
            f.write(f"**Flatten:** Yes\n")
        f.write(f"**Total Tables:** {total_tables}\n")
        f.write(f"**Source:** {doc.metadata.source}\n\n")
        f.write("---\n\n")
        
        # Write content
        f.write(doc.content)
    
    console.print(f"\n[bold green]✓ Content saved to:[/bold green] {output_file}")
    console.print("\n[dim]You can now open this file to check the table extraction quality[/dim]")


def compare_modes(file_path: str):
    """Compare Lattice and Stream modes side by side."""
    from zag.readers import CamelotReader
    
    file_path = normalize_path(file_path)
    
    console.print("[bold cyan]Comparing Lattice vs Stream modes[/bold cyan]\n")
    
    # Test both modes
    for flavor in ["lattice", "stream"]:
        console.print(f"\n[yellow]Testing {flavor.upper()} mode...[/yellow]")
        reader = CamelotReader(flavor=flavor)
        doc = reader.read(str(file_path))
        
        total_tables = sum(page.metadata.get('table_count', 0) for page in doc.pages)
        console.print(f"  Tables detected: {total_tables}")
        console.print(f"  Pages with tables: {len([p for p in doc.pages if p.metadata.get('table_count', 0) > 0])}")


if __name__ == "__main__":
    console.print("[bold cyan]Camelot Reader Example[/bold cyan]")
    console.print("[dim]Best for: PDF with complex tables (bordered or borderless)[/dim]\n")
    
    # Display mode selection table
    mode_table = RichTable(title="Extraction Modes", show_header=True)
    mode_table.add_column("Mode", style="cyan", width=12)
    mode_table.add_column("Best For", style="green")
    mode_table.add_column("Pros", style="yellow")
    
    mode_table.add_row(
        "Lattice",
        "Tables with borders",
        "High accuracy, handles merged cells"
    )
    mode_table.add_row(
        "Stream",
        "Tables without borders",
        "Detects borderless tables"
    )
    
    console.print(mode_table)
    console.print()
    
    # Get file path from user
    console.print("[yellow]Enter PDF file path (or drag & drop):[/yellow] ", end="")
    file_path = input().strip()
    
    if not file_path:
        console.print("[red]No file specified[/red]")
        exit(1)
    
    # Ask for mode
    console.print("\n[yellow]Select extraction mode:[/yellow]")
    console.print("  1. Lattice (default - for bordered tables)")
    console.print("  2. Stream (for borderless tables)")
    console.print("  3. Compare both modes")
    console.print("[yellow]Choice [1/2/3]:[/yellow] ", end="")
    choice = input().strip() or "1"
    
    # Ask for flatten option
    flatten = False
    if choice in ["1", "2"]:
        console.print("\n[yellow]Flatten complex tables (handle merged cells)?[/yellow]")
        console.print("  This will copy merged cell text to all spanning cells")
        console.print("[yellow][y/N]:[/yellow] ", end="")
        flatten_input = input().strip().lower()
        flatten = flatten_input == 'y'
    
    try:
        if choice == "3":
            compare_modes(file_path)
        else:
            flavor = "stream" if choice == "2" else "lattice"
            test_read_pdf(file_path, flavor, flatten)
    
    except ImportError as e:
        console.print(f"\n[red]✗ Missing dependency: {e}[/red]")
        console.print("\n[yellow]Install Camelot:[/yellow]")
        console.print("  pip install 'camelot-py[base]'")
        console.print("\n[dim]Note: You may also need:[/dim]")
        console.print("  - Ghostscript: brew install ghostscript (macOS)")
        console.print("  - cv2: pip install opencv-python")
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
