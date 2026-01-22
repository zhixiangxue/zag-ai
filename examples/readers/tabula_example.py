"""Example of using Tabula reader for table extraction."""

from pathlib import Path
from rich.console import Console
import tempfile

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


def test_tabula_basic(file_path: str):
    """Test basic Tabula extraction (Stream mode)."""
    from zag.readers import TabulaReader
    
    file_path = normalize_path(file_path)
    
    if not file_path.exists():
        console.print(f"[red]✗ File not found: {file_path}[/red]")
        return
    
    console.print(f"[cyan]Reading PDF with Tabula (Stream mode): {file_path.name}[/cyan]")
    
    # Create reader with Stream mode (default)
    reader = TabulaReader(
        stream=True,
        lattice=False,
        multiple_tables=True
    )
    
    # Read document
    console.print("[yellow]Extracting tables...[/yellow]")
    doc = reader.read(str(file_path))
    
    # Print results
    console.print(f"\n[bold green]✓ Extraction complete![/bold green]")
    console.print(f"  Tables found: {doc.metadata.custom.get('total_tables', 0)}")
    console.print(f"  Extraction method: {doc.metadata.custom.get('extraction_method')}")
    
    # Show table previews
    if doc.pages:
        console.print(f"\n[bold]Table Previews:[/bold]")
        for page in doc.pages[:3]:  # Show first 3 tables
            console.print(f"\n[yellow]Page {page.page_number}:[/yellow]")
            console.print(f"  Rows: {page.metadata.get('rows', 'N/A')}")
            console.print(f"  Columns: {page.metadata.get('columns', 'N/A')}")
            preview = page.content[:300]
            console.print(preview)
            if len(page.content) > 300:
                console.print("[dim]...(truncated)[/dim]")
    
    # Save to temp directory
    temp_dir = Path(tempfile.gettempdir())
    output_file = temp_dir / f"{file_path.stem}_tabula_stream.md"
    
    with output_file.open('w', encoding='utf-8') as f:
        f.write(f"# {file_path.name}\n\n")
        f.write(f"**Extracted by:** Tabula (Stream mode)\n")
        f.write(f"**Tables found:** {doc.metadata.custom.get('total_tables', 0)}\n\n")
        f.write("---\n\n")
        f.write(doc.content)
    
    console.print(f"\n[bold green]✓ Content saved to:[/bold green] {output_file}")
    return doc


def test_tabula_lattice(file_path: str):
    """Test Tabula extraction with Lattice mode (for bordered tables)."""
    from zag.readers import TabulaReader
    
    file_path = normalize_path(file_path)
    
    console.print(f"\n[cyan]Reading PDF with Tabula (Lattice mode): {file_path.name}[/cyan]")
    
    # Create reader with Lattice mode
    reader = TabulaReader(
        stream=False,
        lattice=True,
        multiple_tables=True
    )
    
    # Read document
    console.print("[yellow]Extracting tables...[/yellow]")
    doc = reader.read(str(file_path))
    
    # Print results
    console.print(f"\n[bold green]✓ Extraction complete![/bold green]")
    console.print(f"  Tables found: {doc.metadata.custom.get('total_tables', 0)}")
    
    # Save to temp directory
    temp_dir = Path(tempfile.gettempdir())
    output_file = temp_dir / f"{file_path.stem}_tabula_lattice.md"
    
    with output_file.open('w', encoding='utf-8') as f:
        f.write(f"# {file_path.name}\n\n")
        f.write(f"**Extracted by:** Tabula (Lattice mode)\n")
        f.write(f"**Tables found:** {doc.metadata.custom.get('total_tables', 0)}\n\n")
        f.write("---\n\n")
        f.write(doc.content)
    
    console.print(f"\n[bold green]✓ Content saved to:[/bold green] {output_file}")
    return doc


def compare_modes(file_path: str):
    """Compare Stream vs Lattice mode results."""
    console.print("\n[bold cyan]Comparing Stream vs Lattice modes...[/bold cyan]\n")
    
    # Test both modes
    stream_doc = test_tabula_basic(file_path)
    lattice_doc = test_tabula_lattice(file_path)
    
    # Compare results
    console.print("\n[bold]Comparison:[/bold]")
    console.print(f"  Stream mode: {len(stream_doc.pages)} tables")
    console.print(f"  Lattice mode: {len(lattice_doc.pages)} tables")
    
    if len(stream_doc.pages) > len(lattice_doc.pages):
        console.print("\n[green]→ Stream mode found more tables[/green]")
    elif len(lattice_doc.pages) > len(stream_doc.pages):
        console.print("\n[green]→ Lattice mode found more tables[/green]")
    else:
        console.print("\n[yellow]→ Both modes found the same number of tables[/yellow]")


if __name__ == "__main__":
    console.print("[bold cyan]Tabula Reader Example[/bold cyan]")
    console.print("[dim]Best for: PDF tables with clear structure[/dim]")
    console.print("[dim]Note: Requires Java Runtime Environment (JRE)[/dim]\n")
    
    # Get file path from user
    console.print("[yellow]Enter PDF file path (or drag & drop):[/yellow] ", end="")
    file_path = input().strip()
    
    if not file_path:
        console.print("[red]No file specified[/red]")
        exit(1)
    
    try:
        # Ask for mode
        console.print("\n[yellow]Choose extraction mode:[/yellow]")
        console.print("  1. Stream mode (for borderless tables)")
        console.print("  2. Lattice mode (for bordered tables)")
        console.print("  3. Compare both modes")
        console.print("[yellow]Enter choice [1/2/3]:[/yellow] ", end="")
        
        choice = input().strip()
        
        if choice == "1":
            test_tabula_basic(file_path)
        elif choice == "2":
            test_tabula_lattice(file_path)
        elif choice == "3":
            compare_modes(file_path)
        else:
            console.print("[red]Invalid choice, using Stream mode[/red]")
            test_tabula_basic(file_path)
    
    except ImportError as e:
        console.print(f"\n[red]✗ Missing dependency: {e}[/red]")
        console.print("\n[yellow]Install tabula-py:[/yellow]")
        console.print("  pip install tabula-py")
        console.print("\n[yellow]Also make sure Java is installed:[/yellow]")
        console.print("  macOS: brew install openjdk")
        console.print("  Linux: sudo apt install default-jre")
        console.print("  Windows: Download from java.com")
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
