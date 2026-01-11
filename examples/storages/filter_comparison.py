"""
Vector Store Filter Capability Comparison

This script demonstrates the filtering capabilities of ChromaDB, Qdrant, LanceDB, and Milvus
using real e-commerce data with complex filtering scenarios.

Scenarios tested:
1. Simple filter (single condition)
2. Range filter (price range)
3. Complex filter (multiple conditions with AND/OR logic)
4. Category + Brand + Price filter
5. Nested conditions

Before running:
1. Start Ollama with jina/jina-embeddings-v2-base-en:latest model
2. Start Chroma server: run playground/start_chroma.bat
3. Start Qdrant server: run playground/start_qdrant.bat
4. Start Milvus server: In WSL2, run bash playground/start_milvus.sh
5. Ensure the dataset file exists at tmp/dataset.jsonl
"""

import json
import time
from typing import List, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

from zag.schemas.unit import TextUnit
from zag.schemas.base import UnitMetadata
from zag.storages.vector import ChromaVectorStore, QdrantVectorStore, LanceDBVectorStore, MilvusVectorStore
from zag import Embedder

console = Console()


def load_dataset(file_path: str, limit: int = 1000) -> List[Dict]:
    """Load e-commerce dataset"""
    console.print(f"[cyan]Loading dataset from {file_path}...[/cyan]")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                continue
    
    console.print(f"[green]✓ Loaded {len(data)} products[/green]\n")
    return data


def prepare_units(data: List[Dict]) -> List[TextUnit]:
    """Convert raw data to TextUnit objects with metadata"""
    units = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Preparing text units...", total=len(data))
        
        for item in data:
            # Create searchable content
            content = f"Product: {item.get('productDisplayName', 'N/A')} | " \
                     f"Brand: {item.get('brandName', 'N/A')} | " \
                     f"Category: {item.get('masterCategory', 'N/A')} - {item.get('subCategory', 'N/A')} | " \
                     f"Type: {item.get('articleType', 'N/A')} | " \
                     f"Color: {item.get('baseColour', 'N/A')} | " \
                     f"Season: {item.get('season', 'N/A')} | " \
                     f"Price: {item.get('price', 0)}"
            
            # Store metadata in custom fields for filtering
            custom_metadata = {
                'product_id': str(item['id']),
                'brand': item.get('brandName', 'Unknown'),
                'category': item.get('masterCategory', 'Unknown'),
                'sub_category': item.get('subCategory', 'Unknown'),
                'article_type': item.get('articleType', 'Unknown'),
                'color': item.get('baseColour', 'Unknown'),
                'season': item.get('season', 'Unknown'),
                'price': float(item.get('price', 0)),
                'year': int(item.get('year', 2000))
            }
            
            unit = TextUnit(
                unit_id=f"product_{item['id']}",
                content=content,
                metadata=UnitMetadata(
                    context_path=f"{item.get('masterCategory')}/{item.get('subCategory')}/{item.get('articleType')}",
                    custom=custom_metadata
                )
            )
            units.append(unit)
            progress.update(task, advance=1)
    
    console.print(f"[green]✓ Prepared {len(units)} text units[/green]\n")
    return units


def test_chroma_filters(store: ChromaVectorStore, query: str):
    """Test ChromaDB filtering capabilities"""
    console.print("\n[bold yellow]═══ CHROMADB FILTER TESTS ═══[/bold yellow]\n")
    
    results = {}
    
    # Test 1: Simple filter (exact match)
    console.print("[cyan]Test 1: Simple filter (brand = 'Nike')[/cyan]")
    try:
        start = time.time()
        # ChromaDB uses $eq for exact match
        filtered = store.search(
            query=query,
            top_k=10,
            filter={"custom_brand": {"$eq": "Nike"}}
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['simple'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['simple'] = {'success': False, 'error': str(e)}
    
    # Test 2: Range filter (price between 1000-2000)
    console.print("\n[cyan]Test 2: Range filter (price: 1000-2000)[/cyan]")
    try:
        start = time.time()
        filtered = store.search(
            query=query,
            top_k=10,
            filter={
                "$and": [
                    {"custom_price": {"$gte": 1000}},
                    {"custom_price": {"$lte": 2000}}
                ]
            }
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['range'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['range'] = {'success': False, 'error': str(e)}
    
    # Test 3: Complex OR condition (multiple brands)
    console.print("\n[cyan]Test 3: OR condition (brand in ['Nike', 'Adidas', 'Puma'])[/cyan]")
    try:
        start = time.time()
        filtered = store.search(
            query=query,
            top_k=10,
            filter={
                "$or": [
                    {"custom_brand": "Nike"},
                    {"custom_brand": "Adidas"},
                    {"custom_brand": "Puma"}
                ]
            }
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['or'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['or'] = {'success': False, 'error': str(e)}
    
    # Test 4: Complex AND + OR (category AND (brand OR brand))
    console.print("\n[cyan]Test 4: Complex filter (category='Apparel' AND brand in ['Nike','Adidas'] AND price<2000)[/cyan]")
    try:
        start = time.time()
        filtered = store.search(
            query=query,
            top_k=10,
            filter={
                "$and": [
                    {"custom_category": "Apparel"},
                    {
                        "$or": [
                            {"custom_brand": "Nike"},
                            {"custom_brand": "Adidas"}
                        ]
                    },
                    {"custom_price": {"$lt": 2000}}
                ]
            }
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['complex'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['complex'] = {'success': False, 'error': str(e)}
    
    return results


def test_qdrant_filters(store: QdrantVectorStore, query: str):
    """Test Qdrant filtering capabilities"""
    console.print("\n[bold green]═══ QDRANT FILTER TESTS ═══[/bold green]\n")
    
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
    
    results = {}
    
    # Test 1: Simple filter (exact match)
    console.print("[cyan]Test 1: Simple filter (brand = 'Nike')[/cyan]")
    try:
        start = time.time()
        filtered = store.search(
            query=query,
            top_k=10,
            filter=Filter(
                must=[
                    FieldCondition(key="custom_brand", match=MatchValue(value="Nike"))
                ]
            )
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['simple'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['simple'] = {'success': False, 'error': str(e)}
    
    # Test 2: Range filter (price between 1000-2000)
    console.print("\n[cyan]Test 2: Range filter (price: 1000-2000)[/cyan]")
    try:
        start = time.time()
        filtered = store.search(
            query=query,
            top_k=10,
            filter=Filter(
                must=[
                    FieldCondition(key="custom_price", range=Range(gte=1000, lte=2000))
                ]
            )
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['range'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['range'] = {'success': False, 'error': str(e)}
    
    # Test 3: Complex OR condition (multiple brands)
    console.print("\n[cyan]Test 3: OR condition (brand in ['Nike', 'Adidas', 'Puma'])[/cyan]")
    try:
        start = time.time()
        filtered = store.search(
            query=query,
            top_k=10,
            filter=Filter(
                should=[
                    FieldCondition(key="custom_brand", match=MatchValue(value="Nike")),
                    FieldCondition(key="custom_brand", match=MatchValue(value="Adidas")),
                    FieldCondition(key="custom_brand", match=MatchValue(value="Puma"))
                ]
            )
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['or'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['or'] = {'success': False, 'error': str(e)}
    
    # Test 4: Complex AND + OR (category AND (brand OR brand))
    console.print("\n[cyan]Test 4: Complex filter (category='Apparel' AND brand in ['Nike','Adidas'] AND price<2000)[/cyan]")
    try:
        start = time.time()
        filtered = store.search(
            query=query,
            top_k=10,
            filter=Filter(
                must=[
                    FieldCondition(key="custom_category", match=MatchValue(value="Apparel")),
                    FieldCondition(key="custom_price", range=Range(lt=2000))
                ],
                should=[
                    FieldCondition(key="custom_brand", match=MatchValue(value="Nike")),
                    FieldCondition(key="custom_brand", match=MatchValue(value="Adidas"))
                ]
            )
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['complex'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['complex'] = {'success': False, 'error': str(e)}
    
    return results


def test_lancedb_filters(store: LanceDBVectorStore, query: str):
    """Test LanceDB filtering capabilities"""
    console.print("\n[bold blue]═══ LANCEDB FILTER TESTS ═══[/bold blue]\n")
    
    results = {}
    
    # Test 1: Simple filter (exact match)
    console.print("[cyan]Test 1: Simple filter (brand = 'Nike')[/cyan]")
    try:
        start = time.time()
        # Use SQL WHERE clause - LanceDB's native and powerful way!
        filtered = store.search(
            query=query,
            top_k=10,
            filter="custom_brand = 'Nike'"
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['simple'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['simple'] = {'success': False, 'error': str(e)}
    
    # Test 2: Range filter (price between 1000-2000)
    console.print("\n[cyan]Test 2: Range filter (price: 1000-2000)[/cyan]")
    try:
        start = time.time()
        # SQL range query - native support!
        filtered = store.search(
            query=query,
            top_k=10,
            filter="custom_price >= 1000 AND custom_price <= 2000"
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['range'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['range'] = {'success': False, 'error': str(e)}
    
    # Test 3: Complex OR condition (multiple brands)
    console.print("\n[cyan]Test 3: OR condition (brand in ['Nike', 'Adidas', 'Puma'])[/cyan]")
    try:
        start = time.time()
        # SQL IN clause - native support!
        filtered = store.search(
            query=query,
            top_k=10,
            filter="custom_brand IN ('Nike', 'Adidas', 'Puma')"
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['or'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['or'] = {'success': False, 'error': str(e)}
    
    # Test 4: Complex filter
    console.print("\n[cyan]Test 4: Complex filter (category='Apparel' AND brand in ['Nike','Adidas'] AND price<2000)[/cyan]")
    try:
        start = time.time()
        # Complex SQL WHERE - full power of SQL!
        filtered = store.search(
            query=query,
            top_k=10,
            filter="custom_category = 'Apparel' AND custom_brand IN ('Nike', 'Adidas') AND custom_price < 2000"
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['complex'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['complex'] = {'success': False, 'error': str(e)}
    
    return results


def test_milvus_filters(store: MilvusVectorStore, query: str):
    """Test Milvus filtering capabilities"""
    console.print("\n[bold magenta]═══ MILVUS FILTER TESTS ═══[/bold magenta]\n")
    
    results = {}
    
    # Test 1: Simple filter (exact match)
    console.print("[cyan]Test 1: Simple filter (brand = 'Nike')[/cyan]")
    try:
        start = time.time()
        # Milvus uses expression syntax similar to SQL
        filtered = store.search(
            query=query,
            top_k=10,
            filter="custom_brand == 'Nike'"
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['simple'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['simple'] = {'success': False, 'error': str(e)}
    
    # Test 2: Range filter (price between 1000-2000)
    console.print("\n[cyan]Test 2: Range filter (price: 1000-2000)[/cyan]")
    try:
        start = time.time()
        # Milvus supports range queries
        filtered = store.search(
            query=query,
            top_k=10,
            filter="custom_price >= 1000 and custom_price <= 2000"
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['range'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['range'] = {'success': False, 'error': str(e)}
    
    # Test 3: Complex OR condition (multiple brands)
    console.print("\n[cyan]Test 3: OR condition (brand in ['Nike', 'Adidas', 'Puma'])[/cyan]")
    try:
        start = time.time()
        # Milvus supports 'in' operator
        filtered = store.search(
            query=query,
            top_k=10,
            filter="custom_brand in ['Nike', 'Adidas', 'Puma']"
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['or'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['or'] = {'success': False, 'error': str(e)}
    
    # Test 4: Complex filter
    console.print("\n[cyan]Test 4: Complex filter (category='Apparel' AND brand in ['Nike','Adidas'] AND price<2000)[/cyan]")
    try:
        start = time.time()
        # Complex boolean expression
        filtered = store.search(
            query=query,
            top_k=10,
            filter="custom_category == 'Apparel' and custom_brand in ['Nike', 'Adidas'] and custom_price < 2000"
        )
        elapsed = (time.time() - start) * 1000
        console.print(f"  ✓ Found {len(filtered)} results in {elapsed:.2f}ms")
        results['complex'] = {'success': True, 'count': len(filtered), 'time': elapsed}
    except Exception as e:
        console.print(f"  ✗ Error: {e}")
        results['complex'] = {'success': False, 'error': str(e)}
    
    return results


def display_comparison(chroma_results, qdrant_results, lancedb_results, milvus_results):
    """Display comparison table"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Filter Capability Comparison Results[/bold cyan]\n"
        f"[dim]Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="cyan"
    ))
    
    table = Table(
        title="[bold]Filter Performance & Capability Comparison[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("Test Scenario", style="cyan", width=35)
    table.add_column("ChromaDB", style="yellow", width=20)
    table.add_column("Qdrant", style="green", width=20)
    table.add_column("LanceDB", style="blue", width=20)
    table.add_column("Milvus", style="magenta", width=20)
    
    tests = ['simple', 'range', 'or', 'complex']
    test_names = [
        "Simple Filter (brand=Nike)",
        "Range Filter (price: 1000-2000)",
        "OR Condition (3 brands)",
        "Complex Filter (category+brand+price)"
    ]
    
    for test, name in zip(tests, test_names):
        chroma = chroma_results.get(test, {})
        qdrant = qdrant_results.get(test, {})
        lancedb = lancedb_results.get(test, {})
        milvus = milvus_results.get(test, {})
        
        def format_result(r):
            if not r:
                return "[dim]N/A[/dim]"
            if not r.get('success'):
                return "[red]✗ Failed[/red]"
            manual = " [dim](manual)[/dim]" if r.get('manual') else ""
            return f"[green]✓[/green] {r['count']} results\n{r['time']:.1f}ms{manual}"
        
        table.add_row(
            name,
            format_result(chroma),
            format_result(qdrant),
            format_result(lancedb),
            format_result(milvus)
        )
    
    console.print("\n", table)
    
    # Summary
    console.print("\n[bold]📊 Summary:[/bold]\n")
    console.print("  [yellow]ChromaDB:[/yellow]")
    console.print("    ✓ Supports basic filters ($eq, $gte, $lte)")
    console.print("    ✓ Supports AND/OR logic")
    console.print("    ⚠ Limited complex query support")
    console.print("    ⭐ Fastest on small datasets")
    console.print()
    console.print("  [green]Qdrant:[/green]")
    console.print("    ✓ Full filter support (must/should/must_not)")
    console.print("    ✓ Native range queries")
    console.print("    ✓ Excellent for complex filters")
    console.print("    ✓ Best for large-scale production")
    console.print()
    console.print("  [blue]LanceDB:[/blue]")
    console.print("    ✓ Full SQL WHERE clause support")
    console.print("    ✓ Native IN, AND, OR, range queries")
    console.print("    ✓ Supports regex, LIKE, DataFusion functions")
    console.print("    ✓ Powerful and flexible filtering")
    console.print("    ✅ Embedded, no server needed")
    console.print()
    console.print("  [magenta]Milvus:[/magenta]")
    console.print("    ✓ Boolean expression syntax")
    console.print("    ✓ Native 'in' operator and range queries")
    console.print("    ✓ Supports AND/OR/NOT logic")
    console.print("    ✓ High-performance at scale")
    console.print("    ✅ Best for large-scale production")


def main():
    """Main execution"""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]Vector Store Filter Capability Comparison[/bold cyan]\n"
        "[yellow]Real-world E-commerce Filtering Scenarios[/yellow]\n"
        "[dim]ChromaDB vs Qdrant vs LanceDB vs Milvus[/dim]",
        border_style="cyan",
        title="[bold]Filter Test[/bold]"
    ))
    
    # Configuration
    DATA_FILE = "tmp/dataset.jsonl"
    DATA_LIMIT = 1000  # Use 1000 products for filtering tests
    CHROMA_HOST = "localhost"
    CHROMA_PORT = 18000
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 16333
    QDRANT_GRPC_PORT = 16334
    LANCEDB_PATH = "./tmp/lancedb_filter_test"
    MILVUS_HOST = "localhost"
    MILVUS_PORT = 19530
    OLLAMA_MODEL = "jina/jina-embeddings-v2-base-en:latest"
    
    try:
        # Step 1: Load data
        console.print("\n[bold]Step 1: Load Dataset[/bold]")
        data = load_dataset(DATA_FILE, limit=DATA_LIMIT)
        
        # Step 2: Prepare units
        console.print("[bold]Step 2: Prepare Text Units[/bold]")
        units = prepare_units(data)
        
        # Step 3: Initialize embedder
        console.print("[bold]Step 3: Initialize Embedder[/bold]")
        console.print(f"[cyan]Creating embedder with model: {OLLAMA_MODEL}[/cyan]")
        embedder = Embedder(f"ollama/{OLLAMA_MODEL}")
        console.print(f"[green]✓ Embedder initialized (dimension: {embedder.dimension})[/green]\n")
        
        # Sample query for testing
        test_query = "running shoes"
        
        # Step 4: Test ChromaDB
        console.print("[bold]Step 4: ChromaDB Filter Tests[/bold]")
        console.print(f"[cyan]Connecting to Chroma server...[/cyan]")
        chroma_store = ChromaVectorStore.server(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            collection_name="filter_test",
            embedder=embedder
        )
        console.print("[green]✓ Connected[/green]")
        console.print("[yellow]Clearing old data...[/yellow]")
        chroma_store.clear()
        console.print("[green]✓ Data cleared, adding new data...[/green]")
        chroma_store.add(units)
        console.print(f"[green]✓ Added {len(units)} products[/green]")
        chroma_results = test_chroma_filters(chroma_store, test_query)
        
        # Step 5: Test Qdrant
        console.print("\n[bold]Step 5: Qdrant Filter Tests[/bold]")
        console.print(f"[cyan]Connecting to Qdrant server...[/cyan]")
        qdrant_store = QdrantVectorStore.server(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=QDRANT_GRPC_PORT,
            prefer_grpc=True,
            collection_name="filter_test",
            embedder=embedder
        )
        console.print("[green]✓ Connected[/green]")
        console.print("[yellow]Clearing old data...[/yellow]")
        qdrant_store.clear()
        console.print("[green]✓ Data cleared, adding new data...[/green]")
        qdrant_store.add(units)
        console.print(f"[green]✓ Added {len(units)} products[/green]")
        qdrant_results = test_qdrant_filters(qdrant_store, test_query)
        
        # Step 6: Test LanceDB
        console.print("\n[bold]Step 6: LanceDB Filter Tests[/bold]")
        console.print(f"[cyan]Creating LanceDB store...[/cyan]")
        lancedb_store = LanceDBVectorStore.local(
            path=LANCEDB_PATH,
            table_name="filter_test",
            embedder=embedder
        )
        console.print("[green]✓ Created[/green]")
        console.print("[yellow]Clearing old data (if exists)...[/yellow]")
        lancedb_store.clear()
        console.print("[green]✓ Data cleared, adding new data...[/green]")
        lancedb_store.add(units)
        console.print(f"[green]✓ Added {len(units)} products[/green]")
        lancedb_results = test_lancedb_filters(lancedb_store, test_query)
        
        # Step 7: Test Milvus
        console.print("\n[bold]Step 7: Milvus Filter Tests[/bold]")
        console.print(f"[cyan]Connecting to Milvus server...[/cyan]")
        milvus_store = MilvusVectorStore.server(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name="filter_test",
            embedder=embedder
        )
        console.print("[green]✓ Connected[/green]")
        console.print("[yellow]Clearing old data...[/yellow]")
        milvus_store.clear()
        console.print("[green]✓ Data cleared, adding new data...[/green]")
        milvus_store.add(units)
        console.print(f"[green]✓ Added {len(units)} products[/green]")
        milvus_results = test_milvus_filters(milvus_store, test_query)
        
        # Display comparison
        display_comparison(chroma_results, qdrant_results, lancedb_results, milvus_results)
        
        console.print("\n[bold green]✓ Filter comparison completed![/bold green]\n")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
