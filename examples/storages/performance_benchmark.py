"""
Vector Store Performance Benchmark

This script compares the performance of ChromaDB, Qdrant, LanceDB, and Milvus
vector stores using the e-commerce dataset with Ollama embeddings.

Before running:
1. Start Ollama with jina/jina-embeddings-v2-base-en:latest model
2. Start Chroma server: run playground/start_chroma.bat
3. Start Qdrant server: run playground/start_qdrant.bat
4. Start Milvus server: In WSL2, run bash playground/start_milvus.sh
5. Ensure the dataset file exists at tmp/dataset.jsonl
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

from zag.schemas.unit import TextUnit
from zag.schemas import UnitMetadata
from zag.storages.vector import ChromaVectorStore, QdrantVectorStore, LanceDBVectorStore, MilvusVectorStore
from zag import Embedder

console = Console()


class BenchmarkResults:
    """Store and display benchmark results"""
    
    def __init__(self):
        self.results = {
            'chroma': {},
            'qdrant': {},
            'lancedb': {},
            'milvus': {}
        }
    
    def add_result(self, store_name: str, metric: str, value: Any):
        """Add a metric result"""
        self.results[store_name][metric] = value
    
    def display_summary(self):
        """Display a comprehensive summary table"""
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]Performance Benchmark Results[/bold cyan]\n"
            f"[dim]Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="cyan"
        ))
        
        # Create main comparison table
        table = Table(
            title="[bold]Vector Store Performance Comparison[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="cyan"
        )
        
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("ChromaDB", style="yellow", justify="right", width=15)
        table.add_column("Qdrant", style="green", justify="right", width=15)
        table.add_column("LanceDB", style="blue", justify="right", width=15)
        table.add_column("Milvus", style="magenta", justify="right", width=15)
        table.add_column("Winner", style="bold", justify="center", width=12)
        
        # Data loading
        chroma_load = self.results['chroma'].get('load_time', 0)
        qdrant_load = self.results['qdrant'].get('load_time', 0)
        lancedb_load = self.results['lancedb'].get('load_time', 0)
        milvus_load = self.results['milvus'].get('load_time', 0)
        load_winner = min([('Chroma', chroma_load), ('Qdrant', qdrant_load), ('LanceDB', lancedb_load), ('Milvus', milvus_load)], key=lambda x: x[1])[0]
        table.add_row(
            "Data Loading Time",
            f"{chroma_load:.2f}s",
            f"{qdrant_load:.2f}s",
            f"{lancedb_load:.2f}s",
            f"{milvus_load:.2f}s",
            f"[green]{load_winner}[/green]"
        )
        
        # Embedding generation
        chroma_embed = self.results['chroma'].get('embed_time', 0)
        qdrant_embed = self.results['qdrant'].get('embed_time', 0)
        lancedb_embed = self.results['lancedb'].get('embed_time', 0)
        milvus_embed = self.results['milvus'].get('embed_time', 0)
        table.add_row(
            "Embedding Generation",
            f"{chroma_embed:.2f}s",
            f"{qdrant_embed:.2f}s",
            f"{lancedb_embed:.2f}s",
            f"{milvus_embed:.2f}s",
            "[dim]Same[/dim]"
        )
        
        # Write performance
        chroma_write = self.results['chroma'].get('write_time', 0)
        qdrant_write = self.results['qdrant'].get('write_time', 0)
        lancedb_write = self.results['lancedb'].get('write_time', 0)
        milvus_write = self.results['milvus'].get('write_time', 0)
        write_winner = min([('Chroma', chroma_write), ('Qdrant', qdrant_write), ('LanceDB', lancedb_write), ('Milvus', milvus_write)], key=lambda x: x[1])[0]
        table.add_row(
            "Vector Write Time",
            f"{chroma_write:.2f}s",
            f"{qdrant_write:.2f}s",
            f"{lancedb_write:.2f}s",
            f"{milvus_write:.2f}s",
            f"[green]{write_winner}[/green]"
        )
        
        chroma_throughput = self.results['chroma'].get('write_throughput', 0)
        qdrant_throughput = self.results['qdrant'].get('write_throughput', 0)
        lancedb_throughput = self.results['lancedb'].get('write_throughput', 0)
        milvus_throughput = self.results['milvus'].get('write_throughput', 0)
        throughput_winner = max([('Chroma', chroma_throughput), ('Qdrant', qdrant_throughput), ('LanceDB', lancedb_throughput), ('Milvus', milvus_throughput)], key=lambda x: x[1])[0]
        table.add_row(
            "Write Throughput",
            f"{chroma_throughput:.1f} items/s",
            f"{qdrant_throughput:.1f} items/s",
            f"{lancedb_throughput:.1f} items/s",
            f"{milvus_throughput:.1f} items/s",
            f"[green]{throughput_winner}[/green]"
        )
        
        # Query performance
        chroma_query = self.results['chroma'].get('avg_query_time', 0)
        qdrant_query = self.results['qdrant'].get('avg_query_time', 0)
        lancedb_query = self.results['lancedb'].get('avg_query_time', 0)
        milvus_query = self.results['milvus'].get('avg_query_time', 0)
        query_winner = min([('Chroma', chroma_query), ('Qdrant', qdrant_query), ('LanceDB', lancedb_query), ('Milvus', milvus_query)], key=lambda x: x[1])[0]
        table.add_row(
            "Average Query Time",
            f"{chroma_query*1000:.2f}ms",
            f"{qdrant_query*1000:.2f}ms",
            f"{lancedb_query*1000:.2f}ms",
            f"{milvus_query*1000:.2f}ms",
            f"[green]{query_winner}[/green]"
        )
        
        chroma_qps = self.results['chroma'].get('queries_per_second', 0)
        qdrant_qps = self.results['qdrant'].get('queries_per_second', 0)
        lancedb_qps = self.results['lancedb'].get('queries_per_second', 0)
        milvus_qps = self.results['milvus'].get('queries_per_second', 0)
        qps_winner = max([('Chroma', chroma_qps), ('Qdrant', qdrant_qps), ('LanceDB', lancedb_qps), ('Milvus', milvus_qps)], key=lambda x: x[1])[0]
        table.add_row(
            "Queries Per Second (QPS)",
            f"{chroma_qps:.1f}",
            f"{qdrant_qps:.1f}",
            f"{lancedb_qps:.1f}",
            f"{milvus_qps:.1f}",
            f"[green]{qps_winner}[/green]"
        )
        
        # Storage info
        chroma_count = self.results['chroma'].get('total_vectors', 0)
        qdrant_count = self.results['qdrant'].get('total_vectors', 0)
        lancedb_count = self.results['lancedb'].get('total_vectors', 0)
        milvus_count = self.results['milvus'].get('total_vectors', 0)
        table.add_row(
            "Total Vectors Stored",
            f"{chroma_count:,}",
            f"{qdrant_count:,}",
            f"{lancedb_count:,}",
            f"{milvus_count:,}",
            "[dim]Equal[/dim]"
        )
        
        console.print("\n", table)
        
        # Overall winner
        self._display_overall_winner()
    
    def _display_overall_winner(self):
        """Calculate and display overall winner"""
        chroma_score = 0
        qdrant_score = 0
        lancedb_score = 0
        milvus_score = 0
        
        # Score based on key metrics
        write_throughputs = [
            ('chroma', self.results['chroma']['write_throughput']),
            ('qdrant', self.results['qdrant']['write_throughput']),
            ('lancedb', self.results['lancedb']['write_throughput']),
            ('milvus', self.results['milvus']['write_throughput'])
        ]
        winner = max(write_throughputs, key=lambda x: x[1])[0]
        if winner == 'chroma': chroma_score += 1
        elif winner == 'qdrant': qdrant_score += 1
        elif winner == 'lancedb': lancedb_score += 1
        else: milvus_score += 1
        
        qps_values = [
            ('chroma', self.results['chroma']['queries_per_second']),
            ('qdrant', self.results['qdrant']['queries_per_second']),
            ('lancedb', self.results['lancedb']['queries_per_second']),
            ('milvus', self.results['milvus']['queries_per_second'])
        ]
        winner = max(qps_values, key=lambda x: x[1])[0]
        if winner == 'chroma': chroma_score += 1
        elif winner == 'qdrant': qdrant_score += 1
        elif winner == 'lancedb': lancedb_score += 1
        else: milvus_score += 1
        
        query_times = [
            ('chroma', self.results['chroma']['avg_query_time']),
            ('qdrant', self.results['qdrant']['avg_query_time']),
            ('lancedb', self.results['lancedb']['avg_query_time']),
            ('milvus', self.results['milvus']['avg_query_time'])
        ]
        winner = min(query_times, key=lambda x: x[1])[0]
        if winner == 'chroma': chroma_score += 1
        elif winner == 'qdrant': qdrant_score += 1
        elif winner == 'lancedb': lancedb_score += 1
        else: milvus_score += 1
        
        # Determine overall winner
        scores = [('ChromaDB', chroma_score), ('Qdrant', qdrant_score), ('LanceDB', lancedb_score), ('Milvus', milvus_score)]
        max_score = max(scores, key=lambda x: x[1])[1]
        winners = [name for name, score in scores if score == max_score]
        
        if len(winners) == 1:
            winner = winners[0]
            color = "yellow" if winner == "ChromaDB" else ("green" if winner == "Qdrant" else ("blue" if winner == "LanceDB" else "magenta"))
        else:
            winner = " & ".join(winners)
            color = "cyan"
        
        console.print(Panel.fit(
            f"[bold {color}]Overall Winner: {winner}[/bold {color}]\n"
            f"[dim]ChromaDB: {chroma_score} | Qdrant: {qdrant_score} | LanceDB: {lancedb_score} | Milvus: {milvus_score}[/dim]",
            border_style=color,
            title="[bold]Final Verdict[/bold]"
        ))


def load_dataset(file_path: str, limit: int = None) -> List[Dict]:
    """Load e-commerce dataset from JSONL file"""
    console.print(f"\n[cyan]Loading dataset from {file_path}...[/cyan]")
    
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
    
    console.print(f"[green]✓ Loaded {len(data)} products[/green]")
    return data


def prepare_units(data: List[Dict]) -> List[TextUnit]:
    """Convert raw data to TextUnit objects"""
    units = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Preparing text units...", total=len(data))
        
        for item in data:
            # Create searchable content from product fields
            content_parts = [
                f"Product: {item.get('productDisplayName', 'N/A')}",
                f"Brand: {item.get('brandName', 'N/A')}",
                f"Category: {item.get('masterCategory', 'N/A')} - {item.get('subCategory', 'N/A')}",
                f"Type: {item.get('articleType', 'N/A')}",
                f"Color: {item.get('baseColour', 'N/A')}",
                f"Season: {item.get('season', 'N/A')}",
                f"Price: {item.get('price', 0)}"
            ]
            
            if item.get('description'):
                # Extract plain text from HTML description (simple version)
                desc = item['description'].replace('<p>', '').replace('</p>', '')
                desc = desc.replace('<br />', ' ').replace('<br/>', ' ')
                desc = desc[:200]  # Limit description length
                content_parts.append(f"Description: {desc}")
            
            content = " | ".join(content_parts)
            
            unit = TextUnit(
                unit_id=f"product_{item['id']}",
                content=content,
                metadata=UnitMetadata(
                    context_path=f"{item.get('masterCategory')}/{item.get('subCategory')}/{item.get('articleType')}"
                )
            )
            units.append(unit)
            progress.update(task, advance=1)
    
    console.print(f"[green]✓ Prepared {len(units)} text units[/green]")
    return units


def benchmark_write(store, units: List[TextUnit], store_name: str, batch_size: int = 100) -> Dict[str, float]:
    """Benchmark vector write performance with batching"""
    console.print(f"\n[yellow]→ Benchmarking {store_name} write performance...[/yellow]")
    console.print(f"[dim]  Writing {len(units)} vectors in batches of {batch_size}...[/dim]")
    
    start_time = time.time()
    
    # Split into batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(f"[cyan]Writing to {store_name}...", total=len(units))
        
        for i in range(0, len(units), batch_size):
            batch = units[i:i + batch_size]
            store.add(batch)
            progress.update(task, advance=len(batch))
    
    end_time = time.time()
    
    write_time = end_time - start_time
    throughput = len(units) / write_time
    
    console.print(f"[green]✓ {store_name} write completed in {write_time:.2f}s ({throughput:.1f} items/s)[/green]")
    
    return {
        'write_time': write_time,
        'write_throughput': throughput
    }


def benchmark_query(store, query_texts: List[str], store_name: str, top_k: int = 10) -> Dict[str, float]:
    """Benchmark query performance"""
    console.print(f"\n[yellow]→ Benchmarking {store_name} query performance...[/yellow]")
    
    query_times = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task(f"[cyan]Running {len(query_texts)} queries...", total=len(query_texts))
        
        for query in query_texts:
            start_time = time.time()
            results = store.search(query, top_k=top_k)
            end_time = time.time()
            
            query_times.append(end_time - start_time)
            progress.update(task, advance=1)
    
    avg_query_time = sum(query_times) / len(query_times)
    min_query_time = min(query_times)
    max_query_time = max(query_times)
    qps = 1 / avg_query_time if avg_query_time > 0 else 0
    
    console.print(f"[green]✓ {store_name} queries completed[/green]")
    console.print(f"  • Average: {avg_query_time*1000:.2f}ms")
    console.print(f"  • Min: {min_query_time*1000:.2f}ms")
    console.print(f"  • Max: {max_query_time*1000:.2f}ms")
    console.print(f"  • QPS: {qps:.1f}")
    
    return {
        'avg_query_time': avg_query_time,
        'min_query_time': min_query_time,
        'max_query_time': max_query_time,
        'queries_per_second': qps
    }


def main():
    """Main benchmark execution"""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]Vector Store Performance Benchmark[/bold cyan]\n"
        "[yellow]ChromaDB vs Qdrant vs LanceDB vs Milvus[/yellow]\n"
        "[dim]Using Ollama Embeddings (jina/jina-embeddings-v2-base-en)[/dim]",
        border_style="cyan",
        title="[bold]Performance Test[/bold]"
    ))
    
    # Configuration
    DATA_FILE = "tmp/dataset.jsonl"
    DATA_LIMIT = 5000  # Limit to 5000 products for reasonable test time
    WRITE_BATCH_SIZE = 100  # Write in batches to avoid connection timeout
    NUM_QUERIES = 50
    CHROMA_HOST = "localhost"
    CHROMA_PORT = 18000
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 16333
    QDRANT_GRPC_PORT = 16334  # gRPC is much faster!
    LANCEDB_PATH = "./tmp/lancedb_benchmark"  # Local embedded database
    MILVUS_HOST = "localhost"  # WSL2 auto-maps to Windows
    MILVUS_PORT = 19530
    OLLAMA_MODEL = "jina/jina-embeddings-v2-base-en:latest"
    
    results = BenchmarkResults()
    
    try:
        # Step 1: Load data
        console.print("\n[bold]Step 1: Data Loading[/bold]")
        load_start = time.time()
        data = load_dataset(DATA_FILE, limit=DATA_LIMIT)
        load_time = time.time() - load_start
        
        # Step 2: Prepare units
        console.print("\n[bold]Step 2: Prepare Text Units[/bold]")
        units = prepare_units(data)
        
        # Step 3: Initialize embedder
        console.print("\n[bold]Step 3: Initialize Ollama Embedder[/bold]")
        console.print(f"[cyan]Creating embedder with model: {OLLAMA_MODEL}[/cyan]")
        embedder = Embedder(f"ollama/{OLLAMA_MODEL}")
        console.print(f"[green]✓ Embedder initialized (dimension: {embedder.dimension})[/green]")
        
        # Step 4: Generate embeddings (shared for both stores)
        console.print("\n[bold]Step 4: Generate Embeddings[/bold]")
        console.print("[cyan]Generating embeddings for all units...[/cyan]")
        embed_start = time.time()
        # This will be cached by Ollama, so both stores benefit equally
        sample_embedding = embedder.embed(units[0].content)
        embed_time = time.time() - embed_start
        console.print(f"[green]✓ Embedding test completed ({embed_time:.2f}s for 1 sample)[/green]")
        
        # Store embedding time
        results.add_result('chroma', 'embed_time', embed_time)
        results.add_result('qdrant', 'embed_time', embed_time)
        results.add_result('lancedb', 'embed_time', embed_time)
        results.add_result('chroma', 'load_time', load_time)
        results.add_result('qdrant', 'load_time', load_time)
        results.add_result('lancedb', 'load_time', load_time)
        
        # ============================================================
        # CHROMA BENCHMARK
        # ============================================================
        console.print("\n" + "="*70)
        console.print("[bold yellow]CHROMA PERFORMANCE TEST[/bold yellow]")
        console.print("="*70)
        
        try:
            console.print(f"\n[cyan]Connecting to Chroma server at {CHROMA_HOST}:{CHROMA_PORT}...[/cyan]")
            chroma_store = ChromaVectorStore.server(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                collection_name="benchmark_test",
                embedder=embedder
            )
            console.print(f"[green]✓ Connected to Chroma server[/green]")
            console.print(f"[yellow]Clearing old data...[/yellow]")
            chroma_store.clear()
            console.print(f"[green]✓ Data cleared[/green]")
            
            # Write benchmark
            chroma_write_results = benchmark_write(chroma_store, units, "ChromaDB", batch_size=WRITE_BATCH_SIZE)
            results.add_result('chroma', 'write_time', chroma_write_results['write_time'])
            results.add_result('chroma', 'write_throughput', chroma_write_results['write_throughput'])
            results.add_result('chroma', 'total_vectors', chroma_store.count())
            
            # Generate query texts from random products
            query_texts = [random.choice(units).content[:100] for _ in range(NUM_QUERIES)]
            
            # Query benchmark
            chroma_query_results = benchmark_query(chroma_store, query_texts, "ChromaDB")
            results.add_result('chroma', 'avg_query_time', chroma_query_results['avg_query_time'])
            results.add_result('chroma', 'queries_per_second', chroma_query_results['queries_per_second'])
            
        except Exception as e:
            console.print(f"[red]✗ Chroma test failed: {e}[/red]")
            console.print("[yellow]Make sure Chroma server is running: playground/start_chroma.bat[/yellow]")
            return
        
        # ============================================================
        # QDRANT BENCHMARK
        # ============================================================
        console.print("\n" + "="*70)
        console.print("[bold green]QDRANT PERFORMANCE TEST[/bold green]")
        console.print("="*70)
        
        try:
            console.print(f"\n[cyan]Connecting to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT} (gRPC: {QDRANT_GRPC_PORT})...[/cyan]")
            qdrant_store = QdrantVectorStore.server(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                grpc_port=QDRANT_GRPC_PORT,
                prefer_grpc=True,
                collection_name="benchmark_test",
                embedder=embedder
            )
            console.print(f"[green]✓ Connected to Qdrant server (using gRPC for better performance)[/green]")
            console.print(f"[yellow]Clearing old data...[/yellow]")
            qdrant_store.clear()
            console.print(f"[green]✓ Data cleared[/green]")
            
            # Write benchmark
            qdrant_write_results = benchmark_write(qdrant_store, units, "Qdrant", batch_size=WRITE_BATCH_SIZE)
            results.add_result('qdrant', 'write_time', qdrant_write_results['write_time'])
            results.add_result('qdrant', 'write_throughput', qdrant_write_results['write_throughput'])
            results.add_result('qdrant', 'total_vectors', qdrant_store.count())
            
            # Query benchmark (use same queries)
            qdrant_query_results = benchmark_query(qdrant_store, query_texts, "Qdrant")
            results.add_result('qdrant', 'avg_query_time', qdrant_query_results['avg_query_time'])
            results.add_result('qdrant', 'queries_per_second', qdrant_query_results['queries_per_second'])
            
        except Exception as e:
            console.print(f"[red]✗ Qdrant test failed: {e}[/red]")
            console.print("[yellow]Make sure Qdrant server is running: playground/start_qdrant.bat[/yellow]")
            return
        
        # ============================================================
        # LANCEDB BENCHMARK
        # ============================================================
        console.print("\n" + "="*70)
        console.print("[bold blue]LANCEDB PERFORMANCE TEST[/bold blue]")
        console.print("="*70)
        
        try:
            console.print(f"\n[cyan]Creating LanceDB store at {LANCEDB_PATH}...[/cyan]")
            lancedb_store = LanceDBVectorStore.local(
                path=LANCEDB_PATH,
                table_name="benchmark_test",
                embedder=embedder
            )
            console.print(f"[green]✓ Created LanceDB store (embedded, no server needed)[/green]")
            console.print(f"[yellow]Clearing old data (if exists)...[/yellow]")
            lancedb_store.clear()
            console.print(f"[green]✓ Data cleared[/green]")
            
            # Write benchmark
            lancedb_write_results = benchmark_write(lancedb_store, units, "LanceDB", batch_size=WRITE_BATCH_SIZE)
            results.add_result('lancedb', 'write_time', lancedb_write_results['write_time'])
            results.add_result('lancedb', 'write_throughput', lancedb_write_results['write_throughput'])
            results.add_result('lancedb', 'total_vectors', lancedb_store.count())
            
            # Query benchmark (use same queries)
            lancedb_query_results = benchmark_query(lancedb_store, query_texts, "LanceDB")
            results.add_result('lancedb', 'avg_query_time', lancedb_query_results['avg_query_time'])
            results.add_result('lancedb', 'queries_per_second', lancedb_query_results['queries_per_second'])
            
        except Exception as e:
            console.print(f"[red]✗ LanceDB test failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return
        
        # ============================================================
        # MILVUS BENCHMARK
        # ============================================================
        console.print("\n" + "="*70)
        console.print("[bold magenta]MILVUS PERFORMANCE TEST[/bold magenta]")
        console.print("="*70)
        
        try:
            console.print(f"\n[cyan]Connecting to Milvus server at {MILVUS_HOST}:{MILVUS_PORT}...[/cyan]")
            milvus_store = MilvusVectorStore.server(
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                collection_name="benchmark_test",
                embedder=embedder
            )
            console.print(f"[green]✓ Connected to Milvus server (running in WSL2)[/green]")
            console.print(f"[yellow]Clearing old data...[/yellow]")
            milvus_store.clear()
            console.print(f"[green]✓ Data cleared[/green]")
            
            # Write benchmark
            milvus_write_results = benchmark_write(milvus_store, units, "Milvus", batch_size=WRITE_BATCH_SIZE)
            results.add_result('milvus', 'write_time', milvus_write_results['write_time'])
            results.add_result('milvus', 'write_throughput', milvus_write_results['write_throughput'])
            results.add_result('milvus', 'total_vectors', milvus_store.count())
            
            # Query benchmark (use same queries)
            milvus_query_results = benchmark_query(milvus_store, query_texts, "Milvus")
            results.add_result('milvus', 'avg_query_time', milvus_query_results['avg_query_time'])
            results.add_result('milvus', 'queries_per_second', milvus_query_results['queries_per_second'])
            
        except Exception as e:
            console.print(f"[red]✗ Milvus test failed: {e}[/red]")
            console.print("[yellow]Make sure Milvus server is running in WSL2:[/yellow]")
            console.print("[dim]  cd playground[/dim]")
            console.print("[dim]  bash start_milvus.sh[/dim]")
            import traceback
            traceback.print_exc()
            return
        
        # Display results
        results.display_summary()
        
        console.print("\n[bold green]✓ Benchmark completed successfully![/bold green]\n")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
