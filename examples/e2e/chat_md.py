"""End-to-end example: Chat with a Markdown document using DocTree retrievers.

This script demonstrates a complete pipeline that:

    1. Reads a Markdown file and builds a DocTree-compatible structure.
    2. Optionally caches the tree on disk keyed by file hash.
    3. Lets the user choose a tree-based retrieval strategy (simple or MCTS).
    4. Uses an LLM answer generator to produce answers with citations.

Dependencies (install as needed):
    pip install rich diskcache xxhash chakpy tiktoken python-dotenv

Run:
    python examples/e2e/chat_md.py
"""

import asyncio
import os
import time

import xxhash
from diskcache import Cache
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

from zag.readers import MarkdownTreeReader
from zag.schemas import DocTree, TreeNode
from zag.retrievers import SimpleRetriever, MCTSRetriever, SkeletonRetriever
from zag.generators import GeneralAnswerGenerator

# Load environment variables from .env file
load_dotenv()

console = Console()

# Use OS temp directory for cross-platform compatibility
_CACHE_DIR = os.path.join(os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp", "zag_cache")
print(f"Cache directory: {_CACHE_DIR}")
cache = Cache(_CACHE_DIR)

# Get API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    console.print("[bold red]Warning: OPENAI_API_KEY not found in environment. Please set it in .env file.[/bold red]")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _compute_file_hash(file_path: str) -> str:
    """Compute xxhash64 for a file path."""

    hasher = xxhash.xxh64()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


async def get_or_build_tree(md_file: str) -> DocTree:
    """Get or build a DocTree representation for the given markdown file."""

    file_hash = _compute_file_hash(md_file)
    cached_tree = cache.get(file_hash)

    if cached_tree is not None:
        console.print("[green]Loaded tree from cache (file unchanged).[/green]")
        return cached_tree

    console.print("[yellow]Building tree (first run or file changed)...[/yellow]")
    reader = MarkdownTreeReader(llm_uri="openai/gpt-4o", api_key=OPENAI_API_KEY)
    tree = await reader.read(path=md_file, generate_summaries=True)

    cache.set(file_hash, tree)
    console.print("[green]Tree cached for future runs.[/green]")
    return tree


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _clean_path(input_path: str) -> str:
    """Clean Windows drag-and-drop path quirks."""

    path = input_path.strip()
    if path.startswith("\"") and path.endswith("\""):
        path = path[1:-1]
    if path.startswith("'") and path.endswith("'"):
        path = path[1:-1]
    if path.startswith("& "):
        path = path[2:].strip()
        if path.startswith("\"") and path.endswith("\""):
            path = path[1:-1]
    return path


def _truncate_text(text: str, head_lines: int = 3, tail_lines: int = 3, max_line_length: int = 100) -> str:
    """Show only the head and tail of a long text block."""

    lines = text.split("\n")
    if len(lines) <= head_lines + tail_lines:
        return "\n".join(
            line[:max_line_length] + "..." if len(line) > max_line_length else line
            for line in lines
        )

    head = lines[:head_lines]
    tail = lines[-tail_lines:]

    head = [line[:max_line_length] + "..." if len(line) > max_line_length else line for line in head]
    tail = [line[:max_line_length] + "..." if len(line) > max_line_length else line for line in tail]

    result = "\n".join(head)
    omitted = len(lines) - head_lines - tail_lines
    result += f"\n\n... ({omitted} lines omitted) ...\n\n"
    result += "\n".join(tail)
    return result


def _choose_retriever():
    """Interactive retriever selection."""

    console.print("[bold yellow]Step 5:[/bold yellow] Choose retriever strategy")
    console.print("  [1] SimpleRetriever - fast, greedy layer-by-layer search")
    console.print("  [2] SkeletonRetriever (summary mode, default) - fast, lossy")
    console.print("  [3] SkeletonRetriever (full text mode) - lossless but slower")
    console.print("  [4] MCTSRetriever (custom) - configurable MCTS search")
    console.print("  [5] MCTSRetriever (preset: fast) - quick tests (~5s)")
    console.print("  [6] MCTSRetriever (preset: balanced) - general use (~10s)")
    console.print("  [7] MCTSRetriever (preset: accurate) - high precision (~20s)")
    console.print("  [8] MCTSRetriever (preset: explore) - maximize recall")

    choice = console.input("\n[cyan]Select retriever (1-8, default=2): [/cyan]").strip()

    use_full_text = False
    
    if choice == "1":
        console.print("[green]Using SimpleRetriever.[/green]")
        retriever = SimpleRetriever(llm_uri="openai/gpt-4o", api_key=OPENAI_API_KEY)
    
    elif choice == "3":
        console.print("[green]Using SkeletonRetriever (full text mode - lossless).[/green]")
        retriever = SkeletonRetriever(llm_uri="openai/gpt-4o", api_key=OPENAI_API_KEY, verbose=True)
        use_full_text = True
    
    elif choice == "4":
        console.print("[green]Using MCTSRetriever (custom).[/green]")
        retriever = MCTSRetriever(llm_uri="openai/gpt-4o", api_key=OPENAI_API_KEY, iterations=50, exploration_c=1.4, top_k=5, verbose=True)
        retriever.print_config()
    
    elif choice == "5":
        console.print("[green]Using MCTSRetriever (preset: fast).[/green]")
        retriever = MCTSRetriever.from_preset("fast", api_key=OPENAI_API_KEY, verbose=True)
        retriever.print_config()
    
    elif choice == "6":
        console.print("[green]Using MCTSRetriever (preset: balanced).[/green]")
        retriever = MCTSRetriever.from_preset("balanced", api_key=OPENAI_API_KEY, verbose=True)
        retriever.print_config()
    
    elif choice == "7":
        console.print("[green]Using MCTSRetriever (preset: accurate).[/green]")
        retriever = MCTSRetriever.from_preset("accurate", api_key=OPENAI_API_KEY, verbose=True)
        retriever.print_config()
    
    elif choice == "8":
        console.print("[green]Using MCTSRetriever (preset: explore).[/green]")
        retriever = MCTSRetriever.from_preset("explore", api_key=OPENAI_API_KEY, verbose=True)
        retriever.print_config()

    else:
        console.print("[green]Using SkeletonRetriever (default - summary mode).[/green]")
        retriever = SkeletonRetriever(llm_uri="openai/gpt-4o", api_key=OPENAI_API_KEY, verbose=True)
    
    return retriever, use_full_text


def _show_retrieved_nodes(nodes: list[TreeNode]) -> None:
    """Pretty-print retrieved nodes with truncated content."""

    console.print("\n[bold yellow]Retrieved nodes:[/bold yellow]\n")

    for idx, node in enumerate(nodes, 1):
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Node ID", f"[yellow]{node.node_id}[/yellow]")
        table.add_row("Level", f"[magenta]{node.level}[/magenta]")

        content = node.summary if node.summary else node.text
        truncated = _truncate_text(content, head_lines=3, tail_lines=2)

        panel = Panel(
            f"{table}\n\n[bold]Content:[/bold]\n{truncated}",
            title=f"[bold green]{idx}. {node.title}[/bold green]",
            border_style="green",
            expand=False,
        )
        console.print(panel)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    console.print("\n[bold cyan]==============================================[/bold cyan]")
    console.print("[bold cyan]        DocTree QA Pipeline (Markdown)[/bold cyan]")
    console.print("[bold cyan]==============================================[/bold cyan]\n")

    console.print("[bold yellow]Step 1:[/bold yellow] Input markdown file path")
    md_file_raw = console.input("[cyan]Enter markdown file path (drag & drop): [/cyan]")
    md_file = _clean_path(md_file_raw)

    if not os.path.isfile(md_file):
        console.print(f"[bold red]File not found:[/bold red] {md_file}")
        return

    if not md_file.lower().endswith(".md"):
        console.print(f"[bold red]Not a markdown file:[/bold red] {md_file}")
        return

    # Run the async main logic
    asyncio.run(_async_main(md_file))

    console.print("\n[bold green]DocTree QA test complete.[/bold green]")


async def _async_main(md_file: str) -> None:
    """Async main logic."""
    console.print("\n[bold yellow]Step 2:[/bold yellow] Build or load tree")
    tree = await get_or_build_tree(md_file)
    
    # Debug: Save tree to JSON
    import json
    from pathlib import Path
    debug_json = Path("tmp") / f"{Path(md_file).stem}_tree_debug.json"
    with open(debug_json, 'w', encoding='utf-8') as f:
        json.dump(tree.to_dict(), f, ensure_ascii=False, indent=2)
    console.print(f"[dim]Debug: Tree saved to {debug_json}[/dim]")

    console.print("\n[bold yellow]Step 3:[/bold yellow] Load DocTree")
    all_nodes = tree.collect_all_nodes()
    console.print(f"[green]Loaded tree:[/green] {tree.doc_name}")
    console.print(f"[green]Total nodes:[/green] {len(all_nodes)}")

    console.print("\n[bold yellow]Step 4:[/bold yellow] Choose retriever and answer generator")
    retriever, use_full_text = _choose_retriever()

    # AnswerGenerator uses API key from environment
    answer_generator = GeneralAnswerGenerator(llm_uri="openai/gpt-4o", api_key=OPENAI_API_KEY)

    console.print("\n[bold cyan]==============================================[/bold cyan]")
    console.print("[bold cyan]  Interactive retrieval and answer generation[/bold cyan]")
    console.print("[bold cyan]==============================================[/bold cyan]\n")

    # Run the interactive loop
    await _interactive_qa(tree, retriever, answer_generator, use_full_text)


async def _interactive_qa(
    tree: DocTree, 
    retriever, 
    answer_generator: GeneralAnswerGenerator, 
    use_full_text: bool
) -> None:
    """Async interactive QA loop."""
    while True:
        query = console.input("\n[bold cyan]Enter your question (or 'quit' to exit): [/bold cyan]").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        console.print("\n[yellow]Step 1: Retrieving relevant nodes...[/yellow]")
        t0 = time.time()

        try:
            if use_full_text and hasattr(retriever, 'search_full'):
                result = await retriever.search_full(query, tree)
            else:
                result = await retriever.search(query, tree)
        except Exception as e:
            console.print(f"[bold red]Retrieval failed: {e}[/bold red]")
            continue

        retrieve_time = time.time() - t0
        console.print(
            f"[green]Found {len(result.nodes)} relevant nodes[/green] "
            f"[dim]({retrieve_time:.2f}s)[/dim]"
        )
        console.print(f"[blue]Path:[/blue] {' -> '.join(result.path)}")

        console.print("\n[yellow]Step 2: Generating answer...[/yellow]")
        t1 = time.time()
        
        # Custom prompt for Chinese answers
        chinese_prompt = (
            "You are a question-answering assistant.\n\n"
            "Given the user's question and the following context sections, "
            "write a detailed answer in Chinese.\n\n"
            "Requirements:\n"
            "1. Base your answer strictly on the provided context. Do not fabricate facts.\n"
            "2. Provide a detailed and complete answer, using all relevant information.\n"
            "3. When you reference specific sections, annotate with node IDs like [0001].\n"
            "4. If the context is insufficient, explicitly state what information is missing.\n\n"
            "Question: {query}\n\n"
            "Context:\n{context}\n\n"
            "Answer in Chinese:"
        )
        
        try:
            answer = await answer_generator.generate(query, result.nodes, prompt_template=chinese_prompt)
        except Exception as e:
            console.print(f"[bold red]Answer generation failed: {e}[/bold red]")
            continue

        gen_time = time.time() - t1

        console.print(f"[green]Answer generated[/green] [dim]({gen_time:.2f}s)[/dim]")
        console.print(f"[dim]Total time: {retrieve_time + gen_time:.2f}s[/dim]")

        answer_panel = Panel(
            Markdown(answer.text),
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
        console.print("\n" + "=" * 60)
        console.print(answer_panel)

        if answer.citations:
            console.print(
                f"\n[bold blue]Citations:[/bold blue] {', '.join(answer.citations)}"
            )

        show_nodes = console.input("\n[dim]Show retrieved nodes? (y/n): [/dim]").strip().lower()
        if show_nodes == "y":
            _show_retrieved_nodes(result.nodes)


if __name__ == "__main__":
    main()
