#!/usr/bin/env python3
"""
SkeletonRetriever Search Example - Build tree from Markdown and search

Features:
- Interactive Markdown file input
- Build DocTree from Markdown using MarkdownTreeReader (with diskcache)
- One-shot skeleton-guided retrieval
- Support for summary mode (fast) and full-text mode (lossless)
- Cache management: auto-detect cached files, allow selection
- Generate answers with citations and view retrieved nodes

Prerequisites:
- OpenAI API Key (set via OPENAI_API_KEY environment variable)
- A Markdown file to search
"""

import asyncio
import os
import time
from pathlib import Path

import xxhash
from diskcache import Cache
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from zag.readers import MarkdownTreeReader
from zag.retrievers.tree import SkeletonRetriever
from zag.schemas import DocTree, TreeNode
from zag.generators import GeneralAnswerGenerator

console = Console()

# Cache directory
_CACHE_DIR = Path(os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp") / "zag_skeleton_cache"
cache = Cache(_CACHE_DIR)


def clean_path(path_str: str) -> str:
    """Clean up file path - remove quotes and whitespace."""
    path = path_str.strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path.startswith("'") and path.endswith("'"):
        path = path[1:-1]
    if path.startswith("& "):
        path = path[2:].strip()
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
    return path


def print_separator(char="=", length=70):
    """Print a separator line."""
    print(char * length)


def print_tree_info(tree):
    """Print DocTree basic information."""
    print(f"\n📄 DocTree Info:")
    print(f"  • Document name: {tree.doc_name}")
    print(f"  • Root nodes: {len(tree.nodes)}")
    print(f"  • Total nodes: {len(tree.collect_all_nodes())}")


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


def _compute_file_hash(file_path: str) -> str:
    """Compute xxhash64 for a file."""
    hasher = xxhash.xxh64()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _list_cached_files():
    """List all cached files with their metadata."""
    cached_files = []
    for key in cache:
        try:
            tree = cache.get(key)
            if isinstance(tree, DocTree):
                cached_files.append({
                    "hash": key,
                    "doc_name": tree.doc_name,
                    "nodes": len(tree.collect_all_nodes())
                })
        except Exception:
            continue
    return cached_files


def print_retrieval_results(result):
    """Print retrieval results in a formatted way."""
    print_separator()
    print("🎯 Retrieval Results")
    print_separator()
    
    print(f"\n✓ Found {len(result.nodes)} relevant nodes\n")
    
    for i, node in enumerate(result.nodes, 1):
        print(f"{i}. [{node.node_id}] {node.title}")
        print(f"   • Level: {node.level}")
        
        content = node.summary if node.summary else node.text
        preview = content[:150].replace("\n", " ")
        if len(content) > 150:
            preview += "..."
        print(f"   • Content preview: {preview}")
        print()
    
    print(f"🛤️  Selected node_ids: {' -> '.join(result.path)}")


def choose_mode():
    """Let user choose retrieval mode."""
    print("\n" + "=" * 70)
    print("Choose Retrieval Mode")
    print("=" * 70)
    
    print("\nAvailable modes:")
    print("  1. summary    - Use node summaries (fast, lossy)")
    print("  2. full       - Use full node text (slower, lossless)")
    
    while True:
        choice = input("\n👉 Choose mode (1-2, default 1): ").strip() or "1"
        
        if choice == "1":
            return "summary"
        elif choice == "2":
            return "full"
        
        print("⚠️  Invalid choice, please enter 1 or 2")


async def main():
    print_separator()
    print("SkeletonRetriever Search Example - Build & Search")
    print_separator()
    
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ Error: OPENAI_API_KEY environment variable not found")
        return
    
    print("\n✓ OpenAI API Key loaded")
    print(f"📦 Cache directory: {_CACHE_DIR}")
    
    # Check for cached files first
    cached_files = _list_cached_files()
    tree = None
    md_path = None
    
    if cached_files:
        print("\n" + "=" * 70)
        print("💾 Found cached DocTrees:")
        print("=" * 70)
        print("\n  0. 📝 Load a new file (skip cache)")
        for i, cached in enumerate(cached_files, 1):
            print(f"  {i}. 📄 {cached['doc_name']} ({cached['nodes']} nodes)")
        
        while True:
            choice = input("\n👉 Select option (0-{}): ".format(len(cached_files))).strip()
            if choice.isdigit() and 0 <= int(choice) <= len(cached_files):
                choice = int(choice)
                break
            print("⚠️  Invalid choice")
        
        if choice > 0:
            # Use cached tree
            cached = cached_files[choice - 1]
            tree = cache.get(cached["hash"])
            print(f"\n✓ Loaded cached tree: {tree.doc_name}")
            print_tree_info(tree)
    
    if tree is None:
        # Step 1: Input Markdown file
        print("\n" + "=" * 70)
        print("Step 1: Input Markdown file")
        print("=" * 70)
        print("\n💡 Tip: Drag & drop file to terminal, or enter path directly")
        
        md_path_raw = input("\n📁 Markdown file path: ")
        md_path = clean_path(md_path_raw)
        
        if not Path(md_path).is_file():
            print(f"\n❌ File not found: {md_path}")
            return
        
        if not md_path.lower().endswith(".md"):
            print(f"\n❌ Not a Markdown file: {md_path}")
            return
        
        print(f"\n✓ File found: {md_path}")
        
        # Check if already cached
        file_hash = _compute_file_hash(md_path)
        cached_tree = cache.get(file_hash)
        
        if cached_tree is not None:
            print(f"\n💾 Found existing cache for this file!")
            use_cache = input("👉 Use cached tree? (y/n, default: y): ").strip().lower() != "n"
            if use_cache:
                tree = cached_tree
                print(f"\n✓ Loaded cached tree: {tree.doc_name}")
                print_tree_info(tree)
        
        if tree is None:
            # Step 2: Build DocTree
            print("\n" + "=" * 70)
            print("Step 2: Build DocTree from Markdown")
            print("=" * 70)
            
            reader = MarkdownTreeReader(llm_uri="openai/gpt-4o", api_key=api_key)
            tree = await reader.read(path=md_path, generate_summaries=True)
            
            # Save to cache
            cache.set(file_hash, tree)
            print("\n✓ DocTree built and cached successfully")
            print_tree_info(tree)
    
    # Step 3: Configure retriever
    print("\n" + "=" * 70)
    print("Step 3: Configure SkeletonRetriever")
    print("=" * 70)
    
    mode = choose_mode()
    
    retriever = SkeletonRetriever(
        llm_uri="openai/gpt-4o-mini",
        api_key=api_key,
        verbose=True
    )
    
    print("\n✓ SkeletonRetriever initialized")
    print(f"  • LLM: openai/gpt-4o-mini")
    print(f"  • Mode: {mode}")
    
    # Step 4: Initialize answer generator
    print("\n" + "=" * 70)
    print("Step 4: Initialize Answer Generator")
    print("=" * 70)
    
    answer_generator = GeneralAnswerGenerator(llm_uri="openai/gpt-4o", api_key=api_key)
    print("\n✓ Answer generator initialized")
    print("  • LLM: openai/gpt-4o")
    
    # Step 5: Interactive query loop
    print("\n" + "=" * 70)
    print("Step 5: Interactive Retrieval & Answer")
    print("=" * 70)
    print("\n💡 Enter query, type 'quit' or 'exit' to exit")
    print("💡 Type 'mode' to switch between summary/full mode\n")
    
    while True:
        query = console.input("[bold cyan]Enter your question (or 'quit' to exit): [/bold cyan]").strip()
        
        if query.lower() in ["quit", "exit", "q"]:
            print("\n👋 Exiting")
            break
        
        if query.lower() == "mode":
            mode = choose_mode()
            print(f"\n✓ Mode switched to: {mode}\n")
            continue
        
        if not query:
            continue
        
        console.print("\n[yellow]Step 1: Retrieving relevant nodes...[/yellow]")
        t0 = time.time()
        
        try:
            if mode == "full" and hasattr(retriever, 'search_full'):
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
        
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
