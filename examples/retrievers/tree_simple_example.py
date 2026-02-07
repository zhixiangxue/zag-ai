#!/usr/bin/env python3
"""
SimpleRetriever Example - Simple LLM-based tree retrieval

Features:
- Interactive DocTree JSON file input
- Hierarchical retrieval using SimpleRetriever
- Display retrieval path and results

Prerequisites:
- OpenAI API Key (set via OPENAI_API_KEY environment variable)
- Prepare a DocTree JSON file (can be generated via MarkdownTreeReader)
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from zag.schemas import DocTree
from zag.retrievers.tree import SimpleRetriever


def clean_path(path_str: str) -> str:
    """Clean up file path - remove quotes and whitespace."""
    return path_str.strip().strip('"').strip("'")


def print_separator(char="=", length=70):
    """Print a separator line."""
    print(char * length)


def print_tree_info(tree: DocTree):
    """Print DocTree basic information."""
    print(f"\n📄 DocTree Info:")
    print(f"  • Document name: {tree.doc_name}")
    print(f"  • Root nodes: {len(tree.nodes)}")
    print(f"  • Total nodes: {len(tree.collect_all_nodes())}")
    
    # Show root nodes
    print(f"\n  Root node list:")
    for i, node in enumerate(tree.nodes, 1):
        print(f"    {i}. [{node.node_id}] {node.title} (level {node.level})")


def print_retrieval_results(result):
    """Print retrieval results in a formatted way."""
    print_separator()
    print("🎯 Retrieval Results")
    print_separator()
    
    print(f"\n✓ Found {len(result.nodes)} relevant nodes\n")
    
    for i, node in enumerate(result.nodes, 1):
        print(f"{i}. [{node.node_id}] {node.title}")
        print(f"   • Level: {node.level}")
        
        # Show content preview (summary or truncated text)
        content = node.summary if node.summary else node.text
        preview = content[:150].replace("\n", " ")
        if len(content) > 150:
            preview += "..."
        print(f"   • Content preview: {preview}")
        print()
    
    print(f"🛤️  Traversal path: {' -> '.join(result.path)}")


async def main():
    print_separator()
    print("SimpleRetriever Example")
    print_separator()
    
    # Load environment variables
    load_dotenv()
    
    # Step 1: Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ Error: OPENAI_API_KEY environment variable not found")
        print("Please set: export OPENAI_API_KEY='your-api-key'")
        return
    
    print("\n✓ OpenAI API Key loaded")
    
    # Step 2: Input JSON file path
    print("\n" + "=" * 70)
    print("Step 1: Input DocTree JSON file")
    print("=" * 70)
    print("\n💡 Tip: Drag & drop file to terminal, or enter path directly")
    
    json_path_raw = input("\n📁 DocTree JSON file path: ")
    json_path = clean_path(json_path_raw)
    
    if not Path(json_path).is_file():
        print(f"\n❌ File not found: {json_path}")
        return
    
    if not json_path.lower().endswith(".json"):
        print(f"\n❌ Not a JSON file: {json_path}")
        return
    
    print(f"\n✓ File found: {json_path}")
    
    # Step 3: Load DocTree
    print("\n" + "=" * 70)
    print("Step 2: Load DocTree")
    print("=" * 70)
    
    try:
        tree = DocTree.from_json(json_path)
        print("\n✓ DocTree loaded successfully")
        print_tree_info(tree)
    except Exception as e:
        print(f"\n❌ Loading failed: {e}")
        return
    
    # Step 4: Initialize SimpleRetriever
    print("\n" + "=" * 70)
    print("Step 3: Initialize SimpleRetriever")
    print("=" * 70)
    
    retriever = SimpleRetriever(
        llm_uri="openai/gpt-4o-mini",
        api_key=api_key,
        max_depth=5
    )
    
    print("\n✓ SimpleRetriever initialized")
    print(f"  • LLM: openai/gpt-4o-mini")
    print(f"  • Max depth: 5")
    
    # Step 5: Interactive query loop
    print("\n" + "=" * 70)
    print("Step 4: Start Retrieval")
    print("=" * 70)
    print("\n💡 Enter query, type 'quit' or 'exit' to exit\n")
    
    while True:
        query = input("🔍 Enter query: ").strip()
        
        if query.lower() in ["quit", "exit", "q"]:
            print("\n👋 Exiting")
            break
        
        if not query:
            print("⚠️  Query cannot be empty, please try again")
            continue
        
        print(f"\n⏳ Retrieving: '{query}'...\n")
        
        try:
            result = await retriever.search(query, tree)
            print_retrieval_results(result)
        except Exception as e:
            print(f"\n❌ Retrieval failed: {e}")
        
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
