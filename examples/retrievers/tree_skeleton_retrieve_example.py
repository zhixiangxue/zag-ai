#!/usr/bin/env python3
"""
SkeletonRetriever Retrieve Example - Retrieve from vector store by unit_id

Features:
- Interactive unit_id input
- Retrieve from Qdrant vector store
- One-shot skeleton-guided retrieval from stored DocTree
- Support for summary mode (fast) and full-text mode (lossless)

Prerequisites:
- OpenAI API Key (set via OPENAI_API_KEY environment variable)
- Qdrant vector store running with indexed LOD units
- A valid unit_id with HIGH view (tree structure)
"""

import asyncio
import os

from dotenv import load_dotenv

from zag.retrievers.tree import SkeletonRetriever
from zag.storages.vector import QdrantVectorStore
from zag.embedders import Embedder


def print_separator(char="=", length=70):
    """Print a separator line."""
    print(char * length)


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
    print("SkeletonRetriever Retrieve Example - Vector Store")
    print_separator()
    
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ Error: OPENAI_API_KEY environment variable not found")
        return
    
    print("\n✓ OpenAI API Key loaded")
    
    # Step 1: Configure vector store
    print("\n" + "=" * 70)
    print("Step 1: Configure Vector Store")
    print("=" * 70)
    
    qdrant_host = input("\n📍 Qdrant host (default: localhost): ").strip() or "localhost"
    qdrant_port = input("🔌 Qdrant port (default: 16333): ").strip() or "16333"
    collection_name = input("📁 Collection name (default: documents): ").strip() or "documents"
    
    print(f"\n⏳ Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    
    embedder = Embedder("openai/text-embedding-3-small", api_key=api_key)
    vector_store = QdrantVectorStore.server(
        host=qdrant_host,
        port=int(qdrant_port),
        collection_name=collection_name,
        embedder=embedder
    )
    
    print("✓ Connected to vector store")
    
    # Step 2: Initialize retriever
    print("\n" + "=" * 70)
    print("Step 2: Initialize SkeletonRetriever")
    print("=" * 70)
    
    mode = choose_mode()
    
    retriever = SkeletonRetriever(
        llm_uri="openai/gpt-4o-mini",
        api_key=api_key,
        verbose=True,
        vector_store=vector_store
    )
    
    print("\n✓ SkeletonRetriever initialized")
    print(f"  • LLM: openai/gpt-4o-mini")
    print(f"  • Mode: {mode}")
    print(f"  • Vector store: {collection_name}")
    
    # Step 3: Interactive query loop
    print("\n" + "=" * 70)
    print("Step 3: Start Retrieval")
    print("=" * 70)
    print("\n💡 Enter unit_id and query")
    print("💡 Type 'quit' or 'exit' to exit")
    print("💡 Type 'mode' to switch between summary/full mode\n")
    
    while True:
        unit_id = input("📄 Enter unit_id: ").strip()
        
        if unit_id.lower() in ["quit", "exit", "q"]:
            print("\n👋 Exiting")
            break
        
        if unit_id.lower() == "mode":
            mode = choose_mode()
            print(f"\n✓ Mode switched to: {mode}\n")
            continue
        
        if not unit_id:
            print("⚠️  unit_id cannot be empty, please try again")
            continue
        
        query = input("🔍 Enter query: ").strip()
        
        if not query:
            print("⚠️  Query cannot be empty, please try again")
            continue
        
        print(f"\n⏳ Retrieving from unit '{unit_id}': '{query}'...")
        
        try:
            if mode == "summary":
                result = await retriever.retrieve(query, unit_id)
            else:
                # retrieve() uses search() internally, need to get tree first
                from zag.schemas import LODLevel
                units = vector_store.get([unit_id])
                if not units:
                    print(f"\n❌ Unit not found: {unit_id}")
                    continue
                unit = units[0]
                high_view = unit.get_view(LODLevel.HIGH)
                if not high_view:
                    print(f"\n❌ Unit {unit_id} has no HIGH view")
                    continue
                from zag.schemas import DocTree
                tree = DocTree.from_dict(high_view)
                result = await retriever.search_full(query, tree)
            
            print_retrieval_results(result)
        except Exception as e:
            print(f"\n❌ Retrieval failed: {e}")
        
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
