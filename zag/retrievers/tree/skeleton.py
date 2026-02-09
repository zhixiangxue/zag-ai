"""Skeleton-guided retriever for DocTree structures.

This retriever shows the entire tree skeleton to an LLM and asks it to directly
identify relevant node_ids, eliminating path-dependent exploration limitations.
"""

import re
from typing import List, Union, Optional

from ...schemas import TreeNode, TreeRetrievalResult, DocTree, LODLevel
from ...storages.vector import BaseVectorStore
import chak


class SkeletonRetriever:
    """Skeleton-guided tree retriever using LLM global decision.
    
    Strategy:
        1. Build compact tree skeleton (node_id, level, title, summary)
        2. Show full skeleton to LLM in one prompt
        3. LLM directly identifies relevant node_ids
        4. Return full content of selected nodes
    
    Advantages:
        - Zero path dependency: LLM sees complete structure
        - One-shot decision: only 1 LLM call for retrieval
        - Precise targeting: direct node_id selection
        - Cost-effective: cheaper than MCTS iterations
    """

    def __init__(
        self,
        llm_uri: str = "openai/gpt-4o",
        api_key: str = None,
        verbose: bool = False,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        """Initialize SkeletonRetriever.
        
        Args:
            llm_uri: LLM model URI for decision making
            api_key: API key for LLM
            verbose: Enable verbose output
            vector_store: Optional vector store for retrieve() method
        """
        self._api_key = api_key
        self.llm_uri = llm_uri
        self.verbose = verbose
        self.vector_store = vector_store

    async def search(
        self, 
        query: str, 
        tree: Union[DocTree, List[TreeNode]], 
        top_k: int = 8
    ) -> TreeRetrievalResult:
        """Search DocTree using skeleton with summaries (fast, lossy).
        
        Args:
            query: Query text
            tree: DocTree object or list of TreeNode (root level nodes)
            top_k: Maximum number of nodes to return
            
        Returns:
            Tree retrieval result with relevant nodes and path
        """
        return await self._do_search(query, tree, use_full_text=False, top_k=top_k)

    async def search_full(
        self, 
        query: str, 
        tree: Union[DocTree, List[TreeNode]], 
        top_k: int = 8
    ) -> TreeRetrievalResult:
        """Search DocTree using full node text (slower, lossless).
        
        Args:
            query: Query text
            tree: DocTree object or list of TreeNode (root level nodes)
            top_k: Maximum number of nodes to return
            
        Returns:
            Tree retrieval result with relevant nodes and path
        """
        return await self._do_search(query, tree, use_full_text=True, top_k=top_k)

    async def retrieve(
        self,
        query: str,
        unit_id: str,
        top_k: int = 8,
        use_full_text: bool = False
    ) -> TreeRetrievalResult:
        """Retrieve from unit ID (primary API for production use).

        This method:
        1. Gets Unit from vector store by ID
        2. Extracts HIGH view (tree structure)
        3. Parses DocTree from view content
        4. Delegates to search() or search_full() for actual retrieval

        Args:
            query: Query text
            unit_id: Unit ID containing tree structure in HIGH view
            top_k: Maximum number of nodes to return
            use_full_text: If True, use full node text; if False, use summary

        Returns:
            Tree retrieval result with relevant nodes and path

        Raises:
            ValueError: If vector_store is not provided or unit has no HIGH view
        """
        if not self.vector_store:
            raise ValueError("vector_store is required for retrieve() method")

        # Get unit from vector store
        units = self.vector_store.get([unit_id])
        if not units:
            raise ValueError(f"Unit not found: {unit_id}")
        unit = units[0]

        # Extract HIGH view
        high_view = unit.get_view(LODLevel.HIGH)
        if not high_view:
            raise ValueError(f"Unit {unit_id} has no HIGH view (tree structure)")

        # Parse DocTree from view content
        tree = DocTree.from_dict(high_view)

        # Delegate to search or search_full based on use_full_text
        if use_full_text:
            return await self.search_full(query, tree, top_k=top_k)
        return await self.search(query, tree, top_k=top_k)

    async def _do_search(
        self, 
        query: str, 
        tree: Union[DocTree, List[TreeNode]], 
        use_full_text: bool,
        top_k: int
    ) -> TreeRetrievalResult:
        """Internal search implementation.
        
        Args:
            query: Query text
            tree: DocTree object or list of TreeNode (root level nodes)
            use_full_text: If True, use node.text instead of summary
            top_k: Maximum number of nodes to return
            
        Returns:
            Tree retrieval result with relevant nodes and path
        """
        # Parse input to extract nodes
        if isinstance(tree, DocTree):
            nodes = tree.nodes
        elif isinstance(tree, list):
            nodes = tree
        else:
            raise TypeError(f"tree must be DocTree or List[TreeNode], got {type(tree)}")
        
        # Step 1: Build compact skeleton
        skeleton = self._build_skeleton(nodes, use_full_text=use_full_text)
        
        if self.verbose:
            print(f"[Skeleton] Built skeleton: {skeleton.count(chr(10))} lines")
        
        # Step 2: Ask LLM to identify relevant node_ids (fresh conversation each time)
        prompt = self._build_prompt(query, skeleton, top_k)
        
        if self.verbose:
            print(f"[Skeleton] Asking LLM to identify relevant nodes...")
        
        conv = chak.Conversation(self.llm_uri, api_key=self._api_key)
        response = await conv.asend(prompt)
        selected_ids = self._parse_node_ids(response.content)
        
        if self.verbose:
            print(f"[Skeleton] LLM selected node_ids: {selected_ids}")
        
        # Step 3: Retrieve full content of selected nodes
        all_nodes = self._collect_all_nodes(nodes)
        node_map = {n.node_id: n for n in all_nodes}
        
        result_nodes = []
        for nid in selected_ids[:top_k]:
            if nid in node_map:
                result_nodes.append(node_map[nid])
        
        # Build path from node_ids
        path = [n.node_id for n in result_nodes]
        
        if self.verbose:
            print(f"[Skeleton] Retrieved {len(result_nodes)} nodes")
        
        return TreeRetrievalResult(nodes=result_nodes, path=path)

    def _build_skeleton(self, nodes: List[TreeNode], indent: int = 0, use_full_text: bool = False) -> str:
        """Build compact tree skeleton representation.
        
        Format:
            [node_id] # Title
                Summary: first N characters...
                [child_id] ## Child Title
                    Summary: ...
        
        Args:
            nodes: List of TreeNode to process
            indent: Current indentation level
            use_full_text: If True, use node.text instead of summary (lossless but larger)
        """
        lines = []
        
        for node in nodes:
            prefix = "  " * indent
            # Title line with level indicator
            level_marker = "#" * node.level
            lines.append(f"{prefix}[{node.node_id}] {level_marker} {node.title}")
            
            # Content line (summary or full text)
            if use_full_text:
                # Use full text without truncation (lossless)
                content = (node.text or "").replace("\n", " ")
            else:
                # Use summary (already compressed)
                content = (node.summary or node.text or "").replace("\n", " ")
            
            if content:
                label = "Content" if use_full_text else "Summary"
                lines.append(f"{prefix}    {label}: {content}...")
            
            # Recursively add children
            if node.children:
                child_skeleton = self._build_skeleton(node.children, indent + 1, use_full_text=use_full_text)
                lines.append(child_skeleton)
        
        return "\n".join(lines)

    def _build_prompt(self, query: str, skeleton: str, top_k: int) -> str:
        """Build LLM prompt with full skeleton."""
        return f"""You are analyzing a document structure to answer a question.

Question: {query}

Below is the complete document skeleton showing all sections and subsections.
Each node is formatted as: [node_id] # Title, followed by a summary.

Document Structure:
{skeleton}

Task: Identify which node_ids contain information most relevant to answer the question.
Reply ONLY with a comma-separated list of node_ids (e.g., "0091,0092,0093").
Select the {top_k} most relevant nodes. Prefer deeper/more specific sections over high-level ones.

Node IDs:"""

    def _parse_node_ids(self, text: str) -> List[str]:
        """Parse node_ids like '0091,0092,0093' from LLM response.
        
        Extracts 4-digit node_ids using regex.
        """
        # Extract all 4-digit node_ids
        node_ids = re.findall(r'\b\d{4}\b', text)
        return node_ids

    def _collect_all_nodes(self, nodes: List[TreeNode]) -> List[TreeNode]:
        """Flatten tree to list of all nodes."""
        result: List[TreeNode] = []

        def _traverse(node_list: List[TreeNode]) -> None:
            for node in node_list:
                result.append(node)
                if node.children:
                    _traverse(node.children)

        _traverse(nodes)
        return result
