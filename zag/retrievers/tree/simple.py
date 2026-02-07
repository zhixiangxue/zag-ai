"""Simple LLM-based retriever over DocTree structures.

This retriever walks the document tree level by level and uses an LLM to
select relevant nodes at each level.
"""

import asyncio
from typing import List, Optional, Union

from ...schemas import TreeNode, TreeRetrievalResult, DocTree, LODLevel
from ...storages.vector.base import BaseVectorStore
import chak


class SimpleRetriever:
    """Simple LLM-based retrieval on a document tree.

    Algorithm:
        1. Start from root level nodes.
        2. Ask LLM which nodes are relevant to the query.
        3. Expand selected nodes to their children and repeat.
        4. Stop when reaching leaf nodes or max depth.
        5. Return all selected nodes and the traversal path.
    
    Two usage patterns:
        1. retrieve(query, unit_id): Production API - retrieve from unit ID
        2. search(query, tree): Examples/tests - search in DocTree directly
    """

    def __init__(
        self,
        llm_uri: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        max_depth: int = 5,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        """Initialize SimpleRetriever.
        
        Args:
            llm_uri: LLM model URI
            api_key: API key for LLM
            max_depth: Maximum tree depth to traverse
            vector_store: Vector store for retrieve() method (optional, only needed for production API)
        """
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
        self.max_depth = max_depth
        self.vector_store = vector_store

    async def retrieve(self, query: str, unit_id: str) -> TreeRetrievalResult:
        """Retrieve from unit ID (primary API for production use).
        
        This method:
        1. Gets Unit from vector store by ID
        2. Extracts HIGH view (tree structure)
        3. Parses DocTree from view content
        4. Delegates to search() for actual retrieval
        
        Args:
            query: Query text
            unit_id: Unit ID containing tree structure in HIGH view
            
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
        
        # Delegate to search
        return await self.search(query, tree)

    async def search(self, query: str, tree: Union[DocTree, List[TreeNode]]) -> TreeRetrievalResult:
        """Search in DocTree or nodes directly (for examples/tests).
        
        This method accepts DocTree or node list and performs tree search.
        Useful for examples where tree is already loaded from JSON.
        
        Args:
            query: Query text
            tree: DocTree object or list of TreeNode (root level nodes)
            
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
        
        # Delegate to internal search implementation
        return await self._do_search(query, nodes)

    async def _do_search(self, query: str, nodes: List[TreeNode]) -> TreeRetrievalResult:
        """Internal: execute tree search algorithm."""
        path: List[str] = []
        relevant_nodes: List[TreeNode] = []

        current_nodes: List[TreeNode] = nodes
        depth = 0

        while current_nodes and depth < self.max_depth:
            selected = await self._select_nodes(query, current_nodes)
            if not selected:
                break

            for node in selected:
                path.append(node.node_id)
                relevant_nodes.append(node)

            next_level: List[TreeNode] = []
            for node in selected:
                next_level.extend(node.children)

            current_nodes = next_level
            depth += 1

        return TreeRetrievalResult(nodes=relevant_nodes, path=path)

    async def _select_nodes(self, query: str, nodes: List[TreeNode]) -> List[TreeNode]:
        """Ask LLM to select relevant nodes from the current level."""

        if not nodes:
            return []

        node_list: List[str] = []
        for node in nodes:
            text = node.summary if node.summary else node.text[:200]
            node_list.append(f"[{node.node_id}] {node.title}: {text}")

        prompt = (
            f"Given the query: \"{query}\"\n\n"
            "Select the most relevant sections from the following list. "
            "Return ONLY the node IDs (e.g., 0001, 0002), separated by commas.\n\n"
            "Available sections:\n" + "\n".join(node_list) + "\n\n"
            "Relevant node IDs:"
        )

        response = await self._conv.asend(prompt)
        selected_ids = [s.strip() for s in response.content.split(",")]

        id_to_node = {node.node_id: node for node in nodes}
        return [id_to_node[node_id] for node_id in selected_ids if node_id in id_to_node]
