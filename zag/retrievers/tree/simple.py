"""Simple LLM-based retriever over DocTree structures.

This retriever walks the document tree level by level and uses an LLM to
select relevant nodes at each level.
"""

import asyncio
from typing import List, Optional

from ...schemas import TreeNode, TreeRetrievalResult
import chak


class SimpleRetriever:
    """Simple LLM-based retrieval on a document tree.

    Algorithm:
        1. Start from root level nodes.
        2. Ask LLM which nodes are relevant to the query.
        3. Expand selected nodes to their children and repeat.
        4. Stop when reaching leaf nodes or max depth.
        5. Return all selected nodes and the traversal path.
    """

    def __init__(
        self,
        llm_uri: str = "openai/gpt-4o",
        api_key: Optional[str] = None,
        max_depth: int = 5,
    ) -> None:
        # chak is required; imported at module load time
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
        self.max_depth = max_depth

    async def retrieve(self, query: str, nodes: List[TreeNode]) -> TreeRetrievalResult:
        """Async retrieval implementation - traverse tree level by level using LLM."""

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
