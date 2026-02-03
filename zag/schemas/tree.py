"""
Tree schemas - hierarchical document structures used by tree-based retrievers.

This module defines:
    - TreeNode: basic tree node structure with optional summary and children
    - DocTree: container for a forest of TreeNode objects with helper methods

The design is intentionally minimal so it can be reused by different readers
(e.g. MarkdownTreeReader) and different retrieval strategies without coupling
to any specific storage or LLM implementation.
"""

from typing import List, Optional, Dict, Any, Iterable
import json

from pydantic import BaseModel, Field


class TreeNode(BaseModel):
    """Generic tree node used by DocTree.

    Attributes:
        title: Human readable title of this section/node.
        node_id: Stable identifier within the document tree.
        level: Logical depth (e.g. markdown header level).
        text: Raw text content associated with this node.
        summary: Optional summarized content for retrieval.
        children: Child nodes in the tree.
        extras: Optional extra metadata (line ranges, custom fields, etc.).
    """

    title: str
    node_id: str
    level: int
    text: str
    summary: Optional[str] = None
    children: List["TreeNode"] = Field(default_factory=list)
    extras: Dict[str, Any] = Field(default_factory=dict)


class TreeRetrievalResult(BaseModel):
    """Retrieval result for tree-based retrievers.

    Attributes:
        nodes: Relevant TreeNode objects.
        path: Sequence of node IDs visited during search.
    """

    nodes: List[TreeNode]
    path: List[str]


class DocTree:
    """Document tree wrapper.

    A DocTree represents a single logical document (or a small set of related
    documents) as a forest of TreeNode roots. It offers convenience methods
    that are useful for retrievers and downstream components.
    """

    def __init__(self, nodes: List[TreeNode], doc_name: str = "") -> None:
        self.nodes = nodes
        self.doc_name = doc_name

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocTree":
        """Create DocTree from a dictionary structure.

        Expected format:
            {"doc_name": str, "nodes": [TreeNode-like dict, ...]}
        """

        nodes_raw = data.get("nodes", []) or []
        nodes = [TreeNode.model_validate(node) for node in nodes_raw]
        doc_name = data.get("doc_name", "")
        return cls(nodes=nodes, doc_name=doc_name)

    @classmethod
    def from_json(cls, json_path: str) -> "DocTree":
        """Load DocTree from a JSON file produced by a reader."""

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert DocTree to a serializable dict."""

        return {
            "doc_name": self.doc_name,
            "nodes": [node.model_dump(exclude_none=True) for node in self.nodes],
        }

    def to_json(self, json_path: str) -> None:
        """Serialize DocTree to JSON file."""

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def collect_all_nodes(self) -> List[TreeNode]:
        """Flatten the tree into a list of all nodes.

        Nodes are returned in preorder traversal order.
        """

        result: List[TreeNode] = []

        def _traverse(nodes: Iterable[TreeNode]) -> None:
            for node in nodes:
                result.append(node)
                if node.children:
                    _traverse(node.children)

        _traverse(self.nodes)
        return result

    def iter_nodes(self) -> Iterable[TreeNode]:
        """Yield all nodes in preorder traversal order."""

        for node in self.collect_all_nodes():
            yield node

    def __len__(self) -> int:  # pragma: no cover - trivial wrapper
        return len(self.collect_all_nodes())

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"DocTree(doc_name={self.doc_name!r}, nodes={len(self.collect_all_nodes())})"
