"""Monte Carlo Tree Search (MCTS) retriever over DocTree structures.

This retriever performs a guided search over the document tree using MCTS
and an LLM-based relevance scorer.
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

from ...schemas import TreeNode, TreeRetrievalResult, DocTree, LODLevel
from ...storages.vector.base import BaseVectorStore
import chak

@dataclass
class MCTSNode:
    """Search node for MCTS (separate from document TreeNode).

    Tracks statistics used by the MCTS algorithm:
        - visits: how many times this node was visited
        - value: accumulated value from simulations
        - relevance_score: LLM relevance score in [0.0, 1.0]
    """

    tree_node: Optional[TreeNode]
    parent: Optional["MCTSNode"]
    children: List["MCTSNode"] = field(default_factory=list)

    visits: int = 0
    value: float = 0.0
    relevance_score: float = 0.0


class MCTSRetriever:
    """Monte Carlo Tree Search retriever for document trees.

    High-level algorithm:
        1. Selection: navigate the search tree using UCB1.
        2. Expansion: expand children of selected nodes.
        3. Simulation: evaluate node value using LLM relevance.
        4. Backpropagation: update statistics along the path.

    Presets:
        - fast: low-cost quick checks
        - balanced: default for most use cases
        - accurate: higher cost, higher precision
        - explore: focus on recall, exploring more of the tree
    """

    PRESETS = {
        "fast": {
            "llm_uri": "openai/gpt-4o-mini",
            "iterations": 10,
            "exploration_c": 1.4,
            "top_k": 3,
            "description": "Fast preset - low cost, quick validation",
        },
        "balanced": {
            "llm_uri": "openai/gpt-4o-mini",
            "iterations": 30,
            "exploration_c": 1.2,
            "top_k": 5,
            "description": "Balanced preset - trade-off between quality and cost",
        },
        "accurate": {
            "llm_uri": "openai/gpt-4o",
            "iterations": 50,
            "exploration_c": 1.0,
            "top_k": 8,
            "description": "Accurate preset - higher precision for critical documents",
        },
        "explore": {
            "llm_uri": "openai/gpt-4o-mini",
            "iterations": 50,
            "exploration_c": 2.0,
            "top_k": 10,
            "description": "Explore preset - maximize recall and coverage",
        },
    }

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        verbose: bool = False,
        api_key: Optional[str] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> "MCTSRetriever":
        """Create retriever from a named preset configuration."""

        if preset_name not in cls.PRESETS:
            available = ", ".join(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

        config = cls.PRESETS[preset_name].copy()
        description = config.pop("description")

        instance = cls(api_key=api_key, verbose=verbose, vector_store=vector_store, **config)
        instance.preset_name = preset_name
        instance.preset_description = description
        return instance

    def __init__(
        self,
        llm_uri: str = "openai/gpt-4o",
        iterations: int = 50,
        exploration_c: float = 1.4,
        top_k: int = 5,
        verbose: bool = False,
        api_key: Optional[str] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        """Initialize MCTSRetriever.
        
        Args:
            llm_uri: LLM model URI
            iterations: Number of MCTS iterations
            exploration_c: UCB1 exploration constant
            top_k: Number of top nodes to return
            verbose: Enable verbose output
            api_key: API key for LLM
            vector_store: Vector store for retrieve() method (optional)
        """
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
        self.llm_uri = llm_uri
        self.iterations = iterations
        self.exploration_c = exploration_c
        self.top_k = top_k
        self.verbose = verbose
        self.vector_store = vector_store

        self.preset_name: Optional[str] = None
        self.preset_description: Optional[str] = None

        self.stats = {
            "init_time": 0.0,
            "selection_time": 0.0,
            "expansion_time": 0.0,
            "simulation_time": 0.0,
            "backprop_time": 0.0,
            "llm_calls": 0,
            "llm_time": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
        
        This method accepts DocTree or node list and performs MCTS search.
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
        """Internal: run the MCTS search process over the document tree."""

        for key in self.stats:
            self.stats[key] = 0 if key != "llm_calls" else 0

        virtual_root = MCTSNode(tree_node=None, parent=None)

        # Initialize root level nodes (evaluate relevance in parallel)
        t0 = time.time()
        tasks = [self._evaluate_relevance(query, node) for node in nodes]
        relevance_scores = await asyncio.gather(*tasks)

        for node, relevance in zip(nodes, relevance_scores):
            mcts_node = MCTSNode(tree_node=node, parent=virtual_root, relevance_score=relevance)
            virtual_root.children.append(mcts_node)

        self.stats["init_time"] = time.time() - t0

        if self.verbose:
            print(f"[MCTS] Init: {len(nodes)} root nodes evaluated in {self.stats['init_time']:.2f}s (parallel)")

        # MCTS iterations
        for i in range(self.iterations):
            # 1. Selection
            t0 = time.time()
            selected = self._select(virtual_root)
            self.stats["selection_time"] += time.time() - t0

            # 2. Expansion
            t0 = time.time()
            if selected.visits > 0 and selected.tree_node and selected.tree_node.children:
                await self._expand(selected, query)
                if selected.children:
                    selected = selected.children[0]
            self.stats["expansion_time"] += time.time() - t0

            # 3. Simulation
            t0 = time.time()
            value = self._simulate(selected)
            self.stats["simulation_time"] += time.time() - t0

            # 4. Backpropagation
            t0 = time.time()
            self._backpropagate(selected, value)
            self.stats["backprop_time"] += time.time() - t0

            if self.verbose and (i + 1) % 10 == 0:
                print(f"[MCTS] Iteration {i + 1}/{self.iterations}, LLM calls: {self.stats['llm_calls']}")

        best_nodes = self._collect_best_nodes(virtual_root)
        path = [node.tree_node.node_id for node in best_nodes if node.tree_node]
        tree_nodes = [node.tree_node for node in best_nodes if node.tree_node]

        if self.verbose:
            self._print_stats()

        return TreeRetrievalResult(nodes=tree_nodes, path=path)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def print_config(self) -> None:
        """Pretty-print configuration using rich if available."""

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            console = Console()

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Parameter", style="yellow", width=20)
            table.add_column("Value", style="green")
            table.add_column("Description", style="dim")

            table.add_row("llm_uri", self.llm_uri, "LLM model for relevance evaluation")
            table.add_row("iterations", str(self.iterations), "Number of MCTS iterations")
            table.add_row("exploration_c", f"{self.exploration_c:.1f}", "UCB1 exploration constant")
            table.add_row("top_k", str(self.top_k), "Number of nodes to return")

            if self.preset_name:
                title = (
                    "[bold cyan]MCTS Retriever Config[/bold cyan] "
                    f"[dim](preset: {self.preset_name})[/dim]"
                )
                subtitle = f"[italic]{self.preset_description}[/italic]" if self.preset_description else None
            else:
                title = "[bold cyan]MCTS Retriever Config[/bold cyan] [dim](custom)[/dim]"
                subtitle = None

            panel = Panel(table, title=title, subtitle=subtitle, border_style="cyan")
            console.print(panel)

        except ImportError:  # pragma: no cover - optional pretty printing
            print("[MCTS Config]")
            if self.preset_name:
                print(f"  Preset: {self.preset_name}")
                if self.preset_description:
                    print(f"  Description: {self.preset_description}")
            print(f"  llm_uri: {self.llm_uri}")
            print(f"  iterations: {self.iterations}")
            print(f"  exploration_c: {self.exploration_c}")
            print(f"  top_k: {self.top_k}")

    def _print_stats(self) -> None:
        """Print timing and token usage statistics."""

        total = (
            self.stats["init_time"]
            + self.stats["selection_time"]
            + self.stats["expansion_time"]
            + self.stats["simulation_time"]
            + self.stats["backprop_time"]
        )
        if total <= 0:
            return

        conv_stats = self._conv.stats() if hasattr(self._conv, "stats") else {}

        print("\n[MCTS Timing]")
        print(
            f"  Init (eval roots):   {self.stats['init_time']:.2f}s "
            f"({self.stats['init_time'] / total * 100:.1f}%)"
        )
        print(
            f"  Selection (UCB1):    {self.stats['selection_time']:.2f}s "
            f"({self.stats['selection_time'] / total * 100:.1f}%)"
        )
        print(
            f"  Expansion (LLM):     {self.stats['expansion_time']:.2f}s "
            f"({self.stats['expansion_time'] / total * 100:.1f}%)"
        )
        print(
            f"  Simulation:          {self.stats['simulation_time']:.2f}s "
            f"({self.stats['simulation_time'] / total * 100:.1f}%)"
        )
        print(
            f"  Backpropagation:     {self.stats['backprop_time']:.2f}s "
            f"({self.stats['backprop_time'] / total * 100:.1f}%)"
        )
        print("  ------------------------------------------")
        print(f"  Total:               {total:.2f}s")
        print(f"  LLM calls:           {self.stats['llm_calls']}")
        print(
            f"  LLM time:            {self.stats['llm_time']:.2f}s "
            f"({self.stats['llm_time'] / total * 100:.1f}%)"
        )
        if self.stats["llm_calls"] > 0:
            print(f"  Avg per LLM call:    {self.stats['llm_time'] / self.stats['llm_calls']:.2f}s")

        # Print conversation stats directly from chak
        conv_stats = self._conv.stats() if hasattr(self._conv, "stats") else {}
        if conv_stats:
            print(f"\n[Conversation Stats]")
            for key, value in conv_stats.items():
                print(f"  {key}: {value}")

    # ------------------------------------------------------------------
    # Core MCTS steps
    # ------------------------------------------------------------------
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection step: traverse the tree using UCB1."""

        while node.children:
            node = max(node.children, key=self._ucb1)
        return node

    def _ucb1(self, node: MCTSNode) -> float:
        """UCB1 scoring function used in selection.

        UCB = avg_value + C * sqrt(ln(parent_visits) / node_visits)
        """

        if node.visits == 0:
            return float("inf")

        if not node.parent or node.parent.visits == 0:
            return node.value / node.visits

        avg_value = node.value / node.visits
        exploration = self.exploration_c * math.sqrt(math.log(node.parent.visits) / node.visits)
        return avg_value + exploration

    async def _expand(self, node: MCTSNode, query: str) -> None:
        """Expansion step: create MCTS nodes for children of a tree node."""

        if not node.tree_node or not node.tree_node.children:
            return

        tasks = [self._evaluate_relevance(query, child) for child in node.tree_node.children]
        relevance_scores = await asyncio.gather(*tasks)

        for tree_child, relevance in zip(node.tree_node.children, relevance_scores):
            child = MCTSNode(tree_node=tree_child, parent=node, relevance_score=relevance)
            node.children.append(child)

    def _simulate(self, node: MCTSNode) -> float:
        """Simulation step: compute value for a node.

        In this simplified version, we directly use the node's relevance score.
        """

        return node.relevance_score

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagation step: update statistics along the path."""

        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    async def _evaluate_relevance(self, query: str, tree_node: TreeNode) -> float:
        """Use LLM to evaluate node relevance in [0.0, 1.0]."""

        text = tree_node.summary if tree_node.summary else tree_node.text[:300]

        prompt = (
            "Rate the relevance of this section to the query on a scale of 0.0 to 1.0.\n\n"
            f"Query: {query}\n\n"
            f"Section: [{tree_node.node_id}] {tree_node.title}\n{text}\n\n"
            "Return ONLY a single number between 0.0 and 1.0, no explanation.\n"
            "Relevance score:"
        )

        t0 = time.time()
        response = await self._conv.asend(prompt)
        llm_time = time.time() - t0

        self.stats["llm_calls"] += 1
        self.stats["llm_time"] += llm_time

        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            return 0.0

    def _collect_best_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        """Collect top-k nodes with highest average value."""

        all_nodes: List[MCTSNode] = []

        def _collect(node: MCTSNode) -> None:
            if node.tree_node is not None:
                all_nodes.append(node)
            for child in node.children:
                _collect(child)

        _collect(root)

        all_nodes.sort(
            key=lambda n: n.value / n.visits if n.visits > 0 else 0.0,
            reverse=True,
        )
        return all_nodes[: self.top_k]
