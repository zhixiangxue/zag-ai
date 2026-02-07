"""
MarkdownTreeReader - build DocTree-compatible structures from Markdown files.

This reader parses a Markdown document into a hierarchical tree structure
(`TreeNode` + `DocTree` schemas) and optionally generates summaries for each
node using an LLM.

The output format is a plain dict compatible with `zag.schemas.DocTree`:
    {"doc_name": str, "nodes": [TreeNode-like dict, ...]}
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import List, Dict, Any

from ..schemas import DocTree, TreeNode

import chak
import tiktoken


class MarkdownTreeReader:
    """Read a markdown file and build a hierarchical tree structure.

    Responsibilities:
        1. Extract headers from markdown (#, ##, ###, ...)
        2. Assign text ranges to each header
        3. Build a nested tree of `TreeNode` objects
        4. Optionally generate summaries using an LLM
        5. Return a DocTree-compatible dict structure
    """

    def __init__(
        self,
        llm_uri: str,
        api_key: str,
        summary_threshold: int = 200,
    ) -> None:
        self.llm_uri = llm_uri
        self.api_key = api_key
        self.summary_threshold = summary_threshold

        # Initialize tokenizer based on llm_uri
        try:
            model_name = self.llm_uri.split("/", 1)[1] if "/" in self.llm_uri else self.llm_uri
            self._tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception:  # pragma: no cover - tokenizer is optional
            # Fallback to a default tokenizer model if available
            try:
                self._tokenizer = tiktoken.encoding_for_model("gpt-4o")
            except Exception:
                self._tokenizer = None

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def extract_headers(self) -> List[Dict[str, Any]]:
        """Extract markdown headers (#, ##, ###, ...) from the document.

        Code blocks (``` fenced blocks) are skipped.
        """

        headers: List[Dict[str, Any]] = []
        in_code_block = False

        for idx, line in enumerate(self.lines):
            stripped = line.strip()

            # Toggle code block flag
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block or not stripped:
                continue

            match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headers.append({
                    "title": title,
                    "level": level,
                    "line_num": idx,
                })

        return headers

    def assign_text_ranges(self, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign text content range to each header.

        For each header, we capture all lines from its own line up to (but not
        including) the next header.
        """

        total_lines = len(self.lines)

        for i, header in enumerate(headers):
            start_line = header["line_num"]
            end_line = headers[i + 1]["line_num"] if i + 1 < len(headers) else total_lines

            text = "".join(self.lines[start_line:end_line]).strip()
            header["text"] = text
            header["line_start"] = start_line
            header["line_end"] = end_line

        return headers

    def build_tree(self, headers: List[Dict[str, Any]]) -> List[TreeNode]:
        """Build a nested tree of TreeNode objects using a stack algorithm."""

        if not headers:
            return []

        stack: List[tuple[TreeNode, int]] = []
        roots: List[TreeNode] = []
        node_counter = 1

        for header in headers:
            node = TreeNode(
                title=header["title"],
                node_id=f"{node_counter:04d}",
                level=header["level"],
                text=header["text"],
                extras={
                    "line_start": header["line_start"],
                    "line_end": header["line_end"],
                },
            )
            node_counter += 1

            # Pop stack until we find the parent level
            while stack and stack[-1][1] >= header["level"]:
                stack.pop()

            if not stack:
                roots.append(node)
            else:
                parent_node, _ = stack[-1]
                parent_node.children.append(node)

            stack.append((node, header["level"]))

        return roots

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        """Approximate token count for the given text."""

        if not self._tokenizer:
            # Fallback: simple word count
            return len(text.split())
        return len(self._tokenizer.encode(text))

    async def _generate_summary(self, text: str) -> str:
        """Generate a concise summary for the given text using the LLM."""

        if not self.llm_uri:
            # Fallback: truncate long text when LLM is not configured
            return text[:200] + "..."

        # Create a fresh conversation for each summary to avoid shared state
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)

        prompt = (
            "Please provide a concise summary of the following text in 2-3 sentences:\n\n"
            f"{text}\n\nSummary:"
        )
        response = await conv.asend(prompt)
        return response.content.strip()

    async def _process_node_summary(self, node: TreeNode) -> None:
        """Generate or assign summary for a single node based on token count."""

        token_count = self._count_tokens(node.text)
        if token_count < self.summary_threshold:
            node.summary = node.text
        else:
            node.summary = await self._generate_summary(node.text)

    async def _generate_summaries_batch(
        self,
        nodes: List[TreeNode],
        max_concurrent: int = 5,
        delay: float = 0.5,
    ) -> None:
        """Generate summaries for all nodes with concurrency control."""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _throttled(node: TreeNode) -> None:
            async with semaphore:
                await self._process_node_summary(node)
                await asyncio.sleep(delay)

        tasks = [_throttled(node) for node in nodes]
        await asyncio.gather(*tasks)

    def _collect_all_nodes(self, tree: List[TreeNode]) -> List[TreeNode]:
        """Flatten tree to a list of all nodes."""

        result: List[TreeNode] = []

        def _traverse(nodes: List[TreeNode]) -> None:
            for node in nodes:
                result.append(node)
                if node.children:
                    _traverse(node.children)

        _traverse(tree)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def read(
        self,
        path: str = None,
        content: str = None,
        generate_summaries: bool = False,
    ) -> DocTree:
        """Build DocTree structure from markdown source.

        Args:
            path: Markdown file path.
            content: Markdown text content.
            generate_summaries: Whether to generate LLM summaries for nodes.
            
        Note: Provide either path or content, not both.
        """

        if path and content:
            raise ValueError("Provide either path or content, not both")
        if not path and not content:
            raise ValueError("Must provide either path or content")

        if path:
            with open(path, "r", encoding="utf-8") as f:
                self.lines = f.readlines()
            self.doc_name = os.path.basename(path).replace(".md", "")
        else:
            self.lines = content.splitlines(keepends=True)
            self.doc_name = "untitled"

        # Step 1: Extract headers
        headers = self.extract_headers()

        # Step 2: Assign text ranges
        headers = self.assign_text_ranges(headers)

        # Step 3: Build tree nodes
        roots = self.build_tree(headers)

        # Step 4: Generate summaries (optional)
        if generate_summaries:
            all_nodes = self._collect_all_nodes(roots)
            await self._generate_summaries_batch(all_nodes)

        return DocTree(nodes=roots, doc_name=self.doc_name)


def print_tree(nodes: List[Dict[str, Any]], indent: int = 0) -> None:
    """Pretty print a tree structure produced by MarkdownTreeReader."""

    prefix = "  " * indent
    for node in nodes:
        text_preview = node.get("text", "")[:50].replace("\n", " ")
        print(f"{prefix}[{node['node_id']}] {node['title']} (Level {node['level']})")
        print(f"{prefix}   → {text_preview}...")
        children = node.get("children") or []
        if children:
            print_tree(children, indent + 1)
