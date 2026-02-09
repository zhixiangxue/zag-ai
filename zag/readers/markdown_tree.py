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
        """Generate a retrieval-optimized summary that maximizes signal retention."""

        if not self.llm_uri:
            # Fallback: truncate long text when LLM is not configured
            return text[:200] + "..."

        # Create a fresh conversation for each summary to avoid shared state
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)

        prompt = (
            "Summarize the following text for search/retrieval purposes. "
            "The summary will be used to find relevant information later, "
            "so it's critical to preserve ALL specific signals.\n\n"
            "**IMPORTANT INSTRUCTIONS:**\n"
            "- DO NOT generalize or abstract lists. Keep all enumerated items.\n"
            "- DO NOT replace specific terms with vague categories "
            '(e.g., don\'t change "race, color, religion" to "various characteristics").\n'
            "- PRESERVE all: regulation codes, proper nouns, numeric values, dates, "
            "enumerated items, technical terms.\n"
            "- Keep the summary in the SAME LANGUAGE as the source text.\n\n"
            "**Required Format:**\n"
            "[OVERVIEW] One sentence describing what this section covers.\n"
            "[KEY_SIGNALS] List ALL specific terms, entities, codes, numbers, "
            "and enumerated items mentioned. Separate with commas.\n\n"
            "**Examples of what to AVOID:**\n"
            'Bad: "prohibits discrimination based on various characteristics"\n'
            'Bad: "lists several requirements"\n'
            'Bad: "references multiple regulations"\n\n'
            "**Examples of what to PRODUCE:**\n"
            'Good: "prohibits discrimination based on race, color, national origin, '
            'sex, religion, marital status, familial status, age, disability"\n'
            'Good: "requires compliance with Fair Housing Act, Equal Credit Opportunity Act, '
            'and 15 U.S.C. 1601"\n\n'
            f"**Text to summarize:**\n{text}\n\n"
            "**Your summary:**"
        )
        response = await conv.asend(prompt)
        return response.content.strip()

    async def _aggregate_summary_bottom_up(
        self, 
        node: TreeNode,
        semaphore: asyncio.Semaphore,
        delay: float = 0.5
    ) -> None:
        """Generate summary for node using bottom-up aggregation.
        
        Process children first, then aggregate their summaries into parent's summary.
        This ensures parent nodes contain semantic information from child nodes.
        """
        # Step 1: Process all children first (recursive)
        if node.children:
            tasks = [
                self._aggregate_summary_bottom_up(child, semaphore, delay) 
                for child in node.children
            ]
            await asyncio.gather(*tasks)
        
        # Step 2: Aggregate current node with children summaries
        async with semaphore:
            children_summaries = [c.summary for c in node.children if c.summary]
            
            if children_summaries:
                # Combine own text + children summaries
                combined = node.text + "\n\nChild sections:\n" + "\n".join(
                    f"- {c.title}: {c.summary[:200]}" for c in node.children if c.summary
                )
                
                # If combined text is too long, generate new summary; otherwise use as-is
                token_count = self._count_tokens(combined)
                if token_count > 500:
                    node.summary = await self._generate_summary(combined)
                else:
                    node.summary = combined
            else:
                # Leaf node: use original logic
                token_count = self._count_tokens(node.text)
                if token_count < self.summary_threshold:
                    node.summary = node.text
                else:
                    node.summary = await self._generate_summary(node.text)
            
            await asyncio.sleep(delay)

    async def _generate_summaries_batch(
        self,
        roots: List[TreeNode],
        max_concurrent: int = 5,
        delay: float = 0.5,
    ) -> None:
        """Generate summaries using bottom-up aggregation with concurrency control."""

        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Process each root tree independently
        tasks = [
            self._aggregate_summary_bottom_up(root, semaphore, delay) 
            for root in roots
        ]
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
            await self._generate_summaries_batch(roots)

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
