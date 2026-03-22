"""
TableContextExpander - expand table units with surrounding context by fetching
prev/next neighbors directly from the vector store.

Unlike ContextAugmentor (which relies on UnitRegistry and only works during
ingestion), this expander fetches neighbors from the backing store at query
time, so it works correctly in the RAG service request path.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from ..base import BasePostprocessor
from ...schemas import BaseUnit
from ...schemas.types import UnitType

if TYPE_CHECKING:
    from ...storages.vector.base import BaseVectorStore


class ExpandMode(str, Enum):
    """Which neighbors to expand around a table unit."""

    PREV = "prev"        # Prepend the preceding unit only
    NEXT = "next"        # Append the following unit only
    SIBLING = "sibling"  # Prepend prev AND append next


class TableContextExpander(BasePostprocessor):
    """
    Expand table units with surrounding context fetched from the vector store.

    For each retrieved unit that contains a table (detected by unit_type or
    HTML tag), the expander fetches the configured neighbors from the store,
    stitches their content together, and replaces the unit's content in-place.
    The unit_id and all other metadata remain unchanged.

    Neighbors that are consumed by stitching can optionally be removed from
    the result list to avoid duplication before reranking.

    Design notes:
        - Requires ``aprocess`` (async); ``process`` raises NotImplementedError.
        - All neighbor fetches are batched into a single ``aget`` call.
        - Content is joined with ``\\n\\n`` so the LLM sees a clear boundary.

    Args:
        vector_store: Vector store used to fetch neighbor units by ID.
        expand_mode: Which neighbors to fetch (PREV, NEXT, or SIBLING).
        remove_consumed: If True, neighbor units absorbed into a table unit
            are removed from the output list so reranker sees no duplicates.
        separator: String inserted between stitched content pieces.

    Examples:
        >>> expander = TableContextExpander(
        ...     vector_store=vector_store,
        ...     expand_mode=ExpandMode.SIBLING,
        ... )
        >>> units = await expander.aprocess(query, retrieved_units)
    """

    def __init__(
        self,
        vector_store: "BaseVectorStore",
        expand_mode: ExpandMode = ExpandMode.SIBLING,
        remove_consumed: bool = True,
        separator: str = "\n\n",
    ):
        self.vector_store = vector_store
        self.expand_mode = expand_mode
        self.remove_consumed = remove_consumed
        self.separator = separator

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_table_unit(unit: BaseUnit) -> bool:
        """Return True if the unit contains a table.

        Checks unit_type first (free), then parses HTML with BeautifulSoup
        only when necessary.
        """
        if unit.unit_type == UnitType.TABLE:
            return True
        content_str = str(unit.content) if unit.content is not None else ""
        if not content_str:
            return False
        soup = BeautifulSoup(content_str, "html.parser")
        return soup.find("table") is not None

    def _collect_table_neighbors(
        self, units: list[BaseUnit]
    ) -> dict[str, dict[str, str | None]]:
        """Scan units once and return a neighbor map for all table units.

        Returns:
            A dict keyed by table unit_id, each value describing which
            neighbor IDs to fetch::

                {
                    "<table_unit_id>": {
                        "prev_id": "<prev_unit_id> or None",
                        "next_id": "<next_unit_id> or None",
                    },
                    ...
                }

            Only IDs allowed by expand_mode are populated; others are None.
        """
        result: dict[str, dict[str, str | None]] = {}
        for unit in units:
            if not self._is_table_unit(unit):
                continue
            prev_id = (
                unit.prev_unit_id
                if self.expand_mode in (ExpandMode.PREV, ExpandMode.SIBLING)
                else None
            )
            next_id = (
                unit.next_unit_id
                if self.expand_mode in (ExpandMode.NEXT, ExpandMode.SIBLING)
                else None
            )
            result[unit.unit_id] = {"prev_id": prev_id, "next_id": next_id}
        return result

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def process(self, query: str, units: list[BaseUnit]) -> list[BaseUnit]:
        """Sync path is not supported; use aprocess instead."""
        raise NotImplementedError(
            "TableContextExpander requires async execution. "
            "Call aprocess() instead of process()."
        )

    async def aprocess(
        self,
        query: str,
        units: list[BaseUnit],
    ) -> list[BaseUnit]:
        """
        Expand table units with neighbor context.

        Args:
            query: Original query text (not used, kept for interface compat).
            units: Retrieved units to process.

        Returns:
            Units with table unit contents stitched with neighbor context.
            If remove_consumed=True, absorbed neighbor units are excluded.
            Falls back to original units on any error.
        """
        if not units:
            return units

        try:
            return await self._expand(units)
        except Exception:
            return units

    async def _expand(self, units: list[BaseUnit]) -> list[BaseUnit]:
        """Core expansion logic. Errors propagate to aprocess for fallback handling."""
        # 1. Scan units once: build {table_unit_id: {prev_id, next_id}}
        table_neighbor_map = self._collect_table_neighbors(units)
        if not table_neighbor_map:
            return units

        # Flatten all neighbor IDs that actually need fetching
        ids_to_fetch = {
            nid
            for neighbors in table_neighbor_map.values()
            for nid in (neighbors["prev_id"], neighbors["next_id"])
            if nid
        }

        # 2. Batch-fetch all neighbors in one round-trip
        fetched = await self.vector_store.aget(list(ids_to_fetch))
        fetched_map: dict[str, BaseUnit] = {u.unit_id: u for u in fetched}

        # 3. Stitch content into each table unit; track consumed IDs
        consumed_ids: set[str] = set()
        for unit in units:
            neighbors = table_neighbor_map.get(unit.unit_id)
            if neighbors is None:
                continue

            parts: list[str] = []

            prev = fetched_map.get(neighbors["prev_id"]) if neighbors["prev_id"] else None
            if prev:
                parts.append(str(prev.content))
                consumed_ids.add(prev.unit_id)

            parts.append(str(unit.content))

            nxt = fetched_map.get(neighbors["next_id"]) if neighbors["next_id"] else None
            if nxt:
                parts.append(str(nxt.content))
                consumed_ids.add(nxt.unit_id)

            if len(parts) > 1:
                unit.content = self.separator.join(parts)

        # 4. Optionally remove consumed units from result
        if self.remove_consumed and consumed_ids:
            retrieved_ids = {u.unit_id for u in units}
            to_remove = consumed_ids & retrieved_ids
            if to_remove:
                units = [u for u in units if u.unit_id not in to_remove]

        return units
