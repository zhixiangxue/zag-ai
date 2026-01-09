"""
Query fusion retriever - composite layer retriever for combining multiple retrievers
"""

from typing import Any, Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..base import BaseRetriever
from ...schemas.base import BaseUnit


class FusionMode(str, Enum):
    """Fusion mode enumeration"""
    
    SIMPLE = "simple"
    """Simple fusion: deduplicate and keep highest score"""
    
    RECIPROCAL_RANK = "reciprocal_rank"
    """Reciprocal Rank Fusion (RRF): suitable for combining different retrieval types"""
    
    RELATIVE_SCORE = "relative_score"
    """Relative score fusion: suitable for combining same type retrievers"""


class QueryFusionRetriever(BaseRetriever):
    """
    Query fusion retriever (Composite Layer)
    
    Combines multiple retrievers and fuses their results:
    1. Calls multiple retrievers to search
    2. Merges results based on fusion mode
    3. Returns fused and ranked units
    
    Supported fusion modes:
        - SIMPLE: Simple deduplication, keeps highest score for each unit
        - RECIPROCAL_RANK: RRF algorithm, suitable for different retrieval types
        - RELATIVE_SCORE: Relative score fusion with normalization, suitable for same types
    
    Design Philosophy:
        - Composition over inheritance: composes multiple BaseRetriever instances
        - Store-agnostic: only depends on BaseRetriever interface
        - Flexible: supports any retriever type (basic or composite)
    
    Examples:
        >>> from zag.retrievers import VectorRetriever, QueryFusionRetriever, FusionMode
        >>> 
        >>> # Create multiple vector retrievers (e.g., different vector stores)
        >>> retriever1 = VectorRetriever(vector_store1)
        >>> retriever2 = VectorRetriever(vector_store2)
        >>> 
        >>> # Combine with RRF fusion
        >>> fusion_retriever = QueryFusionRetriever(
        ...     retrievers=[retriever1, retriever2],
        ...     mode=FusionMode.RECIPROCAL_RANK,
        ...     top_k=10
        ... )
        >>> 
        >>> # Retrieve
        >>> units = fusion_retriever.retrieve("What is machine learning?")
        >>> for unit in units:
        ...     print(f"Score: {unit.score}, Content: {unit.content[:50]}")
    """
    
    def __init__(
        self, 
        retrievers: list[BaseRetriever],
        mode: FusionMode = FusionMode.SIMPLE,
        top_k: int = 10,
        retriever_weights: Optional[list[float]] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Initialize query fusion retriever
        
        Args:
            retrievers: List of retrievers to combine
            mode: Fusion mode (SIMPLE, RECIPROCAL_RANK, or RELATIVE_SCORE)
            top_k: Default number of results to return
            retriever_weights: Weight for each retriever (only used in RELATIVE_SCORE mode)
                               If None, equal weights are used
            executor: Optional thread pool for concurrent retrieval
                     If None, creates a default executor with 4 workers
        
        Raises:
            ValueError: If retrievers list is empty or weights length doesn't match
        """
        if not retrievers:
            raise ValueError("Must provide at least one retriever")
        
        if retriever_weights and len(retriever_weights) != len(retrievers):
            raise ValueError(
                f"Length of retriever_weights ({len(retriever_weights)}) "
                f"must match number of retrievers ({len(retrievers)})"
            )
        
        self.retrievers = retrievers
        self.mode = mode
        self.default_top_k = top_k
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
        
        # Normalize weights
        if retriever_weights is None:
            self.retriever_weights = [1.0 / len(retrievers)] * len(retrievers)
        else:
            total_weight = sum(retriever_weights)
            self.retriever_weights = [w / total_weight for w in retriever_weights]
    
    def retrieve(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None
    ) -> list[BaseUnit]:
        """
        Execute fusion retrieval
        
        Args:
            query: Query text
            top_k: Number of results to return (overrides default if provided)
            filters: Optional metadata filters
            
        Returns:
            Fused list of units, sorted by fusion score
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # 1. Retrieve from all retrievers concurrently
        def safe_retrieve(retriever: BaseRetriever) -> list[BaseUnit]:
            try:
                return retriever.retrieve(query, top_k=k * 2, filters=filters)
            except Exception as e:
                print(f"Warning: Retriever failed with error: {e}")
                return []
        
        # Use ThreadPoolExecutor for concurrent execution
        all_results = []
        futures = [self._executor.submit(safe_retrieve, r) for r in self.retrievers]
        
        for future in as_completed(futures):
            all_results.append(future.result())
        
        # 2. Fuse results based on mode
        if self.mode == FusionMode.RECIPROCAL_RANK:
            merged_units = self._reciprocal_rank_fusion(all_results)
        elif self.mode == FusionMode.RELATIVE_SCORE:
            merged_units = self._relative_score_fusion(all_results)
        else:  # SIMPLE
            merged_units = self._simple_fusion(all_results)
        
        return merged_units[:k]
    
    async def aretrieve(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None
    ) -> list[BaseUnit]:
        """
        Async fusion retrieval
        
        Args:
            query: Query text
            top_k: Number of results to return (overrides default if provided)
            filters: Optional metadata filters
            
        Returns:
            Fused list of units, sorted by fusion score
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # 1. Retrieve from all retrievers concurrently
        import asyncio
        
        async def safe_retrieve(retriever: BaseRetriever) -> list[BaseUnit]:
            try:
                return await retriever.aretrieve(query, top_k=k * 2, filters=filters)
            except Exception as e:
                print(f"Warning: Async retriever failed with error: {e}")
                return []
        
        all_results = await asyncio.gather(
            *[safe_retrieve(r) for r in self.retrievers]
        )
        
        # 2. Fuse results based on mode
        if self.mode == FusionMode.RECIPROCAL_RANK:
            merged_units = self._reciprocal_rank_fusion(list(all_results))
        elif self.mode == FusionMode.RELATIVE_SCORE:
            merged_units = self._relative_score_fusion(list(all_results))
        else:  # SIMPLE
            merged_units = self._simple_fusion(list(all_results))
        
        return merged_units[:k]
    
    def _reciprocal_rank_fusion(self, results: list[list[BaseUnit]]) -> list[BaseUnit]:
        """
        RRF (Reciprocal Rank Fusion) algorithm
        
        Suitable for: Combining different types of retrieval (e.g., vector + keyword)
        
        Formula: score(d) = sum(1 / (k + rank_i(d))) for all retrievers i
        where k is a constant (typically 60)
        
        Args:
            results: List of result lists from different retrievers
            
        Returns:
            Fused units sorted by RRF score
        """
        k = 60.0  # RRF constant
        fused_scores: dict[str, float] = {}
        id_to_unit: dict[str, BaseUnit] = {}
        
        for units in results:
            # Sort by original score (if present)
            sorted_units = sorted(
                units, 
                key=lambda x: getattr(x, 'score', 0.0) if hasattr(x, 'score') else 0.0,
                reverse=True
            )
            
            for rank, unit in enumerate(sorted_units):
                unit_id = unit.unit_id
                id_to_unit[unit_id] = unit
                
                if unit_id not in fused_scores:
                    fused_scores[unit_id] = 0.0
                
                # RRF formula
                fused_scores[unit_id] += 1.0 / (k + rank)
        
        # Sort by fused score
        sorted_ids = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Build result list with updated scores
        result_units = []
        for unit_id, score in sorted_ids:
            unit = id_to_unit[unit_id]
            # Attach fused score (create a copy to avoid modifying original)
            unit_copy = unit.model_copy()
            unit_copy.score = score
            result_units.append(unit_copy)
        
        return result_units
    
    def _relative_score_fusion(self, results: list[list[BaseUnit]]) -> list[BaseUnit]:
        """
        Relative score fusion with MinMax normalization
        
        Suitable for: Combining same type retrievers (e.g., multiple vector stores)
        
        Process:
        1. Normalize scores within each retriever using MinMax
        2. Apply retriever weights
        3. Sum scores for duplicate units
        
        Args:
            results: List of result lists from different retrievers
            
        Returns:
            Fused units sorted by weighted normalized score
        """
        all_units: dict[str, BaseUnit] = {}
        
        for i, units in enumerate(results):
            if not units:
                continue
            
            # Extract scores
            scores = [
                getattr(unit, 'score', 0.0) if hasattr(unit, 'score') else 0.0 
                for unit in units
            ]
            
            if not scores:
                continue
            
            min_score = min(scores)
            max_score = max(scores)
            
            # Normalize and weight
            for unit in units:
                unit_id = unit.unit_id
                original_score = getattr(unit, 'score', 0.0) if hasattr(unit, 'score') else 0.0
                
                # MinMax normalization
                if max_score == min_score:
                    normalized_score = 1.0 if max_score > 0 else 0.0
                else:
                    normalized_score = (original_score - min_score) / (max_score - min_score)
                
                # Apply weight
                weighted_score = normalized_score * self.retriever_weights[i]
                
                # Accumulate scores for duplicates
                if unit_id in all_units:
                    current_score = getattr(all_units[unit_id], 'score', 0.0)
                    unit_copy = all_units[unit_id]
                    unit_copy.score = current_score + weighted_score
                    all_units[unit_id] = unit_copy
                else:
                    unit_copy = unit.model_copy()
                    unit_copy.score = weighted_score
                    all_units[unit_id] = unit_copy
        
        # Sort by fused score
        return sorted(
            all_units.values(),
            key=lambda x: getattr(x, 'score', 0.0) if hasattr(x, 'score') else 0.0,
            reverse=True
        )
    
    def _simple_fusion(self, results: list[list[BaseUnit]]) -> list[BaseUnit]:
        """
        Simple fusion: deduplicate and keep highest score
        
        Suitable for: Quick deduplication without complex scoring
        
        Process:
        1. For each unit, keep the highest score across all retrievers
        2. Sort by score
        
        Args:
            results: List of result lists from different retrievers
            
        Returns:
            Fused units sorted by highest score
        """
        all_units: dict[str, BaseUnit] = {}
        
        for units in results:
            for unit in units:
                unit_id = unit.unit_id
                unit_score = getattr(unit, 'score', 0.0) if hasattr(unit, 'score') else 0.0
                
                if unit_id in all_units:
                    # Keep highest score
                    existing_score = getattr(all_units[unit_id], 'score', 0.0)
                    if unit_score > existing_score:
                        unit_copy = unit.model_copy()
                        unit_copy.score = unit_score
                        all_units[unit_id] = unit_copy
                else:
                    unit_copy = unit.model_copy()
                    unit_copy.score = unit_score
                    all_units[unit_id] = unit_copy
        
        # Sort by score
        return sorted(
            all_units.values(),
            key=lambda x: getattr(x, 'score', 0.0) if hasattr(x, 'score') else 0.0,
            reverse=True
        )
