"""
Full-text retriever - basic layer retriever for full-text search
"""

from typing import Any, Optional, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

import meilisearch
from meilisearch.errors import MeilisearchApiError

from ..base import BaseRetriever
from ...schemas.base import BaseUnit, UnitMetadata


class FullTextRetriever(BaseRetriever):
    """
    Full-text search retriever (Basic Layer)
    
    Provides full-text search capabilities powered by Meilisearch:
    - Typo tolerance and fuzzy matching
    - Filter by metadata attributes
    - Sort by custom fields
    - Faceted search
    
    Design:
        - Direct integration with Meilisearch service
        - Converts Meilisearch results to BaseUnit objects
        - Supports advanced search features (filters, sorting, facets)
    
    Responsibilities:
        - Execute full-text searches via Meilisearch
        - Convert search results to Units with relevance scores
        - Provide retriever interface for composition
    
    Examples:
        >>> from zag.retrievers import FullTextRetriever
        >>> 
        >>> # Create retriever
        >>> retriever = FullTextRetriever(
        ...     url="http://127.0.0.1:7700",
        ...     index_name="documents"
        ... )
        >>> 
        >>> # Simple search
        >>> units = retriever.retrieve("python programming", top_k=5)
        >>> 
        >>> # Search with filters
        >>> units = retriever.retrieve(
        ...     "machine learning",
        ...     top_k=10,
        ...     filters={"category": "ai", "difficulty": "beginner"}
        ... )
        >>> 
        >>> # Search with sorting
        >>> units = retriever.retrieve(
        ...     "docker tutorial",
        ...     top_k=5,
        ...     sort=["timestamp:desc"]
        ... )
    """
    
    def __init__(
        self,
        url: str = "http://127.0.0.1:7700",
        index_name: str = "documents",
        api_key: Optional[str] = None,
        top_k: int = 10,
        primary_key: str = "unit_id",
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize full-text retriever
        
        Args:
            url: Meilisearch server URL
            index_name: Name of the Meilisearch index
            api_key: Optional API key for authentication
            top_k: Default number of results to return
            primary_key: Primary key field name (default: "unit_id")
            executor: Optional thread pool for async operations
        """
        self.url = url
        self.index_name = index_name
        self.primary_key = primary_key
        self.default_top_k = top_k
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
        
        # Initialize Meilisearch client
        self.client = meilisearch.Client(url, api_key)
        
        # Get index
        try:
            self.index = self.client.get_index(index_name)
        except MeilisearchApiError as e:
            raise ValueError(
                f"Failed to get index '{index_name}': {e}. "
                f"Make sure the index exists and Meilisearch is running."
            )
    
    def _document_to_unit(self, doc: Dict[str, Any], score: Optional[float] = None) -> BaseUnit:
        """
        Convert Meilisearch document to BaseUnit
        
        Args:
            doc: Meilisearch document dictionary
            score: Relevance score (from _rankingScore if available)
        
        Returns:
            BaseUnit instance
        """
        # Extract unit_id (primary key)
        unit_id = doc.get(self.primary_key)
        
        # Extract content
        content = doc.get("content", "")
        
        # Extract context_path if exists
        context_path = doc.get("context_path")
        
        # Build custom metadata from remaining fields
        custom = {}
        for key, value in doc.items():
            if key not in [self.primary_key, "content", "context_path", "_rankingScore"]:
                custom[key] = value
        
        # Get score from _rankingScore if available
        if score is None and "_rankingScore" in doc:
            score = doc["_rankingScore"]
        
        # Create unit with UnitMetadata
        unit = BaseUnit(
            unit_id=unit_id,
            content=content,
            metadata=UnitMetadata(
                context_path=context_path,
                custom=custom
            )
        )
        
        # Attach score if available
        if score is not None:
            unit.score = score
        
        return unit
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        sort: Optional[list[str]] = None,
        **kwargs
    ) -> list[BaseUnit]:
        """
        Execute full-text search
        
        Args:
            query: Search query string
            top_k: Number of results to return (overrides default if provided)
            filters: Metadata filters as dict (converted to Meilisearch filter string)
                    Example: {"category": "tech", "difficulty": "beginner"}
            sort: List of sort criteria (e.g., ["timestamp:desc", "price:asc"])
            **kwargs: Additional Meilisearch search parameters:
                - facets: List of attributes to facet on
                - attributes_to_retrieve: List of attributes to include
                - attributes_to_highlight: List of attributes to highlight
                - show_ranking_score: Whether to include ranking scores (default: True)
        
        Returns:
            List of retrieved units, sorted by relevance
        
        Examples:
            >>> # Simple search
            >>> units = retriever.retrieve("python", top_k=5)
            >>> 
            >>> # With filters
            >>> units = retriever.retrieve(
            ...     "tutorial",
            ...     filters={"category": "programming", "difficulty": "beginner"}
            ... )
            >>> 
            >>> # With sorting
            >>> units = retriever.retrieve(
            ...     "docker",
            ...     sort=["timestamp:desc"]
            ... )
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # Build search parameters
        search_params = {
            "limit": k,
            "showRankingScore": True,  # Always get ranking scores
        }
        
        # Convert filters dict to Meilisearch filter string
        if filters:
            filter_parts = []
            for key, value in filters.items():
                if isinstance(value, str):
                    filter_parts.append(f"{key} = '{value}'")
                elif isinstance(value, (int, float)):
                    filter_parts.append(f"{key} = {value}")
                elif isinstance(value, list):
                    # Handle list as OR condition
                    or_parts = []
                    for v in value:
                        if isinstance(v, str):
                            or_parts.append(f"{key} = '{v}'")
                        else:
                            or_parts.append(f"{key} = {v}")
                    filter_parts.append(f"({' OR '.join(or_parts)})")
            
            if filter_parts:
                search_params["filter"] = " AND ".join(filter_parts)
        
        # Add sort if provided
        if sort:
            search_params["sort"] = sort
        
        # Add any additional kwargs
        search_params.update(kwargs)
        
        # Execute search
        results = self.index.search(query, search_params)
        
        # Convert hits to Units
        hits = results.get("hits", [])
        units = []
        for hit in hits:
            score = hit.get("_rankingScore")
            unit = self._document_to_unit(hit, score)
            units.append(unit)
        
        return units
    
    async def aretrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        sort: Optional[list[str]] = None,
        **kwargs
    ) -> list[BaseUnit]:
        """
        Async version of full-text search
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Metadata filters
            sort: Sort criteria
            **kwargs: Additional search parameters
        
        Returns:
            List of retrieved units
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.retrieve(query, top_k, filters, sort, **kwargs)
        )
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"FullTextRetriever("
            f"url='{self.url}', "
            f"index='{self.index_name}'"
            f")"
        )
