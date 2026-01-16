"""Filter converter for RAG Service query API.

This module converts intuitive, MongoDB-style filter syntax to various vector store formats.
We adopt MongoDB's query syntax as our API interface because it's widely known and easy to use,
then convert it internally to each vector store's native format.

Note: This module has NO dependency on MongoDB. We simply use its query syntax convention.

Architecture:
    - FilterConverter: Abstract base class defining the interface
    - QdrantFilterConverter: Converts to Qdrant format
    - LanceDBFilterConverter: Converts to LanceDB format (future)
    - MilvusFilterConverter: Converts to Milvus format (future)

Examples:
    Input filter (MongoDB-style syntax):
        {
            "status": "active",
            "age": {"$gte": 18, "$lt": 65},
            "city": {"$in": ["SF", "NY"]},
            "$or": [
                {"premium": True},
                {"credits": {"$gte": 100}}
            ]
        }
    
    Usage:
        # For Qdrant
        converter = QdrantFilterConverter()
        qdrant_filter = converter.convert(filter_dict)
        
        # For LanceDB (future)
        converter = LanceDBFilterConverter()
        lancedb_filter = converter.convert(filter_dict)
"""
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range


class FilterConverter(ABC):
    """Abstract base class for filter converters.
    
    Each vector store should implement its own converter that inherits from this class.
    Adopts MongoDB-style query syntax as input format (no MongoDB dependency).
    """
    
    # Comparison operators (MongoDB-style)
    COMPARISON_OPS = {
        "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"
    }
    
    # Logical operators (MongoDB-style)
    LOGICAL_OPS = {
        "$and", "$or", "$not", "$nor"
    }
    
    @abstractmethod
    def convert(self, filter_dict: Dict[str, Any]) -> Any:
        """Convert filter dict to target vector store's format.
        
        Args:
            filter_dict: Filter dict using MongoDB-style syntax
            
        Returns:
            Target vector store's filter object
            
        Examples:
            >>> converter = QdrantFilterConverter()
            >>> filter_dict = {"status": "active", "age": {"$gte": 18}}
            >>> result = converter.convert(filter_dict)
        """
        pass
    
    @abstractmethod
    def _convert_field_condition(self, key: str, value: Any) -> Any:
        """Convert a single field condition to target format.
        
        Args:
            key: Field name (can use dot notation for nested fields)
            value: Field value or operator dict
            
        Returns:
            Target vector store's field condition object
        """
        pass


class QdrantFilterConverter(FilterConverter):
    """Convert filters to Qdrant format.
    
    Converts MongoDB-style filter syntax to Qdrant's must/should/must_not format.
    """
    
    def convert(self, filter_dict: Dict[str, Any]) -> Optional[Filter]:
        """Convert filter dict to Qdrant Filter.
        
        Args:
            filter_dict: Filter dict using MongoDB-style syntax
            
        Returns:
            Qdrant Filter object, or None if empty filter
            
        Examples:
            >>> converter = FilterConverter()
            >>> filter_dict = {"status": "active", "age": {"$gte": 18}}
            >>> qdrant_filter = converter.convert(filter_dict)
        """
        if not filter_dict:
            return None
        
        must_conditions = []
        should_conditions = []
        must_not_conditions = []
        
        for key, value in filter_dict.items():
            if key == "$and":
                # $and: [condition1, condition2, ...]
                for condition in value:
                    sub_filter = self.convert(condition)
                    if sub_filter:
                        must_conditions.append(sub_filter)
                        
            elif key == "$or":
                # $or: [condition1, condition2, ...]
                for condition in value:
                    sub_filter = self.convert(condition)
                    if sub_filter:
                        should_conditions.append(sub_filter)
                        
            elif key == "$nor":
                # $nor: [condition1, condition2, ...]
                for condition in value:
                    sub_filter = self.convert(condition)
                    if sub_filter:
                        must_not_conditions.append(sub_filter)
                        
            elif key == "$not":
                # $not: {condition}
                sub_filter = FilterConverter.convert(value)
                if sub_filter:
                    must_not_conditions.append(sub_filter)
                    
            else:
                # Regular field condition
                field_condition = self._convert_field_condition(key, value)
                if field_condition:
                    must_conditions.append(field_condition)
        
        # Build Qdrant Filter
        if not must_conditions and not should_conditions and not must_not_conditions:
            return None
        
        return Filter(
            must=must_conditions if must_conditions else None,
            should=should_conditions if should_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None
        )
    
    def _convert_field_condition(self, key: str, value: Any) -> Optional[FieldCondition]:
        """Convert a single field condition.
        
        Args:
            key: Field name (can use dot notation for nested fields)
            value: Field value or operator dict
            
        Returns:
            Qdrant FieldCondition or None
        """
        # Simple equality: {"status": "active"}
        if not isinstance(value, dict):
            return FieldCondition(
                key=key,
                match=MatchValue(value=value)
            )
        
        # Operator-based conditions
        conditions = []
        
        for op, op_value in value.items():
            if op == "$eq":
                return FieldCondition(key=key, match=MatchValue(value=op_value))
                
            elif op == "$ne":
                # Use must_not with match
                return FieldCondition(key=key, match=MatchValue(value=op_value))
                
            elif op == "$in":
                # {"city": {"$in": ["SF", "NY", "LA"]}}
                return FieldCondition(key=key, match=MatchAny(any=op_value))
                
            elif op == "$nin":
                # {"city": {"$nin": ["deleted", "banned"]}}
                # This needs to be wrapped in must_not at higher level
                return FieldCondition(key=key, match=MatchAny(any=op_value))
                
            elif op in ["$gt", "$gte", "$lt", "$lte"]:
                # Range conditions: {"age": {"$gte": 18, "$lt": 65}}
                range_params = {}
                
                # Collect all range operators for this field
                for range_op, range_val in value.items():
                    if range_op == "$gt":
                        range_params["gt"] = range_val
                    elif range_op == "$gte":
                        range_params["gte"] = range_val
                    elif range_op == "$lt":
                        range_params["lt"] = range_val
                    elif range_op == "$lte":
                        range_params["lte"] = range_val
                
                if range_params:
                    return FieldCondition(key=key, range=Range(**range_params))
        
        return None


class LanceDBFilterConverter(FilterConverter):
    """Convert filters to LanceDB format.
    
    LanceDB uses SQL-like filter syntax.
    TODO: Implement conversion logic when LanceDB is adopted.
    """
    
    def convert(self, filter_dict: Dict[str, Any]) -> Optional[str]:
        """Convert filter dict to LanceDB SQL-like filter string.
        
        Args:
            filter_dict: Filter dict using MongoDB-style syntax
            
        Returns:
            LanceDB filter string (SQL-like) or None
            
        Examples:
            >>> converter = LanceDBFilterConverter()
            >>> filter_dict = {"status": "active", "age": {"$gte": 18}}
            >>> result = converter.convert(filter_dict)
            >>> # Expected: "status = 'active' AND age >= 18"
        """
        # TODO: Implement LanceDB conversion
        # LanceDB uses SQL-like syntax: "column = value AND column >= value"
        raise NotImplementedError("LanceDB filter converter not yet implemented")
    
    def _convert_field_condition(self, key: str, value: Any) -> Optional[str]:
        """Convert a single field condition to LanceDB format.
        
        Args:
            key: Field name
            value: Field value or operator dict
            
        Returns:
            SQL-like condition string
        """
        # TODO: Implement field condition conversion
        raise NotImplementedError("LanceDB field condition converter not yet implemented")


class MilvusFilterConverter(FilterConverter):
    """Convert filters to Milvus format.
    
    Milvus uses expression-based filter syntax.
    TODO: Implement conversion logic when Milvus is adopted.
    """
    
    def convert(self, filter_dict: Dict[str, Any]) -> Optional[str]:
        """Convert filter dict to Milvus expression.
        
        Args:
            filter_dict: Filter dict using MongoDB-style syntax
            
        Returns:
            Milvus expression string or None
            
        Examples:
            >>> converter = MilvusFilterConverter()
            >>> filter_dict = {"status": "active", "age": {"$gte": 18}}
            >>> result = converter.convert(filter_dict)
            >>> # Expected: "status == 'active' && age >= 18"
        """
        # TODO: Implement Milvus conversion
        # Milvus uses expression syntax: "column == value && column >= value"
        raise NotImplementedError("Milvus filter converter not yet implemented")
    
    def _convert_field_condition(self, key: str, value: Any) -> Optional[str]:
        """Convert a single field condition to Milvus format.
        
        Args:
            key: Field name
            value: Field value or operator dict
            
        Returns:
            Milvus expression string
        """
        # TODO: Implement field condition conversion
        raise NotImplementedError("Milvus field condition converter not yet implemented")


def convert_filter(filter_dict: Dict[str, Any], engine: str = "qdrant") -> Any:
    """Convert filter dict to target vector store format.
    
    Factory function that selects the appropriate converter based on the engine type.
    Uses MongoDB-style query syntax as input (no MongoDB dependency).
    
    Args:
        filter_dict: Filter dict using MongoDB-style syntax
        engine: Vector store engine type ("qdrant", "lancedb", "milvus", etc.)
        
    Returns:
        Target vector store's filter object or None
        
    Examples:
        >>> # For Qdrant (default)
        >>> filter = convert_filter({"status": "active", "age": {"$gte": 18}})
        
        >>> # For LanceDB (future)
        >>> filter = convert_filter({"status": "active"}, engine="lancedb")
    """
    if engine == "qdrant":
        converter = QdrantFilterConverter()
        return converter.convert(filter_dict)
    elif engine == "lancedb":
        converter = LanceDBFilterConverter()
        return converter.convert(filter_dict)
    elif engine == "milvus":
        converter = MilvusFilterConverter()
        return converter.convert(filter_dict)
    else:
        raise ValueError(f"Unsupported engine: {engine}. Supported: qdrant, lancedb, milvus")


# ============================================
# Test Examples (can be run directly)
# ============================================

if __name__ == "__main__":
    """Run test examples."""
    
    print("=" * 80)
    print("Filter Converter - Test Examples")
    print("=" * 80)
    
    # Example 1: Simple equality
    print("\n[Example 1] Simple equality")
    print("-" * 80)
    filter1 = {"status": "active"}
    qdrant1 = convert_filter(filter1)
    print(f"Input:   {filter1}")
    print(f"Qdrant:  {qdrant1}")
    
    # Example 2: Range query
    print("\n[Example 2] Range query")
    print("-" * 80)
    filter2 = {"age": {"$gte": 18, "$lt": 65}}
    qdrant2 = convert_filter(filter2)
    print(f"Input:   {filter2}")
    print(f"Qdrant:  {qdrant2}")
    
    # Example 3: IN operator
    print("\n[Example 3] IN operator")
    print("-" * 80)
    filter3 = {"city": {"$in": ["San Francisco", "New York", "Berlin"]}}
    qdrant3 = convert_filter(filter3)
    print(f"Input:   {filter3}")
    print(f"Qdrant:  {qdrant3}")
    
    # Example 4: Multiple conditions (implicit AND)
    print("\n[Example 4] Multiple conditions (implicit AND)")
    print("-" * 80)
    filter4 = {
        "status": "active",
        "age": {"$gte": 18},
        "city": {"$in": ["SF", "NY"]}
    }
    qdrant4 = convert_filter(filter4)
    print(f"Input:   {filter4}")
    print(f"Qdrant:  {qdrant4}")
    
    # Example 5: OR condition
    print("\n[Example 5] OR condition")
    print("-" * 80)
    filter5 = {
        "$or": [
            {"status": "active"},
            {"status": "pending"}
        ]
    }
    qdrant5 = convert_filter(filter5)
    print(f"Input:   {filter5}")
    print(f"Qdrant:  {qdrant5}")
    
    # Example 6: Complex nested conditions
    print("\n[Example 6] Complex nested conditions")
    print("-" * 80)
    filter6 = {
        "dataset_id": "abc123",
        "$or": [
            {"premium": True},
            {"credits": {"$gte": 100}}
        ],
        "age": {"$gte": 18, "$lt": 65}
    }
    qdrant6 = convert_filter(filter6)
    print(f"Input:   {filter6}")
    print(f"Qdrant:  {qdrant6}")
    
    # Example 7: RAG Service use case
    print("\n[Example 7] RAG Service typical filter")
    print("-" * 80)
    filter7 = {
        "doc_id": {"$in": ["doc1", "doc2", "doc3"]},
        "metadata.category": "legal",
        "metadata.confidence": {"$gte": 0.8}
    }
    qdrant7 = convert_filter(filter7)
    print(f"Input:   {filter7}")
    print(f"Qdrant:  {qdrant7}")
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
