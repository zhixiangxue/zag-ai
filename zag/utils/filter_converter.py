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
            >>> # Returns: "status = 'active' AND age >= 18"
        """
        if not filter_dict:
            return None
        
        # Separate logical operators from field conditions
        logical_ops = []
        field_conditions = []
        
        for key, value in filter_dict.items():
            if key in self.LOGICAL_OPS:
                # Process logical operator
                if key == "$or":
                    or_conditions = [self.convert(cond) for cond in value if self.convert(cond)]
                    if or_conditions:
                        logical_ops.append(f"({' OR '.join(or_conditions)})")
                elif key == "$and":
                    and_conditions = [self.convert(cond) for cond in value if self.convert(cond)]
                    if and_conditions:
                        logical_ops.append(f"({' AND '.join(and_conditions)})")
            else:
                # Process field condition
                mapped_key = self._map_field_name(key)
                condition = self._convert_field_condition(mapped_key, value)
                if condition:
                    field_conditions.append(condition)
        
        # Combine all conditions
        all_conditions = field_conditions + logical_ops
        
        if not all_conditions:
            return None
        
        if len(all_conditions) == 1:
            return all_conditions[0]
        else:
            return " AND ".join(all_conditions)
    
    def _map_field_name(self, key: str) -> str:
        """Map field name from MongoDB-style to LanceDB format.
        
        Args:
            key: Field name (may have metadata.custom prefix)
            
        Returns:
            Mapped field name for LanceDB SQL
            
        Examples:
            >>> converter = LanceDBFilterConverter()
            >>> converter._map_field_name("metadata.custom.category")
            'category'
            >>> converter._map_field_name("doc_id")
            'doc_id'
        """
        # Remove metadata.custom prefix if present
        if key.startswith("metadata.custom."):
            return key.replace("metadata.custom.", "")
        
        # Remove metadata prefix for other metadata fields
        if key.startswith("metadata."):
            return key.replace("metadata.", "")
        
        # Return as-is for top-level fields
        return key
    
    def _convert_field_condition(self, key: str, value: Any) -> Optional[str]:
        """Convert a single field condition to SQL format.
        
        Args:
            key: Field name
            value: Field value or operator dict
            
        Returns:
            SQL condition string
        """
        # Simple value (equality)
        if not isinstance(value, dict):
            return self._format_sql_condition(key, "=", value)
        
        # Operator dict
        conditions = []
        for op, op_value in value.items():
            if op == "$eq":
                conditions.append(self._format_sql_condition(key, "=", op_value))
            elif op == "$ne":
                conditions.append(self._format_sql_condition(key, "!=", op_value))
            elif op == "$gt":
                conditions.append(self._format_sql_condition(key, ">", op_value))
            elif op == "$gte":
                conditions.append(self._format_sql_condition(key, ">=", op_value))
            elif op == "$lt":
                conditions.append(self._format_sql_condition(key, "<", op_value))
            elif op == "$lte":
                conditions.append(self._format_sql_condition(key, "<=", op_value))
            elif op == "$in":
                # IN operator: field IN (val1, val2, ...)
                if isinstance(op_value, list) and op_value:
                    formatted_values = [self._format_sql_value(v) for v in op_value]
                    conditions.append(f"{key} IN ({', '.join(formatted_values)})")
            elif op == "$nin":
                # NOT IN operator
                if isinstance(op_value, list) and op_value:
                    formatted_values = [self._format_sql_value(v) for v in op_value]
                    conditions.append(f"{key} NOT IN ({', '.join(formatted_values)})")
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        else:
            return " AND ".join(conditions)
    
    def _format_sql_condition(self, key: str, operator: str, value: Any) -> str:
        """Format a SQL condition.
        
        Args:
            key: Field name
            operator: SQL operator (=, !=, >, <, etc.)
            value: Field value
            
        Returns:
            Formatted SQL condition string
        """
        formatted_value = self._format_sql_value(value)
        return f"{key} {operator} {formatted_value}"
    
    def _format_sql_value(self, value: Any) -> str:
        """Format a value for SQL.
        
        Args:
            value: Value to format
            
        Returns:
            SQL-formatted value string
        """
        if value is None:
            return "NULL"
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Escape single quotes
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        else:
            # Convert to string and escape
            escaped = str(value).replace("'", "''")
            return f"'{escaped}'"


class ChromaFilterConverter(FilterConverter):
    """Convert filters to Chroma format.
    
    Chroma uses MongoDB-like where clause syntax, very similar to our input format.
    Main difference: field names need to be mapped (metadata.custom.xxx → xxx).
    """
    
    def convert(self, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert filter dict to Chroma where clause.
        
        Args:
            filter_dict: Filter dict using MongoDB-style syntax
            
        Returns:
            Chroma where clause dict, or None if empty filter
            
        Examples:
            >>> converter = ChromaFilterConverter()
            >>> filter_dict = {"metadata.custom.category": "legal"}
            >>> result = converter.convert(filter_dict)
            >>> # Returns: {"category": "legal"}
        """
        if not filter_dict:
            return None
        
        # Separate logical operators from field conditions
        logical_ops = {}
        field_conditions = {}
        
        for key, value in filter_dict.items():
            if key in self.LOGICAL_OPS:
                logical_ops[key] = value
            else:
                # Map field name and store condition
                mapped_key = self._map_field_name(key)
                field_conditions[mapped_key] = value
        
        # Handle field conditions
        result = None
        if field_conditions:
            if len(field_conditions) == 1:
                # Single field: return as-is
                result = field_conditions
            else:
                # Multiple fields: wrap in $and
                result = {"$and": [{k: v} for k, v in field_conditions.items()]}
        
        # Handle logical operators
        for op_key, op_value in logical_ops.items():
            if isinstance(op_value, list):
                # Recursively convert each condition in the list
                converted_conditions = [self.convert(cond) for cond in op_value if self.convert(cond)]
                if result:
                    # Merge with existing result
                    if "$and" in result:
                        result["$and"].append({op_key: converted_conditions})
                    else:
                        result = {"$and": [result, {op_key: converted_conditions}]}
                else:
                    result = {op_key: converted_conditions}
            else:
                # Single condition
                converted = self.convert(op_value)
                if converted:
                    if result:
                        if "$and" in result:
                            result["$and"].append({op_key: converted})
                        else:
                            result = {"$and": [result, {op_key: converted}]}
                    else:
                        result = {op_key: converted}
        
        return result
    
    def _map_field_name(self, key: str) -> str:
        """Map field name from MongoDB-style to Chroma format.
        
        Args:
            key: Field name (may have metadata.custom prefix)
            
        Returns:
            Mapped field name for Chroma
            
        Examples:
            >>> converter = ChromaFilterConverter()
            >>> converter._map_field_name("metadata.custom.category")
            'category'
            >>> converter._map_field_name("doc_id")
            'doc_id'
        """
        # Remove metadata.custom prefix if present
        if key.startswith("metadata.custom."):
            return key.replace("metadata.custom.", "")
        
        # Remove metadata prefix for other metadata fields
        if key.startswith("metadata."):
            return key.replace("metadata.", "")
        
        # Return as-is for top-level fields
        return key
    
    def _convert_field_condition(self, key: str, value: Any) -> Any:
        """Convert a single field condition.
        
        Note: Chroma uses the same operator syntax as MongoDB,
        so we only need to map field names, not operators.
        
        Args:
            key: Field name
            value: Field value or operator dict
            
        Returns:
            Value as-is (Chroma uses same syntax)
        """
        # Chroma uses same operator syntax, no conversion needed
        return value


class MeilisearchFilterConverter(FilterConverter):
    """Convert filters to Meilisearch filter expression string.
    
    Meilisearch uses SQL-like filter syntax and natively supports dot-notation
    field paths (e.g. metadata.custom.mode) on nested JSON documents.
    Field paths are passed through unchanged, unlike LanceDB/Chroma which
    need prefix-stripping.
    
    Supported operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
    Supported logical: $and, $or, $nor
    
    Examples:
        Input : {"metadata.custom.mode": "classic", "unit_type": {"$in": ["text", "table"]}}
        Output: "metadata.custom.mode = 'classic' AND (unit_type = 'text' OR unit_type = 'table')"
    """
    
    def convert(self, filter_dict: Dict[str, Any]) -> Optional[str]:
        """Convert filter dict to Meilisearch filter string.
        
        Args:
            filter_dict: Filter dict using MongoDB-style syntax
            
        Returns:
            Meilisearch filter string, or None if empty
        """
        if not filter_dict:
            return None
        
        parts: List[str] = []
        
        for key, value in filter_dict.items():
            if key == "$and":
                sub_parts = [self.convert(c) for c in value]
                sub_parts = [s for s in sub_parts if s]
                if sub_parts:
                    parts.append("(" + " AND ".join(sub_parts) + ")")
            elif key == "$or":
                sub_parts = [self.convert(c) for c in value]
                sub_parts = [s for s in sub_parts if s]
                if sub_parts:
                    parts.append("(" + " OR ".join(sub_parts) + ")")
            elif key == "$nor":
                sub_parts = [self.convert(c) for c in value]
                sub_parts = [s for s in sub_parts if s]
                if sub_parts:
                    # NOR = NOT (a OR b)
                    parts.append("NOT (" + " OR ".join(sub_parts) + ")")
            else:
                cond = self._convert_field_condition(key, value)
                if cond:
                    parts.append(cond)
        
        if not parts:
            return None
        return " AND ".join(parts)
    
    def _convert_field_condition(self, key: str, value: Any) -> Optional[str]:
        """Convert a single field condition to Meilisearch filter expression.
        
        Args:
            key: Field path, dot-notation preserved as-is (e.g. metadata.custom.mode)
            value: Scalar value (equality) or operator dict
            
        Returns:
            Meilisearch filter condition string, or None
        """
        # Simple equality: {"status": "active"}
        if not isinstance(value, dict):
            return f"{key} = {self._fmt(value)}"
        
        conditions: List[str] = []
        for op, op_value in value.items():
            if op == "$eq":
                conditions.append(f"{key} = {self._fmt(op_value)}")
            elif op == "$ne":
                conditions.append(f"{key} != {self._fmt(op_value)}")
            elif op == "$gt":
                conditions.append(f"{key} > {self._fmt(op_value)}")
            elif op == "$gte":
                conditions.append(f"{key} >= {self._fmt(op_value)}")
            elif op == "$lt":
                conditions.append(f"{key} < {self._fmt(op_value)}")
            elif op == "$lte":
                conditions.append(f"{key} <= {self._fmt(op_value)}")
            elif op == "$in":
                if isinstance(op_value, list) and op_value:
                    or_parts = [f"{key} = {self._fmt(v)}" for v in op_value]
                    conditions.append("(" + " OR ".join(or_parts) + ")")
            elif op == "$nin":
                if isinstance(op_value, list) and op_value:
                    or_parts = [f"{key} = {self._fmt(v)}" for v in op_value]
                    conditions.append("NOT (" + " OR ".join(or_parts) + ")")
        
        if not conditions:
            return None
        return " AND ".join(conditions)
    
    def _fmt(self, value: Any) -> str:
        """Format a scalar value for Meilisearch filter expression."""
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            return str(value)
        # String: wrap in single quotes, escape internal single quotes
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"


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
        engine: Vector store engine type ("qdrant", "chroma", "lancedb", "milvus", "meilisearch")
        
    Returns:
        Target vector store's filter object or None
        
    Examples:
        >>> # For Qdrant (default)
        >>> filter = convert_filter({"status": "active", "age": {"$gte": 18}})
        
        >>> # For Meilisearch (returns filter string)
        >>> filter = convert_filter({"metadata.custom.mode": "classic"}, engine="meilisearch")
        
        >>> # For Chroma
        >>> filter = convert_filter({"status": "active"}, engine="chroma")
        
        >>> # For LanceDB
        >>> filter = convert_filter({"status": "active"}, engine="lancedb")
    """
    if engine == "qdrant":
        converter = QdrantFilterConverter()
        return converter.convert(filter_dict)
    elif engine == "chroma":
        converter = ChromaFilterConverter()
        return converter.convert(filter_dict)
    elif engine == "lancedb":
        converter = LanceDBFilterConverter()
        return converter.convert(filter_dict)
    elif engine == "milvus":
        converter = MilvusFilterConverter()
        return converter.convert(filter_dict)
    elif engine == "meilisearch":
        converter = MeilisearchFilterConverter()
        return converter.convert(filter_dict)
    else:
        raise ValueError(f"Unsupported engine: {engine}. Supported: qdrant, chroma, lancedb, milvus, meilisearch")


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
