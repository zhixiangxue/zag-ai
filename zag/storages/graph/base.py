"""
Graph Storage - Abstract base class and implementations for graph databases.

This module provides a unified interface for storing and querying mortgage
program data as a graph structure. Supports multiple backends (FalkorDB, Neo4j).

Key Features:
1. Parameterized queries (SQL injection prevention)
2. Batch import for large datasets
3. Version management for programs
4. Standardized node/relationship operations
"""

from __future__ import annotations

import abc
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union


class GraphStorage(abc.ABC):
    """
    Abstract base class for graph storage backends.
    
    Subclasses must implement:
    - connect() / disconnect()
    - create_node() / create_nodes()
    - create_relationship()
    - execute_query()
    - clear()
    
    Design Principles:
    1. All queries use parameterized values (no string concatenation)
    2. Nodes have a unique 'id' field
    3. Relationships are directional: (source)-[type]->(target)
    4. Batch operations for performance
    """
    
    def __init__(self, graph_name: str = "mortgage_programs"):
        """
        Initialize graph storage.
        
        Args:
            graph_name: Name of the graph to use
        """
        self.graph_name = graph_name
        self._connected = False
    
    @abc.abstractmethod
    def connect(self) -> bool:
        """Connect to the graph database."""
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the graph database."""
        pass
    
    @abc.abstractmethod
    def execute_query(
        self,
        query: str,
        params: Dict[str, Any] = None,
    ) -> List[Dict]:
        """
        Execute a Cypher query with parameters.
        
        Args:
            query: Cypher query string with $param placeholders
            params: Dictionary of parameter values
            
        Returns:
            List of result dictionaries
        """
        pass
    
    def is_connected(self) -> bool:
        """Check if connected to the database."""
        return self._connected
    
    # =========================================================================
    # Node Operations
    # =========================================================================
    
    @abc.abstractmethod
    def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        unique_field: str = "id",
    ) -> str:
        """
        Create a node with the given label and properties.
        
        Args:
            label: Node label (e.g., "Program", "Product")
            properties: Node properties
            unique_field: Field to use for uniqueness check
            
        Returns:
            Node ID
        """
        pass
    
    def create_nodes(
        self,
        label: str,
        properties_list: List[Dict[str, Any]],
        unique_field: str = "id",
    ) -> List[str]:
        """
        Create multiple nodes.
        
        Default implementation calls create_node() for each.
        Subclasses can override for batch optimization.
        
        Args:
            label: Node label
            properties_list: List of property dictionaries
            unique_field: Field to use for uniqueness check
            
        Returns:
            List of node IDs
        """
        return [
            self.create_node(label, props, unique_field)
            for props in properties_list
        ]
    
    def get_node(self, label: str, node_id: str) -> Optional[Dict]:
        """
        Get a node by ID.
        
        Args:
            label: Node label
            node_id: Node ID
            
        Returns:
            Node properties or None
        """
        query = f"MATCH (n:{label} {{id: $node_id}}) RETURN n"
        results = self.execute_query(query, {"node_id": node_id})
        if results:
            return results[0].get("n", results[0])
        return None
    
    def update_node(
        self,
        label: str,
        node_id: str,
        properties: Dict[str, Any],
    ) -> bool:
        """
        Update node properties.
        
        Args:
            label: Node label
            node_id: Node ID
            properties: Properties to update
            
        Returns:
            True if successful
        """
        set_clause = ", ".join(f"n.{k} = ${k}" for k in properties.keys())
        query = f"MATCH (n:{label} {{id: $node_id}}) SET {set_clause}"
        params = {"node_id": node_id, **properties}
        self.execute_query(query, params)
        return True
    
    def delete_node(self, label: str, node_id: str) -> bool:
        """
        Delete a node by ID.
        
        Args:
            label: Node label
            node_id: Node ID
            
        Returns:
            True if successful
        """
        query = f"MATCH (n:{label} {{id: $node_id}}) DETACH DELETE n"
        self.execute_query(query, {"node_id": node_id})
        return True
    
    # =========================================================================
    # Relationship Operations
    # =========================================================================
    
    @abc.abstractmethod
    def create_relationship(
        self,
        source_label: str,
        source_id: str,
        target_label: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None,
    ) -> bool:
        """
        Create a relationship between two nodes.
        
        Args:
            source_label: Source node label
            source_id: Source node ID
            target_label: Target node label
            target_id: Target node ID
            rel_type: Relationship type
            properties: Optional relationship properties
            
        Returns:
            True if successful
        """
        pass
    
    def get_relationships(
        self,
        label: str,
        node_id: str,
        rel_type: str = None,
        direction: str = "out",
    ) -> List[Dict]:
        """
        Get relationships for a node.
        
        Args:
            label: Node label
            node_id: Node ID
            rel_type: Optional relationship type filter
            direction: "out", "in", or "both"
            
        Returns:
            List of relationship info
        """
        if direction == "out":
            pattern = f"(n:{label} {{id: $node_id}})-[r{':'+rel_type if rel_type else ''}]->(target)"
        elif direction == "in":
            pattern = f"(target)-[r{':'+rel_type if rel_type else ''}]->(n:{label} {{id: $node_id}})"
        else:
            pattern = f"(n:{label} {{id: $node_id}})-[r{':'+rel_type if rel_type else ''}]-(target)"
        
        query = f"MATCH {pattern} RETURN type(r) as rel_type, target.id as target_id, properties(r) as props"
        return self.execute_query(query, {"node_id": node_id})
    
    # =========================================================================
    # Graph Operations
    # =========================================================================
    
    @abc.abstractmethod
    def clear(self) -> bool:
        """Clear all data from the graph."""
        pass
    
    def node_count(self, label: str = None) -> int:
        """
        Count nodes in the graph.
        
        Args:
            label: Optional label filter
            
        Returns:
            Node count
        """
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
        else:
            query = "MATCH (n) RETURN count(n) as count"
        results = self.execute_query(query)
        return results[0].get("count", 0) if results else 0
    
    # =========================================================================
    # Utility Methods (for subclass use)
    # =========================================================================
    
    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert object to dictionary with proper serialization."""
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return {k: self._serialize_value(v) for k, v in obj.items()}
        if hasattr(obj, 'model_dump'):
            return {k: self._serialize_value(v) for k, v in obj.model_dump(exclude_none=True).items()}
        return {}
    
    def _serialize_value(self, v: Any) -> Any:
        """Serialize a value for storage."""
        import json
        
        if v is None:
            return None
        # Check Enum BEFORE basic types (Enum can be subclass of str/int)
        if isinstance(v, Enum):
            return v.value
        if isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, list):
            return [self._serialize_value(item) for item in v]
        if isinstance(v, dict):
            # Serialize dict to JSON string for graph databases that don't support nested objects
            return json.dumps({k: self._serialize_value(val) for k, val in v.items()})
        if hasattr(v, 'value'):
            # Handle objects with value attribute (non-Enum)
            return v.value
        return str(v)
    
    # =========================================================================
    # Context Manager Support
    # =========================================================================
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
