"""
FalkorDB Storage Backend

Implementation of GraphStorage for FalkorDB (Redis-based graph database).

Installation:
    pip install redis graphdatascience

Usage:
    storage = FalkorDBGraphStorage(host="localhost", port=6379)
    storage.connect()
    program_id = storage.store_program(program_data)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import GraphStorage


class FalkorDBGraphStorage(GraphStorage):
    """
    FalkorDB implementation of GraphStorage.
    
    FalkorDB is a Redis module that provides graph database capabilities
    with Cypher query support.
    
    Args:
        host: Redis host
        port: Redis port
        graph_name: Name of the graph to use
        password: Optional Redis password
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph_name: str = "mortgage_programs",
        password: str = None,
    ):
        super().__init__(graph_name)
        self.host = host
        self.port = port
        self.password = password
        self._db = None
        self._graph = None
    
    def connect(self) -> bool:
        """Connect to FalkorDB."""
        try:
            from falkordb import FalkorDB
            
            self._db = FalkorDB(
                host=self.host,
                port=self.port,
                password=self.password,
            )
            self._graph = self._db.select_graph(self.graph_name)
            self._connected = True
            return True
        except ImportError:
            raise ImportError(
                "falkordb is required for FalkorDBGraphStorage. "
                "Install with: pip install redis"
            )
        except Exception as e:
            print(f"Failed to connect to FalkorDB: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from FalkorDB."""
        self._db = None
        self._graph = None
        self._connected = False
        return True
    
    def execute_query(
        self,
        query: str,
        params: Dict[str, Any] = None,
    ) -> List[Dict]:
        """
        Execute a Cypher query.
        
        Note: FalkorDB uses $param syntax but has limited parameterization.
        We do basic parameter substitution for safety.
        """
        if not self._connected:
            raise RuntimeError("Not connected to database")
        
        # For FalkorDB, we need to do safe parameter substitution
        # since it doesn't fully support parameterized queries
        if params:
            query = self._substitute_params(query, params)
        
        try:
            result = self._graph.query(query)
            return self._process_result(result)
        except Exception as e:
            import traceback
            print(f"Query execution failed: {e}")
            traceback.print_exc()
            print(f"Query: {query}")
            return []
    
    def _substitute_params(self, query: str, params: Dict[str, Any]) -> str:
        """
        Safely substitute parameters into query.
        
        This is a critical security function - must handle all edge cases.
        """
        for key, value in params.items():
            placeholder = f"${key}"
            
            if value is None:
                replacement = "NULL"
            elif isinstance(value, bool):
                replacement = "true" if value else "false"
            elif isinstance(value, int):
                replacement = str(value)
            elif isinstance(value, float):
                replacement = str(value)
            elif isinstance(value, str):
                # Escape single quotes for Cypher
                escaped = value.replace("'", "\\'")
                replacement = f"'{escaped}'"
            elif isinstance(value, list):
                # Handle list values
                items = []
                for item in value:
                    if isinstance(item, str):
                        escaped = item.replace("'", "\\'")
                        items.append(f"'{escaped}'")
                    else:
                        items.append(str(item))
                replacement = "[" + ", ".join(items) + "]"
            else:
                # Fallback to string representation
                escaped = str(value).replace("'", "\\'")
                replacement = f"'{escaped}'"
            
            query = query.replace(placeholder, replacement)
        
        return query
    
    def _process_result(self, result) -> List[Dict]:
        """Process FalkorDB result into list of dictionaries."""
        if not result or not result.result_set:
            return []
        
        # Get header (column names)
        # FalkorDB header format: [[type, alias], [type, alias], ...]
        # where type is an integer and alias is the column name
        header = result.header if hasattr(result, 'header') else []
        
        # Normalize header - extract alias from [type, alias] format
        normalized_header = []
        for col in header:
            if isinstance(col, str):
                normalized_header.append(col)
            elif isinstance(col, (list, tuple)) and len(col) >= 2:
                # FalkorDB returns [type_code, alias]
                normalized_header.append(str(col[1]) if isinstance(col[1], str) else str(col[-1]))
            elif isinstance(col, (list, tuple)) and len(col) == 1:
                normalized_header.append(str(col[0]))
            else:
                normalized_header.append(str(col))
        
        # Process each row
        processed = []
        for row in result.result_set:
            if normalized_header:
                row_dict = {}
                for i, col in enumerate(normalized_header):
                    if i < len(row):
                        row_dict[col] = self._convert_value(row[i])
                processed.append(row_dict)
            else:
                processed.append({"result": self._convert_value(row[0]) if row else None})
        
        return processed
    
    def _convert_value(self, value) -> Any:
        """Convert FalkorDB value to Python type."""
        if value is None:
            return None
        if isinstance(value, (int, float, str, bool)):
            return value
        if isinstance(value, list):
            return [self._convert_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        return str(value)
    
    def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        unique_field: str = "id",
    ) -> str:
        """
        Create a node with MERGE for idempotency.
        """
        node_id = properties.get(unique_field)
        if not node_id:
            import uuid
            node_id = str(uuid.uuid4())[:8]
            properties[unique_field] = node_id
        
        # Build property string safely
        props_str = self._build_props_string(properties)
        
        query = f"""
        MERGE (n:{label} {{{unique_field}: '{node_id}'}})
        SET n += {{{props_str}}}
        RETURN n.id as id
        """
        
        result = self.execute_query(query)
        return node_id
    
    def create_nodes(
        self,
        label: str,
        properties_list: List[Dict[str, Any]],
        unique_field: str = "id",
    ) -> List[str]:
        """
        Create multiple nodes efficiently using UNWIND.
        """
        if not properties_list:
            return []
        
        # Build values list
        values = []
        node_ids = []
        for props in properties_list:
            node_id = props.get(unique_field)
            if not node_id:
                import uuid
                node_id = str(uuid.uuid4())[:8]
                props[unique_field] = node_id
            node_ids.append(node_id)
            
            # Build property object
            prop_items = []
            for k, v in props.items():
                prop_items.append(f"{k}: {self._format_value(v)}")
            values.append("{" + ", ".join(prop_items) + "}")
        
        values_str = ", ".join(values)
        
        query = f"""
        UNWIND [{values_str}] as props
        MERGE (n:{label} {{{unique_field}: props.{unique_field}}})
        SET n += props
        RETURN n.id as id
        """
        
        self.execute_query(query)
        return node_ids
    
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
        """
        props_str = ""
        if properties:
            props_str = " {" + self._build_props_string(properties) + "}"
        
        query = f"""
        MATCH (s:{source_label} {{id: '{source_id}'}})
        MATCH (t:{target_label} {{id: '{target_id}'}})
        MERGE (s)-[r:{rel_type}{props_str}]->(t)
        """
        
        self.execute_query(query)
        return True
    
    def clear(self) -> bool:
        """Clear all data from the graph."""
        try:
            self._graph.delete()
            self._graph = self._db.select_graph(self.graph_name)
            return True
        except Exception:
            # Graph might not exist
            return True
    
    def _build_props_string(self, properties: Dict[str, Any]) -> str:
        """Build Cypher property string safely."""
        items = []
        for k, v in properties.items():
            items.append(f"{k}: {self._format_value(v)}")
        return ", ".join(items)
    
    def _format_value(self, value: Any) -> str:
        """Format a value for Cypher."""
        if value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            escaped = value.replace("'", "\\'")
            return f"'{escaped}'"
        elif isinstance(value, list):
            items = [self._format_value(item) for item in value]
            return "[" + ", ".join(items) + "]"
        elif isinstance(value, dict):
            # Handle dict values
            items = [f"{k}: {self._format_value(v)}" for k, v in value.items()]
            return "{" + ", ".join(items) + "}"
        elif hasattr(value, 'model_dump'):
            # Handle Pydantic models
            return self._format_value(value.model_dump())
        elif hasattr(value, 'value'):
            # Handle Enums
            return f"'{value.value}'"
        else:
            escaped = str(value).replace("'", "\\'")
            return f"'{escaped}'"


# =============================================================================
# Convenience Functions
# =============================================================================


def create_storage(
    backend: str = "falkordb",
    **kwargs,
) -> GraphStorage:
    """
    Factory function to create a graph storage instance.
    
    Args:
        backend: Storage backend ("falkordb" or "neo4j")
        **kwargs: Backend-specific arguments
        
    Returns:
        GraphStorage instance
    """
    if backend == "falkordb":
        return FalkorDBGraphStorage(**kwargs)
    elif backend == "neo4j":
        # Import Neo4j implementation if available
        try:
            from .neo4j import Neo4jGraphStorage
            return Neo4jGraphStorage(**kwargs)
        except ImportError:
            raise ImportError(
                "Neo4j support requires additional dependencies. "
                "Install with: pip install neo4j"
            )
    else:
        raise ValueError(f"Unknown backend: {backend}")
