"""
Graph Storage Module

Provides unified interface for graph database operations.
"""

from .base import GraphStorage
from .falkordb import FalkorDBGraphStorage, create_storage

__all__ = [
    "GraphStorage",
    "FalkorDBGraphStorage",
    "create_storage",
]
