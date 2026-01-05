"""
Filters - postprocessors that filter units based on certain criteria
"""

from .similarity import SimilarityFilter
from .deduplicator import Deduplicator

__all__ = [
    "SimilarityFilter",
    "Deduplicator",
]
