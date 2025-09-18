"""
Neighborhood Algorithm Module

This module provides taste neighbor discovery and scoring algorithms based on
multi-signal interaction analysis.

Components:
- discovery: ETL-time neighbor discovery based on shared interactions
- scoring: Ranking-time content scoring using discovered neighbors
- utils: Shared utilities for neighborhood algorithms
"""

from .discovery import discover_taste_neighbors
from .scoring import collect_neighbor_content

__all__ = [
    'discover_taste_neighbors',
    'collect_neighbor_content'
]