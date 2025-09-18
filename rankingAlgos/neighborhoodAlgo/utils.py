"""
Shared utilities for neighborhood algorithms
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dateutil import parser

logger = logging.getLogger(__name__)


def calculate_jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Calculate Jaccard similarity between two sets
    
    Args:
        set_a: First set
        set_b: Second set
    
    Returns:
        Jaccard similarity coefficient (0.0 to 1.0)
    """
    if not set_a and not set_b:
        return 1.0
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    return intersection / union if union > 0 else 0.0


def apply_temporal_decay(score: float, content_age_hours: float) -> float:
    """
    Apply temporal decay to reduce weight of older content
    
    Args:
        score: Original score
        content_age_hours: Age of content in hours
    
    Returns:
        Decay-adjusted score
    """
    if content_age_hours <= 6:
        return score  # Full weight for last 6 hours
    elif content_age_hours <= 24:
        return score * 0.9  # 90% weight for last day
    elif content_age_hours <= 72:
        return score * 0.7  # 70% weight for last 3 days
    elif content_age_hours <= 168:
        return score * 0.5  # 50% weight for last week
    else:
        return score * 0.2  # 20% weight for older content


def calculate_content_age_hours(indexed_at: str) -> Optional[float]:
    """
    Calculate age of content in hours from indexed_at timestamp
    
    Args:
        indexed_at: ISO timestamp string
    
    Returns:
        Age in hours, or None if parsing fails
    """
    try:
        content_time = parser.isoparse(indexed_at.replace('Z', '+00:00'))
        now = datetime.now(content_time.tzinfo)
        age_delta = now - content_time
        return age_delta.total_seconds() / 3600  # Convert to hours
    except Exception as e:
        logger.debug(f"Failed to parse timestamp {indexed_at}: {e}")
        return None


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to 0-1 range using min-max scaling
    
    Args:
        scores: List of raw scores
    
    Returns:
        List of normalized scores
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if min_score == max_score:
        return [1.0] * len(scores)
    
    range_score = max_score - min_score
    return [(score - min_score) / range_score for score in scores]


def filter_by_engagement_threshold(
    content_list: List[Dict], 
    min_likes: int = 0, 
    min_reposts: int = 0
) -> List[Dict]:
    """
    Filter content by minimum engagement thresholds
    
    Args:
        content_list: List of content dicts with engagement metrics
        min_likes: Minimum like count
        min_reposts: Minimum repost count
    
    Returns:
        Filtered content list
    """
    filtered = []
    for content in content_list:
        likes = content.get('like_count', 0)
        reposts = content.get('repost_count', 0)
        
        if likes >= min_likes and reposts >= min_reposts:
            filtered.append(content)
    
    return filtered


def deduplicate_content(content_list: List[Dict], key: str = 'uri') -> List[Dict]:
    """
    Remove duplicate content based on specified key
    
    Args:
        content_list: List of content dicts
        key: Key to use for deduplication
    
    Returns:
        Deduplicated content list
    """
    seen = set()
    deduplicated = []
    
    for content in content_list:
        content_key = content.get(key)
        if content_key and content_key not in seen:
            seen.add(content_key)
            deduplicated.append(content)
    
    return deduplicated