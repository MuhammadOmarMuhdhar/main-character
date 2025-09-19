"""
Post collection functions for the ranking system
"""
import logging
import sys
import os
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from etl.ranking.config import DEFAULT_TIME_HOURS
from rankingAlgos.neighborhoodAlgo.scoring import collect_neighbor_content
from rankingAlgos.networkAlgo.scoring import collect_mutual_content

logger = logging.getLogger(__name__)


def collect_posts_for_user(user_id: str, user_data: Dict, bq_client, redis_client, client=None) -> List[Dict]:
    """
    Collect posts that will be candidates for ranking using taste neighbors
    
    Args:
        user_id: User's DID
        user_data: User profile data from userData.py
        bq_client: BigQuery client for data queries
        redis_client: Redis client for caching
        client: Bluesky client for API calls (optional)
        
    Returns:
        List of post dictionaries to be ranked
    """
    try:
        logger.info(f"Collecting posts for user {user_id}")
        
        collected_posts = []
        
        # Get both taste neighbors and mutual connections from user profile data (discovered during ETL)
        
        # Extract taste neighbors from dedicated column
        taste_neighbors = user_data.get('taste_neighbors', [])
        if taste_neighbors:
            logger.info(f"Found {len(taste_neighbors)} taste neighbors for user {user_id}")
        else:
            logger.warning(f"No taste neighbors found for user {user_id}")
        
        # Extract mutual connections from dedicated column
        mutual_connections = []
        network_data = user_data.get('network_relationships', {})
        if network_data and isinstance(network_data, dict):
            mutual_connections = network_data.get('mutual_connections', [])
            logger.info(f"Found {len(mutual_connections)} mutual connections for user {user_id}")
        else:
            logger.warning(f"No mutual connections found for user {user_id}")
        
        # Collect content from taste neighbors if available
        if taste_neighbors and client:
            neighbor_content = collect_neighbor_content(
                taste_neighbors=taste_neighbors,
                client=client,
                collection_config={
                    'max_posts_per_neighbor': 15,  # Collect 15 posts per neighbor
                    'time_hours': 48,              # Look back 48 hours
                    'collect_authored': True,      # Collect neighbor's posts
                    'collect_liked': False,        # Skip liked content for now
                    'min_engagement': 2            # Minimum 2 likes
                }
            )
            collected_posts.extend(neighbor_content)
            logger.info(f"Collected {len(neighbor_content)} posts from taste neighbors")
        
        # Collect content from mutual connections if available
        if mutual_connections and client:
            mutual_content = collect_mutual_content(
                mutual_connections=mutual_connections,
                client=client,
                collection_config={
                    'max_posts_per_mutual': 20,    # Collect 20 posts per mutual
                    'time_hours': 24,              # Look back 24 hours
                    'include_reposts': True,       # Include reposted content
                    'repost_weight': 0.7,         # Weight reposts at 70%
                    'min_engagement': 1            # Minimum 1 like
                }
            )
            collected_posts.extend(mutual_content)
            logger.info(f"Collected {len(mutual_content)} posts from mutual connections")
        
        if not client:
            logger.warning(f"No client provided for content collection for user {user_id}")
        
        # TODO: Add other collection strategies here:
        # - Posts from users they follow directly
        # - Trending posts for diversity
        # - Posts from specific feeds
        
        logger.info(f"Total collected posts for user {user_id}: {len(collected_posts)}")
        return collected_posts
        
    except Exception as e:
        logger.error(f"Failed to collect posts for user {user_id}: {e}")
        return []


def get_trending_posts(bq_client, time_hours: int = DEFAULT_TIME_HOURS, limit: int = 100) -> List[Dict]:
    """
    Get trending/popular posts for default feed
    TODO: Implement trending post detection
    
    Args:
        bq_client: BigQuery client
        time_hours: Hours back to look for trending posts
        limit: Maximum number of posts to return
        
    Returns:
        List of trending post dictionaries
    """
    try:
        logger.info(f"Collecting trending posts from last {time_hours} hours")
        
        # TODO: Implement trending post logic
        # This could be based on:
        # - Engagement velocity (likes/reposts per hour)
        # - Total engagement above threshold
        # - Growth rate in engagement
        # - etc.
        
        trending_posts = []
        
        logger.info(f"Found {len(trending_posts)} trending posts")
        return trending_posts
        
    except Exception as e:
        logger.error(f"Failed to get trending posts: {e}")
        return []