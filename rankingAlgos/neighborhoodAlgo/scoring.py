"""
Neighborhood content collection for ranking pipeline

This module collects content from discovered taste neighbors to expand 
the content pool for ranking. Focus is on broad collection, not scoring.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dateutil import parser
from .utils import deduplicate_content, calculate_content_age_hours

logger = logging.getLogger(__name__)


def collect_neighbor_content(
    taste_neighbors: List[Dict],
    client,
    collection_config: Optional[Dict] = None
) -> List[Dict]:
    """
    Collect content from discovered taste neighbors
    
    Args:
        taste_neighbors: Pre-discovered neighbors from ETL with similarity scores
        client: Bluesky client for API calls  
        collection_config: Settings for content collection
    
    Returns:
        List of content from neighbors (deduplicated)
    """
    if not taste_neighbors:
        logger.warning("No taste neighbors provided for content collection")
        return []
    
    # Set default collection configuration
    if collection_config is None:
        collection_config = {
            'max_posts_per_neighbor': 20,    # Posts to collect per neighbor
            'time_hours': 48,                # Look back 48 hours  
            'collect_authored': True,        # Collect neighbor's own posts
            'collect_liked': True,           # Collect posts neighbors liked
            'collect_reposted': False,       # Skip reposts for now (complex to get originals)
            'min_engagement': 1              # Minimum likes to include content
        }
    
    logger.info(f"Collecting content from {len(taste_neighbors)} taste neighbors")
    
    all_content = []
    cutoff_time = datetime.now() - timedelta(hours=collection_config['time_hours'])
    
    # Collect authored content from neighbors
    if collection_config['collect_authored']:
        authored_content = _collect_neighbor_authored_content(
            taste_neighbors, client, collection_config, cutoff_time
        )
        all_content.extend(authored_content)
        logger.info(f"Collected {len(authored_content)} authored posts from neighbors")
    
    # Collect liked content from neighbors  
    if collection_config['collect_liked']:
        liked_content = _collect_neighbor_liked_content(
            taste_neighbors, client, collection_config, cutoff_time
        )
        all_content.extend(liked_content)
        logger.info(f"Collected {len(liked_content)} liked posts from neighbors")
    
    # Deduplicate and filter
    deduplicated_content = deduplicate_content(all_content, key='uri')
    filtered_content = _apply_basic_filters(deduplicated_content, collection_config)
    
    logger.info(f"Final collection: {len(filtered_content)} posts from taste neighbors")
    return filtered_content


def _collect_neighbor_authored_content(
    taste_neighbors: List[Dict],
    client,
    config: Dict,
    cutoff_time: datetime
) -> List[Dict]:
    """Collect recent posts authored by taste neighbors"""
    authored_content = []
    
    for neighbor in taste_neighbors:
        neighbor_did = neighbor['did']
        similarity_score = neighbor['similarity_score']
        
        try:
            # Get recent posts from this neighbor
            params = {
                'actor': neighbor_did,
                'limit': min(config['max_posts_per_neighbor'], 50),
                'filter': 'posts_no_replies'  # Only original posts, no replies
            }
            
            response = client.client.app.bsky.feed.get_author_feed(params)
            
            for feed_item in response.feed:
                post = feed_item.post
                
                # Check if post is within time window
                post_time = parser.isoparse(post.record.created_at.replace('Z', '+00:00'))
                if post_time.replace(tzinfo=None) < cutoff_time:
                    continue
                
                # Create post data with neighbor context
                post_data = {
                    'uri': post.uri,
                    'cid': post.cid,
                    'author': {
                        'did': post.author.did,
                        'handle': post.author.handle,
                        'display_name': getattr(post.author, 'display_name', ''),
                    },
                    'text': post.record.text,
                    'created_at': post.record.created_at,
                    'indexed_at': post.indexed_at,
                    'like_count': getattr(post, 'like_count', 0),
                    'repost_count': getattr(post, 'repost_count', 0),
                    'reply_count': getattr(post, 'reply_count', 0),
                    'neighbor_source': {
                        'type': 'authored',
                        'neighbor_did': neighbor_did,
                        'similarity_score': similarity_score
                    }
                }
                
                authored_content.append(post_data)
            
        except Exception as e:
            logger.debug(f"Error collecting authored content from neighbor {neighbor_did}: {e}")
            continue
    
    return authored_content


def _collect_neighbor_liked_content(
    taste_neighbors: List[Dict],
    client,
    config: Dict,
    cutoff_time: datetime
) -> List[Dict]:
    """Collect posts that taste neighbors have liked"""
    liked_content = []
    
    # Note: This is more complex as we need to get neighbor's likes
    # For now, we'll use a simplified approach focusing on recent popular content
    # that neighbors might have liked. A full implementation would require
    # accessing each neighbor's like history.
    
    logger.info("Neighbor-liked content collection: Simplified implementation")
    
    # TODO: Implement full neighbor likes collection
    # This would require:
    # 1. Get each neighbor's recent likes (if accessible)
    # 2. Collect the original posts they liked
    # 3. Filter by time window and engagement
    
    # For now, return empty list - focus on authored content first
    return liked_content


def _apply_basic_filters(content_list: List[Dict], config: Dict) -> List[Dict]:
    """Apply basic quality and engagement filters"""
    filtered_content = []
    
    for content in content_list:
        try:
            # Filter by minimum engagement
            like_count = content.get('like_count', 0)
            if like_count < config.get('min_engagement', 1):
                continue
            
            # Filter out very old content (additional safety check)
            content_age = calculate_content_age_hours(content.get('indexed_at', ''))
            if content_age and content_age > config.get('time_hours', 48):
                continue
            
            # Filter out empty or very short posts
            text = content.get('text', '')
            if len(text.strip()) < 10:  # Skip very short posts
                continue
            
            filtered_content.append(content)
            
        except Exception as e:
            logger.debug(f"Error filtering content {content.get('uri', 'unknown')}: {e}")
            continue
    
    return filtered_content


def get_neighbor_content_stats(content_list: List[Dict]) -> Dict:
    """Get statistics about collected neighbor content"""
    if not content_list:
        return {}
    
    authored_count = len([c for c in content_list if c.get('neighbor_source', {}).get('type') == 'authored'])
    liked_count = len([c for c in content_list if c.get('neighbor_source', {}).get('type') == 'liked'])
    
    total_engagement = sum(
        c.get('like_count', 0) + c.get('repost_count', 0) + c.get('reply_count', 0) 
        for c in content_list
    )
    
    unique_neighbors = len(set(
        c.get('neighbor_source', {}).get('neighbor_did') 
        for c in content_list 
        if c.get('neighbor_source', {}).get('neighbor_did')
    ))
    
    return {
        'total_posts': len(content_list),
        'authored_posts': authored_count,
        'liked_posts': liked_count,
        'total_engagement': total_engagement,
        'unique_neighbors_contributing': unique_neighbors,
        'avg_engagement_per_post': total_engagement / len(content_list) if content_list else 0
    }