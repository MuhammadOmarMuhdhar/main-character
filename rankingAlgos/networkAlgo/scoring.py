"""
Network-based content collection for ranking pipeline

This module collects content from discovered mutual connections to expand 
the content pool for ranking. Focus is on leveraging pre-discovered network relationships.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dateutil import parser

logger = logging.getLogger(__name__)


def collect_mutual_content(
    mutual_connections: List[Dict],
    client,
    collection_config: Optional[Dict] = None
) -> List[Dict]:
    """
    Collect content from pre-discovered mutual connections
    
    Args:
        mutual_connections: Pre-discovered mutuals from ETL with connection metadata
        client: Bluesky client for API calls (NewPosts client)
        collection_config: Settings for content collection
    
    Returns:
        List of content from mutual connections (deduplicated)
    """
    if not mutual_connections:
        logger.warning("No mutual connections provided for content collection")
        return []
    
    # Set default collection configuration
    if collection_config is None:
        collection_config = {
            'max_posts_per_mutual': 25,      # Posts to collect per mutual
            'time_hours': 24,                # Look back 24 hours
            'include_reposts': True,         # Include reposted content
            'repost_weight': 0.7,           # Weight for reposts vs original posts
            'min_engagement': 1              # Minimum likes to include content
        }
    
    logger.info(f"Collecting content from {len(mutual_connections)} mutual connections")
    
    try:
        # Use the existing get_following_timeline function for efficient collection
        mutual_posts = client.get_following_timeline(
            following_list=mutual_connections,
            target_count=len(mutual_connections) * collection_config['max_posts_per_mutual'],
            time_hours=collection_config['time_hours'],
            include_reposts=collection_config['include_reposts'],
            repost_weight=collection_config['repost_weight']
        )
        
        # Add metadata to indicate these came from mutual connections
        processed_posts = []
        for post in mutual_posts:
            # Create enhanced post with mutual connection metadata
            enhanced_post = post.copy()
            enhanced_post['source'] = 'mutual'
            enhanced_post['collection_method'] = 'network_mutual'
            enhanced_post['is_mutual'] = True
            
            # Find which mutual connection this post came from
            post_author_did = post.get('author', {}).get('did')
            for mutual in mutual_connections:
                if mutual.get('did') == post_author_did:
                    enhanced_post['mutual_connection_info'] = {
                        'connection_type': mutual.get('connection_type', 'mutual'),
                        'discovered_at': mutual.get('discovered_at', ''),
                        'mutual_handle': mutual.get('handle', '')
                    }
                    break
            
            # Apply engagement filter
            if enhanced_post.get('like_count', 0) >= collection_config['min_engagement']:
                processed_posts.append(enhanced_post)
        
        logger.info(f"Collected {len(processed_posts)} posts from mutual connections")
        return processed_posts
        
    except Exception as e:
        logger.error(f"Error collecting mutual content: {e}")
        return []


def collect_mutual_posts_legacy(
    user_id: str, 
    client,
    time_hours: float = 24, 
    target_count: int = 500
) -> List[Dict]:
    """
    Legacy function: Collect posts from mutual connections (discovery + collection in one step)
    
    Note: This function maintains the original behavior but it's recommended to use
    the new ETL discovery + ranking collection approach instead.
    
    Args:
        user_id: User's DID
        client: UserData client for discovery, NewPosts client for collection
        time_hours: Time window for post collection in hours
        target_count: Target number of posts to collect
        
    Returns:
        List of posts from mutual connections
    """
    try:
        logger.info(f"Legacy: Collecting mutual posts for user {user_id}")
        
        # Import discovery function for legacy support
        from .discovery import find_mutual_connections
        
        # Step 1: Find mutual connections (should be done in ETL)
        mutuals = find_mutual_connections(user_id, client)
        
        if not mutuals:
            logger.info(f"No mutual connections found for user {user_id}")
            return []
        
        # Step 2: Collect posts using the new collection function
        return collect_mutual_content(
            mutual_connections=mutuals,
            client=client,
            collection_config={
                'max_posts_per_mutual': target_count // len(mutuals) if mutuals else 20,
                'time_hours': time_hours,
                'include_reposts': True,
                'repost_weight': 0.7,
                'min_engagement': 1
            }
        )
        
    except Exception as e:
        logger.error(f"Error in legacy mutual posts collection for user {user_id}: {e}")
        return []


def get_mutual_content_stats(content_list: List[Dict]) -> Dict:
    """Get statistics about collected mutual connection content"""
    if not content_list:
        return {}
    
    mutual_posts = [c for c in content_list if c.get('source') == 'mutual']
    original_posts = [c for c in mutual_posts if c.get('post_type') == 'original']
    repost_posts = [c for c in mutual_posts if c.get('post_type') == 'repost']
    
    total_engagement = sum(
        c.get('like_count', 0) + c.get('repost_count', 0) + c.get('reply_count', 0) 
        for c in mutual_posts
    )
    
    unique_mutuals = len(set(
        c.get('mutual_connection_info', {}).get('mutual_handle', '')
        for c in mutual_posts 
        if c.get('mutual_connection_info', {}).get('mutual_handle')
    ))
    
    return {
        'total_mutual_posts': len(mutual_posts),
        'original_posts': len(original_posts),
        'repost_posts': len(repost_posts),
        'total_engagement': total_engagement,
        'unique_mutuals_contributing': unique_mutuals,
        'avg_engagement_per_post': total_engagement / len(mutual_posts) if mutual_posts else 0
    }


def filter_mutual_content_by_quality(content_list: List[Dict], quality_config: Optional[Dict] = None) -> List[Dict]:
    """Apply quality filters specific to mutual connection content"""
    if quality_config is None:
        quality_config = {
            'min_text_length': 15,           # Minimum characters in post
            'max_age_hours': 48,             # Maximum age of content
            'min_engagement': 2,             # Minimum total engagement
            'exclude_very_popular': False,   # Exclude posts with > 1000 likes (noise)
            'popular_threshold': 1000
        }
    
    filtered_content = []
    
    for content in content_list:
        try:
            # Text length filter
            text = content.get('text', '')
            if len(text.strip()) < quality_config['min_text_length']:
                continue
            
            # Age filter
            if quality_config['max_age_hours']:
                created_at = content.get('created_at', '')
                if created_at:
                    post_time = parser.isoparse(created_at.replace('Z', '+00:00'))
                    age_hours = (datetime.now(post_time.tzinfo) - post_time).total_seconds() / 3600
                    if age_hours > quality_config['max_age_hours']:
                        continue
            
            # Engagement filter
            total_engagement = (
                content.get('like_count', 0) + 
                content.get('repost_count', 0) + 
                content.get('reply_count', 0)
            )
            if total_engagement < quality_config['min_engagement']:
                continue
            
            # Popular content filter (optional)
            if quality_config['exclude_very_popular']:
                if content.get('like_count', 0) > quality_config['popular_threshold']:
                    continue
            
            filtered_content.append(content)
            
        except Exception as e:
            logger.debug(f"Error filtering content {content.get('uri', 'unknown')}: {e}")
            continue
    
    logger.info(f"Quality filtering: {len(content_list)} -> {len(filtered_content)} mutual posts")
    return filtered_content