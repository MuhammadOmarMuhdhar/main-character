"""
Cache management for user feeds and default content
"""
import logging
from typing import Dict, List
from datetime import datetime

from ETL.ranking.config import CACHE_TTL_MINUTES, DEFAULT_FEED_TTL_HOURS, MAX_POSTS_PER_USER

logger = logging.getLogger(__name__)


def cache_user_rankings(redis_client, user_id: str, ranked_posts: List[Dict]) -> bool:
    """
    Cache user's ranked feed in Redis
    
    Args:
        redis_client: Redis client instance
        user_id: User's DID
        ranked_posts: Sorted list of posts with scores
        
    Returns:
        True if caching succeeded
    """
    try:
        if not ranked_posts:
            logger.warning(f"No posts to cache for user {user_id}")
            return False
        
        # Get existing cache to preserve unconsumed posts
        existing_feed = redis_client.get_user_feed(user_id) or []
        existing_posts = {post.get('post_uri', post.get('uri', '')): post for post in existing_feed} if existing_feed else {}
        
        # Format posts for feed server with minimal required fields
        new_posts = []
        for post in ranked_posts:
            post_uri = post.get('uri', '')
            if not post_uri:
                continue
                
            # Store only essential fields for feed server
            formatted_post = {
                'post_uri': post_uri,
                'uri': post_uri,  # Required for consumption tracking
                'score': float(post.get('ranking_score', 0)),
                'post_type': post.get('post_type', 'original'),
                'followed_user': post.get('followed_user', None)
            }
            new_posts.append(formatted_post)
        
        # Merge new posts with existing unconsumed posts (deduplication via URI)
        merged_posts = {}
        duplicate_count = 0
        
        # Add new posts (they get priority with fresh scores)
        for post in new_posts:
            uri = post['post_uri']
            if uri in existing_posts:
                duplicate_count += 1
            merged_posts[uri] = post  # Always use new score for duplicates
        
        # Add existing posts that aren't in new batch (preserve unconsumed)
        for uri, post_data in existing_posts.items():
            if uri not in merged_posts:
                # Preserve existing post but extract only essential fields
                if isinstance(post_data, dict) and ('score' in post_data):
                    score = post_data.get('score', 0)
                    merged_posts[uri] = {
                        'post_uri': uri,
                        'uri': uri,
                        'score': float(score),
                        'post_type': post_data.get('post_type', 'original'),
                        'followed_user': post_data.get('followed_user', None)
                    }
        
        # Sort by score and limit total posts
        final_feed = sorted(merged_posts.values(), key=lambda x: x['score'], reverse=True)[:MAX_POSTS_PER_USER]
        
        if duplicate_count > 0:
            logger.info(f"Deduplication: {duplicate_count} duplicate posts found and updated with fresh scores")
        
        # Cache with TTL
        ttl_seconds = CACHE_TTL_MINUTES * 60
        success = redis_client.set_user_feed(user_id, final_feed, ttl=ttl_seconds)
        
        if success:
            new_count = len(new_posts)
            total_count = len(final_feed)
            logger.info(f"Cached {total_count} posts for user {user_id} ({new_count} new)")
        else:
            logger.error(f"Failed to cache posts for user {user_id}")
            
        return success
        
    except Exception as e:
        logger.error(f"Failed to cache rankings for {user_id}: {e}")
        return False


def update_default_feed_if_needed(redis_client):
    """
    Update default feed if expired or missing
    
    Args:
        redis_client: Redis client instance
    """
    try:
        # Check if default feed exists and is fresh
        existing_default = redis_client.get_default_feed()
        if existing_default:
            logger.info("Default feed already cached and fresh, skipping update")
            return
        
        logger.info("Default feed expired or missing, generating new default feed...")
        
        # TODO: Implement default feed generation
        # This could use trending posts, popular content, etc.
        default_posts = generate_default_feed_content()
        
        if not default_posts:
            logger.warning("No posts generated for default feed")
            return
        
        # Cache for specified hours
        ttl_seconds = DEFAULT_FEED_TTL_HOURS * 3600
        success = redis_client.set_default_feed(default_posts, ttl=ttl_seconds)
        
        if success:
            logger.info(f"Successfully updated default feed with {len(default_posts)} posts")
        else:
            logger.error("Failed to cache default feed")
            
    except Exception as e:
        logger.error(f"Error updating default feed: {e}")


def generate_default_feed_content() -> List[Dict]:
    """
    Generate content for default feed shown to new users
    TODO: Implement default content generation
    
    Returns:
        List of posts for default feed
    """
    try:
        # TODO: Implement default feed content generation
        # This could include:
        # - Trending posts from last 24 hours
        # - Popular posts across the network
        # - Curated high-quality content
        # - Posts from verified/trusted accounts
        # - etc.
        
        default_posts = []
        
        # PLACEHOLDER: Return empty list for now
        logger.info(f"Generated {len(default_posts)} posts for default feed")
        return default_posts
        
    except Exception as e:
        logger.error(f"Failed to generate default feed content: {e}")
        return []


def clear_user_cache(redis_client, user_id: str) -> bool:
    """
    Clear cached data for a specific user
    
    Args:
        redis_client: Redis client instance
        user_id: User's DID
        
    Returns:
        True if clearing succeeded
    """
    try:
        # Clear user's feed cache
        success = redis_client.delete_user_feed(user_id)
        
        if success:
            logger.info(f"Cleared cache for user {user_id}")
        else:
            logger.warning(f"Failed to clear cache for user {user_id}")
            
        return success
        
    except Exception as e:
        logger.error(f"Error clearing cache for user {user_id}: {e}")
        return False