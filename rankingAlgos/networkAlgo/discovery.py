"""
Network discovery for finding mutual connections during ETL processing.
Core idea: Find users with bidirectional follow relationships (mutual connections).
"""
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def find_mutual_connections(user_id: str, client) -> List[Dict]:
    """
    Find users who follow each other (bidirectional relationships)
    
    Args:
        user_id: User's DID to find mutuals for
        client: UserData client for API calls
        
    Returns:
        List of mutual connection user objects with metadata
    """
    try:
        logger.info(f"Finding mutual connections for user {user_id}")
        
        # Get who the user follows
        user_follows = client.get_all_user_follows(user_id)
        if not user_follows:
            logger.warning(f"User {user_id} follows no one")
            return []
        
        # Get who follows the user
        user_followers = client.get_all_user_followers(user_id)
        if not user_followers:
            logger.warning(f"User {user_id} has no followers")
            return []
        
        # Create sets of DIDs for efficient intersection
        follows_dids = {user['did'] for user in user_follows if user.get('did')}
        followers_dids = {user['did'] for user in user_followers if user.get('did')}
        
        # Find mutual DIDs (intersection)
        mutual_dids = follows_dids.intersection(followers_dids)
        
        if not mutual_dids:
            logger.info(f"User {user_id} has no mutual connections")
            return []
        
        # Get full user objects for mutuals with metadata
        mutuals = []
        for user in user_follows:
            if user.get('did') in mutual_dids:
                mutual_user = user.copy()
                mutual_user['connection_type'] = 'mutual'
                mutual_user['discovered_at'] = user.get('indexed_at', '')
                mutuals.append(mutual_user)
        
        logger.info(f"Found {len(mutuals)} mutual connections for user {user_id}")
        logger.info(f"Network stats: {len(user_follows)} follows, {len(user_followers)} followers, {len(mutuals)} mutuals")
        
        return mutuals
        
    except Exception as e:
        logger.error(f"Error finding mutual connections for user {user_id}: {e}")
        return []


def discover_network_relationships(user_id: str, client) -> Dict:
    """
    Discover comprehensive network relationships for a user during ETL
    
    Args:
        user_id: User's DID
        client: UserData client for API calls
        
    Returns:
        Dictionary with mutual connections and network statistics
    """
    try:
        logger.info(f"Discovering network relationships for user {user_id}")
        
        # Find mutual connections
        mutual_connections = find_mutual_connections(user_id, client)
        
        # Calculate network statistics
        network_stats = _calculate_network_stats(user_id, client)
        
        # Combine results
        network_data = {
            'mutual_connections': mutual_connections,
            'network_stats': network_stats,
            'discovery_timestamp': logger.handlers[0].formatter.formatTime(logger.makeRecord('', 0, '', 0, '', (), None)) if logger.handlers else ''
        }
        
        logger.info(f"Network discovery complete for user {user_id}: {len(mutual_connections)} mutuals found")
        return network_data
        
    except Exception as e:
        logger.error(f"Error discovering network relationships for user {user_id}: {e}")
        return {
            'mutual_connections': [],
            'network_stats': {},
            'discovery_timestamp': '',
            'error': str(e)
        }


def _calculate_network_stats(user_id: str, client) -> Dict:
    """Calculate network statistics for a user"""
    try:
        # Get follow/follower data
        user_follows = client.get_all_user_follows(user_id)
        user_followers = client.get_all_user_followers(user_id)
        
        follows_count = len(user_follows) if user_follows else 0
        followers_count = len(user_followers) if user_followers else 0
        
        # Calculate mutual count
        if follows_count > 0 and followers_count > 0:
            follows_dids = {user['did'] for user in user_follows if user.get('did')}
            followers_dids = {user['did'] for user in user_followers if user.get('did')}
            mutuals_count = len(follows_dids.intersection(followers_dids))
        else:
            mutuals_count = 0
        
        # Calculate ratios
        mutual_to_follows_ratio = mutuals_count / follows_count if follows_count > 0 else 0
        mutual_to_followers_ratio = mutuals_count / followers_count if followers_count > 0 else 0
        
        stats = {
            'total_follows': follows_count,
            'total_followers': followers_count,
            'mutual_connections': mutuals_count,
            'mutual_to_follows_ratio': round(mutual_to_follows_ratio, 3),
            'mutual_to_followers_ratio': round(mutual_to_followers_ratio, 3),
            'network_reciprocity': round(mutual_to_follows_ratio, 3)  # How much of their following is reciprocated
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating network stats for user {user_id}: {e}")
        return {
            'total_follows': 0,
            'total_followers': 0,
            'mutual_connections': 0,
            'mutual_to_follows_ratio': 0,
            'mutual_to_followers_ratio': 0,
            'network_reciprocity': 0,
            'error': str(e)
        }