"""
Neighborhood Algorithm for discovering taste neighbors based on multi-signal interactions.
Core idea: Find users who liked/engaged with the same content as the target user.
"""
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dateutil import parser

logger = logging.getLogger(__name__)


def discover_taste_neighbors(
    user_data: Dict,
    client,
    signal_weights: Optional[Dict[str, float]] = None,
    min_overlap_threshold: int = 3,
    max_neighbors: int = 50,
    time_hours: int = 168  # 7 days
) -> List[Dict]:
    """
    Discover taste neighbors based on multi-signal interactions
    
    Args:
        user_data: User's engagement data with posts, reposts, likes, replies
        client: Bluesky client for API calls
        signal_weights: Weight for each signal type
        min_overlap_threshold: Minimum shared interactions to qualify as neighbor
        max_neighbors: Maximum neighbors to return
        time_hours: Hours back to consider for recency
    
    Returns:
        List of neighbor dicts with DID and similarity scores
    """
    if signal_weights is None:
        signal_weights = {
            'repost': 1.0,    # Highest intent - user amplifies content
            'like': 0.6,      # Medium intent - approval signal
            'reply': 0.8,     # High engagement - active participation  
            'quote_post': 0.9 # Very high - adds commentary
        }
    
    logger.info(f"Starting neighborhood discovery with {min_overlap_threshold} min overlap threshold")
    
    try:
        # Collect all interaction data for the user's content
        interaction_data = _collect_user_interaction_data(user_data, client, signal_weights, time_hours)
        
        if not interaction_data:
            logger.warning("No interaction data collected, no neighbors found")
            return []
        
        # Find potential neighbors based on interaction overlaps
        neighbor_scores = _calculate_neighbor_scores(interaction_data, signal_weights, min_overlap_threshold)
        
        if not neighbor_scores:
            logger.warning("No neighbors found meeting overlap threshold")
            return []
        
        # Sort neighbors by score and return top results
        sorted_neighbors = sorted(neighbor_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        top_neighbors = sorted_neighbors[:max_neighbors]
        
        # Format results
        result = []
        for neighbor_did, score_data in top_neighbors:
            result.append({
                'did': neighbor_did,
                'similarity_score': score_data['total_score'],
                'shared_interactions': score_data['interaction_count'],
                'signal_breakdown': score_data['signal_breakdown']
            })
        
        logger.info(f"Discovered {len(result)} taste neighbors for user")
        return result
        
    except Exception as e:
        logger.error(f"Error in neighborhood discovery: {e}")
        return []


def _collect_user_interaction_data(user_data: Dict, client, signal_weights: Dict, time_hours: int) -> Dict:
    """
    Collect interaction data for all user's content across different signal types
    
    Returns:
        Dict mapping content_uri -> {signal_type -> [list of user_dids]}
    """
    interaction_data = defaultdict(lambda: defaultdict(list))
    cutoff_time = datetime.now() - timedelta(hours=time_hours)
    
    # Process reposts (highest priority signal)
    if 'repost' in signal_weights and user_data.get('reposts'):
        logger.info(f"Processing {len(user_data['reposts'])} reposts")
        for repost in user_data['reposts']:
            content_uri = repost.get('uri')
            if content_uri and _is_recent_enough(repost.get('indexed_at'), cutoff_time):
                # Get users who liked this reposted content
                likers = _get_content_likers(content_uri, client)
                if likers:
                    interaction_data[content_uri]['repost_likers'].extend(likers)
    
    # Process likes (if available)
    if 'like' in signal_weights and user_data.get('likes'):
        logger.info(f"Processing {len(user_data['likes'])} likes")
        for like in user_data['likes']:
            content_uri = like.get('subject', {}).get('uri')
            if content_uri and _is_recent_enough(like.get('indexed_at'), cutoff_time):
                # Get other users who also liked this content
                other_likers = _get_content_likers(content_uri, client)
                if other_likers:
                    interaction_data[content_uri]['like_overlap'].extend(other_likers)
    
    # Process replies (high engagement signal)
    if 'reply' in signal_weights and user_data.get('replies'):
        logger.info(f"Processing {len(user_data['replies'])} replies")
        for reply in user_data['replies']:
            # Get the parent post URI that was replied to
            parent_uri = reply.get('record', {}).get('reply', {}).get('parent', {}).get('uri')
            if parent_uri and _is_recent_enough(reply.get('indexed_at'), cutoff_time):
                # Get users who also engaged with the parent post
                engagers = _get_content_likers(parent_uri, client)
                if engagers:
                    interaction_data[parent_uri]['reply_context'].extend(engagers)
    
    # Process user's own posts to see who engaged with them
    if user_data.get('posts'):
        logger.info(f"Processing {len(user_data['posts'])} user posts")
        for post in user_data['posts']:
            content_uri = post.get('uri')
            if content_uri and _is_recent_enough(post.get('indexed_at'), cutoff_time):
                # Get users who liked the user's own posts
                likers = _get_content_likers(content_uri, client)
                if likers:
                    interaction_data[content_uri]['post_likers'].extend(likers)
    
    logger.info(f"Collected interaction data for {len(interaction_data)} content pieces")
    return dict(interaction_data)


def _get_content_likers(content_uri: str, client, limit: int = 100) -> List[str]:
    """
    Get list of users who liked specific content
    
    Args:
        content_uri: AT-URI of the content
        client: Bluesky client
        limit: Maximum number of likers to retrieve
    
    Returns:
        List of user DIDs who liked the content
    """
    try:
        # Use getLikes API to get users who liked this content
        response = client.client.app.bsky.feed.get_likes({
            'uri': content_uri,
            'limit': min(limit, 100)  # API limit
        })
        
        likers = []
        for like in response.likes:
            actor_did = like.actor.did
            if actor_did:
                likers.append(actor_did)
        
        return likers
        
    except Exception as e:
        logger.debug(f"Failed to get likers for {content_uri}: {e}")
        return []


def _calculate_neighbor_scores(
    interaction_data: Dict, 
    signal_weights: Dict, 
    min_overlap_threshold: int
) -> Dict[str, Dict]:
    """
    Calculate similarity scores for potential neighbors based on interaction overlaps
    
    Returns:
        Dict mapping neighbor_did -> {total_score, interaction_count, signal_breakdown}
    """
    neighbor_scores = defaultdict(lambda: {
        'total_score': 0.0,
        'interaction_count': 0,
        'signal_breakdown': defaultdict(int)
    })
    
    # Count interactions for each potential neighbor
    for content_uri, signals in interaction_data.items():
        for signal_type, user_dids in signals.items():
            # Determine signal weight based on type
            if 'repost' in signal_type:
                weight = signal_weights.get('repost', 1.0)
            elif 'like' in signal_type:
                weight = signal_weights.get('like', 0.6)
            elif 'reply' in signal_type:
                weight = signal_weights.get('reply', 0.8)
            else:
                weight = 0.5  # Default weight for unknown signals
            
            # Count occurrences of each user
            user_counts = Counter(user_dids)
            
            for user_did, count in user_counts.items():
                if user_did:  # Ensure valid DID
                    # Add weighted score based on interaction frequency
                    score_increment = weight * count
                    neighbor_scores[user_did]['total_score'] += score_increment
                    neighbor_scores[user_did]['interaction_count'] += count
                    neighbor_scores[user_did]['signal_breakdown'][signal_type] += count
    
    # Filter neighbors that meet minimum overlap threshold
    qualified_neighbors = {}
    for neighbor_did, score_data in neighbor_scores.items():
        if score_data['interaction_count'] >= min_overlap_threshold:
            # Convert defaultdict to regular dict for JSON serialization
            score_data['signal_breakdown'] = dict(score_data['signal_breakdown'])
            qualified_neighbors[neighbor_did] = score_data
    
    logger.info(f"Found {len(qualified_neighbors)} neighbors meeting threshold of {min_overlap_threshold}")
    return qualified_neighbors


def _is_recent_enough(indexed_at: str, cutoff_time: datetime) -> bool:
    """Check if content is recent enough to be considered"""
    if not indexed_at:
        return False
    
    try:
        content_time = parser.isoparse(indexed_at.replace('Z', '+00:00'))
        return content_time.replace(tzinfo=None) >= cutoff_time
    except Exception:
        return False


def _apply_time_decay(score: float, content_age_hours: float) -> float:
    """Apply time decay to reduce weight of older interactions"""
    if content_age_hours <= 24:
        return score  # Full weight for last 24 hours
    elif content_age_hours <= 72:
        return score * 0.8  # 80% weight for last 3 days
    elif content_age_hours <= 168:
        return score * 0.6  # 60% weight for last week
    else:
        return score * 0.3  # 30% weight for older content