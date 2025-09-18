"""
Network analysis utilities for mutual connection algorithms
"""
import logging
from typing import Dict, List, Set, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


def analyze_network_overlap(user_follows: List[Dict], user_followers: List[Dict]) -> Dict:
    """
    Analyze overlap between follows and followers to understand network structure
    
    Args:
        user_follows: List of users the target user follows
        user_followers: List of users who follow the target user
    
    Returns:
        Dictionary with network overlap analysis
    """
    if not user_follows or not user_followers:
        return {
            'mutual_count': 0,
            'follow_only_count': len(user_follows) if user_follows else 0,
            'follower_only_count': len(user_followers) if user_followers else 0,
            'reciprocity_ratio': 0.0,
            'network_density': 0.0
        }
    
    # Create sets for efficient operations
    follows_dids = {user['did'] for user in user_follows if user.get('did')}
    followers_dids = {user['did'] for user in user_followers if user.get('did')}
    
    # Calculate overlap metrics
    mutual_dids = follows_dids.intersection(followers_dids)
    follow_only_dids = follows_dids - followers_dids
    follower_only_dids = followers_dids - follows_dids
    
    total_connections = len(follows_dids.union(followers_dids))
    
    analysis = {
        'mutual_count': len(mutual_dids),
        'follow_only_count': len(follow_only_dids),
        'follower_only_count': len(follower_only_dids),
        'total_unique_connections': total_connections,
        'reciprocity_ratio': len(mutual_dids) / len(follows_dids) if follows_dids else 0.0,
        'network_density': len(mutual_dids) / total_connections if total_connections else 0.0,
        'follow_back_ratio': len(mutual_dids) / len(followers_dids) if followers_dids else 0.0
    }
    
    return analysis


def calculate_network_strength(mutual_connections: List[Dict]) -> Dict:
    """
    Calculate network strength metrics based on mutual connections
    
    Args:
        mutual_connections: List of mutual connection user objects
    
    Returns:
        Dictionary with network strength metrics
    """
    if not mutual_connections:
        return {
            'strength_score': 0.0,
            'avg_follower_count': 0,
            'connection_diversity': 0.0,
            'network_influence': 0.0
        }
    
    # Analyze mutual connections' profiles
    follower_counts = []
    handle_domains = []
    
    for mutual in mutual_connections:
        # Extract follower count if available
        follower_count = mutual.get('followersCount', mutual.get('followers_count', 0))
        if follower_count:
            follower_counts.append(follower_count)
        
        # Extract domain diversity from handles
        handle = mutual.get('handle', '')
        if '.' in handle:
            domain = handle.split('.')[-1]
            handle_domains.append(domain)
    
    # Calculate metrics
    avg_follower_count = sum(follower_counts) / len(follower_counts) if follower_counts else 0
    
    # Domain diversity (how many different domains in handles)
    unique_domains = len(set(handle_domains)) if handle_domains else 0
    connection_diversity = unique_domains / len(mutual_connections) if mutual_connections else 0
    
    # Network influence (based on mutual connections' follower counts)
    total_reach = sum(follower_counts) if follower_counts else 0
    network_influence = total_reach / len(mutual_connections) if mutual_connections else 0
    
    # Overall strength score (normalized)
    strength_score = min(1.0, (
        (len(mutual_connections) / 100) * 0.4 +  # Number of mutuals (normalized to 100)
        (connection_diversity) * 0.3 +            # Diversity bonus
        (min(network_influence, 10000) / 10000) * 0.3  # Influence (capped and normalized)
    ))
    
    return {
        'strength_score': round(strength_score, 3),
        'avg_follower_count': int(avg_follower_count),
        'connection_diversity': round(connection_diversity, 3),
        'network_influence': round(network_influence, 2),
        'total_mutual_reach': total_reach
    }


def identify_network_clusters(mutual_connections: List[Dict]) -> Dict:
    """
    Identify clusters or communities within mutual connections
    
    Args:
        mutual_connections: List of mutual connection user objects
    
    Returns:
        Dictionary with cluster analysis
    """
    if not mutual_connections:
        return {'clusters': [], 'cluster_count': 0}
    
    # Simple clustering based on handle domains and display names
    domain_clusters = {}
    name_clusters = {}
    
    for mutual in mutual_connections:
        # Cluster by handle domain
        handle = mutual.get('handle', '')
        if '.' in handle:
            domain = handle.split('.')[-1]
            if domain not in domain_clusters:
                domain_clusters[domain] = []
            domain_clusters[domain].append(mutual)
        
        # Cluster by common words in display names
        display_name = mutual.get('displayName', mutual.get('display_name', ''))
        if display_name and len(display_name) > 3:
            # Extract potential keywords from display name
            words = display_name.lower().split()
            for word in words:
                if len(word) > 3:  # Only consider words longer than 3 chars
                    if word not in name_clusters:
                        name_clusters[word] = []
                    name_clusters[word].append(mutual)
    
    # Find significant clusters (more than 1 member)
    significant_clusters = []
    
    # Domain clusters
    for domain, members in domain_clusters.items():
        if len(members) > 1:
            significant_clusters.append({
                'type': 'domain',
                'identifier': domain,
                'member_count': len(members),
                'members': [m.get('handle', '') for m in members]
            })
    
    # Name clusters (only if multiple people have the same word)
    for word, members in name_clusters.items():
        if len(members) > 1:
            significant_clusters.append({
                'type': 'name_keyword',
                'identifier': word,
                'member_count': len(members),
                'members': [m.get('handle', '') for m in members]
            })
    
    return {
        'clusters': significant_clusters,
        'cluster_count': len(significant_clusters),
        'domain_diversity': len(domain_clusters),
        'largest_cluster_size': max([c['member_count'] for c in significant_clusters]) if significant_clusters else 0
    }


def calculate_mutual_quality_score(mutual_connections: List[Dict]) -> float:
    """
    Calculate a quality score for mutual connections based on various factors
    
    Args:
        mutual_connections: List of mutual connection user objects
    
    Returns:
        Quality score between 0.0 and 1.0
    """
    if not mutual_connections:
        return 0.0
    
    quality_factors = []
    
    # Factor 1: Number of mutual connections (more is better, up to a point)
    count_score = min(1.0, len(mutual_connections) / 50)  # Normalize to 50 mutuals = 1.0
    quality_factors.append(count_score * 0.3)
    
    # Factor 2: Network strength
    strength_metrics = calculate_network_strength(mutual_connections)
    quality_factors.append(strength_metrics['strength_score'] * 0.4)
    
    # Factor 3: Cluster diversity (more diverse connections = higher quality)
    cluster_metrics = identify_network_clusters(mutual_connections)
    diversity_score = min(1.0, cluster_metrics['cluster_count'] / 5)  # Normalize to 5 clusters = 1.0
    quality_factors.append(diversity_score * 0.3)
    
    # Calculate overall quality score
    overall_quality = sum(quality_factors)
    
    return round(overall_quality, 3)


def get_network_health_report(user_id: str, network_data: Dict) -> Dict:
    """
    Generate a comprehensive network health report
    
    Args:
        user_id: User's DID
        network_data: Network data from discovery (mutual_connections, network_stats)
    
    Returns:
        Comprehensive network health analysis
    """
    mutual_connections = network_data.get('mutual_connections', [])
    network_stats = network_data.get('network_stats', {})
    
    # Calculate all metrics
    strength_metrics = calculate_network_strength(mutual_connections)
    cluster_metrics = identify_network_clusters(mutual_connections)
    quality_score = calculate_mutual_quality_score(mutual_connections)
    
    # Generate health assessment
    health_status = "poor"
    if quality_score >= 0.7:
        health_status = "excellent"
    elif quality_score >= 0.5:
        health_status = "good"
    elif quality_score >= 0.3:
        health_status = "fair"
    
    # Recommendations
    recommendations = []
    if len(mutual_connections) < 10:
        recommendations.append("Consider following more users who might follow back")
    if strength_metrics['connection_diversity'] < 0.3:
        recommendations.append("Diversify connections across different communities")
    if cluster_metrics['cluster_count'] < 2:
        recommendations.append("Engage with users from different interest areas")
    
    report = {
        'user_id': user_id,
        'health_status': health_status,
        'quality_score': quality_score,
        'mutual_count': len(mutual_connections),
        'network_stats': network_stats,
        'strength_metrics': strength_metrics,
        'cluster_analysis': cluster_metrics,
        'recommendations': recommendations,
        'generated_at': logger.handlers[0].formatter.formatTime(logger.makeRecord('', 0, '', 0, '', (), None)) if logger.handlers else ''
    }
    
    return report