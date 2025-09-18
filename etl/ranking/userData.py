"""
User data retrieval and management functions
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def get_active_users_with_data(redis_client, bq_client, test_mode: bool = False) -> List[Dict]:
    """
    Get active users with their data from Redis + BigQuery
    
    Args:
        redis_client: Redis client for activity tracking
        bq_client: BigQuery client for user data
        test_mode: Limit users for testing
        
    Returns:
        List of user dictionaries with basic data
    """
    try:
        # Get active users from Redis (fast, real-time activity)
        active_user_ids = redis_client.get_active_users(days=30)
        
        if not active_user_ids:
            logger.warning("No active users found in Redis, falling back to BigQuery")
            return get_users_from_bigquery_fallback(bq_client, test_mode)
        
        # Limit in test mode
        if test_mode:
            active_user_ids = active_user_ids[:5]
        
        logger.info(f"Found {len(active_user_ids)} active users from Redis")
        
        # Get user data for active users from BigQuery
        if not active_user_ids:
            return []
        
        # Create parameterized query for active users
        user_ids_str = "', '".join(active_user_ids)
        query = f"""
        SELECT 
            user_id,
            handle
        FROM `{bq_client.project_id}.data.users`
        WHERE user_id IN ('{user_ids_str}')
        """
        
        result = bq_client.query(query)
        users = result.to_dict('records') if not result.empty else []
        
        logger.info(f"Retrieved data for {len(users)} active users from BigQuery")
        return users
        
    except Exception as e:
        logger.error(f"Error getting active users: {e}")
        logger.warning("Falling back to BigQuery-only approach")
        return get_users_from_bigquery_fallback(bq_client, test_mode)


def get_users_from_bigquery_fallback(bq_client, test_mode: bool = False) -> List[Dict]:
    """Fallback: Get active users from BigQuery only"""
    try:
        # Only limit in test mode for development
        limit_clause = "LIMIT 5" if test_mode else ""
        
        query = f"""
        SELECT 
            user_id,
            handle
        FROM `{bq_client.project_id}.data.users`
        WHERE last_request_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY last_request_at DESC
        {limit_clause}
        """
        
        result = bq_client.query(query)
        users = result.to_dict('records') if not result.empty else []
        
        logger.info(f"Retrieved {len(users)} active users from BigQuery (fallback)")
        return users
        
    except Exception as e:
        logger.warning(f"Could not get users from BigQuery: {e}")
        return []


def get_user_profile_data(user_id: str, bq_client) -> Optional[Dict]:
    """
    Get user profile data for ranking
    TODO: Implement based on what data your algorithm needs
    
    Args:
        user_id: User's DID
        bq_client: BigQuery client
        
    Returns:
        User profile data or None
    """
    try:
        # TODO: Query whatever user data your algorithm needs
        # Examples: preferences, keywords, embeddings, behavior patterns, etc.
        
        query = f"""
        SELECT 
            user_id,
            handle
        FROM `{bq_client.project_id}.data.users`
        WHERE user_id = '{user_id}'
        LIMIT 1
        """
        
        result = bq_client.query(query)
        
        if result.empty:
            logger.warning(f"No profile data found for user {user_id}")
            return None
        
        profile_data = result.iloc[0].to_dict()
        logger.debug(f"Retrieved profile data for user {user_id}")
        return profile_data
        
    except Exception as e:
        logger.error(f"Failed to retrieve profile data for user {user_id}: {e}")
        return None