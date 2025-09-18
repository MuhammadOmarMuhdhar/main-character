"""
Main ETL orchestrator that coordinates all ranking modules
"""
import os
import sys
import argparse
import json
import logging
from datetime import datetime

# Add parent directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
env_file_path = os.path.join(project_root, '.env')
if os.path.exists(env_file_path):
    with open(env_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

from client.bigQuery import Client as BigQueryClient
from client.redis import Client as RedisClient

from ETL.ranking.userData import get_active_users_with_data, get_user_profile_data
from ETL.ranking.postCollector import collect_posts_for_user
from ETL.ranking.rankingEngine import rank_posts_for_user, apply_post_filters
from ETL.ranking.cacheManager import cache_user_rankings, update_default_feed_if_needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_user_feed(user_id: str, user_handle: str, bq_client, redis_client, bluesky_client=None) -> bool:
    """
    Process feed generation for a single user
    
    Args:
        user_id: User's DID
        user_handle: User's handle for logging
        bq_client: BigQuery client
        redis_client: Redis client
        
    Returns:
        True if processing succeeded
    """
    try:
        logger.info(f"Processing user: {user_handle}")
        
        # Step 1: Get user profile data
        user_data = get_user_profile_data(user_id, bq_client)
        if not user_data:
            logger.warning(f"No profile data found for {user_handle}, using defaults")
            user_data = {'user_id': user_id, 'handle': user_handle}
        
        # Step 2: Collect posts for ranking
        posts_to_rank = collect_posts_for_user(user_id, user_data, bq_client, redis_client, bluesky_client)
        
        if not posts_to_rank:
            logger.warning(f"No posts collected for {user_handle}, skipping")
            return False
        
        # Step 3: Apply content filters
        filtered_posts = apply_post_filters(posts_to_rank)
        
        if not filtered_posts:
            logger.warning(f"No posts remaining after filtering for {user_handle}, skipping")
            return False
        
        # Step 4: Rank posts using algorithm
        ranked_posts = rank_posts_for_user(user_id, user_data, filtered_posts)
        
        if not ranked_posts:
            logger.warning(f"No rankings generated for {user_handle}, skipping")
            return False
        
        # Step 5: Cache results
        cache_success = cache_user_rankings(redis_client, user_id, ranked_posts)
        
        if cache_success:
            logger.info(f"Successfully processed {user_handle} - {len(ranked_posts)} posts ranked and cached")
        else:
            logger.error(f"Failed to cache results for {user_handle}")
        
        return cache_success
        
    except Exception as e:
        logger.error(f"Error processing {user_handle}: {e}")
        return False


def main():
    """Main ETL orchestration process"""
    parser = argparse.ArgumentParser(description='Feed Ranking ETL - Clean Slate Version')
    parser.add_argument('--test-mode', default='false', help='Run in test mode with limited users')
    args = parser.parse_args()
    
    test_mode = args.test_mode.lower() == 'true'
    batch_id = f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting Feed Ranking ETL (test_mode={test_mode}, batch_id={batch_id})")
    logger.info("=== CLEAN SLATE VERSION - Ready for new algorithm implementation ===")
    
    try:
        # Initialize clients
        credentials_json = json.loads(os.environ['BIGQUERY_CREDENTIALS_JSON'])
        bq_client = BigQueryClient(credentials_json, os.environ['BIGQUERY_PROJECT_ID'])
        redis_client = RedisClient()
        
        # Get users to process
        users = get_active_users_with_data(redis_client, bq_client, test_mode)
        
        if not users:
            logger.warning("No users found to process, updating default feed only")
            success_count, error_count = 0, 0
        else:
            logger.info(f"Processing {len(users)} users")
            success_count = 0
            error_count = 0
        
        # Process each user
        for user in users:
            user_handle = user.get('handle', '')
            user_id = user.get('user_id', user_handle)
            
            if process_user_feed(user_id, user_handle, bq_client, redis_client, None):
                success_count += 1
            else:
                error_count += 1
        
        # Update default feed for new users
        try:
            update_default_feed_if_needed(redis_client)
        except Exception as e:
            logger.error(f"Failed to update default feed: {e}")
        
        # Final summary
        logger.info(f"ETL Complete! Success: {success_count}, Errors: {error_count}")
        
        # Show cache stats
        try:
            stats = redis_client.get_stats()
            cached_users = redis_client.get_cached_users()
            logger.info(f"Redis Stats: {stats}")
            logger.info(f"Total cached feeds: {len(cached_users)}")
        except Exception as e:
            logger.warning(f"Could not retrieve cache stats: {e}")
        
        logger.info("=== ETL run complete - ready for algorithm implementation ===")
        
    except Exception as e:
        logger.error(f"ETL failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()