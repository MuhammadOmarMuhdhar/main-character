import os
import sys
import argparse
import json
import logging
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from client.userData import Client as BlueskyUserDataClient
from client.bigQuery import Client as BigQueryClient
from .timeUtils import is_update_time
from .queryManager import UserQueryManager
from .profileCollector import UserProfileCollector
from .dataProcessor import UserDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UserDiscoveryOrchestrator:
    def __init__(self):
        self.bq_client = None
        self.bluesky_client = None
        self.query_manager = None
        self.profile_collector = None
        self.data_processor = None
    
    def initialize_clients(self):
        """Initialize all required clients"""
        try:
            # Initialize BigQuery client
            credentials_json = json.loads(os.environ['BIGQUERY_CREDENTIALS_JSON'])
            self.bq_client = BigQueryClient(credentials_json, os.environ['BIGQUERY_PROJECT_ID'])
            
            # Initialize Bluesky client
            self.bluesky_client = BlueskyUserDataClient()
            self.bluesky_client.login()
            
            # Initialize managers
            self.query_manager = UserQueryManager(self.bq_client)
            self.profile_collector = UserProfileCollector(self.bluesky_client)
            self.data_processor = UserDataProcessor()
            
            logger.info("All clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    def process_user(self, user_request: dict, is_bulk_update: bool, dry_run: bool) -> bool:
        """Process a single user and return success status"""
        user_did = user_request['user_did']
        
        try:
            if is_bulk_update:
                logger.info(f"Bulk updating user: {user_request.get('handle', user_did)}")
                # For bulk updates, use existing handle data
                user_profile = {
                    'did': user_did,
                    'handle': user_request.get('handle', '')
                }
            else:
                logger.info(f"Processing new user: {user_did}")
                # Get user profile for new users
                user_profile = self.profile_collector.get_user_profile_from_did(user_did)
                if not user_profile:
                    logger.warning(f"Could not get profile for user {user_did}")
                    return False
            
            # Collect user engagement data
            user_handle = user_profile.get('handle', user_did)
            user_engagement_data = self.profile_collector.collect_user_engagement_data(
                user_handle, user_request.get('app_password')
            )
            if not user_engagement_data:
                logger.warning(f"Could not collect engagement data for {user_handle}")
                return False
            
            # Process user data with client for neighborhood discovery
            processed_user = self.data_processor.process_user_data(user_did, user_profile, user_engagement_data, self.bluesky_client)
            if not processed_user:
                logger.warning(f"Could not process user data for {user_handle}")
                return False
            
            # Update user profile in BigQuery
            if not dry_run:
                batch_id = f"user_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if self.query_manager.update_user_profile_in_bigquery(processed_user, batch_id):
                    logger.info(f"Successfully updated user profile for {processed_user['handle']}")
                    return True
                else:
                    logger.error(f"Failed to update user profile for {processed_user['handle']}")
                    return False
            else:
                logger.info(f"DRY RUN: Would update user {processed_user['handle']} with {len(processed_user['keywords'])} keywords")
                return True
                
        except Exception as e:
            logger.error(f"Error processing user {user_did}: {e}")
            return False
    
    def run_etl(self, hours_back: int = 1, dry_run: bool = False, bulk_update: bool = False):
        """Main ETL execution"""
        batch_id = f"user_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        since_timestamp = datetime.now() - timedelta(hours=hours_back)
        
        # Check if it's time for daily bulk updates or manual bulk update requested
        is_auto_bulk_update_time = is_update_time()
        is_manual_bulk_update = bulk_update
        is_bulk_update_mode = is_auto_bulk_update_time or is_manual_bulk_update
        
        if is_manual_bulk_update:
            logger.info(f"Starting Manual Bulk Update ETL (batch_id={batch_id})")
            logger.info("Manually updating ALL users' keywords and embeddings")
        elif is_auto_bulk_update_time:
            logger.info(f"Starting Daily Bulk Update ETL (batch_id={batch_id})")
            logger.info("Updating ALL users' keywords and embeddings at 2 AM Pacific")
        else:
            logger.info(f"Starting User Discovery ETL (batch_id={batch_id})")
            logger.info(f"Processing logs since: {since_timestamp}")
        
        try:
            self.initialize_clients()
            
            # Choose workflow based on time or manual flag
            if is_bulk_update_mode:
                # Get all users for bulk update
                users_to_process = self.query_manager.get_all_users_for_update()
            else:
                # Get users needing profile data from BigQuery
                users_to_process = self.query_manager.get_users_needing_profile_data(since_timestamp)
            
            if not users_to_process:
                if is_bulk_update_mode:
                    logger.info("No users found for bulk update")
                else:
                    logger.info("No users needing profile data found")
                return
            
            if is_bulk_update_mode:
                logger.info(f"Found {len(users_to_process)} users for bulk update")
            else:
                logger.info(f"Found {len(users_to_process)} users needing profile data")
            
            success_count = 0
            error_count = 0
            
            for user_request in users_to_process:
                if self.process_user(user_request, is_bulk_update_mode, dry_run):
                    success_count += 1
                else:
                    error_count += 1
            
            # Final summary
            if is_manual_bulk_update:
                logger.info(f"Manual Bulk Update ETL Complete!")
                logger.info(f"Success: {success_count}, Errors: {error_count}")
                logger.info(f"All users' keywords and embeddings manually refreshed")
            elif is_auto_bulk_update_time:
                logger.info(f"Daily Bulk Update ETL Complete!")
                logger.info(f"Success: {success_count}, Errors: {error_count}")
                logger.info(f"All users' keywords and embeddings refreshed")
            else:
                logger.info(f"User Discovery ETL Complete!")
                logger.info(f"Success: {success_count}, Errors: {error_count}")
                logger.info(f"New active users ready for feed ranking")
            
        except Exception as e:
            logger.error(f"User Discovery ETL failed: {e}")
            sys.exit(1)


def main():
    """Main user discovery ETL process"""
    parser = argparse.ArgumentParser(description='User Discovery ETL')
    parser.add_argument('--log-file', default='/var/log/feed-server.log', help='Path to feed server log file')
    parser.add_argument('--hours-back', type=int, default=1, help='Hours to look back in logs')
    parser.add_argument('--dry-run', action='store_true', help='Run without storing to BigQuery')
    parser.add_argument('--bulk-update', action='store_true', help='Manually update all users keywords/embeddings regardless of time')
    
    args = parser.parse_args()
    
    orchestrator = UserDiscoveryOrchestrator()
    orchestrator.run_etl(
        hours_back=args.hours_back,
        dry_run=args.dry_run,
        bulk_update=args.bulk_update
    )


if __name__ == "__main__":
    main()