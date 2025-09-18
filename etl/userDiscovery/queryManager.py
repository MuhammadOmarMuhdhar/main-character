import logging
from datetime import datetime
from typing import Dict, List
from google.cloud import bigquery

logger = logging.getLogger(__name__)


class UserQueryManager:
    def __init__(self, bq_client):
        self.bq_client = bq_client
    
    def get_users_needing_profile_data(self, since_timestamp: datetime) -> List[Dict]:
        """
        Get users from BigQuery who need their profile data populated
        
        Args:
            since_timestamp: Only process users who made requests after this time
            
        Returns:
            List of user data that needs profile information
        """
        try:
            query = f"""
            SELECT DISTINCT user_id, last_request_at, request_count, password
            FROM `{self.bq_client.project_id}.data.users`
            WHERE (handle = '' OR handle IS NULL) OR embeddings IS NULL
            AND last_request_at >= '{since_timestamp.isoformat()}'
            AND user_id LIKE 'did:plc:%'
            ORDER BY last_request_at DESC
            """
            
            result = self.bq_client.query(query)
            
            if result.empty:
                logger.info("No users needing profile data found")
                return []
            
            users_needing_data = []
            for _, row in result.iterrows():
                users_needing_data.append({
                    'user_did': row['user_id'],
                    'request_count': row['request_count'],
                    'app_password': row.get('password', '')
                })
            
            logger.info(f"Found {len(users_needing_data)} users needing profile data")
            return users_needing_data
            
        except Exception as e:
            logger.error(f"Failed to get users needing profile data: {e}")
            return []

    def get_existing_users_from_bigquery(self) -> set:
        """Get set of existing user DIDs from BigQuery"""
        try:
            query = f"""
            SELECT user_id
            FROM `{self.bq_client.project_id}.data.users`
            """
            
            result = self.bq_client.query(query)
            existing_dids = set(result['user_id'].tolist()) if not result.empty else set()
            
            logger.info(f"Found {len(existing_dids)} existing users in BigQuery")
            return existing_dids
            
        except Exception as e:
            logger.warning(f"Failed to get existing users from BigQuery: {e}")
            return set()

    def get_all_users_for_update(self) -> List[Dict]:
        """
        Get all users from BigQuery for daily bulk updates
        
        Returns:
            List of all user data for bulk updates
        """
        try:
            query = f"""
            SELECT user_id, handle, password
            FROM `{self.bq_client.project_id}.data.users`
            WHERE user_id LIKE 'did:plc:%'
            AND handle IS NOT NULL
            AND handle != ''
            ORDER BY updated_at ASC
            """
            
            result = self.bq_client.query(query)
            
            if result.empty:
                logger.info("No users found for bulk update")
                return []
            
            users_for_update = []
            for _, row in result.iterrows():
                users_for_update.append({
                    'user_did': row['user_id'],
                    'handle': row.get('handle', ''),
                    'app_password': row.get('password', '')
                })
            
            logger.info(f"Found {len(users_for_update)} users for bulk update")
            return users_for_update
            
        except Exception as e:
            logger.error(f"Failed to get users for bulk update: {e}")
            return []

    def update_user_profile_in_bigquery(self, user_data: Dict, batch_id: str) -> bool:
        """Update existing user record with profile data in BigQuery"""
        try:
            # Use parameterized query to properly handle JSON
            update_query = f"""
            UPDATE `{self.bq_client.project_id}.data.users`
            SET handle = @handle,
                keywords = @keywords,
                embeddings = @embeddings,
                reading_level = @reading_level,
                updated_at = CURRENT_TIMESTAMP()
            WHERE user_id = @user_id
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("handle", "STRING", user_data['handle']),
                    bigquery.ScalarQueryParameter("keywords", "JSON", user_data['keywords']),
                    bigquery.ScalarQueryParameter("embeddings", "JSON", user_data['embeddings']),
                    bigquery.ScalarQueryParameter("reading_level", "INT64", user_data['reading_level']),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_data['user_id'])
                ]
            )
            
            query_job = self.bq_client.client.query(update_query, job_config=job_config)
            query_job.result()
            
            logger.info(f"Updated user profile for {user_data['handle']} in BigQuery")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user profile in BigQuery: {e}")
            return False