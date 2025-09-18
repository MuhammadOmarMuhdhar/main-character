import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class UserProfileCollector:
    def __init__(self, bluesky_client):
        self.client = bluesky_client
    
    def get_user_profile_from_did(self, user_did: str) -> Optional[Dict]:
        """Get user profile information from their DID"""
        try:
            # Use DID to get profile - this requires resolving DID to handle first
            # For now, we'll use the DID directly (Bluesky client should handle this)
            profile = self.client.client.app.bsky.actor.get_profile({'actor': user_did})
            
            return {
                'did': profile.did,
                'handle': profile.handle or ''
            }
            
        except Exception as e:
            logger.error(f"Failed to get profile for DID {user_did}: {e}")
            return None

    def collect_user_engagement_data(self, user_handle: str, app_password: str = None) -> Optional[Dict]:
        """Collect user posts and engagement data (including likes if app password provided)"""
        try:
            logger.info(f"Processing user data for {user_handle}")
            
            # Collect user engagement data with likes if app password available
            if app_password:
                logger.info(f"Using app password to include likes for {user_handle}")
                user_data = self.client.get_comprehensive_user_data(
                    actor=user_handle,            # Use handle instead of DID for likes compatibility
                    include_likes=True,           # Include likes with app password
                    likes_username=user_handle,   # User's handle for likes auth
                    likes_password=app_password,  # User's app password
                    likes_limit=300,              
                    posts_limit=200,              
                    reposts_limit=100,
                    replies_limit=100
                )
                logger.info(f"Retrieved user data WITH likes for {user_handle}")
            else:
                logger.info(f"No app password provided, collecting without likes for {user_handle}")
                user_data = self.client.get_comprehensive_user_data(
                    actor=user_handle,    # Use handle for consistency
                    include_likes=False,  # Skip likes when no app password
                    posts_limit=200,
                    reposts_limit=100,
                    replies_limit=100
                )
                logger.info(f"Retrieved user data WITHOUT likes for {user_handle}")
            
            # Check if we have enough content for keyword extraction
            total_items = len(user_data['posts']) + len(user_data['reposts']) + len(user_data['replies'])
            if total_items < 5:
                logger.warning(f"User {user_handle} has insufficient content ({total_items} items)")
                return None
            
            return user_data
            
        except Exception as e:
            logger.error(f"Failed to collect user engagement data for {user_handle}: {e}")
            return None