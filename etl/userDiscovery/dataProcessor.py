import logging
import sys
import os
from datetime import datetime
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rankingAlgos.neighborhoodAlgo import discover_taste_neighbors
from rankingAlgos.networkAlgo import discover_network_relationships

logger = logging.getLogger(__name__)


class UserDataProcessor:
    def __init__(self):
        pass
    
    def process_user_data(self, user_did: str, user_profile: Dict, user_engagement_data: Dict, client=None) -> Optional[Dict]:
        """Process user data to extract keywords and generate embeddings"""
        try:
            # Discover taste neighbors using multi-signal interaction analysis
            taste_neighbors = []
            if client and user_engagement_data:
                logger.info(f"Discovering taste neighbors for user {user_profile.get('handle')}")
                taste_neighbors = discover_taste_neighbors(
                    user_data=user_engagement_data,
                    client=client,
                    signal_weights={
                        'repost': 1.0,    # Highest intent - user amplifies content
                        'like': 0.6,      # Medium intent - approval signal
                        'reply': 0.8,     # High engagement - active participation  
                        'quote_post': 0.9 # Very high - adds commentary
                    },
                    min_overlap_threshold=3,  # Need 3+ shared interactions
                    max_neighbors=50,         # Top 50 taste neighbors
                    time_hours=168           # Look back 7 days
                )
                logger.info(f"Found {len(taste_neighbors)} taste neighbors for user {user_profile.get('handle')}")
            else:
                logger.warning(f"No client provided or no engagement data for neighborhood discovery for {user_profile.get('handle')}")
            
            # Discover mutual connections using network relationship analysis
            network_relationships = {}
            if client:
                logger.info(f"Discovering network relationships for user {user_profile.get('handle')}")
                network_relationships = discover_network_relationships(
                    user_id=user_did,
                    client=client
                )
                mutual_count = len(network_relationships.get('mutual_connections', []))
                logger.info(f"Found {mutual_count} mutual connections for user {user_profile.get('handle')}")
            else:
                logger.warning(f"No client provided for network discovery for {user_profile.get('handle')}")
            
            # TODO: Extract keywords and generate embeddings with new algorithm
            # enhanced_keywords, reading_level = extract_enhanced_user_keywords_with_fallback(user_engagement_data, top_k=100, min_freq=1)
            # all_posts = embed_posts(user_engagement_data)
            
            # PLACEHOLDER: Return both neighborhood and network data until new algorithm is implemented
            enhanced_keywords = {
                'taste_neighbors': taste_neighbors,
                'network_relationships': network_relationships
            }
            reading_level = 8  # Default reading level
            user_embedding = None
            
            logger.info("User discovery: Using neighborhood + network discovery with placeholder data")
            
            # For now, we'll proceed if we have either neighbors or network data
            if not taste_neighbors and not network_relationships.get('mutual_connections'):
                logger.warning(f"No taste neighbors or mutual connections found for user {user_profile.get('handle')}")
                return None
            
            # Prepare user data for BigQuery
            processed_user = {
                'user_id': user_did,
                'handle': user_profile.get('handle', ''),
                'keywords': enhanced_keywords,  # Store enhanced keywords as JSON dict
                'embeddings': user_embedding,  # Single embedding vector
                'reading_level': reading_level,  # User's reading level
                'updated_at': datetime.utcnow()
            }
            
            logger.info(f"Extracted {len(enhanced_keywords)} enhanced keywords and reading level {reading_level} for user {user_profile.get('handle')}")
            return processed_user
            
        except Exception as e:
            logger.error(f"Failed to process user data for {user_did}: {e}")
            return None