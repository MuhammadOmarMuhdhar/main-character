#!/usr/bin/env python3
"""
Topic persistence and evolution tracking system
Enables topics to persist across analysis cycles with trend tracking
"""

import uuid
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TopicMatch:
    """Represents a match between new and previous topic"""
    new_topic: Dict[str, Any]
    previous_topic: Dict[str, Any]
    similarity_score: float
    match_confidence: str  # 'high', 'medium', 'low'


class TopicPersistenceManager:
    """Manages topic persistence, evolution tracking, and trend analysis"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 keyword_overlap_threshold: float = 0.3,
                 trend_threshold: float = 0.2):
        """
        Initialize topic persistence manager
        
        Args:
            similarity_threshold: Cosine similarity threshold for topic matching (0.8 = very similar)
            keyword_overlap_threshold: Keyword overlap threshold for fallback matching
            trend_threshold: Post count change threshold for trend classification (0.2 = 20%)
        """
        self.similarity_threshold = similarity_threshold
        self.keyword_overlap_threshold = keyword_overlap_threshold
        self.trend_threshold = trend_threshold
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First topic embedding
            embedding2: Second topic embedding
            
        Returns:
            Cosine similarity score (0-1, higher = more similar)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Handle zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def keyword_overlap_similarity(self, keywords1: List[str], keywords2: List[str]) -> float:
        """
        Calculate keyword overlap similarity as fallback
        
        Args:
            keywords1: First topic keywords
            keywords2: Second topic keywords
            
        Returns:
            Overlap similarity score (0-1)
        """
        if not keywords1 or not keywords2:
            return 0.0
        
        set1 = set(keywords1[:10])  # Use top 10 keywords
        set2 = set(keywords2[:10])
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def find_best_match(self, new_topic: Dict[str, Any], 
                       previous_topics: List[Dict[str, Any]]) -> Optional[TopicMatch]:
        """
        Find the best matching previous topic for a new topic
        
        Args:
            new_topic: New topic to match
            previous_topics: List of previous topics
            
        Returns:
            TopicMatch object if good match found, None otherwise
        """
        best_match = None
        best_score = 0.0
        
        new_embedding = new_topic.get('topic_embedding', [])
        new_keywords = new_topic.get('keywords', [])
        
        for prev_topic in previous_topics:
            if prev_topic.get('status') != 'active':
                continue  # Skip non-active topics
            
            prev_embedding = prev_topic.get('topic_embedding', [])
            prev_keywords = prev_topic.get('keywords', [])
            
            # Primary matching: embedding similarity
            embedding_similarity = 0.0
            if new_embedding and prev_embedding:
                embedding_similarity = self.cosine_similarity(new_embedding, prev_embedding)
            
            # Secondary matching: keyword overlap
            keyword_similarity = self.keyword_overlap_similarity(new_keywords, prev_keywords)
            
            # Combined score (prefer embedding similarity)
            combined_score = embedding_similarity * 0.8 + keyword_similarity * 0.2
            
            if combined_score > best_score and combined_score >= self.similarity_threshold:
                # Determine match confidence
                if embedding_similarity >= 0.9:
                    confidence = 'high'
                elif embedding_similarity >= 0.8:
                    confidence = 'medium' 
                else:
                    confidence = 'low'
                
                best_match = TopicMatch(
                    new_topic=new_topic,
                    previous_topic=prev_topic,
                    similarity_score=combined_score,
                    match_confidence=confidence
                )
                best_score = combined_score
        
        return best_match
    
    def calculate_trend(self, current_count: int, previous_count: int) -> str:
        """
        Calculate trend based on post count change
        
        Args:
            current_count: Current post count
            previous_count: Previous post count
            
        Returns:
            Trend string: 'rising', 'falling', 'stable', or 'new'
        """
        if previous_count == 0:
            return 'new'
        
        change_ratio = (current_count - previous_count) / previous_count
        
        if change_ratio >= self.trend_threshold:
            return 'rising'
        elif change_ratio <= -self.trend_threshold:
            return 'falling'
        else:
            return 'stable'
    
    def create_persistent_topic(self, topic: Dict[str, Any], 
                               previous_topic: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create or update a topic with persistence fields
        
        Args:
            topic: New topic data
            previous_topic: Previous version of this topic (if matched)
            
        Returns:
            Enhanced topic with persistence fields
        """
        current_timestamp = datetime.now(timezone.utc).isoformat()
        current_count = topic.get('post_count', 0)
        
        if previous_topic:
            # Update existing topic
            persistent_topic = topic.copy()
            
            # Preserve persistence fields
            persistent_topic['persistent_id'] = previous_topic['persistent_id']
            persistent_topic['first_detected'] = previous_topic['first_detected']
            persistent_topic['last_updated'] = current_timestamp
            
            # Update analysis windows
            prev_windows = previous_topic.get('analysis_windows', [])
            persistent_topic['analysis_windows'] = prev_windows + [current_timestamp]
            
            # Calculate trend
            prev_count = previous_topic.get('post_count', 0)
            trend = self.calculate_trend(current_count, prev_count)
            persistent_topic['trend'] = trend
            persistent_topic['post_count_change'] = current_count - prev_count
            
            # Update peak
            prev_peak = previous_topic.get('peak_post_count', prev_count)
            persistent_topic['peak_post_count'] = max(current_count, prev_peak)
            
            # Keep active status
            persistent_topic['status'] = 'active'
            
        else:
            # Create new topic
            persistent_topic = topic.copy()
            persistent_topic['persistent_id'] = str(uuid.uuid4())
            persistent_topic['first_detected'] = current_timestamp
            persistent_topic['last_updated'] = current_timestamp
            persistent_topic['analysis_windows'] = [current_timestamp]
            persistent_topic['trend'] = 'new'
            persistent_topic['post_count_change'] = current_count  # All posts are new
            persistent_topic['peak_post_count'] = current_count
            persistent_topic['status'] = 'active'
        
        return persistent_topic
    
    def archive_topic(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create archived version of a topic that's no longer detected
        
        Args:
            topic: Topic to archive
            
        Returns:
            Lightweight archived topic record
        """
        current_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Calculate days active
        first_detected = topic.get('first_detected', current_timestamp)
        try:
            first_dt = datetime.fromisoformat(first_detected.replace('Z', '+00:00'))
            current_dt = datetime.now(timezone.utc)
            days_active = (current_dt - first_dt).days
        except:
            days_active = 0
        
        return {
            'persistent_id': topic['persistent_id'],
            'label': topic['label'],
            'last_seen': topic.get('last_updated', current_timestamp),
            'peak_post_count': topic.get('peak_post_count', topic.get('post_count', 0)),
            'days_active': max(1, days_active),  # Minimum 1 day
            'total_appearances': len(topic.get('analysis_windows', [1]))
        }
    
    def cleanup_archived_topics(self, archived_topics: List[Dict[str, Any]], 
                               max_age_days: int = 7) -> List[Dict[str, Any]]:
        """
        Remove archived topics older than max_age_days
        
        Args:
            archived_topics: List of archived topics
            max_age_days: Maximum age in days before removal
            
        Returns:
            Cleaned list of archived topics
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cleaned_topics = []
        
        for topic in archived_topics:
            try:
                last_seen = topic.get('last_seen', '')
                last_seen_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                
                if last_seen_dt >= cutoff_date:
                    cleaned_topics.append(topic)
                    
            except Exception:
                # Keep topics with invalid timestamps for safety
                cleaned_topics.append(topic)
        
        # Also limit to 50 most recent archived topics
        return sorted(cleaned_topics, 
                     key=lambda x: x.get('last_seen', ''), 
                     reverse=True)[:50]
    
    def update_topics_with_persistence(self, new_topics: List[Dict[str, Any]], 
                                     existing_topics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main function to update topics with persistence logic
        
        Args:
            new_topics: Newly detected topics from current analysis
            existing_topics_data: Current topics.json data
            
        Returns:
            Updated topics data with persistence applied
        """
        print(f"🔄 Applying topic persistence to {len(new_topics)} new topics...")
        
        # Extract previous active topics
        previous_topics = existing_topics_data.get('topics', [])
        archived_topics = existing_topics_data.get('archived_topics', [])
        
        updated_topics = []
        matched_previous_ids = set()
        
        # Step 1: Match new topics with previous topics
        for new_topic in new_topics:
            match = self.find_best_match(new_topic, previous_topics)
            
            if match:
                # Update existing topic
                persistent_topic = self.create_persistent_topic(new_topic, match.previous_topic)
                updated_topics.append(persistent_topic)
                matched_previous_ids.add(match.previous_topic['persistent_id'])
                print(f"  📈 Updated topic: '{new_topic['label']}' (similarity: {match.similarity_score:.2f})")
            else:
                # Create new topic
                persistent_topic = self.create_persistent_topic(new_topic)
                updated_topics.append(persistent_topic)
                print(f"  🆕 New topic: '{new_topic['label']}'")
        
        # Step 2: Handle unmatched previous topics (archive them)
        for prev_topic in previous_topics:
            if prev_topic.get('persistent_id') not in matched_previous_ids:
                archived_topic = self.archive_topic(prev_topic)
                archived_topics.append(archived_topic)
                print(f"  👻 Archived topic: '{prev_topic['label']}'")
        
        # Step 3: Clean up old archived topics
        archived_topics = self.cleanup_archived_topics(archived_topics)
        
        # Step 4: Re-rank topics (rising topics get slight boost)
        def get_ranking_score(topic):
            base_score = topic.get('post_count', 0)
            
            # Apply trend multipliers
            trend = topic.get('trend', 'stable')
            if trend == 'rising':
                return base_score * 1.15
            elif trend == 'new':
                return base_score * 1.1
            elif trend == 'falling':
                return base_score * 0.9
            else:  # stable
                return base_score
        
        updated_topics.sort(key=get_ranking_score, reverse=True)
        
        # Reassign IDs based on new ranking
        for i, topic in enumerate(updated_topics, 1):
            topic['id'] = i
        
        # Step 5: Create updated topics data
        updated_data = existing_topics_data.copy()
        updated_data['topics'] = updated_topics
        updated_data['archived_topics'] = archived_topics
        
        # Update metadata
        metadata = updated_data.get('metadata', {})
        metadata['persistence_enabled'] = True
        metadata['last_cleanup'] = datetime.now(timezone.utc).isoformat()
        metadata['topics_found'] = len(updated_topics)
        metadata['archived_count'] = len(archived_topics)
        updated_data['metadata'] = metadata
        
        print(f"✅ Topic persistence complete: {len(updated_topics)} active, {len(archived_topics)} archived")
        return updated_data