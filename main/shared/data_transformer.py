#!/usr/bin/env python3
"""
Data transformer for converting pipeline results to main characters format
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from .utils import (
    format_engagement_number, 
    format_all_engagement_metrics,
    calculate_ratio_display,
    generate_initials,
    extract_handle_from_did,
    clean_text_for_display,
    get_current_timestamp,
    resolve_did_to_handle
)


class MainCharacterTransformer:
    """Transforms pipeline results into main characters format for frontend"""
    
    def __init__(self, client=None):
        self.current_time = datetime.now(timezone.utc)
        self.client = client
    
    def transform_pipeline_results(self, pipeline_results: Dict[str, Any], 
                                 previous_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Convert pipeline results to main characters format
        
        Args:
            pipeline_results: Output from RatioPipeline.run_full_pipeline()
            previous_data: Previous analysis for change detection
            
        Returns:
            List of main character entries
        """
        if not pipeline_results.get('success', False):
            print("Pipeline results indicate failure, returning empty list")
            return []
        
        deep_dive_results = pipeline_results.get('deep_dive_results', [])
        if not deep_dive_results:
            print("No deep dive results found")
            return []
        
        main_characters = []
        
        for rank, ratio_result in enumerate(deep_dive_results, 1):
            try:
                character = self._transform_single_ratio(ratio_result, rank, previous_data)
                if character:
                    main_characters.append(character)
            except Exception as e:
                print(f"Error transforming ratio result {rank}: {e}")
                continue
        
        return main_characters
    
    def _transform_single_ratio(self, ratio_result: Dict[str, Any], rank: int, 
                               previous_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Transform a single ratio result to main character format"""
        
        original_post = ratio_result.get('original_post', {})
        ratio_metrics = ratio_result.get('ratio_metrics', {})
        overall_sentiment = ratio_result.get('overall_sentiment', {})
        
        # Extract user information
        user_info = self._extract_user_info(original_post)
        
        # Calculate engagement metrics
        engagement = original_post.get('engagement', {})
        engagement_formatted = format_all_engagement_metrics(engagement)
        
        # Calculate controversy score
        controversy = self._calculate_controversy_score(ratio_metrics, overall_sentiment)
        
        # Calculate ratio display
        ratio_display = calculate_ratio_display(engagement)
        
        # Calculate change indicator
        change_info = self._calculate_change_indicator(user_info['handle'], controversy, 
                                                      rank, previous_data)
        
        # Extract sample replies
        sample_replies = self._select_sample_replies(ratio_result.get('top_5_replies', []))
        
        # Create main character entry
        character = {
            "rank": rank,
            "user": user_info,
            "ratio": ratio_display,
            "controversy": round(controversy, 1),
            "change": change_info['display'],
            "change_type": change_info['type'],
            "post": {
                "text": clean_text_for_display(original_post.get('text', ''), 200),
                "uri": original_post.get('uri', ''),
                "created_at": original_post.get('created_at', '')
            },
            "engagement": {
                **engagement,
                "formatted": engagement_formatted
            },
            "sample_replies": sample_replies,
            "metrics": {
                "ratio_score": ratio_metrics.get('score', 0),
                "confidence": ratio_metrics.get('confidence', 0),
                "category": ratio_metrics.get('category', 'Unknown'),
                "post_age_minutes": ratio_metrics.get('post_age_minutes', 0)
            },
            "previous_analysis": {
                "controversy": change_info.get('previous_controversy', 0),
                "rank": change_info.get('previous_rank', None)
            }
        }
        
        return character
    
    def _extract_user_info(self, post: Dict[str, Any]) -> Dict[str, str]:
        """Extract user information from post data"""
        
        # Try to get from author field first
        if 'author' in post and isinstance(post['author'], dict):
            author = post['author']
            handle = author.get('handle', '').replace('.bsky.social', '').replace('@', '')
            display_name = author.get('display_name', '')
            did = author.get('did', '')
        else:
            # Fallback to post-level fields
            handle = extract_handle_from_did(post)
            display_name = ''
            did = post.get('author_did', post.get('did', ''))
        
        # If we don't have a real handle, try to resolve from DID
        if (not handle or handle.startswith('user_')) and did and self.client:
            print(f"Resolving DID {did[-8:]}... to real handle")
            profile_info = resolve_did_to_handle(did, self.client)
            if profile_info:
                handle = profile_info['handle']
                display_name = profile_info['display_name']
                print(f"✅ Resolved to @{handle}")
        
        # Generate initials
        initials = generate_initials(display_name, handle)
        
        # Clean handle
        if not handle or handle == 'unknown':
            handle = f"user_{did[-8:]}" if did else "unknown_user"
        
        return {
            "handle": handle,
            "display_name": display_name or handle.replace('_', ' ').title(),
            "initials": initials,
            "did": did
        }
    
    def _calculate_controversy_score(self, ratio_metrics: Dict[str, Any], 
                                   sentiment: Dict[str, Any]) -> float:
        """
        Calculate controversy score (0-10 scale) from ratio metrics and sentiment
        
        Args:
            ratio_metrics: Ratio detection metrics
            sentiment: Sentiment analysis results
            
        Returns:
            Controversy score between 0 and 10
        """
        ratio_score = ratio_metrics.get('score', 0)
        confidence = ratio_metrics.get('confidence', 0)
        
        # Base score from ratio (cap at 8 to leave room for sentiment boost)
        base_score = min(ratio_score * 0.4, 8.0)  # Scale down ratio score
        
        # Sentiment boost (0-2 points possible)
        avg_anger = sentiment.get('avg_anger', 0)
        avg_disgust = sentiment.get('avg_disgust', 0)
        sentiment_boost = (avg_anger + avg_disgust) * 2
        
        # Apply confidence multiplier
        raw_score = (base_score + sentiment_boost) * confidence
        
        # Category-based adjustments
        category = ratio_metrics.get('category', '')
        if category == 'Complete Ratio':
            raw_score *= 1.2  # 20% boost for complete ratios
        elif category == 'Fresh Ratio':
            raw_score *= 1.1  # 10% boost for fresh ratios
        
        # Cap at 10 and round
        return min(raw_score, 10.0)
    
    def _calculate_change_indicator(self, handle: str, current_controversy: float,
                                  current_rank: int, previous_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate change indicator like '+3', '-1', 'NEW', '='"""
        
        if not previous_data or 'main_characters' not in previous_data:
            return {
                'display': 'NEW',
                'type': 'new',
                'previous_controversy': 0,
                'previous_rank': None
            }
        
        # Find this user in previous data
        previous_character = None
        for char in previous_data['main_characters']:
            if char.get('user', {}).get('handle') == handle:
                previous_character = char
                break
        
        if not previous_character:
            return {
                'display': 'NEW',
                'type': 'new',
                'previous_controversy': 0,
                'previous_rank': None
            }
        
        # Calculate change
        previous_controversy = previous_character.get('controversy', 0)
        previous_rank = previous_character.get('rank', current_rank)
        
        controversy_change = current_controversy - previous_controversy
        rank_change = previous_rank - current_rank  # Positive = moved up in ranking
        
        # Determine display based on primary metric (controversy)
        if abs(controversy_change) < 0.1:
            display = '='
            change_type = 'neutral'
        elif controversy_change > 0:
            display = f"+{controversy_change:.1f}"
            change_type = 'positive'
        else:
            display = f"{controversy_change:.1f}"  # Negative sign included
            change_type = 'negative'
        
        return {
            'display': display,
            'type': change_type,
            'previous_controversy': previous_controversy,
            'previous_rank': previous_rank,
            'rank_change': rank_change
        }
    
    def _select_sample_replies(self, replies: List[Dict[str, Any]], max_count: int = 4) -> List[str]:
        """
        Select most critical/engaging replies for display
        
        Args:
            replies: List of reply objects
            max_count: Maximum number of replies to return
            
        Returns:
            List of reply text strings
        """
        if not replies:
            return []
        
        # Sort by engagement (already sorted in pipeline, but just in case)
        sorted_replies = sorted(replies, 
                              key=lambda r: r.get('engagement', {}).get('total', 0), 
                              reverse=True)
        
        # Extract text and clean it
        sample_replies = []
        for reply in sorted_replies[:max_count]:
            text = reply.get('text', '').strip()
            if text:
                # Clean and truncate for display
                cleaned_text = clean_text_for_display(text, 100)
                sample_replies.append(cleaned_text)
        
        return sample_replies
    
    def create_metadata(self, pipeline_results: Dict[str, Any], 
                       main_characters_count: int) -> Dict[str, Any]:
        """
        Create metadata section for today.json
        
        Args:
            pipeline_results: Full pipeline results
            main_characters_count: Number of main characters identified
            
        Returns:
            Metadata dictionary
        """
        summary = pipeline_results.get('summary', {})
        collection_stats = summary.get('collection_stats', {})
        
        return {
            "last_full_analysis": get_current_timestamp(),
            "last_metrics_update": get_current_timestamp(),
            "collection_period_hours": 6,  # Default, should be configurable
            "total_posts_analyzed": collection_stats.get('total_posts_collected', 0),
            "ratios_detected": len(pipeline_results.get('deep_dive_results', [])),
            "main_characters_count": main_characters_count,
            "analysis_settings": {
                "min_ratio_score": 1.5,
                "min_engagement": 10,
                "deep_dive_top_n": 5
            }
        }
    
    def update_metrics_only(self, current_data: Dict[str, Any], 
                           fresh_engagement_data: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Update only engagement metrics in existing data
        
        Args:
            current_data: Current today.json data
            fresh_engagement_data: Dict mapping post URIs to fresh engagement metrics
            
        Returns:
            Updated data with fresh metrics
        """
        updated_data = current_data.copy()
        
        for character in updated_data.get('main_characters', []):
            post_uri = character.get('post', {}).get('uri')
            
            if post_uri and post_uri in fresh_engagement_data:
                fresh_metrics = fresh_engagement_data[post_uri]
                
                # Update engagement data
                character['engagement'].update(fresh_metrics)
                character['engagement']['formatted'] = format_all_engagement_metrics(fresh_metrics)
                
                # Recalculate ratio display
                character['ratio'] = calculate_ratio_display(fresh_metrics)
                
                # Recalculate controversy (keep sentiment data from original analysis)
                ratio_metrics = character.get('metrics', {})
                overall_sentiment = {
                    'avg_anger': 0,  # We don't re-analyze sentiment
                    'avg_disgust': 0
                }
                
                # Create updated ratio metrics with fresh engagement
                updated_ratio_metrics = ratio_metrics.copy()
                total_engagement = sum(fresh_metrics.values())
                negative_engagement = fresh_metrics.get('replies', 0) + fresh_metrics.get('quotes', 0)
                positive_engagement = fresh_metrics.get('likes', 0) + fresh_metrics.get('reposts', 0)
                
                if positive_engagement > 0:
                    updated_ratio_metrics['score'] = negative_engagement / positive_engagement
                else:
                    updated_ratio_metrics['score'] = negative_engagement
                
                # Recalculate controversy with updated metrics
                character['controversy'] = round(
                    self._calculate_controversy_score(updated_ratio_metrics, overall_sentiment), 1
                )
        
        # Update timestamp
        updated_data['metadata']['last_metrics_update'] = get_current_timestamp()
        
        return updated_data