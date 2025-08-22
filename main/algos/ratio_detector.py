#!/usr/bin/env python3
"""
Bluesky Ratio Detection Algorithm - Simplified Version
Identifies controversial posts using simple time bucket + engagement ranking
"""

import json
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
try:
    from ..shared.config import get_config
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.config import get_config


@dataclass
class RatioResult:
    post: Dict[str, Any]
    ratio_score: float
    explanation: str
    total_engagement: int
    negative_engagement: int
    positive_engagement: int
    post_age_minutes: float
    time_bucket: int  # 0-3 for 15-minute buckets


class RatioDetector:
    def __init__(self, 
                 min_quotes_threshold: int = None,
                 min_likes_threshold: int = None,
                 min_replies_threshold: int = None,
                 min_ratio_threshold: float = None,
                 test_mode: bool = False):
        
        config = get_config(test_mode=test_mode)
        ratio_config = config.get_ratio_detection_config()
        ranking_config = config.get_ranking_config()
        
        self.min_quotes_threshold = min_quotes_threshold or ratio_config.get('min_quotes_threshold', 5)
        self.min_likes_threshold = min_likes_threshold or ratio_config.get('min_likes_threshold', 3)
        self.min_replies_threshold = min_replies_threshold or ratio_config.get('min_replies_threshold', 2)
        self.min_ratio_threshold = min_ratio_threshold or ratio_config.get('min_ratio_threshold', 0.5)
        
        # New simplified ranking config
        self.time_bucket_minutes = ranking_config.get('time_bucket_minutes', 15)
        self.max_post_age_hours = ranking_config.get('max_post_age_hours', 1.0)
        self.min_engagement = ranking_config.get('min_engagement', 10)
        
        self.current_time = datetime.now(timezone.utc)
    
    def calculate_post_age_minutes(self, post: Dict[str, Any]) -> float:
        """Calculate how old the post is in minutes"""
        try:
            post_time = datetime.fromisoformat(post['timestamp'].replace('Z', '+00:00'))
            age_delta = self.current_time - post_time
            return age_delta.total_seconds() / 60
        except:
            return 0.0
    
    def get_time_bucket(self, post_age_minutes: float) -> int:
        """Get time bucket (0-3) for post age. Lower bucket = more recent"""
        if post_age_minutes <= self.time_bucket_minutes:
            return 0  # 0-15 minutes (most recent)
        elif post_age_minutes <= self.time_bucket_minutes * 2:
            return 1  # 15-30 minutes
        elif post_age_minutes <= self.time_bucket_minutes * 3:
            return 2  # 30-45 minutes
        elif post_age_minutes <= self.time_bucket_minutes * 4:
            return 3  # 45-60 minutes
        else:
            return 4  # Over 1 hour (will be filtered out)
    
    def calculate_engagement_metrics(self, post: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """Calculate quotes, reposts, replies, and total engagement"""
        quotes = post.get('quote_count', 0)
        reposts = post.get('repost_count', 0) 
        replies = post.get('reply_count', 0)
        likes = post.get('like_count', 0)
        
        # Total engagement for ranking
        total_engagement = quotes + reposts + replies + likes
        
        return quotes, reposts, replies, total_engagement
    
    def calculate_ratio_score(self, quotes: int, reposts: int, replies: int) -> float:
        """Calculate discussion vs sharing score - higher = more conversation"""
        # Discussion (quotes + replies) vs sharing (reposts)
        # +1 to reposts prevents division by zero
        return (quotes + replies) / (reposts + 1)
    
    def generate_explanation(self, quotes: int, reposts: int, replies: int, 
                           post_age_minutes: float, time_bucket: int) -> str:
        """Generate simple explanation for the discussion score"""
        bucket_labels = ["0-15m", "15-30m", "30-45m", "45-60m"]
        bucket_str = bucket_labels[time_bucket] if time_bucket < 4 else "60m+"
        
        discussion_count = quotes + replies
        
        if reposts == 0:
            return f"Discussion: {discussion_count} responses ({quotes} quotes + {replies} replies), no shares ({bucket_str})"
        else:
            return f"Discussion: {discussion_count} responses vs {reposts} shares ({bucket_str})"
    
    def analyze_post(self, post: Dict[str, Any]) -> RatioResult:
        """Analyze a single post for ratio patterns"""
        post_age_minutes = self.calculate_post_age_minutes(post)
        time_bucket = self.get_time_bucket(post_age_minutes)
        quotes, reposts, replies, total = self.calculate_engagement_metrics(post)
        ratio_score = self.calculate_ratio_score(quotes, reposts, replies)
        explanation = self.generate_explanation(quotes, reposts, replies, post_age_minutes, time_bucket)
        
        return RatioResult(
            post=post,
            ratio_score=ratio_score,
            explanation=explanation,
            total_engagement=total,
            negative_engagement=quotes + replies,
            positive_engagement=reposts,
            post_age_minutes=post_age_minutes,
            time_bucket=time_bucket
        )
    
    def detect_ratios(self, posts_data: Dict[str, Any], 
                     min_ratio_score: float = None,
                     top_n: int = None) -> List[RatioResult]:
        """Detect ratios using simplified time bucket + engagement ranking"""
        posts = posts_data.get('posts', [])
        candidates = []
        
        config = get_config()
        ratio_config = config.get_ratio_detection_config()
        
        # Use instance threshold if not provided
        if min_ratio_score is None:
            min_ratio_score = self.min_ratio_threshold
        if top_n is None:
            top_n = ratio_config.get('default_top_n', 50)
        
        for post in posts:
            quotes, reposts, replies, total = self.calculate_engagement_metrics(post)
            likes = post.get('like_count', 0)
            
            # Basic engagement requirements
            if quotes < self.min_quotes_threshold:
                continue
            if likes < self.min_likes_threshold:
                continue
            if replies < self.min_replies_threshold:
                continue
            if total < self.min_engagement:
                continue
            
            # Skip very old posts (over max age)
            post_age_minutes = self.calculate_post_age_minutes(post)
            if post_age_minutes > self.max_post_age_hours * 60:
                continue
            
            # Apply ratio score filter
            ratio_score = self.calculate_ratio_score(quotes, reposts, replies)
            if ratio_score < min_ratio_score:
                continue
            
            # Analyze the post
            result = self.analyze_post(post)
            candidates.append(result)
        
        # Sort by time bucket first (0=most recent), then by engagement descending
        candidates.sort(key=lambda x: (x.time_bucket, -x.total_engagement))
        
        return candidates[:top_n]
    
    def format_results(self, results: List[RatioResult]) -> Dict[str, Any]:
        """Format results for output"""
        return {
            'total_ratios_found': len(results),
            'ranking_method': 'time_bucket_then_engagement',
            'time_bucket_minutes': self.time_bucket_minutes,
            'max_age_hours': self.max_post_age_hours,
            'ratios': [
                {
                    'rank': i + 1,
                    'ratio_score': round(result.ratio_score, 2),
                    'time_bucket': result.time_bucket,
                    'explanation': result.explanation,
                    'post': {
                        'uri': result.post['uri'],
                        'text': result.post['text'],
                        'likes': result.post['like_count'],
                        'replies': result.post['reply_count'],
                        'reposts': result.post['repost_count'],
                        'quotes': result.post['quote_count'],
                        'character_count': result.post['character_count'],
                        'timestamp': result.post['timestamp']
                    },
                    'metrics': {
                        'total_engagement': result.total_engagement,
                        'positive_engagement': result.positive_engagement,
                        'negative_engagement': result.negative_engagement,
                        'post_age_minutes': round(result.post_age_minutes, 1)
                    }
                }
                for i, result in enumerate(results)
            ]
        }


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect ratio patterns in Bluesky posts (simplified)')
    parser.add_argument('input_file', help='JSON file from collect_posts.py')
    
    config = get_config()
    cli_defaults = config.get_cli_defaults('ratio_detection')
    
    parser.add_argument('--min-ratio', type=float, 
                       default=cli_defaults.get('min_ratio_score', 0.5), 
                       help=f'Minimum ratio score (default: {cli_defaults.get("min_ratio_score", 0.5)})')
    parser.add_argument('--min-quotes', type=int, 
                       default=cli_defaults.get('min_quotes', 5), 
                       help=f'Minimum quotes (default: {cli_defaults.get("min_quotes", 5)})')
    parser.add_argument('--top-n', type=int, 
                       default=cli_defaults.get('top_n', 20), 
                       help=f'Number of top ratios to show (default: {cli_defaults.get("top_n", 20)})')
    parser.add_argument('--output', help='Output file for results (default: print to console)')
    
    args = parser.parse_args()
    
    # Load posts data
    with open(args.input_file, 'r') as f:
        posts_data = json.load(f)
    
    # Initialize detector
    detector = RatioDetector(min_quotes_threshold=args.min_quotes)
    
    # Detect ratios
    results = detector.detect_ratios(
        posts_data, 
        min_ratio_score=args.min_ratio,
        top_n=args.top_n
    )
    
    # Format and output results
    formatted_results = detector.format_results(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(formatted_results, indent=2))


if __name__ == "__main__":
    main()