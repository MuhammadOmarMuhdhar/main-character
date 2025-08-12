#!/usr/bin/env python3
"""
Bluesky Ratio Detection Algorithm
Identifies posts getting "ratio'd" based on engagement patterns
"""

import json
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class RatioCategory(Enum):
    FRESH_RATIO = "Fresh Ratio"
    CLASSIC_RATIO = "Classic Ratio" 
    QUOTE_DUNK = "Quote Dunk"
    COMPLETE_RATIO = "Complete Ratio"
    BREWING_RATIO = "Brewing Ratio"


@dataclass
class RatioResult:
    post: Dict[str, Any]
    ratio_score: float
    confidence: float
    category: RatioCategory
    explanation: str
    total_engagement: int
    negative_engagement: int
    positive_engagement: int
    post_age_minutes: float


class RatioDetector:
    def __init__(self, 
                 min_engagement_threshold: int = 10,
                 max_post_age_hours: float = 6.0):
        self.min_engagement_threshold = min_engagement_threshold
        self.max_post_age_hours = max_post_age_hours
        self.current_time = datetime.now(timezone.utc)
    
    def calculate_post_age_minutes(self, post: Dict[str, Any]) -> float:
        """Calculate how old the post is in minutes"""
        try:
            post_time = datetime.fromisoformat(post['timestamp'].replace('Z', '+00:00'))
            age_delta = self.current_time - post_time
            return age_delta.total_seconds() / 60
        except:
            return 0.0
    
    def calculate_engagement_metrics(self, post: Dict[str, Any]) -> Tuple[int, int, int]:
        """Calculate positive, negative, and total engagement"""
        likes = post.get('like_count', 0)
        reposts = post.get('repost_count', 0) 
        replies = post.get('reply_count', 0)
        quotes = post.get('quote_count', 0)
        
        positive_engagement = likes + reposts
        negative_engagement = replies + quotes
        total_engagement = positive_engagement + negative_engagement
        
        return positive_engagement, negative_engagement, total_engagement
    
    def calculate_ratio_score(self, positive: int, negative: int) -> float:
        """Calculate basic ratio score"""
        if positive == 0 and negative == 0:
            return 0.0
        
        # Avoid division by zero
        denominator = max(positive, 1)
        return negative / denominator
    
    def calculate_confidence(self, post: Dict[str, Any], 
                           total_engagement: int,
                           post_age_minutes: float) -> float:
        """Calculate confidence score for ratio detection"""
        confidence_factors = []
        
        # Factor 1: Engagement volume (more engagement = higher confidence)
        volume_score = min(total_engagement / 100, 1.0)  # Cap at 100 engagements
        confidence_factors.append(volume_score * 0.4)
        
        # Factor 2: Post freshness (newer posts = higher confidence for ratios)
        max_age_minutes = self.max_post_age_hours * 60
        freshness_score = max(0, 1 - (post_age_minutes / max_age_minutes))
        confidence_factors.append(freshness_score * 0.3)
        
        # Factor 3: Engagement velocity (rapid accumulation = higher confidence)
        if post_age_minutes > 0:
            velocity = total_engagement / post_age_minutes  # engagements per minute
            velocity_score = min(velocity / 5, 1.0)  # Cap at 5 engagements/minute
        else:
            velocity_score = 0
        confidence_factors.append(velocity_score * 0.2)
        
        # Factor 4: Character count (longer posts often more controversial)
        char_count = post.get('character_count', 0)
        length_score = min(char_count / 280, 1.0)  # Normalize to Twitter-like length
        confidence_factors.append(length_score * 0.1)
        
        return sum(confidence_factors)
    
    def classify_ratio(self, post: Dict[str, Any], 
                      ratio_score: float,
                      positive: int,
                      negative: int,
                      post_age_minutes: float) -> RatioCategory:
        """Classify the type of ratio occurring"""
        replies = post.get('reply_count', 0)
        quotes = post.get('quote_count', 0)
        likes = post.get('like_count', 0)
        
        # Fresh ratio - happening very recently
        if post_age_minutes < 30 and ratio_score >= 3.0:
            return RatioCategory.FRESH_RATIO
        
        # Complete ratio - overwhelming negative engagement
        elif ratio_score >= 10.0:
            return RatioCategory.COMPLETE_RATIO
            
        # Quote dunk - mainly getting quoted negatively
        elif quotes > replies and quotes > likes * 2:
            return RatioCategory.QUOTE_DUNK
            
        # Classic ratio - mainly replies overwhelming likes
        elif replies > likes * 3:
            return RatioCategory.CLASSIC_RATIO
            
        # Brewing ratio - early signs
        elif ratio_score >= 1.5 and negative >= 5:
            return RatioCategory.BREWING_RATIO
            
        else:
            return RatioCategory.CLASSIC_RATIO
    
    def generate_explanation(self, post: Dict[str, Any], 
                           category: RatioCategory,
                           post_age_minutes: float) -> str:
        """Generate human-readable explanation of the ratio"""
        likes = post.get('like_count', 0)
        reposts = post.get('repost_count', 0)
        replies = post.get('reply_count', 0) 
        quotes = post.get('quote_count', 0)
        
        age_str = f"{int(post_age_minutes)} minutes" if post_age_minutes < 60 else f"{post_age_minutes/60:.1f} hours"
        
        if category == RatioCategory.FRESH_RATIO:
            return f"{replies} replies + {quotes} quotes vs {likes} likes in just {age_str}"
        elif category == RatioCategory.QUOTE_DUNK:
            return f"Getting dunked on: {quotes} quotes vs {likes} likes in {age_str}"
        elif category == RatioCategory.COMPLETE_RATIO:
            return f"Completely ratio'd: {replies + quotes} negative vs {likes + reposts} positive in {age_str}"
        elif category == RatioCategory.BREWING_RATIO:
            return f"Brewing ratio: {replies} replies + {quotes} quotes building up vs {likes} likes in {age_str}"
        else:
            return f"Classic ratio: {replies} replies vs {likes} likes in {age_str}"
    
    def analyze_post(self, post: Dict[str, Any]) -> RatioResult:
        """Analyze a single post for ratio patterns"""
        post_age_minutes = self.calculate_post_age_minutes(post)
        positive, negative, total = self.calculate_engagement_metrics(post)
        ratio_score = self.calculate_ratio_score(positive, negative)
        confidence = self.calculate_confidence(post, total, post_age_minutes)
        category = self.classify_ratio(post, ratio_score, positive, negative, post_age_minutes)
        explanation = self.generate_explanation(post, category, post_age_minutes)
        
        return RatioResult(
            post=post,
            ratio_score=ratio_score,
            confidence=confidence,
            category=category,
            explanation=explanation,
            total_engagement=total,
            negative_engagement=negative,
            positive_engagement=positive,
            post_age_minutes=post_age_minutes
        )
    
    def detect_ratios(self, posts_data: Dict[str, Any], 
                     min_ratio_score: float = 1.5,
                     top_n: int = 50) -> List[RatioResult]:
        """Detect ratios in a collection of posts"""
        posts = posts_data.get('posts', [])
        candidates = []
        
        for post in posts:
            # Skip posts with insufficient engagement
            positive, negative, total = self.calculate_engagement_metrics(post)
            if total < self.min_engagement_threshold:
                continue
            
            # Skip very old posts
            post_age_minutes = self.calculate_post_age_minutes(post)
            if post_age_minutes > self.max_post_age_hours * 60:
                continue
            
            # Calculate ratio score
            ratio_score = self.calculate_ratio_score(positive, negative)
            if ratio_score < min_ratio_score:
                continue
            
            # Analyze the post
            result = self.analyze_post(post)
            candidates.append(result)
        
        # Sort by ratio severity, then confidence, then total engagement
        candidates.sort(key=lambda x: (x.ratio_score * x.confidence, x.total_engagement), reverse=True)
        
        return candidates[:top_n]
    
    def format_results(self, results: List[RatioResult]) -> Dict[str, Any]:
        """Format results for output"""
        return {
            'analysis_timestamp': self.current_time.isoformat(),
            'total_ratios_found': len(results),
            'ratios': [
                {
                    'rank': i + 1,
                    'ratio_score': round(result.ratio_score, 2),
                    'confidence': round(result.confidence, 3),
                    'category': result.category.value,
                    'explanation': result.explanation,
                    'post': {
                        'uri': result.post['uri'],
                        'text': result.post['text'][:200] + '...' if len(result.post['text']) > 200 else result.post['text'],
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
    
    parser = argparse.ArgumentParser(description='Detect ratio patterns in Bluesky posts')
    parser.add_argument('input_file', help='JSON file from collect_posts.py')
    parser.add_argument('--min-ratio', type=float, default=1.5, help='Minimum ratio score (default: 1.5)')
    parser.add_argument('--min-engagement', type=int, default=10, help='Minimum total engagement (default: 10)')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top ratios to show (default: 20)')
    parser.add_argument('--output', help='Output file for results (default: print to console)')
    
    args = parser.parse_args()
    
    # Load posts data
    with open(args.input_file, 'r') as f:
        posts_data = json.load(f)
    
    # Initialize detector
    detector = RatioDetector(min_engagement_threshold=args.min_engagement)
    
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
    
    # Print summary
    print(f"\n--- RATIO DETECTION SUMMARY ---")
    print(f"Total posts analyzed: {len(posts_data.get('posts', []))}")
    print(f"Ratios detected: {len(results)}")
    
    if results:
        print(f"\nTop 5 Ratios:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. {result.category.value} (Score: {result.ratio_score:.1f}) - {result.explanation}")
            print(f"   Text: {result.post['text'][:100]}...")
            print()


if __name__ == "__main__":
    main()