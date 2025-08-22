#!/usr/bin/env python3
"""
Bluesky Sentiment Analysis
Analyzes polarity, subjectivity, and emotions (anger, disgust, frustration) in posts
"""

import json
import re
from typing import Dict, List, Any
from dataclasses import dataclass
try:
    from ..shared.config import get_config
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.config import get_config

try:
    from textblob import TextBlob
except ImportError:
    print("Installing TextBlob...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'textblob'])
    from textblob import TextBlob

try:
    from nrclex import NRCLex
except ImportError:
    print("Installing NRCLex...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'NRCLex'])
    from nrclex import NRCLex


@dataclass
class SentimentResult:
    text: str
    polarity: float  # -1 (negative) to +1 (positive)
    subjectivity: float  # 0 (objective) to 1 (subjective)
    anger: float  # 0 to 1 intensity
    disgust: float  # 0 to 1 intensity
    total_negative_emotion: float  # Sum of anger + disgust


class SentimentAnalyzer:
    """Analyzes sentiment and emotions in text"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        pass
    
    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment and emotions in text
        
        Args:
            text: The text to analyze
            
        Returns:
            SentimentResult with polarity, subjectivity, and emotion scores
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                polarity=0.0,
                subjectivity=0.0,
                anger=0.0,
                disgust=0.0,
                total_negative_emotion=0.0
            )
        
        # Clean text for better analysis
        clean_text = self._clean_text(text)
        
        # Polarity and subjectivity using TextBlob
        blob = TextBlob(clean_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Emotion analysis using NRCLex
        emotion_analyzer = NRCLex(clean_text)
        emotion_scores = emotion_analyzer.affect_frequencies
        
        # Extract specific emotions (NRCLex provides anger and disgust)
        anger = emotion_scores.get('anger', 0.0)
        disgust = emotion_scores.get('disgust', 0.0)
        
        # Calculate total negative emotion
        total_negative_emotion = anger + disgust
        
        return SentimentResult(
            text=text,
            polarity=polarity,
            subjectivity=subjectivity,
            anger=anger,
            disgust=disgust,
            total_negative_emotion=total_negative_emotion
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags for sentiment (keep the words)
        text = re.sub(r'[@#](\w+)', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_posts(self, posts_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a collection of posts
        
        Args:
            posts_data: Posts data from collect_posts.py
            
        Returns:
            List of posts with sentiment analysis added
        """
        posts = posts_data.get('posts', [])
        analyzed_posts = []
        
        for post in posts:
            text = post.get('text', '')
            sentiment = self.analyze_text(text)
            
            # Add sentiment data to post
            post_with_sentiment = post.copy()
            post_with_sentiment['sentiment'] = {
                'polarity': round(sentiment.polarity, 3),
                'subjectivity': round(sentiment.subjectivity, 3),
                'anger': round(sentiment.anger, 3),
                'disgust': round(sentiment.disgust, 3),
                'total_negative_emotion': round(sentiment.total_negative_emotion, 3)
            }
            
            analyzed_posts.append(post_with_sentiment)
        
        return analyzed_posts
    
    def get_sentiment_summary(self, analyzed_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for sentiment analysis
        
        Args:
            analyzed_posts: Posts with sentiment analysis
            
        Returns:
            Summary statistics
        """
        if not analyzed_posts:
            return {}
        
        sentiments = [post['sentiment'] for post in analyzed_posts]
        
        # Calculate averages
        avg_polarity = sum(s['polarity'] for s in sentiments) / len(sentiments)
        avg_subjectivity = sum(s['subjectivity'] for s in sentiments) / len(sentiments)
        avg_anger = sum(s['anger'] for s in sentiments) / len(sentiments)
        avg_disgust = sum(s['disgust'] for s in sentiments) / len(sentiments)
        
        config = get_config()
        sentiment_config = config.get_sentiment_config()
        high_emotion_threshold = sentiment_config.get('high_emotion_threshold', 0.3)
        
        # Count posts with significant emotions
        high_anger = len([s for s in sentiments if s['anger'] > high_emotion_threshold])
        high_disgust = len([s for s in sentiments if s['disgust'] > high_emotion_threshold])
        
        # Most emotional posts
        most_negative = max(sentiments, key=lambda x: x['total_negative_emotion'])
        most_positive = max(sentiments, key=lambda x: x['polarity'])
        most_subjective = max(sentiments, key=lambda x: x['subjectivity'])
        
        return {
            'total_posts_analyzed': len(analyzed_posts),
            'averages': {
                'polarity': round(avg_polarity, 3),
                'subjectivity': round(avg_subjectivity, 3),
                'anger': round(avg_anger, 3),
                'disgust': round(avg_disgust, 3)
            },
            'emotion_counts': {
                'high_anger_posts': high_anger,
                'high_disgust_posts': high_disgust
            },
            'extremes': {
                'most_negative_emotion_score': round(most_negative['total_negative_emotion'], 3),
                'most_positive_polarity': round(most_positive['polarity'], 3),
                'most_subjective_score': round(most_subjective['subjectivity'], 3)
            }
        }


def main():
    """Example usage and CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze sentiment in Bluesky posts')
    parser.add_argument('input_file', help='JSON file from collect_posts.py')
    parser.add_argument('--output', help='Output file for results (default: print summary)')
    parser.add_argument('--top-emotional', type=int, default=10, help='Show top N most emotional posts')
    
    args = parser.parse_args()
    
    # Load posts data
    with open(args.input_file, 'r') as f:
        posts_data = json.load(f)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze posts
    print("🧠 Analyzing sentiment and emotions...")
    analyzed_posts = analyzer.analyze_posts(posts_data)
    
    # Generate summary
    summary = analyzer.get_sentiment_summary(analyzed_posts)
    
    # Show most emotional posts
    emotional_posts = sorted(
        analyzed_posts, 
        key=lambda x: x['sentiment']['total_negative_emotion'], 
        reverse=True
    )[:args.top_emotional]
    
    # Print results
    print("\n📊 SENTIMENT ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Posts analyzed: {summary['total_posts_analyzed']}")
    print(f"Average polarity: {summary['averages']['polarity']} (-1=negative, +1=positive)")
    print(f"Average subjectivity: {summary['averages']['subjectivity']} (0=objective, 1=subjective)")
    print(f"Average anger: {summary['averages']['anger']}")
    print(f"Average disgust: {summary['averages']['disgust']}")
    
    print(f"\n🔥 High Emotion Posts:")
    print(f"High anger: {summary['emotion_counts']['high_anger_posts']}")
    print(f"High disgust: {summary['emotion_counts']['high_disgust_posts']}")
    
    print(f"\n😡 Top {args.top_emotional} Most Emotional Posts:")
    for i, post in enumerate(emotional_posts):
        s = post['sentiment']
        print(f"{i+1}. Anger:{s['anger']:.2f} Disgust:{s['disgust']:.2f} Polarity:{s['polarity']:.2f}")
        print(f"   Text: {post['text'][:100]}...")
        print()
    
    # Save results if requested
    if args.output:
        output_data = {
            'analysis_summary': summary,
            'analyzed_posts': analyzed_posts
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()