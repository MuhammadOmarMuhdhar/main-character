#!/usr/bin/env python3
"""
Bluesky Ratio Detection Pipeline
Main entry point for collecting posts and detecting ratios
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from client.collect_posts import JetstreamCollector
from client.post_responses_collector import PostResponsesCollector
from algos.ratio_detector import RatioDetector, RatioResult
from algos.sentiment_analyzer import SentimentAnalyzer
from algos.topic_modeler import EnglishTopicModeler


class RatioPipeline:
    """Main pipeline for collecting posts and detecting ratios"""
    
    def __init__(self, 
                 hours_back: int = 1,
                 collection_timeout: int = 3600,
                 min_ratio_score: float = 1.5,
                 min_engagement: int = 10,
                 top_n_ratios: int = 20,
                 deep_dive_top_n: int = 5):
        """
        Initialize the ratio detection pipeline
        
        Args:
            hours_back: How many hours back to collect posts
            collection_timeout: Max timeout for post collection in seconds
            min_ratio_score: Minimum ratio score to consider
            min_engagement: Minimum total engagement to consider
            top_n_ratios: Number of top ratios to keep in memory
            deep_dive_top_n: Number of top ratios to deep dive with responses
        """
        self.hours_back = hours_back
        self.collection_timeout = collection_timeout
        self.min_ratio_score = min_ratio_score
        self.min_engagement = min_engagement
        self.top_n_ratios = top_n_ratios
        self.deep_dive_top_n = deep_dive_top_n
        
        # In-memory storage
        self.collected_posts = None
        self.detected_ratios = []
        self.deep_dive_results = []
        self.collection_stats = {}
        self.analysis_timestamp = None
        
        # Initialize components
        self.collector = JetstreamCollector(
            hours_back=hours_back,
            max_timeout=collection_timeout
        )
        
        self.detector = RatioDetector(
            min_quotes_threshold=5,
            min_likes_threshold=3,
            min_replies_threshold=2,
            min_ratio_threshold=0.5,
            test_mode=False
        )
        
        self.responses_collector = PostResponsesCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = EnglishTopicModeler()
    
    def collect_posts(self) -> Dict[str, Any]:
        """
        Collect posts from the last hour via Jetstream firehose
        
        Returns:
            Dict containing collected posts and metadata
        """
        print(f"🚀 Starting post collection for the last {self.hours_back} hour(s)...")
        
        try:
            # Collect posts
            posts = self.collector.collect_posts()
            
            # Create posts data structure
            posts_data = {
                'success': True,
                'collection_info': {
                    'start_time': datetime.fromtimestamp(
                        self.collector.start_time / 1000000, tz=timezone.utc
                    ).isoformat(),
                    'end_time': datetime.fromtimestamp(
                        self.collector.end_time / 1000000, tz=timezone.utc
                    ).isoformat(),
                    'hours_back': self.hours_back,
                    'total_posts': len(posts),
                    'collection_timestamp': datetime.now(tz=timezone.utc).isoformat()
                },
                'posts': posts
            }
            
            # Store in memory
            self.collected_posts = posts_data
            
            # Update collection stats
            self.collection_stats = {
                'total_posts_collected': len(posts),
                'posts_with_engagement': len([p for p in posts if 
                    (p['like_count'] + p['reply_count'] + p['repost_count'] + p['quote_count']) > 0]),
                'collection_duration_seconds': self.collection_timeout,
                'success': True
            }
            
            print(f"✅ Collection complete: {len(posts)} posts collected")
            return posts_data
            
        except Exception as e:
            print(f"❌ Collection failed: {e}")
            self.collection_stats = {
                'total_posts_collected': 0,
                'posts_with_engagement': 0,
                'collection_duration_seconds': 0,
                'success': False,
                'error': str(e)
            }
            
            # Return failure structure instead of raising
            return {
                'success': False,
                'error': str(e),
                'collection_info': {
                    'total_posts': 0,
                    'collection_timestamp': datetime.now(tz=timezone.utc).isoformat()
                },
                'posts': []
            }
    
    def detect_ratios(self) -> List[RatioResult]:
        """
        Detect ratios in the collected posts
        
        Returns:
            List of RatioResult objects
        """
        if not self.collected_posts:
            raise ValueError("No posts collected yet. Run collect_posts() first.")
        
        print(f"🔍 Analyzing posts for ratio patterns...")
        
        try:
            # Run ratio detection
            ratio_results = self.detector.detect_ratios(
                self.collected_posts,
                min_ratio_score=self.min_ratio_score,
                top_n=self.top_n_ratios
            )
            
            # Store in memory
            self.detected_ratios = ratio_results
            self.analysis_timestamp = datetime.now(tz=timezone.utc)
            
            print(f"✅ Ratio analysis complete: {len(ratio_results)} ratios detected")
            return ratio_results
            
        except Exception as e:
            print(f"❌ Ratio detection failed: {e}")
            raise
    
    def deep_dive_responses(self) -> List[Dict[str, Any]]:
        """
        Deep dive into top ratios by collecting responses and analyzing sentiment
        
        Returns:
            List of deep dive results with responses and sentiment analysis
        """
        if not self.detected_ratios:
            raise ValueError("No ratios detected yet. Run detect_ratios() first.")
        
        print(f"🔍 Deep diving into top {self.deep_dive_top_n} ratios...")
        
        deep_dive_results = []
        top_ratios = self.detected_ratios[:self.deep_dive_top_n]
        
        for i, ratio_result in enumerate(top_ratios):
            print(f"\n📊 Analyzing ratio {i+1}/{len(top_ratios)}: {ratio_result.post['uri']}")
            
            try:
                # Collect all responses for this post
                responses_data = self.responses_collector.collect_all_responses(
                    post_uri=ratio_result.post['uri'],
                    include_quotes=True,
                    max_quote_results=200,
                    thread_depth=10
                )
                
                if not responses_data.get('success'):
                    print(f"⚠️  Failed to collect responses: {responses_data.get('error', 'Unknown error')}")
                    continue
                
                all_responses = responses_data['replies'] + responses_data['quotes']
                
                if not all_responses:
                    print("📭 No responses found for this post")
                    deep_dive_results.append({
                        'original_post': ratio_result.post,
                        'ratio_metrics': {
                            'score': ratio_result.ratio_score,
                            'time_bucket': ratio_result.time_bucket,
                            'total_engagement': ratio_result.total_engagement,
                            'explanation': ratio_result.explanation,
                            'post_age_minutes': ratio_result.post_age_minutes
                        },
                        'top_5_replies': [],
                        'top_5_quotes': [],
                        'overall_sentiment': {
                            'analyzed_total_responses': 0,
                            'avg_polarity': 0.0,
                            'avg_anger': 0.0,
                            'avg_disgust': 0.0
                        },
                        'collection_success': True,
                        'responses_found': 0
                    })
                    continue
                
                # Analyze sentiment on ALL responses
                print(f"🧠 Analyzing sentiment on {len(all_responses)} responses...")
                
                # Create temporary posts data structure for sentiment analysis
                responses_for_sentiment = {
                    'posts': [{'text': response.get('text', '')} for response in all_responses]
                }
                
                # Run sentiment analysis on all responses
                analyzed_responses = self.sentiment_analyzer.analyze_posts(responses_for_sentiment)
                sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(analyzed_responses)
                
                # Add sentiment data back to original responses
                for i, response in enumerate(all_responses):
                    if i < len(analyzed_responses):
                        response['sentiment'] = analyzed_responses[i]['sentiment']
                
                # Separate replies and quotes
                replies = [r for r in all_responses if 'reply_to' in r]
                quotes = [r for r in all_responses if 'quoted_post_uri' in r]
                
                # Sort by engagement (likes + replies + reposts)
                def get_engagement_score(post):
                    return (post.get('like_count', 0) + 
                           post.get('reply_count', 0) + 
                           post.get('repost_count', 0))
                
                replies.sort(key=get_engagement_score, reverse=True)
                quotes.sort(key=get_engagement_score, reverse=True)
                
                # Get top 5 of each with engagement metrics
                top_5_replies = []
                for reply in replies[:5]:
                    top_5_replies.append({
                        'uri': reply.get('uri'),
                        'author': reply.get('author', {}),
                        'text': reply.get('text', ''),
                        'created_at': reply.get('created_at'),
                        'engagement': {
                            'likes': reply.get('like_count', 0),
                            'replies': reply.get('reply_count', 0),
                            'reposts': reply.get('repost_count', 0),
                            'total': get_engagement_score(reply)
                        },
                        'sentiment': reply.get('sentiment', {})
                    })
                
                top_5_quotes = []
                for quote in quotes[:5]:
                    top_5_quotes.append({
                        'uri': quote.get('uri'),
                        'author': quote.get('author', {}),
                        'text': quote.get('text', ''),
                        'created_at': quote.get('created_at'),
                        'engagement': {
                            'likes': quote.get('like_count', 0),
                            'replies': quote.get('reply_count', 0),
                            'reposts': quote.get('repost_count', 0),
                            'total': get_engagement_score(quote)
                        },
                        'sentiment': quote.get('sentiment', {})
                    })
                
                # Compile deep dive result
                deep_dive_result = {
                    'original_post': {
                        'uri': ratio_result.post['uri'],
                        'text': ratio_result.post['text'],
                        'author_did': ratio_result.post['did'],
                        'created_at': ratio_result.post['created_at'],
                        'engagement': {
                            'likes': ratio_result.post['like_count'],
                            'replies': ratio_result.post['reply_count'],
                            'reposts': ratio_result.post['repost_count'],
                            'quotes': ratio_result.post['quote_count']
                        }
                    },
                    'ratio_metrics': {
                        'score': round(ratio_result.ratio_score, 2),
                        'time_bucket': ratio_result.time_bucket,
                        'total_engagement': ratio_result.total_engagement,
                        'explanation': ratio_result.explanation,
                        'post_age_minutes': round(ratio_result.post_age_minutes, 1)
                    },
                    'top_5_replies': top_5_replies,
                    'top_5_quotes': top_5_quotes,
                    'overall_sentiment': {
                        'analyzed_total_responses': len(all_responses),
                        'avg_polarity': sentiment_summary.get('averages', {}).get('polarity', 0.0),
                        'avg_anger': sentiment_summary.get('averages', {}).get('anger', 0.0),
                        'avg_disgust': sentiment_summary.get('averages', {}).get('disgust', 0.0),
                        'high_emotion_counts': sentiment_summary.get('emotion_counts', {})
                    },
                    'collection_success': True,
                    'responses_found': len(all_responses)
                }
                
                deep_dive_results.append(deep_dive_result)
                
                print(f"✅ Analysis complete: {len(replies)} replies, {len(quotes)} quotes")
                print(f"   Avg sentiment - Polarity: {deep_dive_result['overall_sentiment']['avg_polarity']:.2f}, Anger: {deep_dive_result['overall_sentiment']['avg_anger']:.2f}")
                
            except Exception as e:
                print(f"❌ Deep dive failed for post {ratio_result.post['uri']}: {e}")
                continue
        
        self.deep_dive_results = deep_dive_results
        print(f"\n🎯 Deep dive complete: {len(deep_dive_results)} ratios analyzed")
        return deep_dive_results
    
    def analyze_main_character_topics(self) -> Dict[str, Any]:
        """
        Analyze topics across main character posts and their responses
        
        Returns:
            Dict containing main character topic analysis results
        """
        if not self.detected_ratios:
            raise ValueError("No ratios detected yet. Run detect_ratios() first.")
        
        if not self.deep_dive_results:
            raise ValueError("No deep dive results yet. Run deep_dive_responses() first.")
        
        print(f"🏷️ Starting main character topic analysis...")
        
        try:
            # Load Gemini API key from environment
            import os
            from dotenv import load_dotenv
            
            # Load environment variables from .env file
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
            if os.path.exists(env_path):
                load_dotenv(env_path)
            
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if gemini_api_key:
                print("✅ Gemini API key loaded")
            else:
                print("⚠️ No Gemini API key found - will use fallback labeling")
            
            # Use the new method that handles corpus building and API key
            topic_results = self.topic_modeler.analyze_main_character_corpus(
                posts_data=self.collected_posts,
                deep_dive_results=self.deep_dive_results,
                api_key=gemini_api_key
            )
            
            # Apply topic persistence if enabled
            try:
                from shared.topic_evolution import TopicPersistenceManager
                from shared.utils import load_topics_json
                
                # Load existing topics for persistence comparison
                existing_topics_data = load_topics_json()
                
                # Check if persistence should be enabled (default: True)
                enable_persistence = existing_topics_data.get('metadata', {}).get('persistence_enabled', True)
                
                if enable_persistence and existing_topics_data.get('topics'):
                    print("🔄 Applying topic persistence...")
                    persistence_manager = TopicPersistenceManager()
                    topic_results = persistence_manager.update_topics_with_persistence(
                        new_topics=topic_results.get('topics', []),
                        existing_topics_data=existing_topics_data
                    )
                else:
                    print("📝 Topic persistence disabled or no previous topics found")
                    
            except Exception as e:
                print(f"⚠️ Topic persistence failed, continuing without: {e}")
                # Continue with original results if persistence fails
            
            print(f"✅ Main character topic analysis complete: {len(topic_results.get('topics', []))} topics identified")
            return topic_results
            
        except Exception as e:
            print(f"❌ Main character topic analysis failed: {e}")
            return {
                "collection_date": datetime.now(tz=timezone.utc).strftime('%Y-%m-%d'),
                "collection_timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "metadata": {
                    "error": str(e),
                    "total_posts": 0,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat()
                },
                "topics": []
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline results
        
        Returns:
            Dict containing pipeline summary
        """
        return {
            'pipeline_timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'collection_stats': self.collection_stats,
            'analysis_stats': {
                'ratios_detected': len(self.detected_ratios),
                'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
                'top_ratio_score': max([r.ratio_score for r in self.detected_ratios]) if self.detected_ratios else 0,
                'time_buckets_found': list(set([r.time_bucket for r in self.detected_ratios])) if self.detected_ratios else []
            },
            'settings': {
                'hours_back': self.hours_back,
                'min_ratio_score': self.min_ratio_score,
                'min_engagement': self.min_engagement,
                'top_n_ratios': self.top_n_ratios
            }
        }
    
    def get_top_ratios(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the top N ratios in a formatted structure
        
        Args:
            n: Number of ratios to return (default: all detected ratios)
            
        Returns:
            List of formatted ratio dictionaries
        """
        if not self.detected_ratios:
            return []
        
        n = n or len(self.detected_ratios)
        top_ratios = self.detected_ratios[:n]
        
        return [
            {
                'rank': i + 1,
                'ratio_score': round(ratio.ratio_score, 2),
                'time_bucket': ratio.time_bucket,
                'total_engagement': ratio.total_engagement,
                'explanation': ratio.explanation,
                'post': {
                    'uri': ratio.post['uri'],
                    'text': ratio.post['text'][:200] + '...' if len(ratio.post['text']) > 200 else ratio.post['text'],
                    'likes': ratio.post['like_count'],
                    'replies': ratio.post['reply_count'],
                    'reposts': ratio.post['repost_count'],
                    'quotes': ratio.post['quote_count'],
                    'character_count': ratio.post['character_count'],
                    'timestamp': ratio.post['timestamp']
                },
                'metrics': {
                    'total_engagement': ratio.total_engagement,
                    'positive_engagement': ratio.positive_engagement,
                    'negative_engagement': ratio.negative_engagement,
                    'post_age_minutes': round(ratio.post_age_minutes, 1)
                }
            }
            for i, ratio in enumerate(top_ratios)
        ]
    
    def save_daily_results(self, output_path: str = None) -> str:
        """
        Save the complete analysis results to today.json
        
        Args:
            output_path: Custom output path (default: main/data/today.json)
            
        Returns:
            Path where results were saved
        """
        if not output_path:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            os.makedirs(data_dir, exist_ok=True)
            output_path = os.path.join(data_dir, 'today.json')
        
        # Compile complete results
        results_data = {
            'collection_date': datetime.now(tz=timezone.utc).strftime('%Y-%m-%d'),
            'collection_timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'pipeline_summary': self.get_summary(),
            'top_ratios': self.deep_dive_results,
            'metadata': {
                'posts_collected': len(self.collected_posts.get('posts', [])) if self.collected_posts else 0,
                'ratios_detected': len(self.detected_ratios),
                'deep_dive_completed': len(self.deep_dive_results),
                'pipeline_settings': {
                    'hours_back': self.hours_back,
                    'min_ratio_score': self.min_ratio_score,
                    'min_engagement': self.min_engagement,
                    'deep_dive_top_n': self.deep_dive_top_n
                }
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Daily results saved to {output_path}")
        return output_path
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline: collect posts and detect ratios
        
        Returns:
            Dict containing full pipeline results
        """
        print("🎯 Starting Bluesky Ratio Detection Pipeline")
        print("=" * 50)
        
        try:
            # Step 1: Collect posts
            posts_data = self.collect_posts()
            
            # Step 2: Detect ratios
            ratio_results = self.detect_ratios()
            
            # Step 3: Deep dive into top ratios
            deep_dive_results = self.deep_dive_responses()
            
            # Step 4: Analyze main character topics
            global_topics = self.analyze_main_character_topics()
            
            # Step 5: Save daily results
            output_path = self.save_daily_results()
            
            # Step 6: Generate summary
            summary = self.get_summary()
            top_ratios = self.get_top_ratios()
            
            print("\n📊 Pipeline Summary:")
            print(f"   Posts collected: {summary['collection_stats']['total_posts_collected']}")
            print(f"   Ratios detected: {summary['analysis_stats']['ratios_detected']}")
            print(f"   Deep dive completed: {len(deep_dive_results)}")
            print(f"   Topics identified: {len(global_topics.get('topics', []))}")
            print(f"   Results saved: {output_path}")
            
            if top_ratios:
                print(f"   Top ratio score: {summary['analysis_stats']['top_ratio_score']:.2f}")
                print(f"   Time buckets found: {', '.join(map(str, summary['analysis_stats']['time_buckets_found']))}")
                
                print("\n🔥 Top 5 Ratios with Response Analysis:")
                for i, result in enumerate(deep_dive_results[:5]):
                    ratio = result['ratio_metrics']
                    sentiment = result['overall_sentiment']
                    bucket_labels = ["0-15m", "15-30m", "30-45m", "45-60m"]
                    bucket_str = bucket_labels[ratio['time_bucket']] if ratio['time_bucket'] < 4 else "60m+"
                    print(f"   {i+1}. {bucket_str} (Score: {ratio['score']}) - Responses: {sentiment['analyzed_total_responses']}")
                    print(f"      Text: {result['original_post']['text'][:80]}...")
                    print(f"      Sentiment: Polarity {sentiment['avg_polarity']:.2f}, Anger {sentiment['avg_anger']:.2f}")
                    print()
            
            return {
                'success': True,
                'summary': summary,
                'top_ratios': top_ratios,
                'deep_dive_results': deep_dive_results,
                'global_topics': global_topics,
                'saved_file': output_path,
                'timestamp': datetime.now(tz=timezone.utc).isoformat()
            }
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': self.get_summary(),
                'timestamp': datetime.now(tz=timezone.utc).isoformat()
            }


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Bluesky Ratio Detection Pipeline')
    parser.add_argument('--hours', type=int, default=1, help='Hours back to collect posts (default: 1)')
    parser.add_argument('--timeout', type=int, default=600, help='Collection timeout in seconds (default: 600)')
    parser.add_argument('--min-ratio', type=float, default=1.5, help='Minimum ratio score (default: 1.5)')
    parser.add_argument('--min-engagement', type=int, default=10, help='Minimum engagement threshold (default: 10)')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top ratios to keep (default: 20)')
    parser.add_argument('--deep-dive', type=int, default=5, help='Number of top ratios to deep dive (default: 5)')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--test', action='store_true', help='Run in test mode (5 minutes, lower thresholds)')
    parser.add_argument('--skip-deep-dive', action='store_true', help='Skip deep dive analysis (faster)')
    
    args = parser.parse_args()
    
    # Test mode adjustments
    if args.test:
        print("🧪 Running in test mode")
        hours_back = 5/60  # 5 minutes
        timeout = 120  # 2 minutes
        min_ratio = 0.5
        min_engagement = 1
    else:
        hours_back = args.hours
        timeout = args.timeout
        min_ratio = args.min_ratio
        min_engagement = args.min_engagement
    
    # Initialize and run pipeline
    pipeline = RatioPipeline(
        hours_back=hours_back,
        collection_timeout=timeout,
        min_ratio_score=min_ratio,
        min_engagement=min_engagement,
        top_n_ratios=args.top_n,
        deep_dive_top_n=args.deep_dive
    )
    
    # Run the pipeline
    results = pipeline.run_full_pipeline()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    # Return exit code
    return 0 if results['success'] else 1


if __name__ == "__main__":
    exit(main())