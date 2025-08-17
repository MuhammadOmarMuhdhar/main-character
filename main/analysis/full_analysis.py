#!/usr/bin/env python3
"""
Full 6-hour analysis pipeline that updates main characters
Designed to run as a cron job every 6 hours
"""

import sys
import os
import argparse
from datetime import datetime, timezone

# Add current directory and parent to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

# Import our modules
from pipeline import RatioPipeline
from shared.data_transformer import MainCharacterTransformer
from shared.rolling_window import RollingWindowManager
from shared.utils import load_today_json, save_today_json, save_topics_json, get_current_timestamp, migrate_legacy_data, is_legacy_format


def load_previous_analysis() -> dict:
    """Load previous analysis data for change detection"""
    try:
        return load_today_json()
    except Exception as e:
        print(f"Could not load previous analysis: {e}")
        return {}


def run_full_analysis(hours_back: int = 6, 
                     min_ratio_score: float = 1.5,
                     min_engagement: int = 10,
                     deep_dive_top_n: int = 10,
                     test_mode: bool = False) -> dict:
    """
    Run complete analysis pipeline and update main characters
    
    Args:
        hours_back: Hours back to collect posts
        min_ratio_score: Minimum ratio score threshold
        min_engagement: Minimum engagement threshold
        deep_dive_top_n: Number of top ratios to analyze deeply
        test_mode: Run in test mode with lower thresholds
        
    Returns:
        Dict with analysis results and success status
    """
    print("🎯 Starting Full Analysis Pipeline")
    print("=" * 50)
    print(f"Collection window: {hours_back} hours")
    print(f"Deep dive limit: {deep_dive_top_n} posts")
    print(f"Test mode: {test_mode}")
    print()
    
    try:
        # Load previous data for change detection
        print("📄 Loading previous analysis data...")
        previous_data = load_previous_analysis()
        
        # Check if we need to migrate legacy data
        if previous_data and is_legacy_format(previous_data):
            print("🔄 Migrating legacy data format to rolling window structure...")
            previous_data = migrate_legacy_data(previous_data)
            print("✅ Data migration completed")
        
        previous_count = len(previous_data.get('main_characters', []))
        print(f"   Previous analysis had {previous_count} main characters")
        
        # Adjust parameters for test mode
        if test_mode:
            hours_back = 0.5  # 30 minutes
            min_ratio_score = 0.5
            min_engagement = 1
            print("🧪 Test mode: Using reduced thresholds")
        
        # Initialize and run pipeline
        print("🚀 Initializing ratio detection pipeline...")
        pipeline = RatioPipeline(
            hours_back=hours_back,
            collection_timeout=3600,  # 1 hour max
            min_ratio_score=min_ratio_score,
            min_engagement=min_engagement,
            top_n_ratios=deep_dive_top_n * 2,  # Get more candidates
            deep_dive_top_n=deep_dive_top_n
        )
        
        # Run the complete pipeline
        print("⚡ Running complete analysis pipeline...")
        results = pipeline.run_full_pipeline()
        
        if not results.get('success', False):
            print("❌ Pipeline execution failed")
            return {
                'success': False,
                'error': 'Pipeline execution failed',
                'timestamp': get_current_timestamp()
            }
        
        # Transform results to main characters format
        print("🔄 Transforming results to main characters format...")
        
        # Initialize authenticated client for handle resolution
        try:
            from atproto import Client
            from dotenv import load_dotenv
            import os
            
            # Load environment variables
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
            if os.path.exists(env_path):
                load_dotenv(env_path)
            
            handle = os.getenv('BLUESKY_HANDLE')
            password = os.getenv('BLUESKY_PASSWORD')
            
            client = None
            if handle and password:
                client = Client()
                client.login(handle, password)
                print(f"✅ Authenticated for handle resolution")
            
        except Exception as e:
            print(f"⚠️  Could not authenticate for handle resolution: {e}")
            client = None
        
        # Transform results using new rolling window approach
        print("🔄 Transforming results for rolling window integration...")
        transformer = MainCharacterTransformer(client=client)
        new_main_characters, metadata = transformer.transform_for_rolling_window(results, previous_data)
        
        if not new_main_characters:
            print("⚠️  No new main characters identified in this 6-hour window")
            # If no new results, keep existing data but update timestamp
            current_data = previous_data
            if 'metadata' not in current_data:
                current_data['metadata'] = {}
            current_data['metadata']['last_full_analysis'] = get_current_timestamp()
            current_data['metadata']['last_metrics_update'] = get_current_timestamp()
        else:
            # Use rolling window manager to merge new results with existing 24-hour data
            print("🎯 Integrating new results into 24-hour rolling window...")
            rolling_window = RollingWindowManager(window_hours=24, update_interval_hours=hours_back)
            current_data = rolling_window.merge_analysis_results(previous_data, new_main_characters)
            
            # Print rolling window stats
            stats = rolling_window.get_window_stats(current_data['main_characters'])
            print(f"📊 Rolling Window Stats:")
            print(f"   Total characters: {stats['total_characters']}")
            print(f"   New: {stats['new_count']}, Rising: {stats['rising_count']}, Falling: {stats['falling_count']}, Stable: {stats['stable_count']}")
            print(f"   Avg controversy: {stats['avg_controversy']:.1f}, Peak: {stats['peak_controversy']:.1f}")
        
        # Save updated data
        print("💾 Saving updated main characters data...")
        success = save_today_json(current_data)
        
        if not success:
            print("❌ Failed to save results")
            return {
                'success': False,
                'error': 'Failed to save results',
                'timestamp': get_current_timestamp()
            }
        
        # Save topics data
        print("💾 Saving topics analysis data...")
        topics_data = results.get('global_topics', {})
        if topics_data:
            topics_success = save_topics_json(topics_data)
            if not topics_success:
                print("⚠️ Warning: Failed to save topics.json (main analysis still successful)")
        else:
            print("⚠️ Warning: No topics data found in pipeline results")
        
        # Print summary
        print("\n📊 Analysis Complete!")
        print("=" * 50)
        print(f"Posts analyzed: {results.get('summary', {}).get('collection_stats', {}).get('total_posts_collected', 0)}")
        print(f"Ratios detected: {len(results.get('deep_dive_results', []))}")
        final_characters = current_data.get('main_characters', [])
        print(f"Main characters identified: {len(final_characters)}")
        
        if final_characters:
            print(f"\n🔥 Top 5 Main Characters:")
            for i, char in enumerate(final_characters[:5]):
                user = char['user']['handle']
                ratio = char['ratio']
                controversy = char['controversy']
                change = char['change']
                print(f"   {i+1}. @{user} - {ratio} ratio, {controversy}/10 controversy ({change})")
                print(f"      Post: {char['post']['text'][:80]}...")
                print()
        
        return {
            'success': True,
            'main_characters_count': len(final_characters),
            'posts_analyzed': results.get('summary', {}).get('collection_stats', {}).get('total_posts_collected', 0),
            'ratios_detected': len(results.get('deep_dive_results', [])),
            'timestamp': get_current_timestamp()
        }
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'timestamp': get_current_timestamp()
        }


def main():
    """Command line interface for full analysis"""
    parser = argparse.ArgumentParser(description='Run full main character analysis')
    parser.add_argument('--hours', type=int, default=6, 
                       help='Hours back to collect posts (default: 6)')
    parser.add_argument('--min-ratio', type=float, default=1.5,
                       help='Minimum ratio score (default: 1.5)')
    parser.add_argument('--min-engagement', type=int, default=10,
                       help='Minimum engagement threshold (default: 10)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top ratios to analyze (default: 10)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with reduced thresholds')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Run the analysis
    result = run_full_analysis(
        hours_back=args.hours,
        min_ratio_score=args.min_ratio,
        min_engagement=args.min_engagement,
        deep_dive_top_n=args.top_n,
        test_mode=args.test
    )
    
    # Print results summary if verbose
    if args.verbose:
        print("\n📋 Final Results:")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Main characters: {result['main_characters_count']}")
            print(f"Posts analyzed: {result['posts_analyzed']}")
            print(f"Ratios detected: {result['ratios_detected']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Timestamp: {result['timestamp']}")
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()