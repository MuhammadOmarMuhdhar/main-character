#!/usr/bin/env python3
"""
Lightweight metrics updater that refreshes engagement numbers
Designed to run frequently (every 2-5 minutes) as a cron job
"""

import sys
import os
import argparse
from datetime import datetime, timezone

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import our modules
from shared.data_transformer import MainCharacterTransformer
from shared.utils import load_today_json, save_today_json, get_fresh_engagement, get_current_timestamp

try:
    from atproto import Client
except ImportError:
    print("Warning: atproto not available, metrics updates will be limited")
    Client = None

try:
    from dotenv import load_dotenv
except ImportError:
    print("Installing python-dotenv...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'python-dotenv'])
    from dotenv import load_dotenv


def init_client() -> object:
    """Initialize AT Protocol client for API calls"""
    if not Client:
        return None
    
    try:
        # Load environment variables from .env file
        # Look for .env in current directory and parent directories
        env_path = find_env_file()
        if env_path:
            load_dotenv(env_path)
            print(f"📄 Loaded environment from {env_path}")
        
        # Get credentials
        handle = os.getenv('BLUESKY_HANDLE')
        password = os.getenv('BLUESKY_PASSWORD')
        
        if handle and password:
            # Create authenticated client
            client = Client()
            client.login(handle, password)
            print(f"✅ Authenticated as {handle}")
            return client
        else:
            # Create unauthenticated client (limited functionality)
            print("⚠️  No credentials found, using unauthenticated client")
            client = Client()
            return client
            
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return None


def find_env_file() -> str:
    """Find .env file in current or parent directories"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check current directory and up to 3 parent directories
    for _ in range(4):
        env_path = os.path.join(current_dir, '.env')
        if os.path.exists(env_path):
            return env_path
        current_dir = os.path.dirname(current_dir)
    
    return None


def fetch_fresh_engagement_batch(post_uris: list, client: object = None) -> dict:
    """
    Fetch fresh engagement metrics for multiple posts
    
    Args:
        post_uris: List of post URIs to fetch metrics for
        client: AT Protocol client
        
    Returns:
        Dict mapping post URIs to fresh engagement metrics
    """
    if not Client or not client:
        print("AT Protocol client not available")
        return {}
    
    fresh_data = {}
    
    # Process in batches (API might have limits)
    batch_size = 25  # Conservative batch size
    
    for i in range(0, len(post_uris), batch_size):
        batch = post_uris[i:i + batch_size]
        
        try:
            from atproto import models
            response = client.app.bsky.feed.get_posts(
                models.AppBskyFeedGetPosts.Params(uris=batch)
            )
            
            for post in response.posts:
                fresh_data[post.uri] = {
                    'likes': getattr(post, 'like_count', 0),
                    'replies': getattr(post, 'reply_count', 0),
                    'reposts': getattr(post, 'repost_count', 0),
                    'quotes': getattr(post, 'quote_count', 0)
                }
                
        except Exception as e:
            print(f"Error fetching batch {i//batch_size + 1}: {e}")
            # Continue with remaining batches
            continue
    
    return fresh_data


def update_metrics(max_age_hours: int = 12, verbose: bool = False) -> dict:
    """
    Update engagement metrics for current main characters
    
    Args:
        max_age_hours: Only update metrics for analyses newer than this
        verbose: Enable verbose output
        
    Returns:
        Dict with update results and success status
    """
    if verbose:
        print("🔄 Starting Metrics Update")
        print("=" * 30)
    
    try:
        # Load current data
        if verbose:
            print("📄 Loading current main characters data...")
        
        data = load_today_json()
        
        if not data or 'main_characters' not in data:
            if verbose:
                print("⚠️  No main characters data found")
            return {
                'success': False,
                'error': 'No main characters data found',
                'timestamp': get_current_timestamp()
            }
        
        main_characters = data['main_characters']
        if not main_characters:
            if verbose:
                print("⚠️  No main characters in data")
            return {
                'success': False,
                'error': 'No main characters to update',
                'timestamp': get_current_timestamp()
            }
        
        # Check if analysis is recent enough
        metadata = data.get('metadata', {})
        last_analysis = metadata.get('last_full_analysis')
        
        if last_analysis:
            try:
                analysis_time = datetime.fromisoformat(last_analysis.replace('Z', '+00:00'))
                age_hours = (datetime.now(timezone.utc) - analysis_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    if verbose:
                        print(f"⏰ Analysis is {age_hours:.1f} hours old (max: {max_age_hours}), skipping update")
                    return {
                        'success': False,
                        'error': f'Analysis too old ({age_hours:.1f} hours)',
                        'timestamp': get_current_timestamp()
                    }
            except Exception as e:
                if verbose:
                    print(f"⚠️  Could not parse analysis timestamp: {e}")
        
        if verbose:
            print(f"📊 Found {len(main_characters)} main characters to update")
        
        # Initialize client
        client = init_client()
        
        if not client:
            if verbose:
                print("❌ Could not initialize AT Protocol client")
            return {
                'success': False,
                'error': 'Could not initialize AT Protocol client',
                'timestamp': get_current_timestamp()
            }
        
        # Collect post URIs
        post_uris = []
        for character in main_characters:
            post_uri = character.get('post', {}).get('uri')
            if post_uri:
                post_uris.append(post_uri)
        
        if not post_uris:
            if verbose:
                print("⚠️  No post URIs found to update")
            return {
                'success': False,
                'error': 'No post URIs found',
                'timestamp': get_current_timestamp()
            }
        
        if verbose:
            print(f"🌐 Fetching fresh engagement for {len(post_uris)} posts...")
        
        # Fetch fresh engagement data
        fresh_engagement_data = fetch_fresh_engagement_batch(post_uris, client)
        
        if not fresh_engagement_data:
            if verbose:
                print("⚠️  No fresh engagement data retrieved (likely due to auth requirements)")
                print("   Keeping existing metrics and updating timestamp only")
            
            # Just update the timestamp if we can't get fresh data
            data['metadata']['last_metrics_update'] = get_current_timestamp()
            success = save_today_json(data)
            
            return {
                'success': True,  # Don't fail completely
                'updated_count': 0,
                'total_count': len(main_characters),
                'timestamp': get_current_timestamp(),
                'warning': 'Could not fetch fresh engagement data'
            }
        
        if verbose:
            print(f"✅ Retrieved fresh data for {len(fresh_engagement_data)} posts")
        
        # Update metrics using transformer
        transformer = MainCharacterTransformer()
        updated_data = transformer.update_metrics_only(data, fresh_engagement_data)
        
        # Save updated data
        if verbose:
            print("💾 Saving updated metrics...")
        
        success = save_today_json(updated_data)
        
        if not success:
            if verbose:
                print("❌ Failed to save updated data")
            return {
                'success': False,
                'error': 'Failed to save updated data',
                'timestamp': get_current_timestamp()
            }
        
        # Calculate update summary
        updated_count = len(fresh_engagement_data)
        total_count = len(main_characters)
        
        if verbose:
            print(f"\n📊 Metrics Update Complete!")
            print(f"   Updated: {updated_count}/{total_count} characters")
            print(f"   Timestamp: {updated_data['metadata']['last_metrics_update']}")
            
            # Show sample updates
            print(f"\n📈 Sample Updates:")
            for i, character in enumerate(updated_data['main_characters'][:3]):
                user = character['user']['handle']
                ratio = character['ratio']
                controversy = character['controversy']
                engagement = character['engagement']
                print(f"   {i+1}. @{user}: {ratio} ratio, {controversy}/10 controversy")
                print(f"      Engagement: {engagement['likes']}L {engagement['replies']}R {engagement['quotes']}Q")
        
        return {
            'success': True,
            'updated_count': updated_count,
            'total_count': total_count,
            'timestamp': get_current_timestamp()
        }
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Metrics update failed: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'timestamp': get_current_timestamp()
        }


def main():
    """Command line interface for metrics updater"""
    parser = argparse.ArgumentParser(description='Update engagement metrics for main characters')
    parser.add_argument('--max-age', type=int, default=12,
                       help='Maximum age of analysis in hours (default: 12)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress all output except errors')
    
    args = parser.parse_args()
    
    # Run the metrics update
    result = update_metrics(
        max_age_hours=args.max_age,
        verbose=args.verbose and not args.quiet
    )
    
    # Print results summary if not quiet
    if not args.quiet:
        if result['success']:
            print(f"✅ Updated {result['updated_count']}/{result['total_count']} characters")
        else:
            print(f"❌ Update failed: {result.get('error', 'Unknown error')}")
    
    # Always print errors
    if not result['success'] and args.quiet:
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()