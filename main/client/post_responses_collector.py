#!/usr/bin/env python3
"""
Bluesky Post Responses Collector
Collects all replies and quotes for a specific post
"""

import json
import time
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import argparse

try:
    from dotenv import load_dotenv
except ImportError:
    print("Installing python-dotenv...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'python-dotenv'])
    from dotenv import load_dotenv

try:
    from atproto import Client
except ImportError:
    print("Installing atproto...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'atproto'])
    from atproto import Client


class PostResponsesCollector:
    """Collects replies and quotes for a specific Bluesky post"""
    
    def __init__(self, handle: str = None, password: str = None, load_env: bool = True):
        """
        Initialize the responses collector
        
        Args:
            handle: Bluesky handle for authentication (optional)
            password: Bluesky password for authentication (optional)
            load_env: Whether to load credentials from .env file (default: True)
        """
        self.client = None
        self.authenticated = False
        
        # Load environment variables
        if load_env:
            # Look for .env file in current directory or parent directories
            env_path = self._find_env_file()
            if env_path:
                load_dotenv(env_path)
                print(f"📄 Loaded environment from {env_path}")
        
        # Get credentials (priority: params > env vars > none)
        final_handle = handle or os.getenv('BLUESKY_HANDLE')
        final_password = password or os.getenv('BLUESKY_PASSWORD')
        
        if final_handle and final_password:
            self.login(final_handle, final_password)
        else:
            print("⚠️  No credentials provided. Some features may require authentication.")
            self.client = Client()  # Unauthenticated client
    
    def _find_env_file(self) -> str:
        """Find .env file in current or parent directories"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check current directory and up to 3 parent directories
        for _ in range(4):
            env_path = os.path.join(current_dir, '.env')
            if os.path.exists(env_path):
                return env_path
            current_dir = os.path.dirname(current_dir)
        
        return None
    
    def login(self, handle: str, password: str):
        """Login to Bluesky for authenticated requests"""
        try:
            self.client = Client()
            self.client.login(handle, password)
            self.authenticated = True
            print(f"✅ Authenticated as {handle}")
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            self.client = Client()  # Use unauthenticated client
            self.authenticated = False
    
    def parse_post_uri(self, post_uri: str) -> Dict[str, str]:
        """
        Parse an AT Protocol post URI
        
        Args:
            post_uri: URI like "at://did:plc:123/app.bsky.feed.post/abc"
            
        Returns:
            Dict with did and rkey
        """
        try:
            # Remove "at://" prefix
            uri_parts = post_uri.replace('at://', '').split('/')
            did = uri_parts[0]
            rkey = uri_parts[-1]
            
            return {
                'did': did,
                'rkey': rkey,
                'collection': 'app.bsky.feed.post'
            }
        except Exception as e:
            raise ValueError(f"Invalid post URI format: {post_uri}") from e
    
    def get_post_details(self, post_uri: str) -> Dict[str, Any]:
        """
        Get the original post details
        
        Args:
            post_uri: The post URI to fetch
            
        Returns:
            Post details
        """
        try:
            parsed = self.parse_post_uri(post_uri)
            
            # Get the post record
            from atproto import models
            post_record = self.client.com.atproto.repo.get_record(
                models.ComAtprotoRepoGetRecord.Params(
                    repo=parsed['did'],
                    collection=parsed['collection'],
                    rkey=parsed['rkey']
                )
            )
            
            # Try to get post view with engagement metrics
            try:
                posts_response = self.client.app.bsky.feed.get_posts(
                    models.AppBskyFeedGetPosts.Params(uris=[post_uri])
                )
                if posts_response.posts:
                    post_view = posts_response.posts[0]
                    return {
                        'uri': post_uri,
                        'record': post_record.value,
                        'author': {
                            'did': post_view.author.did,
                            'handle': post_view.author.handle,
                            'display_name': getattr(post_view.author, 'display_name', ''),
                        },
                        'like_count': getattr(post_view, 'like_count', 0),
                        'reply_count': getattr(post_view, 'reply_count', 0),
                        'repost_count': getattr(post_view, 'repost_count', 0),
                        'quote_count': getattr(post_view, 'quote_count', 0),
                        'indexed_at': getattr(post_view, 'indexed_at', ''),
                        'text': post_record.value.text,
                        'created_at': post_record.value.created_at
                    }
            except:
                pass
            
            # Fallback to basic record data
            return {
                'uri': post_uri,
                'record': post_record.value,
                'text': post_record.value.text,
                'created_at': post_record.value.created_at,
                'like_count': 0,
                'reply_count': 0,
                'repost_count': 0,
                'quote_count': 0
            }
            
        except Exception as e:
            print(f"❌ Failed to get post details: {e}")
            return None
    
    def get_post_thread(self, post_uri: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get the post thread (replies) using the official thread API
        
        Args:
            post_uri: The post URI to get thread for
            depth: Maximum depth of replies to fetch
            
        Returns:
            Thread data with replies
        """
        try:
            from atproto import models
            thread_response = self.client.app.bsky.feed.get_post_thread(
                models.AppBskyFeedGetPostThread.Params(
                    uri=post_uri,
                    depth=depth
                )
            )
            
            return {
                'thread': thread_response.thread,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Failed to get post thread: {e}")
            return {
                'thread': None,
                'success': False,
                'error': str(e)
            }
    
    def extract_replies_from_thread(self, thread_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all replies from thread data
        
        Args:
            thread_data: Thread data from get_post_thread
            
        Returns:
            List of reply posts
        """
        replies = []
        
        def extract_replies_recursive(thread_node):
            """Recursively extract replies from thread structure"""
            if not thread_node:
                return
            
            # Check if this node has replies
            if hasattr(thread_node, 'replies') and thread_node.replies:
                for reply in thread_node.replies:
                    if hasattr(reply, 'post'):
                        reply_data = {
                            'uri': reply.post.uri,
                            'cid': reply.post.cid,
                            'author': {
                                'did': reply.post.author.did,
                                'handle': reply.post.author.handle,
                                'display_name': getattr(reply.post.author, 'display_name', ''),
                            },
                            'text': reply.post.record.text,
                            'created_at': reply.post.record.created_at,
                            'like_count': getattr(reply.post, 'like_count', 0),
                            'reply_count': getattr(reply.post, 'reply_count', 0),
                            'repost_count': getattr(reply.post, 'repost_count', 0),
                            'indexed_at': getattr(reply.post, 'indexed_at', ''),
                            'reply_to': reply.post.record.reply.parent.uri if reply.post.record.reply else None
                        }
                        replies.append(reply_data)
                    
                    # Recursively process nested replies
                    extract_replies_recursive(reply)
        
        if thread_data.get('success') and thread_data.get('thread'):
            extract_replies_recursive(thread_data['thread'])
        
        return replies
    
    def search_quote_posts(self, post_uri: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search for posts that quote the given post
        
        Args:
            post_uri: The original post URI
            max_results: Maximum number of quote posts to return
            
        Returns:
            List of quote posts
        """
        quotes = []
        
        try:
            # Search for posts that mention the URI (quote posts contain the URI)
            search_query = post_uri
            
            # Use search API to find posts containing the URI
            from atproto import models
            search_response = self.client.app.bsky.feed.search_posts(
                models.AppBskyFeedSearchPosts.Params(
                    q=search_query,
                    limit=min(max_results, 100)  # API limit
                )
            )
            
            for post in search_response.posts:
                # Check if this post actually quotes the target post
                if hasattr(post.record, 'embed') and post.record.embed:
                    embed = post.record.embed
                    if (hasattr(embed, 'record') and 
                        hasattr(embed.record, 'uri') and 
                        embed.record.uri == post_uri):
                        
                        quote_data = {
                            'uri': post.uri,
                            'cid': post.cid,
                            'author': {
                                'did': post.author.did,
                                'handle': post.author.handle,
                                'display_name': getattr(post.author, 'display_name', ''),
                            },
                            'text': post.record.text,
                            'created_at': post.record.created_at,
                            'like_count': getattr(post, 'like_count', 0),
                            'reply_count': getattr(post, 'reply_count', 0),
                            'repost_count': getattr(post, 'repost_count', 0),
                            'indexed_at': getattr(post, 'indexed_at', ''),
                            'quoted_post_uri': post_uri
                        }
                        quotes.append(quote_data)
            
        except Exception as e:
            print(f"⚠️  Quote search failed (may not be fully supported): {e}")
        
        return quotes
    
    def collect_all_responses(self, 
                            post_uri: str, 
                            include_quotes: bool = True,
                            max_quote_results: int = 100,
                            thread_depth: int = 10) -> Dict[str, Any]:
        """
        Collect all responses (replies and quotes) for a post
        
        Args:
            post_uri: The post URI to collect responses for
            include_quotes: Whether to search for quote posts
            max_quote_results: Maximum quote posts to collect
            thread_depth: Maximum reply thread depth
            
        Returns:
            Complete response data
        """
        print(f"🔍 Collecting responses for post: {post_uri}")
        
        # Initialize client if not authenticated
        if not self.client:
            self.client = Client()
        
        # Get original post details
        print("📄 Fetching original post...")
        original_post = self.get_post_details(post_uri)
        if not original_post:
            return {
                'success': False,
                'error': 'Failed to fetch original post'
            }
        
        # Get replies via thread API
        print("💬 Fetching replies...")
        thread_data = self.get_post_thread(post_uri, depth=thread_depth)
        replies = self.extract_replies_from_thread(thread_data)
        
        # Get quotes if requested
        quotes = []
        if include_quotes:
            print("🔄 Searching for quote posts...")
            quotes = self.search_quote_posts(post_uri, max_quote_results)
        
        # Compile results
        results = {
            'success': True,
            'collection_timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'original_post': original_post,
            'replies': replies,
            'quotes': quotes,
            'summary': {
                'original_post_uri': post_uri,
                'replies_found': len(replies),
                'quotes_found': len(quotes),
                'total_responses': len(replies) + len(quotes),
                'thread_depth_used': thread_depth,
                'original_engagement': {
                    'likes': original_post.get('like_count', 0),
                    'replies': original_post.get('reply_count', 0),
                    'reposts': original_post.get('repost_count', 0),
                    'quotes': original_post.get('quote_count', 0)
                }
            }
        }
        
        print(f"✅ Collection complete:")
        print(f"   Replies found: {len(replies)}")
        print(f"   Quotes found: {len(quotes)}")
        print(f"   Total responses: {len(replies) + len(quotes)}")
        
        return results


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Collect replies and quotes for a Bluesky post')
    parser.add_argument('post_uri', help='Post URI (at://...)')
    parser.add_argument('--handle', help='Bluesky handle for authentication')
    parser.add_argument('--password', help='Bluesky password for authentication')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--no-quotes', action='store_true', help='Skip quote post collection')
    parser.add_argument('--max-quotes', type=int, default=100, help='Max quote posts to collect')
    parser.add_argument('--thread-depth', type=int, default=10, help='Max reply thread depth')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = PostResponsesCollector(args.handle, args.password)
    
    # Collect responses
    results = collector.collect_all_responses(
        post_uri=args.post_uri,
        include_quotes=not args.no_quotes,
        max_quote_results=args.max_quotes,
        thread_depth=args.thread_depth
    )
    
    if not results.get('success'):
        print(f"❌ Collection failed: {results.get('error')}")
        return 1
    
    # Save or print results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {args.output}")
    else:
        # Print summary
        summary = results['summary']
        print(f"\n📊 RESPONSE COLLECTION SUMMARY")
        print("=" * 50)
        print(f"Original post: {summary['original_post_uri']}")
        print(f"Replies collected: {summary['replies_found']}")
        print(f"Quotes collected: {summary['quotes_found']}")
        print(f"Total responses: {summary['total_responses']}")
        
        # Show sample replies
        if results['replies']:
            print(f"\n💬 Sample Replies:")
            for i, reply in enumerate(results['replies'][:3]):
                print(f"{i+1}. @{reply['author']['handle']}: {reply['text'][:100]}...")
        
        # Show sample quotes
        if results['quotes']:
            print(f"\n🔄 Sample Quotes:")
            for i, quote in enumerate(results['quotes'][:3]):
                print(f"{i+1}. @{quote['author']['handle']}: {quote['text'][:100]}...")
    
    return 0


if __name__ == "__main__":
    exit(main())