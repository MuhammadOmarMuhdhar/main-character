import asyncio
import json
import os
import websockets
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from atproto import Client as AtprotoClient, models
from dateutil import parser
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables
load_dotenv()


class Client:
    def __init__(self, service_url: str = "https://bsky.social"):
        self.client = AtprotoClient(service_url)
        self.authenticated = False
    
    def login(self, identifier: str = None, password: str = None):
        """Login to Bluesky using provided credentials or environment variables"""
        # Use environment variables if credentials not provided
        if not identifier:
            identifier = os.getenv('BLUESKY_IDENTIFIER')
        if not password:
            password = os.getenv('BLUESKY_PASSWORD')
            
        if not identifier or not password:
            raise Exception("Credentials required. Provide them as parameters or set BLUESKY_IDENTIFIER and BLUESKY_PASSWORD in .env file")
        
        try:
            self.client.login(identifier, password)
            self.authenticated = True
            print(f"Successfully logged in as {identifier}")
        except Exception as e:
            print(f"Login failed: {e}")
            raise
    
    def search_top_posts(
        self,
        query: str = "the",
        limit: int = 100,
        since: Optional[datetime] = None,
        max_requests: int = 10
    ) -> List[Dict]:
        """
        Search for top posts using the searchPosts API
        
        Args:
            query: Search query string
            limit: Posts per request (max 100)
            since: Filter posts after this datetime
            max_requests: Maximum number of API requests to make
        
        Returns:
            List of posts with engagement metrics
        """
        all_posts = []
        cursor = None
        requests_made = 0
        
        while requests_made < max_requests:
            try:
                params = {
                    'q': query,
                    'sort': 'top',
                    'limit': min(limit, 100)
                }
                
                if since:
                    params['since'] = since.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                
                if cursor:
                    params['cursor'] = cursor
                
                response = self.client.app.bsky.feed.search_posts(params)
                posts = response.posts
                
                if not posts:
                    break
                
                # Convert posts to dictionaries with engagement metrics (filter out replies and low engagement)
                for post in posts:
                    # Skip replies - only include original posts
                    if hasattr(post.record, 'reply') and post.record.reply:
                        continue
                    
                    # Skip low-engagement posts from search (< 10 likes)
                    like_count = getattr(post, 'like_count', 0)
                    if like_count < 10:
                        continue
                        
                    post_data = {
                        'uri': post.uri,
                        'cid': post.cid,
                        'author': {
                            'did': post.author.did,
                            'handle': post.author.handle,
                            'display_name': getattr(post.author, 'display_name', ''),
                        },
                        'text': post.record.text,
                        'created_at': post.record.created_at,
                        'indexed_at': post.indexed_at,
                        'like_count': getattr(post, 'like_count', 0),
                        'repost_count': getattr(post, 'repost_count', 0),
                        'reply_count': getattr(post, 'reply_count', 0),
                        'engagement_score': (
                            getattr(post, 'like_count', 0) + 
                            getattr(post, 'repost_count', 0) * 2 + 
                            getattr(post, 'reply_count', 0)
                        )
                    }
                    all_posts.append(post_data)
                
                cursor = getattr(response, 'cursor', None)
                requests_made += 1
                
                if not cursor:
                    break
                    
                print(f"Fetched {len(posts)} posts (total: {len(all_posts)})")
                
            except Exception as e:
                print(f"Error fetching posts: {e}")
                break
        
        return all_posts
    
    def get_top_posts_multiple_queries(
        self,
        queries: List[str] = ["the", "a", "I", "you", "this", "that", "to", "in", "if", "and", "my", "me", "but", "of"],
        target_count: int = 1000,
        time_hours: int = 24
    ) -> List[Dict]:
        """
        Get top posts using multiple search queries to maximize coverage
        
        Args:
            queries: List of search terms to use
            target_count: Target number of posts to collect
            time_hours: Time window in hours to search within
        
        Returns:
            Deduplicated list of top posts sorted by engagement
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
            
        since_time = datetime.now() - timedelta(hours=time_hours)
        all_posts = []
        seen_uris = set()
        
        posts_per_query = max(200, target_count // len(queries))
        
        for query in queries:
            print(f"Searching with query: '{query}'")
            posts = self.search_top_posts(
                query=query,
                limit=500,
                since=since_time,
                max_requests=posts_per_query // 500 + 1
            )
            
            # Deduplicate posts
            for post in posts:
                if post['uri'] not in seen_uris:
                    all_posts.append(post)
                    seen_uris.add(post['uri'])
            
            if len(all_posts) >= target_count:
                break
        
        # Sort by engagement score
        all_posts.sort(key=lambda x: x['engagement_score'], reverse=True)
        
        return all_posts[:target_count]
    
    def get_posts_with_user_keywords(
        self,
        user_keywords: List[str],
        target_count: int = 1000,
        time_hours: int = 6,
        mix_generic: bool = True,
        generic_ratio: float = 0.3
    ) -> List[Dict]:
        """
        Get top posts using personalized user keywords with optional generic mixing
        
        Args:
            user_keywords: List of user-specific search terms
            target_count: Target number of posts to collect
            time_hours: Time window in hours to search within
            mix_generic: Whether to mix in some generic queries for discovery
            generic_ratio: Ratio of generic queries to include (0.0 to 1.0)
        
        Returns:
            Deduplicated list of top posts
        """
        if not user_keywords:
            print("No user keywords provided, falling back to generic queries")
            return self.get_top_posts_multiple_queries(target_count=target_count, time_hours=time_hours)
        
        print(f"Collecting posts with user keywords: {user_keywords[:5]}...")  # Show first 5
        
        # Prepare query list
        if mix_generic and generic_ratio > 0:
            # Mix user keywords with some generic terms for discovery
            generic_queries = ["the", "a", "I", "you", "this", "today", "new"]
            num_generic = int(len(user_keywords) * generic_ratio)
            
            if num_generic > 0:
                final_queries = user_keywords + generic_queries[:num_generic]
                print(f"Mixing {len(user_keywords)} user keywords with {num_generic} generic terms")
            else:
                final_queries = user_keywords
        else:
            final_queries = user_keywords
        
        # Use existing multiple queries method with personalized terms
        return self.get_top_posts_multiple_queries(
            queries=final_queries,
            target_count=target_count,
            time_hours=time_hours
        )
    
    def get_following_timeline(
        self,
        following_list: List[Dict],
        target_count: int = 500,
        time_hours: float = 0.5,
        include_reposts: bool = True,
        repost_weight: float = 0.7
    ) -> List[Dict]:
        """
        Get recent posts from users in the following list
        
        Args:
            following_list: List of followed users (already filtered by ratio upstream)
            target_count: Target number of posts to collect
            time_hours: Time window in hours (0.5 = 30 minutes for incremental updates)
            include_reposts: Whether to include reposts from followed users
            repost_weight: Weight multiplier for reposts (lower = less priority)
            
        Returns:
            List of recent posts from followed users with weighted engagement scores
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        if not following_list:
            print("No following list provided")
            return []
        
        # Use filtered following list directly (filtering done upstream by ratio)
        sampled_users = following_list
        print(f"Using {len(sampled_users)} ratio-filtered users from following list")
        
        since_time = datetime.now() - timedelta(hours=time_hours)
        all_posts = []
        seen_uris = set()
        
        posts_per_user = max(5, target_count // len(sampled_users))
        
        print(f"Collecting posts from {len(sampled_users)} followed users (last {time_hours} hours)")
        
        for i, user in enumerate(sampled_users):
            try:
                user_handle = user.get('handle', '')
                if not user_handle:
                    continue
                
                print(f"   ({i+1}/{len(sampled_users)}) Fetching from @{user_handle}")
                
                # Get recent posts from this user
                params = {
                    'actor': user_handle,
                    'limit': min(posts_per_user, 50),
                    'filter': 'posts_no_replies'
                }
                
                response = self.client.app.bsky.feed.get_author_feed(params)
                
                for feed_item in response.feed:
                    post = feed_item.post
                    is_repost = hasattr(feed_item, 'reason') and feed_item.reason
                    
                    # Skip reposts if not including them
                    if is_repost and not include_reposts:
                        continue
                    
                    # Check if post is within time window
                    post_time = parser.isoparse(post.record.created_at.replace('Z', '+00:00'))
                    if post_time < since_time.replace(tzinfo=post_time.tzinfo):
                        continue
                    
                    # Deduplicate
                    if post.uri in seen_uris:
                        continue
                    
                    # Calculate base engagement score
                    base_engagement = (
                        getattr(post, 'like_count', 0) + 
                        getattr(post, 'repost_count', 0) * 2 + 
                        getattr(post, 'reply_count', 0)
                    )
                    
                    # Apply repost weight if this is a repost
                    weighted_engagement = base_engagement * (repost_weight if is_repost else 1.0)
                    
                    post_data = {
                        'uri': post.uri,
                        'cid': post.cid,
                        'author': {
                            'did': post.author.did,
                            'handle': post.author.handle,
                            'display_name': getattr(post.author, 'display_name', ''),
                        },
                        'text': post.record.text,
                        'created_at': post.record.created_at,
                        'indexed_at': post.indexed_at,
                        'like_count': getattr(post, 'like_count', 0),
                        'repost_count': getattr(post, 'repost_count', 0),
                        'reply_count': getattr(post, 'reply_count', 0),
                        'engagement_score': weighted_engagement,
                        'base_engagement_score': base_engagement,
                        'source': 'following',
                        'post_type': 'repost' if is_repost else 'original',
                        'followed_user': user_handle,
                        'repost_weight_applied': repost_weight if is_repost else 1.0
                    }
                    
                    all_posts.append(post_data)
                    seen_uris.add(post.uri)
                    
                    if len(all_posts) >= target_count:
                        break
                
                if len(all_posts) >= target_count:
                    break
                    
            except Exception as e:
                print(f"   Error fetching from @{user.get('handle', 'unknown')}: {e}")
                continue
        
        # Sort by engagement score
        all_posts.sort(key=lambda x: x['engagement_score'], reverse=True)
        
        # Count post types for reporting
        original_count = len([p for p in all_posts if p.get('post_type') == 'original'])
        repost_count = len([p for p in all_posts if p.get('post_type') == 'repost'])
        
        print(f"Collected {len(all_posts)} posts from following timeline")
        if include_reposts:
            print(f"   Original posts: {original_count}, Reposts: {repost_count} (weight: {repost_weight})")
        else:
            print(f"   Original posts only: {original_count}")
        
        return all_posts[:target_count]
    
    def get_posts_hybrid(
        self,
        user_data: Dict,
        following_list: List[Dict] = None,
        target_count: int = 1000,
        time_hours: float = 2.0,
        following_ratio: float = 0.6,
        keyword_ratio: float = 0.4,
        keyword_extraction_method: str = "with_fallback",
        include_reposts: bool = True,
        repost_weight: float = 0.7
    ) -> List[Dict]:
        """
        Get posts using hybrid approach: following timeline + keyword search
        
        Args:
            user_data: User's engagement data
            following_list: List of followed users (optional, will fetch if not provided)
            target_count: Total target number of posts
            time_hours: Time window in hours
            following_ratio: Ratio of posts from following timeline (0.6 = 60%)
            keyword_ratio: Ratio of posts from keywords (0.4 = 40%)
            keyword_extraction_method: Method for keyword extraction
            include_reposts: Whether to include reposts from following timeline
            repost_weight: Weight multiplier for reposts (0.7 = 70% of original weight)
            
        Returns:
            Combined list of posts from following timeline and keyword search
        """
        following_count = int(target_count * following_ratio)
        keyword_count = int(target_count * keyword_ratio)
        
        all_posts = []
        seen_uris = set()
        
        # Get posts from following timeline
        if following_list:
            print(f"Collecting {following_count} posts from following timeline...")
            following_posts = self.get_following_timeline(
                following_list=following_list,
                target_count=following_count, 
                time_hours=time_hours,
                include_reposts=include_reposts,
                repost_weight=repost_weight
            )
            
            for post in following_posts:
                if post['uri'] not in seen_uris:
                    all_posts.append(post)
                    seen_uris.add(post['uri'])
        else:
            print("No following list provided, skipping following timeline")
        
        # Get posts from keyword search
        print(f"Collecting {keyword_count} posts from personalized keywords...")
        keyword_posts = self.get_posts_personalized(
            user_data=user_data,
            target_count=keyword_count,
            time_hours=time_hours,
            keyword_extraction_method=keyword_extraction_method
        )
        
        for post in keyword_posts:
            if post['uri'] not in seen_uris:
                post['source'] = 'keywords'  # Mark source
                all_posts.append(post)
                seen_uris.add(post['uri'])
        
        # Sort by engagement score
        all_posts.sort(key=lambda x: x['engagement_score'], reverse=True)
        
        # Enhanced breakdown reporting
        following_posts_final = [p for p in all_posts if p.get('source') == 'following']
        keyword_posts_final = [p for p in all_posts if p.get('source') == 'keywords']
        original_posts = [p for p in following_posts_final if p.get('post_type') == 'original']
        repost_posts = [p for p in following_posts_final if p.get('post_type') == 'repost']
        
        print(f"Hybrid collection complete: {len(all_posts)} total posts")
        print(f"   From following: {len(following_posts_final)} (original: {len(original_posts)}, reposts: {len(repost_posts)})")
        print(f"   From keywords: {len(keyword_posts_final)}")
        
        return all_posts[:target_count]

    def get_posts_personalized(
        self,
        user_data: Dict,
        target_count: int = 1000,
        time_hours: int = 24,
        keyword_extraction_method: str = "fallback"
    ) -> List[Dict]:
        """
        PLACEHOLDER: Get personalized posts based on user's engagement data
        TODO: Implement your new personalization algorithm here
        
        Args:
            user_data: User's posts, reposts, replies, likes data
            target_count: Target number of posts to collect
            time_hours: Time window in hours to search within
            keyword_extraction_method: Method to use (placeholder)
        
        Returns:
            List of posts (currently falls back to generic collection)
        """
        print("get_posts_personalized: Algorithm removed, falling back to generic collection")
        return self.get_top_posts_multiple_queries(target_count=target_count, time_hours=time_hours)

    async def connect_to_jetstream(
        self,
        websocket_url: str = "wss://jetstream1.us-east.bsky.network/subscribe",
        collections: List[str] = ["app.bsky.feed.post"]
    ):
        """
        Connect to Jetstream firehose for real-time post data
        
        Args:
            websocket_url: Jetstream WebSocket URL
            collections: Collections to subscribe to
        """
        params = {
            "wantedCollections": collections
        }
        
        try:
            async with websockets.connect(f"{websocket_url}?{self._build_query_string(params)}") as websocket:
                print("Connected to Jetstream firehose")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if data.get('kind') == 'commit' and 'commit' in data:
                            commit = data['commit']
                            
                            for op in commit.get('ops', []):
                                if op.get('action') == 'create' and 'record' in op:
                                    post_data = {
                                        'did': data['did'],
                                        'collection': op['path'].split('/')[0],
                                        'rkey': op['path'].split('/')[1],
                                        'record': op['record'],
                                        'timestamp': commit['rev']
                                    }
                                    
                                    yield post_data
                                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error processing message: {e}")
                        
        except Exception as e:
            print(f"WebSocket connection error: {e}")
    
    
    def extract_posts_from_feed(
        self, 
        feed_url_or_uri: str = None,
        query: str = "python", 
        limit: int = 25, 
        sort: str = "top",
        ranking: bool = False,
        time_hours: int = None
    ) -> List[Dict]:
        """
        Extract posts from a Bluesky feed (custom feed or search)
        
        Args:
            feed_url_or_uri: Custom feed URL or AT-URI (if provided, uses getFeed)
            query: Search term (used if no feed_url_or_uri provided)
            limit: Number of posts to return (max 100)
            sort: Sort order for search ('top' or 'latest')
        
        Returns:
            List of post dictionaries
        """
        try:
            # Calculate time cutoff if specified (matching pattern from other functions)
            since_time = None
            if time_hours is not None:
                since_time = datetime.now() - timedelta(hours=time_hours)
            
            # If feed URL/URI provided, use getFeed API
            if feed_url_or_uri:
                return self._extract_from_custom_feed(feed_url_or_uri, limit, ranking, since_time)
            
            # Otherwise use search API
            params = {
                'q': query,
                'sort': sort,
                'limit': min(limit, 100)
            }
            
            response = self.client.app.bsky.feed.search_posts(params)
            posts = []
            
            for post in response.posts:
                # Apply time filtering if specified (matching pattern from other functions)
                if since_time:
                    post_time = parser.isoparse(post.record.created_at.replace('Z', '+00:00'))
                    if post_time < since_time.replace(tzinfo=post_time.tzinfo):
                        continue
                
                post_data = {
                    'uri': post.uri,
                    'author_handle': post.author.handle,
                    'author_display_name': getattr(post.author, 'display_name', ''),
                    'text': post.record.text,
                    'created_at': post.record.created_at,
                    'like_count': getattr(post, 'like_count', 0),
                    'repost_count': getattr(post, 'repost_count', 0),
                    'reply_count': getattr(post, 'reply_count', 0)
                }
                if ranking and post_data['like_count'] > 10:
                    posts.append(post_data)
                else:
                    posts.append(post_data)
                                
            return posts
            
        except Exception as e:
            print(f"Error extracting posts: {e}")
            return []
    
    def _extract_from_custom_feed(self, 
                                  feed_url_or_uri: str, 
                                  limit: int, 
                                  ranking: bool = False,
                                  since_time: datetime = None) -> List[Dict]:
        """Extract posts from a custom feed using getFeed API"""
        try:
            # Convert URL to AT-URI if needed
            feed_uri = self._convert_feed_url_to_uri(feed_url_or_uri)
            
            params = {
                'feed': feed_uri,
                'limit': min(limit, 100)
            }
            
            response = self.client.app.bsky.feed.get_feed(params)
            posts = []
            
            for feed_item in response.feed:
                post = feed_item.post
                
                # Apply time filtering if specified (matching pattern from other functions)
                if since_time:
                    post_time = parser.isoparse(post.record.created_at.replace('Z', '+00:00'))
                    if post_time < since_time.replace(tzinfo=post_time.tzinfo):
                        continue
                
                post_data = {
                    'uri': post.uri,
                    'author_handle': post.author.handle,
                    'author_display_name': getattr(post.author, 'display_name', ''),
                    'text': post.record.text,
                    'created_at': post.record.created_at,
                    'like_count': getattr(post, 'like_count', 0),
                    'repost_count': getattr(post, 'repost_count', 0),
                    'reply_count': getattr(post, 'reply_count', 0)
                }
                if ranking and post_data['like_count'] > 10:
                    posts.append(post_data)
                else:
                    posts.append(post_data)
                    
            return posts
            
        except Exception as e:
            print(f"Error extracting from custom feed: {e}")
            return []
    
    def _convert_feed_url_to_uri(self, feed_url_or_uri: str) -> str:
        """Convert feed URL to AT-URI format"""
        if feed_url_or_uri.startswith('at://'):
            return feed_url_or_uri
        
        # Extract components from URL like: https://bsky.app/profile/bossett.social/feed/for-science
        if 'bsky.app/profile/' in feed_url_or_uri and '/feed/' in feed_url_or_uri:
            parts = feed_url_or_uri.split('/')
            handle = parts[parts.index('profile') + 1]
            feed_id = parts[parts.index('feed') + 1]
            
            # For now, we need to resolve the handle to DID
            # This is a simplified approach - in production you'd want to resolve the handle properly
            try:
                profile = self.client.app.bsky.actor.get_profile({'actor': handle})
                did = profile.did
                return f"at://{did}/app.bsky.feed.generator/{feed_id}"
            except Exception as e:
                print(f"Error resolving handle to DID: {e}")
                raise
        
        return feed_url_or_uri
    
    def _build_query_string(self, params: Dict) -> str:
        """Build query string from parameters"""
        return "&".join([f"{k}={v}" if isinstance(v, str) else f"{k}={','.join(v)}" 
                        for k, v in params.items()])
    
    def print_post_summary(self, posts: List[Dict], limit: int = 10):
        """Print a summary of top posts"""
        print(f"\n=== Top {min(limit, len(posts))} Posts ===\n")
        
        for i, post in enumerate(posts[:limit]):
            print(f"{i+1}. @{post['author']['handle']}")
            print(f"   Likes: {post['like_count']} | Reposts: {post['repost_count']} | Replies: {post['reply_count']}")
            print(f"   Score: {post['engagement_score']}")
            print(f"   Text: {post['text'][:100]}...")
            print(f"   Created: {post['created_at']}")
            print()