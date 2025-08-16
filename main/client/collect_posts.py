#!/usr/bin/env python3
"""
Bluesky Jetstream Posts Collector
Collects all posts from the last hour using Jetstream firehose
"""

import json
import time
import websocket
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any
import argparse
import sys


class JetstreamCollector:
    def __init__(self, hours_back: int = 1, max_timeout: int = 600):
        self.hours_back = hours_back
        self.max_timeout = max_timeout
        self.posts = []
        self.posts_dict = {}  # URI -> post data for quick lookup
        self.collected_replies = []  # Store reply info for later processing
        self.engagement_events = []
        self.connected = False
        self.finished = False
        self.start_time = None
        self.end_time = None
        
    def get_cursor_timestamp(self) -> int:
        """Get Unix microseconds timestamp for hours_back ago"""
        current_time = time.time()
        target_time = current_time - (self.hours_back * 3600)  # hours_back * seconds_per_hour
        return int(target_time * 1000000)  # Convert to microseconds
    
    def get_current_timestamp(self) -> int:
        """Get current Unix microseconds timestamp"""
        return int(time.time() * 1000000)
    
    def on_message(self, ws, message):
        """Handle incoming Jetstream messages"""
        try:
            data = json.loads(message)
            
            if data.get('kind') == 'commit' and data.get('commit', {}).get('operation') == 'create':
                commit = data.get('commit', {})
                collection = commit.get('collection')
                
                # Handle new posts (exclude replies)
                if collection == 'app.bsky.feed.post':
                    record = commit.get('record', {})
                    
                    # Check if this is a quote post
                    embed = record.get('embed', {})
                    is_quote_post = (embed.get('$type') == 'app.bsky.embed.record' and 
                                   embed.get('record', {}).get('uri'))
                    
                    # If it's a quote post, increment quote count for the referenced post
                    if is_quote_post:
                        quoted_uri = embed['record']['uri']
                        if quoted_uri in self.posts_dict:
                            self.posts_dict[quoted_uri]['quote_count'] += 1
                    
                    # Store replies for later processing (no real-time modification)
                    if record.get('reply'):
                        parent_uri = record.get('reply', {}).get('parent', {}).get('uri')
                        if parent_uri:
                            reply_info = {
                                'parent_uri': parent_uri,
                                'reply_time': data['time_us'],
                                'reply_author': data['did']
                            }
                            self.collected_replies.append(reply_info)
                        return  # Don't collect reply as its own post
                    
                    post_uri = f"at://{data['did']}/app.bsky.feed.post/{commit.get('rkey', '')}"
                    post_data = {
                        'uri': post_uri,
                        'did': data['did'],
                        'rkey': commit.get('rkey', ''),
                        'time_us': data['time_us'],
                        'timestamp': datetime.fromtimestamp(data['time_us'] / 1000000, tz=timezone.utc).isoformat(),
                        'text': record.get('text', ''),
                        'created_at': record.get('createdAt', ''),
                        'character_count': len(record.get('text', '')),
                        'langs': record.get('langs', []),
                        'embed': record.get('embed'),
                        'facets': record.get('facets'),
                        'tags': record.get('tags'),
                        'like_count': 0,
                        'reply_count': 0, 
                        'repost_count': 0,
                        'quote_count': 0,
                        'is_quote_post': is_quote_post,
                        'quoted_uri': embed['record']['uri'] if is_quote_post else None
                    }
                    self.posts.append(post_data)
                    self.posts_dict[post_uri] = post_data
                
                # Handle engagement events
                elif collection == 'app.bsky.feed.like':
                    record = commit.get('record', {})
                    subject_uri = record.get('subject', {}).get('uri')
                    if subject_uri and subject_uri in self.posts_dict:
                        self.posts_dict[subject_uri]['like_count'] += 1
                
                elif collection == 'app.bsky.feed.repost':
                    record = commit.get('record', {})
                    subject_uri = record.get('subject', {}).get('uri')
                    if subject_uri and subject_uri in self.posts_dict:
                        self.posts_dict[subject_uri]['repost_count'] += 1
                
                
                # Check if we've reached the current time (end of our window)
                if data['time_us'] >= self.end_time:
                    print(f"Reached end time. Collected {len(self.posts)} posts.")
                    self.finished = True
                    ws.close()
                    
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except KeyError as e:
            print(f"Missing key in message: {e}")
        except Exception as e:
            print(f"Error processing message: {e}")
            # Print first few chars of message for debugging
            print(f"Message preview: {str(message)[:100]}...")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print(f"WebSocket closed. Status: {close_status_code}, Message: {close_msg}")
        self.connected = False
    
    def on_open(self, ws):
        """Handle WebSocket open"""
        print("WebSocket connection opened")
        self.connected = True
    
    def collect_posts(self) -> List[Dict[str, Any]]:
        """Collect posts from the last hour via Jetstream"""
        cursor = self.get_cursor_timestamp()
        self.start_time = cursor
        self.end_time = self.get_current_timestamp()
        
        start_dt = datetime.fromtimestamp(cursor / 1000000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(self.end_time / 1000000, tz=timezone.utc)
        
        print(f"Collecting posts from {start_dt} to {end_dt}")
        print(f"Cursor: {cursor}")
        
        # Jetstream WebSocket URL with cursor (no collection filter to get all events)
        url = f"wss://jetstream2.us-east.bsky.network/subscribe?cursor={cursor}"
        
        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Start connection in a separate thread with timeout
        def run_with_timeout():
            ws.run_forever()
        
        thread = threading.Thread(target=run_with_timeout)
        thread.daemon = True
        thread.start()
        
        # Wait for connection or timeout
        start_wait = time.time()
        while not self.connected and time.time() - start_wait < 30:
            time.sleep(0.1)
        
        if not self.connected:
            print("Failed to connect to Jetstream")
            return []
        
        # Wait for collection to finish or timeout
        wait_start = time.time()
        while not self.finished and time.time() - wait_start < self.max_timeout:
            time.sleep(1)
            if len(self.posts) > 0 and len(self.posts) % 1000 == 0:
                print(f"Collected {len(self.posts)} posts so far...")
        
        # Close connection if still open
        if self.connected:
            ws.close()
        
        # Wait for thread to finish
        thread.join(timeout=5)
        
        print(f"Collection finished. Total posts: {len(self.posts)}")
        
        # Process collected replies after firehose collection is complete
        self.process_replies()
        
        return self.posts
    
    def process_replies(self):
        """Process collected replies to count replies for each post"""
        print(f"📊 Processing {len(self.collected_replies)} collected replies...")
        
        reply_count = 0
        for reply in self.collected_replies:
            parent_uri = reply['parent_uri']
            if parent_uri and parent_uri in self.posts_dict:
                self.posts_dict[parent_uri]['reply_count'] += 1
                reply_count += 1
        
        print(f"✅ Reply processing complete: {reply_count} replies counted for collected posts")
    
    def save_posts(self, filename: str = None) -> str:
        """Save collected posts to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"posts_{timestamp}.json"
        
        output_data = {
            'collection_info': {
                'start_time': datetime.fromtimestamp(self.start_time / 1000000, tz=timezone.utc).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time / 1000000, tz=timezone.utc).isoformat(),
                'hours_back': self.hours_back,
                'total_posts': len(self.posts),
                'collection_timestamp': datetime.now(tz=timezone.utc).isoformat()
            },
            'posts': self.posts
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Posts saved to {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(description='Collect Bluesky posts from Jetstream firehose')
    parser.add_argument('--hours', type=int, default=1, help='Hours back to collect posts from (default: 1)')
    parser.add_argument('--timeout', type=int, default=600, help='Max timeout in seconds (default: 600)')
    parser.add_argument('--output', type=str, help='Output filename (default: auto-generated)')
    parser.add_argument('--test', action='store_true', help='Run a quick test with last 5 minutes')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running test mode - collecting posts from last 5 minutes")
        collector = JetstreamCollector(hours_back=5/60, max_timeout=60)  # 5 minutes
    else:
        collector = JetstreamCollector(hours_back=args.hours, max_timeout=args.timeout)
    
    try:
        posts = collector.collect_posts()
        
        if posts:
            filename = collector.save_posts(args.output)
            print(f"Successfully collected {len(posts)} posts")
            print(f"Output file: {filename}")
        else:
            print("No posts collected")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
        if collector.posts:
            filename = collector.save_posts(args.output)
            print(f"Partial collection saved to {filename}")
    except Exception as e:
        print(f"Error during collection: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()