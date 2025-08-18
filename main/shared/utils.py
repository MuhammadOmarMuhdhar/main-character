#!/usr/bin/env python3
"""
Shared utility functions for main character detection system
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional

try:
    from atproto import Client
except ImportError:
    print("Warning: atproto not available for fresh engagement fetching")
    Client = None


def format_engagement_number(num: int) -> str:
    """
    Format numbers like 8.2K, 47K, 1.2M
    
    Args:
        num: Raw number to format
        
    Returns:
        Formatted string with K/M suffix
    """
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K" 
    else:
        return str(num)


def generate_initials(display_name: str = "", handle: str = "") -> str:
    """
    Generate 2-letter initials for profile circle
    
    Args:
        display_name: User's display name
        handle: User's handle as fallback
        
    Returns:
        2-character initials (uppercase)
    """
    # Try display name first
    if display_name and display_name.strip():
        words = display_name.strip().split()
        if len(words) >= 2:
            return (words[0][0] + words[1][0]).upper()
        elif len(words) == 1:
            name = words[0]
            if len(name) >= 2:
                return (name[0] + name[1]).upper()
            else:
                return (name[0] + name[0]).upper()
    
    # Fallback to handle
    if handle:
        clean_handle = handle.replace('@', '')
        # For initials, use just the username part (before domain)
        username = clean_handle.split('.')[0] if '.' in clean_handle else clean_handle
        if len(username) >= 2:
            return (username[0] + username[1]).upper()
        elif len(username) == 1:
            return (username[0] + username[0]).upper()
    
    # Ultimate fallback
    return "MC"


def calculate_time_ago(timestamp: str) -> str:
    """
    Calculate relative time like '2 min ago', '3 hours ago'
    
    Args:
        timestamp: ISO timestamp string
        
    Returns:
        Human readable time difference
    """
    try:
        # Parse timestamp
        post_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        
        diff = current_time - post_time
        total_seconds = diff.total_seconds()
        
        if total_seconds < 60:
            return "just now"
        elif total_seconds < 3600:  # Less than 1 hour
            minutes = int(total_seconds / 60)
            return f"{minutes} min ago"
        elif total_seconds < 86400:  # Less than 1 day
            hours = int(total_seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:  # Days
            days = int(total_seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
            
    except Exception as e:
        print(f"Error calculating time ago: {e}")
        return "unknown"


def load_today_json(file_path: str = None) -> Dict[str, Any]:
    """
    Load and parse today.json safely
    
    Args:
        file_path: Custom file path (default: frontend/today.json)
        
    Returns:
        Parsed JSON data or empty structure if file doesn't exist
    """
    if not file_path:
        # Get path relative to this file - go to frontend directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', '..', 'frontend', 'today.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return empty structure if file doesn't exist
        return {
            "metadata": {
                "last_full_analysis": None,
                "last_metrics_update": None,
                "collection_period_hours": 6,
                "total_posts_analyzed": 0,
                "ratios_detected": 0
            },
            "main_characters": []
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing today.json: {e}")
        return {"metadata": {}, "main_characters": []}
    except Exception as e:
        print(f"Error loading today.json: {e}")
        return {"metadata": {}, "main_characters": []}


def save_today_json(data: Dict[str, Any], file_path: str = None) -> bool:
    """
    Save data to today.json with error handling
    
    Args:
        data: Data to save
        file_path: Custom file path (default: frontend/today.json)
        
    Returns:
        True if successful, False otherwise
    """
    if not file_path:
        # Get path relative to this file - go to frontend directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', '..', 'frontend', 'today.json')
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save with proper formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error saving today.json: {e}")
        return False


def load_topics_json(file_path: str = None) -> Dict[str, Any]:
    """
    Load and parse topics.json safely
    
    Args:
        file_path: Custom file path (default: frontend/topics.json)
        
    Returns:
        Parsed JSON data or empty structure if file doesn't exist
    """
    if not file_path:
        # Get path relative to this file - go to frontend directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', '..', 'frontend', 'topics.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return empty structure if file doesn't exist
        return {
            "collection_date": "",
            "collection_timestamp": "",
            "metadata": {
                "total_posts": 0,
                "topics_found": 0,
                "persistence_enabled": True
            },
            "topics": [],
            "archived_topics": []
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing topics.json: {e}")
        return {"metadata": {}, "topics": [], "archived_topics": []}
    except Exception as e:
        print(f"Error loading topics.json: {e}")
        return {"metadata": {}, "topics": [], "archived_topics": []}


def save_topics_json(data: Dict[str, Any], file_path: str = None) -> bool:
    """
    Save topics data to topics.json with error handling
    
    Args:
        data: Topics data to save
        file_path: Custom file path (default: frontend/topics.json)
        
    Returns:
        True if successful, False otherwise
    """
    if not file_path:
        # Get path relative to this file - go to frontend directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', '..', 'frontend', 'topics.json')
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save with proper formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Topics data saved to {file_path}")
        return True
        
    except Exception as e:
        print(f"Error saving topics.json: {e}")
        return False


def get_fresh_engagement(post_uri: str, client: Optional[Any] = None) -> Optional[Dict[str, int]]:
    """
    Fetch current engagement metrics for a post
    
    Args:
        post_uri: AT Protocol URI of the post
        client: Optional authenticated atproto client
        
    Returns:
        Dict with current engagement numbers or None if failed
    """
    if not Client:
        print("Warning: atproto not available, cannot fetch fresh engagement")
        return None
    
    try:
        # Use provided client or create new one
        if not client:
            client = Client()
        
        # Get posts to fetch engagement (this endpoint requires auth for some cases)
        from atproto import models
        response = client.app.bsky.feed.get_posts(
            models.AppBskyFeedGetPosts.Params(uris=[post_uri])
        )
        
        if response.posts:
            post = response.posts[0]
            return {
                'likes': getattr(post, 'like_count', 0),
                'replies': getattr(post, 'reply_count', 0),
                'reposts': getattr(post, 'repost_count', 0),
                'quotes': getattr(post, 'quote_count', 0)
            }
        else:
            print(f"No post found for URI: {post_uri}")
            return None
            
    except Exception as e:
        # If we get auth errors, skip this post but don't fail completely
        if "401" in str(e) or "AuthMissing" in str(e):
            print(f"Authentication required for {post_uri}, skipping...")
            return None
        else:
            print(f"Error fetching engagement for {post_uri}: {e}")
            return None


def format_all_engagement_metrics(engagement: Dict[str, int]) -> Dict[str, str]:
    """
    Format all engagement metrics for display
    
    Args:
        engagement: Dict with raw engagement numbers
        
    Returns:
        Dict with formatted versions of all metrics
    """
    return {
        'likes': format_engagement_number(engagement.get('likes', 0)),
        'replies': format_engagement_number(engagement.get('replies', 0)), 
        'reposts': format_engagement_number(engagement.get('reposts', 0)),
        'quotes': format_engagement_number(engagement.get('quotes', 0))
    }


def calculate_ratio_display(engagement: Dict[str, int]) -> str:
    """
    Create ratio display like '37:1' for quotes vs reposts
    
    Args:
        engagement: Dict with engagement numbers
        
    Returns:
        Ratio string formatted for display (quotes:reposts)
    """
    quotes = engagement.get('quotes', 0)
    reposts = engagement.get('reposts', 0)
    
    # Handle edge cases
    if quotes == 0 and reposts == 0:
        return "0:0"
    elif reposts == 0:
        return f"{quotes}:0"
    elif quotes == 0:
        return f"0:{reposts}"
    else:
        ratio = quotes / reposts
        if ratio >= 1:
            return f"{ratio:.0f}:1"
        else:
            return f"1:{1/ratio:.0f}"


def extract_handle_from_did(post: Dict[str, Any]) -> str:
    """
    Extract a clean handle from post data
    
    Args:
        post: Post data containing author info
        
    Returns:
        Clean handle string
    """
    # Try to get handle from various possible locations
    handle = None
    
    # Check if author info is available
    if 'author' in post and isinstance(post['author'], dict):
        handle = post['author'].get('handle')
    
    # Fallback to DID-based generation
    if not handle and 'did' in post:
        did = post['did']
        # Extract a readable part from DID if possible
        if 'did:plc:' in did:
            did_suffix = did.replace('did:plc:', '')[:8]
            handle = f"user_{did_suffix}"
        else:
            handle = "unknown_user"
    elif not handle and 'author_did' in post:
        did = post['author_did']
        if 'did:plc:' in did:
            did_suffix = did.replace('did:plc:', '')[:8]
            handle = f"user_{did_suffix}"
        else:
            handle = "unknown_user"
    
    # Clean the handle - only remove @ symbol, preserve domain
    if handle:
        handle = handle.replace('@', '')
    
    return handle or "unknown"


def clean_text_for_display(text: str, max_length: int = 200) -> str:
    """
    Clean and truncate text for display
    
    Args:
        text: Original text
        max_length: Maximum length before truncation
        
    Returns:
        Cleaned and potentially truncated text
    """
    if not text:
        return ""
    
    # Basic cleaning
    cleaned = text.strip()
    
    # Truncate if necessary
    if len(cleaned) > max_length:
        # Try to break at word boundary
        truncated = cleaned[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # If we can break reasonably close to limit
            cleaned = truncated[:last_space] + "..."
        else:
            cleaned = truncated + "..."
    
    return cleaned


def get_current_timestamp() -> str:
    """Get current UTC timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()


def resolve_did_to_handle(did: str, client: Optional[Any] = None) -> Optional[Dict[str, str]]:
    """
    Resolve a DID to get real handle and profile info
    
    Args:
        did: The DID to resolve
        client: Optional authenticated atproto client
        
    Returns:
        Dict with handle, display_name, and other profile info
    """
    if not Client or not did:
        return None
    
    try:
        if not client:
            client = Client()
        
        from atproto import models
        
        # Get profile information
        profile_response = client.app.bsky.actor.get_profile(
            models.AppBskyActorGetProfile.Params(actor=did)
        )
        
        if profile_response:
            return {
                'handle': profile_response.handle,  # Keep full handle with domain
                'display_name': getattr(profile_response, 'display_name', '') or '',
                'description': getattr(profile_response, 'description', '') or '',
                'avatar': getattr(profile_response, 'avatar', '') or '',
                'did': did
            }
        
        return None
        
    except Exception as e:
        print(f"Error resolving DID {did}: {e}")
        return None


def migrate_legacy_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate legacy data structure to rolling window format
    
    Args:
        data: Legacy today.json data
        
    Returns:
        Migrated data with rolling window fields
    """
    if not data or 'main_characters' not in data:
        return data
    
    current_timestamp = get_current_timestamp()
    migrated = data.copy()
    
    # Migrate main characters
    for char in migrated.get('main_characters', []):
        # Add rolling window fields if missing
        if 'first_detected' not in char:
            char['first_detected'] = current_timestamp
        
        if 'last_updated' not in char:
            char['last_updated'] = current_timestamp
        
        if 'analysis_windows' not in char:
            char['analysis_windows'] = [current_timestamp]
        
        if 'peak_controversy' not in char:
            char['peak_controversy'] = char.get('controversy', 0)
        
        if 'trend' not in char:
            char['trend'] = 'stable'  # Assume existing data is stable
    
    # Migrate metadata
    metadata = migrated.get('metadata', {})
    
    # Add rolling window metadata if missing
    if 'rolling_window' not in metadata:
        metadata['rolling_window'] = {
            'enabled': True,
            'window_hours': 24,
            'update_interval_hours': 6,
            'last_update': current_timestamp,
            'total_updates': 1,
            'first_update': current_timestamp
        }
    
    # Update collection period info
    if 'window_period_hours' not in metadata:
        metadata['window_period_hours'] = 24
    
    if 'collection_period_hours' not in metadata:
        metadata['collection_period_hours'] = 6
    
    migrated['metadata'] = metadata
    
    return migrated


def is_legacy_format(data: Dict[str, Any]) -> bool:
    """
    Check if data is in legacy format and needs migration
    
    Args:
        data: Data to check
        
    Returns:
        True if legacy format, False if already migrated
    """
    if not data or 'main_characters' not in data:
        return False
    
    # Check if any main character is missing rolling window fields
    for char in data['main_characters']:
        if 'first_detected' not in char or 'trend' not in char:
            return True
    
    # Check if metadata is missing rolling window info
    metadata = data.get('metadata', {})
    if 'rolling_window' not in metadata:
        return True
    
    return False