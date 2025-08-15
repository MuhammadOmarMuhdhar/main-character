#!/usr/bin/env python3
"""
Rolling Window Manager for 24-hour controversy tracking
Merges new 6-hour results with existing data and manages aging
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from .utils import get_current_timestamp

class RollingWindowManager:
    """Manages rolling 24-hour window with 6-hour updates"""
    
    def __init__(self, window_hours: int = 24, update_interval_hours: int = 6):
        self.window_hours = window_hours
        self.update_interval_hours = update_interval_hours
        self.current_time = datetime.now(timezone.utc)
    
    def merge_analysis_results(self, 
                              current_data: Dict[str, Any], 
                              new_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge new 6-hour analysis results with existing 24-hour data
        
        Args:
            current_data: Existing today.json data
            new_analysis: New main characters from latest 6-hour analysis
            
        Returns:
            Updated data with merged results
        """
        print(f"🔄 Merging {len(new_analysis)} new results with existing data")
        
        # Get existing main characters and age them out
        existing_chars = current_data.get('main_characters', [])
        aged_chars = self._age_out_old_entries(existing_chars)
        
        print(f"   Kept {len(aged_chars)} characters after aging (removed {len(existing_chars) - len(aged_chars)})")
        
        # Enhance new analysis with rolling window metadata
        enhanced_new = self._enhance_new_entries(new_analysis)
        
        # Merge and deduplicate
        merged_chars = self._merge_and_deduplicate(aged_chars, enhanced_new)
        
        # Calculate trends and re-rank
        final_chars = self._calculate_trends_and_rank(merged_chars)
        
        # Update metadata
        updated_metadata = self._update_metadata(current_data.get('metadata', {}), 
                                               len(new_analysis), len(final_chars))
        
        print(f"✅ Final result: {len(final_chars)} main characters in rolling window")
        
        return {
            'metadata': updated_metadata,
            'main_characters': final_chars
        }
    
    def _age_out_old_entries(self, existing_chars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove entries older than window_hours"""
        cutoff_time = self.current_time - timedelta(hours=self.window_hours)
        
        aged_chars = []
        for char in existing_chars:
            # Get the timestamp when this character was first detected
            first_detected = char.get('first_detected')
            if not first_detected:
                # Legacy data - assume it was detected now and keep it
                char['first_detected'] = get_current_timestamp()
                aged_chars.append(char)
                continue
            
            try:
                detected_time = datetime.fromisoformat(first_detected.replace('Z', '+00:00'))
                if detected_time > cutoff_time:
                    aged_chars.append(char)
                else:
                    print(f"   Aging out @{char.get('user', {}).get('handle', 'unknown')} (age: {(self.current_time - detected_time).total_seconds() / 3600:.1f}h)")
            except Exception as e:
                print(f"   Warning: Could not parse timestamp for {char.get('user', {}).get('handle', 'unknown')}: {e}")
                # Keep it to be safe
                aged_chars.append(char)
        
        return aged_chars
    
    def _enhance_new_entries(self, new_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add rolling window metadata to new entries"""
        current_timestamp = get_current_timestamp()
        
        enhanced = []
        for char in new_analysis:
            # Add rolling window fields if not present
            if 'first_detected' not in char:
                char['first_detected'] = current_timestamp
            
            if 'last_updated' not in char:
                char['last_updated'] = current_timestamp
            
            if 'analysis_windows' not in char:
                char['analysis_windows'] = [current_timestamp]
            
            if 'peak_controversy' not in char:
                char['peak_controversy'] = char.get('controversy', 0)
            
            if 'trend' not in char:
                char['trend'] = 'new'
            
            enhanced.append(char)
        
        return enhanced
    
    def _merge_and_deduplicate(self, 
                              existing_chars: List[Dict[str, Any]], 
                              new_chars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge existing and new characters, handling duplicates intelligently"""
        merged = {}
        current_timestamp = get_current_timestamp()
        
        # Add existing characters to merged dict
        for char in existing_chars:
            handle = char.get('user', {}).get('handle', 'unknown')
            merged[handle] = char
        
        # Process new characters
        for new_char in new_chars:
            handle = new_char.get('user', {}).get('handle', 'unknown')
            
            if handle in merged:
                # User already exists - update with new data
                existing = merged[handle]
                
                print(f"   Updating existing user @{handle}")
                
                # Update core metrics
                existing['controversy'] = new_char.get('controversy', existing['controversy'])
                existing['ratio'] = new_char.get('ratio', existing['ratio'])
                existing['engagement'] = new_char.get('engagement', existing['engagement'])
                existing['post'] = new_char.get('post', existing['post'])
                existing['sample_replies'] = new_char.get('sample_replies', existing['sample_replies'])
                existing['metrics'] = new_char.get('metrics', existing['metrics'])
                
                # Update rolling window metadata
                existing['last_updated'] = current_timestamp
                existing['analysis_windows'] = existing.get('analysis_windows', []) + [current_timestamp]
                
                # Track peak controversy
                current_controversy = new_char.get('controversy', 0)
                existing['peak_controversy'] = max(
                    existing.get('peak_controversy', 0), 
                    current_controversy
                )
                
                # Calculate trend (simplified)
                prev_controversy = existing.get('previous_analysis', {}).get('controversy', 0)
                if current_controversy > prev_controversy * 1.2:
                    existing['trend'] = 'rising'
                elif current_controversy < prev_controversy * 0.8:
                    existing['trend'] = 'falling'
                else:
                    existing['trend'] = 'stable'
                
                # Update change indicators
                existing['change'] = new_char.get('change', existing['change'])
                existing['change_type'] = new_char.get('change_type', existing['change_type'])
                
            else:
                # New user - add to merged
                print(f"   Adding new user @{handle}")
                merged[handle] = new_char
        
        return list(merged.values())
    
    def _calculate_trends_and_rank(self, characters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate final rankings with trend bonuses and sort"""
        
        for char in characters:
            # Calculate trending score (boosts recent high controversy)
            base_controversy = char.get('controversy', 0)
            peak_controversy = char.get('peak_controversy', base_controversy)
            trend = char.get('trend', 'stable')
            
            # Apply trend bonuses
            trend_multiplier = {
                'new': 1.1,      # 10% boost for new entries
                'rising': 1.15,   # 15% boost for rising trends
                'stable': 1.0,    # No change for stable
                'falling': 0.9    # 10% penalty for falling trends
            }.get(trend, 1.0)
            
            # Apply peak controversy factor (rewards sustained controversy)
            peak_factor = 1.0 + (peak_controversy - base_controversy) * 0.1
            
            # Calculate final ranking score
            char['_ranking_score'] = base_controversy * trend_multiplier * peak_factor
        
        # Sort by ranking score and assign ranks
        sorted_chars = sorted(characters, key=lambda x: x.get('_ranking_score', 0), reverse=True)
        
        for i, char in enumerate(sorted_chars, 1):
            char['rank'] = i
            # Remove internal ranking score
            char.pop('_ranking_score', None)
        
        return sorted_chars
    
    def _update_metadata(self, 
                        current_metadata: Dict[str, Any], 
                        new_entries_count: int, 
                        final_count: int) -> Dict[str, Any]:
        """Update metadata with rolling window information"""
        
        current_timestamp = get_current_timestamp()
        
        # Update basic metadata
        metadata = current_metadata.copy()
        metadata.update({
            'last_full_analysis': current_timestamp,
            'last_metrics_update': current_timestamp,
            'collection_period_hours': self.update_interval_hours,  # 6 hours per cycle
            'window_period_hours': self.window_hours,  # 24-hour rolling window
            'main_characters_count': final_count,
            'rolling_window': {
                'last_update': current_timestamp,
                'new_entries_added': new_entries_count,
                'total_in_window': final_count,
                'window_hours': self.window_hours,
                'update_interval_hours': self.update_interval_hours
            }
        })
        
        return metadata
    
    def get_window_stats(self, characters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the current rolling window"""
        if not characters:
            return {
                'total_characters': 0,
                'new_count': 0,
                'rising_count': 0,
                'falling_count': 0,
                'stable_count': 0
            }
        
        trends = [char.get('trend', 'stable') for char in characters]
        
        return {
            'total_characters': len(characters),
            'new_count': trends.count('new'),
            'rising_count': trends.count('rising'),
            'falling_count': trends.count('falling'),
            'stable_count': trends.count('stable'),
            'avg_controversy': sum(char.get('controversy', 0) for char in characters) / len(characters),
            'peak_controversy': max((char.get('peak_controversy', 0) for char in characters), default=0)
        }