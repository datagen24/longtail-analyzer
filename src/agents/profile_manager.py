"""
Profile manager for maintaining and evolving attacker profiles over time.

This module provides persistent storage and management of attacker profiles
using SQLite for structured data and support for vector embeddings.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..models.profile import AttackerProfile

logger = logging.getLogger(__name__)


class ProfileManager:
    """
    Manages attacker profiles with persistent SQLite storage.
    
    This class provides CRUD operations for attacker profiles with support for:
    - Incremental profile updates
    - Vector embedding storage
    - Analysis state tracking
    - Profile merging and correlation
    """
    
    def __init__(self, db_path: str = "data/profiles.db"):
        """
        Initialize the profile manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database connection
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        self._init_db()
        logger.info(f"ProfileManager initialized with database: {self.db_path}")
        
    def _init_db(self) -> None:
        """Initialize SQLite tables for profile storage."""
        # Main profiles table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                profile_data TEXT NOT NULL,
                embedding BLOB,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL DEFAULT 0.0,
                total_events INTEGER DEFAULT 0,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                analysis_depth TEXT DEFAULT 'shallow'
            )
        """)
        
        # Analysis state tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_state (
                analysis_id TEXT PRIMARY KEY,
                state_data TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Profile relationships
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profile_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity_id) REFERENCES profiles (entity_id),
                FOREIGN KEY (target_entity_id) REFERENCES profiles (entity_id),
                UNIQUE (source_entity_id, target_entity_id, relationship_type)
            )
        """)
        
        # Create indexes for performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_profiles_entity_type ON profiles (entity_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_profiles_last_seen ON profiles (last_seen)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_profiles_confidence ON profiles (confidence_score)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON profile_relationships (source_entity_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON profile_relationships (target_entity_id)")
        
        self.conn.commit()
        logger.debug("Database tables initialized")
    
    def save_profile(self, profile: AttackerProfile) -> bool:
        """
        Save or update an attacker profile.
        
        Args:
            profile: AttackerProfile instance to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert profile to JSON, handling numpy arrays
            profile_dict = profile.to_dict()
            
            # Handle embedding separately
            embedding_bytes = None
            if profile.behavioral_embedding is not None:
                embedding_bytes = profile.behavioral_embedding.tobytes()
            
            # Remove embedding from JSON data
            profile_dict.pop('behavioral_embedding', None)
            
            self.conn.execute("""
                INSERT OR REPLACE INTO profiles 
                (entity_id, entity_type, profile_data, embedding, last_updated, 
                 confidence_score, total_events, first_seen, last_seen, analysis_depth)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.entity_id,
                profile.entity_type,
                json.dumps(profile_dict, default=str),
                embedding_bytes,
                datetime.now(),
                profile.get_confidence_score('overall'),
                profile.total_events,
                profile.first_seen,
                profile.last_seen,
                profile.analysis_depth
            ))
            
            self.conn.commit()
            logger.debug(f"Saved profile: {profile.entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving profile {profile.entity_id}: {e}")
            self.conn.rollback()
            return False
    
    def get_profile(self, entity_id: str) -> Optional[AttackerProfile]:
        """
        Retrieve a profile by entity ID.
        
        Args:
            entity_id: ID of the entity to retrieve
            
        Returns:
            AttackerProfile instance or None if not found
        """
        try:
            cursor = self.conn.execute(
                "SELECT * FROM profiles WHERE entity_id = ?", 
                (entity_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Reconstruct profile from database row
            profile_data = json.loads(row['profile_data'])
            
            # Add embedding if present
            if row['embedding']:
                profile_data['behavioral_embedding'] = np.frombuffer(
                    row['embedding'], dtype=np.float32
                )
            
            profile = AttackerProfile.from_dict(profile_data)
            logger.debug(f"Retrieved profile: {entity_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error retrieving profile {entity_id}: {e}")
            return None
    
    def get_profiles_by_type(self, entity_type: str) -> List[AttackerProfile]:
        """
        Get all profiles of a specific type.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            List of AttackerProfile instances
        """
        try:
            cursor = self.conn.execute(
                "SELECT * FROM profiles WHERE entity_type = ? ORDER BY last_seen DESC",
                (entity_type,)
            )
            rows = cursor.fetchall()
            
            profiles = []
            for row in rows:
                profile_data = json.loads(row['profile_data'])
                
                if row['embedding']:
                    profile_data['behavioral_embedding'] = np.frombuffer(
                        row['embedding'], dtype=np.float32
                    )
                
                profile = AttackerProfile.from_dict(profile_data)
                profiles.append(profile)
            
            logger.debug(f"Retrieved {len(profiles)} profiles of type {entity_type}")
            return profiles
            
        except Exception as e:
            logger.error(f"Error retrieving profiles by type {entity_type}: {e}")
            return []
    
    def get_active_profiles(self, hours: int = 24) -> List[AttackerProfile]:
        """
        Get profiles that have been active within the specified hours.
        
        Args:
            hours: Number of hours to check for activity
            
        Returns:
            List of active AttackerProfile instances
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            cursor = self.conn.execute(
                "SELECT * FROM profiles WHERE last_seen > ? ORDER BY last_seen DESC",
                (cutoff_time,)
            )
            rows = cursor.fetchall()
            
            profiles = []
            for row in rows:
                profile_data = json.loads(row['profile_data'])
                
                if row['embedding']:
                    profile_data['behavioral_embedding'] = np.frombuffer(
                        row['embedding'], dtype=np.float32
                    )
                
                profile = AttackerProfile.from_dict(profile_data)
                profiles.append(profile)
            
            logger.debug(f"Retrieved {len(profiles)} active profiles")
            return profiles
            
        except Exception as e:
            logger.error(f"Error retrieving active profiles: {e}")
            return []
    
    def update_profile_incrementally(
        self,
        entity_id: str,
        new_events: List[Dict[str, Any]],
        llm_analysis: Optional[Dict[str, Any]] = None
    ) -> Optional[AttackerProfile]:
        """
        Incrementally update a profile with new events.
        
        This is crucial for handling large datasets efficiently by avoiding
        full profile reconstruction on each update.
        
        Args:
            entity_id: ID of the entity to update
            new_events: List of new events to incorporate
            llm_analysis: Optional LLM analysis results
            
        Returns:
            Updated AttackerProfile instance or None if error
        """
        try:
            # Get existing profile or create new one
            profile = self.get_profile(entity_id)
            if not profile:
                profile = self._create_profile_from_events(entity_id, new_events)
            else:
                # Update existing profile
                self._merge_new_events(profile, new_events)
            
            # Apply LLM analysis if provided
            if llm_analysis:
                self._apply_llm_insights(profile, llm_analysis)
            
            # Save updated profile
            if self.save_profile(profile):
                logger.debug(f"Incrementally updated profile: {entity_id}")
                return profile
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error updating profile {entity_id}: {e}")
            return None
    
    def _create_profile_from_events(
        self, 
        entity_id: str, 
        events: List[Dict[str, Any]]
    ) -> AttackerProfile:
        """
        Create a new profile from events.
        
        Args:
            entity_id: ID for the new profile
            events: Events to base the profile on
            
        Returns:
            New AttackerProfile instance
        """
        # Determine entity type from entity_id
        if entity_id.startswith('ip_'):
            entity_type = 'ip'
        elif entity_id.startswith('asn_'):
            entity_type = 'asn'
        else:
            entity_type = 'composite'
        
        # Extract first and last timestamps
        timestamps = []
        for event in events:
            if '@timestamp' in event:
                try:
                    timestamps.append(datetime.fromisoformat(event['@timestamp']))
                except ValueError:
                    continue
        
        if timestamps:
            first_seen = min(timestamps)
            last_seen = max(timestamps)
        else:
            first_seen = last_seen = datetime.now()
        
        # Create new profile
        profile = AttackerProfile(
            entity_id=entity_id,
            entity_type=entity_type,
            first_seen=first_seen,
            last_seen=last_seen
        )
        
        # Update with events
        profile.update_activity(events)
        
        return profile
    
    def _merge_new_events(
        self, 
        profile: AttackerProfile, 
        new_events: List[Dict[str, Any]]
    ) -> None:
        """
        Merge new events into an existing profile.
        
        Args:
            profile: Existing profile to update
            new_events: New events to merge
        """
        # Update activity tracking
        profile.update_activity(new_events)
        
        # Update patterns based on new events
        # This is where pattern analysis would be integrated
        # For now, we'll just update basic metrics
        
        # Update confidence based on data volume
        if profile.total_events > 100:
            profile.set_confidence_score('data_volume', 0.8)
        elif profile.total_events > 50:
            profile.set_confidence_score('data_volume', 0.6)
        else:
            profile.set_confidence_score('data_volume', 0.4)
    
    def _apply_llm_insights(
        self, 
        profile: AttackerProfile, 
        analysis: Dict[str, Any]
    ) -> None:
        """
        Apply LLM analysis insights to a profile.
        
        Args:
            profile: Profile to update
            analysis: LLM analysis results
        """
        # Update confidence scores
        if 'confidence' in analysis:
            profile.set_confidence_score('llm_analysis', analysis['confidence'])
        
        # Add TTPs
        if 'ttps' in analysis:
            for ttp in analysis['ttps']:
                profile.add_ttp(ttp)
        
        # Update analysis depth
        if 'analysis_depth' in analysis:
            profile.analysis_depth = analysis['analysis_depth']
        
        # Update data quality score
        if 'data_quality' in analysis:
            profile.data_quality_score = analysis['data_quality']
    
    def delete_profile(self, entity_id: str) -> bool:
        """
        Delete a profile by entity ID.
        
        Args:
            entity_id: ID of the profile to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("DELETE FROM profiles WHERE entity_id = ?", (entity_id,))
            self.conn.execute(
                "DELETE FROM profile_relationships WHERE source_entity_id = ? OR target_entity_id = ?",
                (entity_id, entity_id)
            )
            self.conn.commit()
            
            logger.info(f"Deleted profile: {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting profile {entity_id}: {e}")
            self.conn.rollback()
            return False
    
    def get_profile_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored profiles.
        
        Returns:
            Dictionary with profile statistics
        """
        try:
            cursor = self.conn.execute("""
                SELECT 
                    entity_type,
                    COUNT(*) as count,
                    AVG(confidence_score) as avg_confidence,
                    AVG(total_events) as avg_events,
                    MAX(last_seen) as most_recent
                FROM profiles 
                GROUP BY entity_type
            """)
            
            stats = {
                'by_type': {},
                'total_profiles': 0,
                'total_events': 0
            }
            
            for row in cursor.fetchall():
                stats['by_type'][row['entity_type']] = {
                    'count': row['count'],
                    'avg_confidence': row['avg_confidence'],
                    'avg_events': row['avg_events'],
                    'most_recent': row['most_recent']
                }
                stats['total_profiles'] += row['count']
            
            # Get total events
            cursor = self.conn.execute("SELECT SUM(total_events) as total FROM profiles")
            total_row = cursor.fetchone()
            stats['total_events'] = total_row['total'] or 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting profile statistics: {e}")
            return {}
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("ProfileManager database connection closed")
