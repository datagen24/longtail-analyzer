"""
State store for analysis state persistence using SQLite.

This module provides persistent storage for analysis state, checkpoints,
and system metadata to enable resumable long-running analyses.
"""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class StateStore:
    """
    State store for analysis state persistence.
    
    This class provides persistent storage for analysis state, checkpoints,
    and system metadata using SQLite.
    """
    
    def __init__(self, db_path: str = "data/analysis_state.db"):
        """
        Initialize the state store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database connection
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        self._init_db()
        logger.info(f"StateStore initialized with database: {self.db_path}")
    
    def _init_db(self) -> None:
        """Initialize SQLite tables for state storage."""
        # Analysis sessions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                session_id TEXT PRIMARY KEY,
                session_type TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status TEXT DEFAULT 'active',
                config_data TEXT,
                metadata TEXT
            )
        """)
        
        # Analysis checkpoints table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                checkpoint_time TIMESTAMP NOT NULL,
                checkpoint_data TEXT NOT NULL,
                progress_percentage REAL DEFAULT 0.0,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
            )
        """)
        
        # System state table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                state_key TEXT PRIMARY KEY,
                state_value TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        
        # Analysis results table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                result_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                result_type TEXT NOT NULL,
                result_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
            )
        """)
        
        # Create indexes for performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON analysis_sessions (status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_type ON analysis_sessions (session_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON analysis_checkpoints (session_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_results_session ON analysis_results (session_id)")
        
        self.conn.commit()
        logger.debug("State store tables initialized")
    
    def create_analysis_session(
        self,
        session_id: str,
        session_type: str,
        config_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new analysis session.
        
        Args:
            session_id: Unique session identifier
            session_type: Type of analysis session
            config_data: Configuration data for the session
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
                INSERT INTO analysis_sessions 
                (session_id, session_type, start_time, config_data, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                session_type,
                datetime.now(),
                json.dumps(config_data) if config_data else None,
                json.dumps(metadata) if metadata else None
            ))
            
            self.conn.commit()
            logger.info(f"Created analysis session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating analysis session {session_id}: {e}")
            self.conn.rollback()
            return False
    
    def update_analysis_session(
        self,
        session_id: str,
        status: Optional[str] = None,
        end_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an analysis session.
        
        Args:
            session_id: Session identifier
            status: New status
            end_time: End time for the session
            metadata: Updated metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            updates = []
            params = []
            
            if status:
                updates.append("status = ?")
                params.append(status)
            
            if end_time:
                updates.append("end_time = ?")
                params.append(end_time)
            
            if metadata:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            if updates:
                params.append(session_id)
                query = f"UPDATE analysis_sessions SET {', '.join(updates)} WHERE session_id = ?"
                self.conn.execute(query, params)
                self.conn.commit()
                
                logger.debug(f"Updated analysis session: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating analysis session {session_id}: {e}")
            self.conn.rollback()
            return False
    
    def get_analysis_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an analysis session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        try:
            cursor = self.conn.execute(
                "SELECT * FROM analysis_sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            session_data = dict(row)
            
            # Parse JSON fields
            if session_data['config_data']:
                session_data['config_data'] = json.loads(session_data['config_data'])
            if session_data['metadata']:
                session_data['metadata'] = json.loads(session_data['metadata'])
            
            return session_data
            
        except Exception as e:
            logger.error(f"Error getting analysis session {session_id}: {e}")
            return None
    
    def save_checkpoint(
        self,
        session_id: str,
        checkpoint_data: Dict[str, Any],
        progress_percentage: float = 0.0
    ) -> bool:
        """
        Save an analysis checkpoint.
        
        Args:
            session_id: Session identifier
            checkpoint_data: Checkpoint data to save
            progress_percentage: Progress percentage (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint_id = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.conn.execute("""
                INSERT INTO analysis_checkpoints 
                (checkpoint_id, session_id, checkpoint_time, checkpoint_data, progress_percentage)
                VALUES (?, ?, ?, ?, ?)
            """, (
                checkpoint_id,
                session_id,
                datetime.now(),
                json.dumps(checkpoint_data),
                progress_percentage
            ))
            
            self.conn.commit()
            logger.debug(f"Saved checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving checkpoint for session {session_id}: {e}")
            self.conn.rollback()
            return False
    
    def get_latest_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Latest checkpoint data or None if not found
        """
        try:
            cursor = self.conn.execute("""
                SELECT * FROM analysis_checkpoints 
                WHERE session_id = ? 
                ORDER BY checkpoint_time DESC 
                LIMIT 1
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            checkpoint_data = dict(row)
            checkpoint_data['checkpoint_data'] = json.loads(checkpoint_data['checkpoint_data'])
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error getting latest checkpoint for session {session_id}: {e}")
            return None
    
    def save_analysis_result(
        self,
        session_id: str,
        result_type: str,
        result_data: Dict[str, Any]
    ) -> bool:
        """
        Save analysis results.
        
        Args:
            session_id: Session identifier
            result_type: Type of result
            result_data: Result data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result_id = f"{session_id}_{result_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.conn.execute("""
                INSERT INTO analysis_results 
                (result_id, session_id, result_type, result_data)
                VALUES (?, ?, ?, ?)
            """, (
                result_id,
                session_id,
                result_type,
                json.dumps(result_data)
            ))
            
            self.conn.commit()
            logger.debug(f"Saved analysis result: {result_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis result for session {session_id}: {e}")
            self.conn.rollback()
            return False
    
    def get_analysis_results(
        self,
        session_id: str,
        result_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get analysis results for a session.
        
        Args:
            session_id: Session identifier
            result_type: Optional result type filter
            
        Returns:
            List of analysis results
        """
        try:
            if result_type:
                cursor = self.conn.execute("""
                    SELECT * FROM analysis_results 
                    WHERE session_id = ? AND result_type = ?
                    ORDER BY created_at DESC
                """, (session_id, result_type))
            else:
                cursor = self.conn.execute("""
                    SELECT * FROM analysis_results 
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                """, (session_id,))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                result_data = dict(row)
                result_data['result_data'] = json.loads(result_data['result_data'])
                results.append(result_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting analysis results for session {session_id}: {e}")
            return []
    
    def set_system_state(self, key: str, value: Any, description: Optional[str] = None) -> bool:
        """
        Set a system state value.
        
        Args:
            key: State key
            value: State value
            description: Optional description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO system_state 
                (state_key, state_value, last_updated, description)
                VALUES (?, ?, ?, ?)
            """, (
                key,
                json.dumps(value),
                datetime.now(),
                description
            ))
            
            self.conn.commit()
            logger.debug(f"Set system state: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting system state {key}: {e}")
            self.conn.rollback()
            return False
    
    def get_system_state(self, key: str) -> Optional[Any]:
        """
        Get a system state value.
        
        Args:
            key: State key
            
        Returns:
            State value or None if not found
        """
        try:
            cursor = self.conn.execute(
                "SELECT state_value FROM system_state WHERE state_key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                return json.loads(row['state_value'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting system state {key}: {e}")
            return None
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all active analysis sessions.
        
        Returns:
            List of active sessions
        """
        try:
            cursor = self.conn.execute("""
                SELECT * FROM analysis_sessions 
                WHERE status = 'active'
                ORDER BY start_time DESC
            """)
            
            rows = cursor.fetchall()
            sessions = []
            
            for row in rows:
                session_data = dict(row)
                
                # Parse JSON fields
                if session_data['config_data']:
                    session_data['config_data'] = json.loads(session_data['config_data'])
                if session_data['metadata']:
                    session_data['metadata'] = json.loads(session_data['metadata'])
                
                sessions.append(session_data)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about analysis sessions.
        
        Returns:
            Dictionary with session statistics
        """
        try:
            # Get session counts by status
            cursor = self.conn.execute("""
                SELECT status, COUNT(*) as count
                FROM analysis_sessions
                GROUP BY status
            """)
            
            status_counts = dict(cursor.fetchall())
            
            # Get session counts by type
            cursor = self.conn.execute("""
                SELECT session_type, COUNT(*) as count
                FROM analysis_sessions
                GROUP BY session_type
            """)
            
            type_counts = dict(cursor.fetchall())
            
            # Get total checkpoints and results
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM analysis_checkpoints")
            total_checkpoints = cursor.fetchone()['count']
            
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM analysis_results")
            total_results = cursor.fetchone()['count']
            
            return {
                "status_counts": status_counts,
                "type_counts": type_counts,
                "total_checkpoints": total_checkpoints,
                "total_results": total_results
            }
            
        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return {}
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("StateStore database connection closed")
