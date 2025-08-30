"""
Time window processor for intelligent data chunking and analysis.

This module provides efficient processing of large datasets through
overlapping time windows and anomaly detection to manage context windows.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter

from ..utils.mcp_client import EnhancedMCPClient

logger = logging.getLogger(__name__)


class TimeWindowProcessor:
    """
    Processes data in overlapping time windows to catch cross-boundary patterns.
    
    This class implements intelligent chunking strategies to handle large datasets
    while managing context window limitations through:
    - Overlapping time windows to catch patterns that span boundaries
    - Anomaly detection to focus analysis on interesting entities
    - Pre-aggregation to reduce data volume before detailed analysis
    """
    
    def __init__(
        self,
        mcp_client: EnhancedMCPClient,
        window_hours: int = 6,
        overlap_hours: int = 1,
        max_entities_per_window: int = 50
    ):
        """
        Initialize the time window processor.
        
        Args:
            mcp_client: Enhanced MCP client for data retrieval
            window_hours: Size of each time window in hours
            overlap_hours: Overlap between windows in hours
            max_entities_per_window: Maximum entities to analyze per window
        """
        self.mcp_client = mcp_client
        self.window_hours = window_hours
        self.overlap_hours = overlap_hours
        self.max_entities_per_window = max_entities_per_window
        self.processed_windows = set()  # Track processed windows
        self.anomaly_threshold = 2.0  # Threshold for anomaly detection
        
        logger.info(f"TimeWindowProcessor initialized: {window_hours}h windows, {overlap_hours}h overlap")
    
    def generate_time_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generate overlapping time windows for analysis.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            List of (start_time, end_time) tuples
        """
        windows = []
        current = start_date
        
        while current < end_date:
            window_end = min(
                current + timedelta(hours=self.window_hours),
                end_date
            )
            windows.append((current, window_end))
            
            # Move forward with overlap
            current += timedelta(hours=self.window_hours - self.overlap_hours)
        
        logger.info(f"Generated {len(windows)} time windows from {start_date} to {end_date}")
        return windows
    
    async def process_window(
        self,
        start_time: datetime,
        end_time: datetime,
        profile_manager,
        pattern_analyzer
    ) -> Dict[str, Any]:
        """
        Process a single time window with proper data retrieval and analysis.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            profile_manager: ProfileManager instance
            pattern_analyzer: PatternAnalyzer instance
            
        Returns:
            Dictionary with processing results
        """
        window_id = self._get_window_id(start_time, end_time)
        
        # Skip if already processed
        if window_id in self.processed_windows:
            logger.debug(f"Window {window_id} already processed, skipping")
            return {"status": "skipped", "window_id": window_id}
        
        logger.info(f"Processing window: {start_time} to {end_time}")
        
        try:
            # Step 1: Get summary statistics first
            summary = await self.mcp_client.get_time_window_summary(
                start_time, end_time
            )
            
            if not summary:
                logger.warning(f"No summary data for window {window_id}")
                return {"status": "no_data", "window_id": window_id}
            
            # Step 2: Identify interesting entities from summary
            interesting_entities = self._identify_anomalies(summary)
            
            if not interesting_entities:
                logger.info(f"No interesting entities found in window {window_id}")
                return {"status": "no_anomalies", "window_id": window_id}
            
            # Step 3: Get detailed data only for interesting entities
            detailed_data = []
            for entity in interesting_entities[:self.max_entities_per_window]:
                entity_data = await self.mcp_client.query_with_pagination(
                    time_range_hours=self.window_hours,
                    page_size=500,
                    fields=["source.ip", "destination.port", "event.action", "@timestamp"],
                    filters={"source.ip": entity}
                )
                detailed_data.extend(entity_data)
            
            if not detailed_data:
                logger.warning(f"No detailed data retrieved for window {window_id}")
                return {"status": "no_detailed_data", "window_id": window_id}
            
            # Step 4: Analyze patterns
            patterns = await pattern_analyzer.analyze(detailed_data)
            
            # Step 5: Update profiles incrementally
            entity_groups = self._group_by_entity(detailed_data)
            profiles_updated = 0
            
            for entity_id, entity_events in entity_groups.items():
                pattern_data = patterns.get(entity_id, {})
                
                updated_profile = profile_manager.update_profile_incrementally(
                    entity_id,
                    entity_events,
                    pattern_data
                )
                
                if updated_profile:
                    profiles_updated += 1
            
            # Mark window as processed
            self.processed_windows.add(window_id)
            
            result = {
                "status": "processed",
                "window_id": window_id,
                "entities_analyzed": len(interesting_entities),
                "profiles_updated": profiles_updated,
                "patterns_found": len(patterns),
                "total_events": len(detailed_data),
                "processing_time": (end_time - start_time).total_seconds()
            }
            
            logger.info(f"Window {window_id} processed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing window {window_id}: {e}")
            return {"status": "error", "window_id": window_id, "error": str(e)}
    
    def _get_window_id(self, start_time: datetime, end_time: datetime) -> str:
        """
        Generate a unique ID for a time window.
        
        Args:
            start_time: Start of window
            end_time: End of window
            
        Returns:
            Unique window identifier
        """
        window_str = f"{start_time.isoformat()}_{end_time.isoformat()}"
        return hashlib.md5(window_str.encode()).hexdigest()[:16]
    
    def _identify_anomalies(self, summary: Dict[str, Any]) -> List[str]:
        """
        Identify entities that warrant detailed analysis.
        
        This is KEY to managing context windows efficiently by focusing
        analysis on the most interesting entities.
        
        Args:
            summary: Summary statistics from MCP server
            
        Returns:
            List of entity IDs that are anomalous
        """
        anomalies = []
        
        try:
            # Look for IPs with unusual activity volumes
            if "top_sources" in summary and "average_count" in summary:
                avg_count = summary["average_count"]
                threshold = avg_count * self.anomaly_threshold
                
                for source in summary["top_sources"]:
                    if source.get("count", 0) > threshold:
                        anomalies.append(source["ip"])
            
            # Look for rare ports (potential reconnaissance)
            if "rare_ports" in summary:
                rare_sources = summary["rare_ports"].get("sources", [])
                anomalies.extend(rare_sources)
            
            # Look for unusual event types
            if "unusual_events" in summary:
                unusual_sources = summary["unusual_events"].get("sources", [])
                anomalies.extend(unusual_sources)
            
            # Look for high-frequency events
            if "high_frequency" in summary:
                hf_sources = summary["high_frequency"].get("sources", [])
                anomalies.extend(hf_sources)
            
            # Remove duplicates and limit to max entities
            anomalies = list(set(anomalies))[:self.max_entities_per_window]
            
            logger.debug(f"Identified {len(anomalies)} anomalous entities")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error identifying anomalies: {e}")
            return []
    
    def _group_by_entity(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group events by source entity (IP address).
        
        Args:
            events: List of events to group
            
        Returns:
            Dictionary mapping entity IDs to their events
        """
        entity_groups = defaultdict(list)
        
        for event in events:
            source_ip = event.get("source.ip")
            if source_ip:
                entity_id = f"ip_{source_ip}"
                entity_groups[entity_id].append(event)
        
        return dict(entity_groups)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processed windows.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "processed_windows": len(self.processed_windows),
            "window_hours": self.window_hours,
            "overlap_hours": self.overlap_hours,
            "max_entities_per_window": self.max_entities_per_window,
            "anomaly_threshold": self.anomaly_threshold
        }
    
    def reset_processed_windows(self) -> None:
        """Reset the set of processed windows (for testing or restart)."""
        self.processed_windows.clear()
        logger.info("Reset processed windows tracking")
    
    def is_window_processed(self, start_time: datetime, end_time: datetime) -> bool:
        """
        Check if a window has already been processed.
        
        Args:
            start_time: Start of window
            end_time: End of window
            
        Returns:
            True if window has been processed
        """
        window_id = self._get_window_id(start_time, end_time)
        return window_id in self.processed_windows
