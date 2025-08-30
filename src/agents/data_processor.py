# src/agents/data_processor.py

import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from src.utils.mcp_client import EnhancedMCPClient, MCPQueryOptimizer, QuerySummary
from src.agents.profile_manager import ProfileManager, EntityType

logger = logging.getLogger(__name__)


@dataclass
class ProcessingWindow:
    """Represents a time window for processing"""
    window_id: str
    start_time: datetime
    end_time: datetime
    overlap_start: Optional[datetime] = None
    overlap_end: Optional[datetime] = None
    processed: bool = False
    summary: Optional[QuerySummary] = None
    entities_found: int = 0
    patterns_found: int = 0


class TimeWindowProcessor:
    """
    Processes data in overlapping time windows to catch cross-boundary patterns.
    This is the key to handling large datasets within context window limitations.
    """
    
    def __init__(
        self,
        mcp_client: EnhancedMCPClient,
        profile_manager: ProfileManager,
        window_hours: int = 6,
        overlap_hours: int = 1,
        max_entities_per_window: int = 50
    ):
        self.mcp_client = mcp_client
        self.profile_manager = profile_manager
        self.query_optimizer = MCPQueryOptimizer(mcp_client)
        
        self.window_hours = window_hours
        self.overlap_hours = overlap_hours
        self.max_entities_per_window = max_entities_per_window
        
        self.processed_windows: Set[str] = set()
        self.window_cache: Dict[str, ProcessingWindow] = {}
        
    def generate_time_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[ProcessingWindow]:
        """
        Generate overlapping time windows for analysis.
        Windows overlap to ensure patterns spanning boundaries are detected.
        """
        windows: List[ProcessingWindow] = []
        current = start_date
        
        while current < end_date:
            window_end = min(
                current + timedelta(hours=self.window_hours),
                end_date
            )
            
            # Calculate overlap period with previous window
            overlap_start = None
            overlap_end = None
            
            if windows:  # If there's a previous window
                overlap_start = current
                overlap_end = current + timedelta(hours=self.overlap_hours)
            
            window = ProcessingWindow(
                window_id=self._get_window_id(current, window_end),
                start_time=current,
                end_time=window_end,
                overlap_start=overlap_start,
                overlap_end=overlap_end
            )
            
            windows.append(window)
            
            # Move forward with overlap
            current += timedelta(hours=self.window_hours - self.overlap_hours)
        
        logger.info(f"Generated {len(windows)} time windows from {start_date} to {end_date}")
        return windows
    
    def _get_window_id(self, start_time: datetime, end_time: datetime) -> str:
        """Generate unique ID for a time window"""
        id_string = f"{start_time.isoformat()}_{end_time.isoformat()}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    async def process_window(
        self,
        window: ProcessingWindow,
        pattern_analyzer=None
    ) -> Dict:
        """
        Process a single time window with intelligent data retrieval.
        
        Key optimization: Get summary first, identify anomalies, then fetch details only for anomalous entities.
        """
        
        # Skip if already processed
        if window.window_id in self.processed_windows:
            logger.info(f"Window {window.window_id} already processed, skipping")
            return {"status": "skipped", "window_id": window.window_id}
        
        logger.info(f"Processing window {window.start_time} to {window.end_time}")
        
        try:
            # Step 1: Get aggregated summary for the window
            summary = await self.mcp_client.get_aggregated_summary(
                window.start_time,
                window.end_time
            )
            window.summary = summary
            
            logger.info(f"Window summary: {summary.total_events} events, "
                       f"{summary.unique_sources} unique sources")
            
            # Step 2: Identify anomalous entities from summary
            anomalies = await self.query_optimizer.identify_anomalies(
                summary,
                threshold_multiplier=2.0
            )
            
            # Limit to max entities per window
            anomalies = anomalies[:self.max_entities_per_window]
            
            logger.info(f"Identified {len(anomalies)} anomalous entities for detailed analysis")
            
            # Step 3: Process overlap region if exists
            overlap_entities = []
            if window.overlap_start and window.overlap_end:
                overlap_entities = await self._process_overlap(
                    window.overlap_start,
                    window.overlap_end,
                    anomalies
                )
            
            # Step 4: Get detailed data for anomalous entities
            entity_events = await self.query_optimizer.fetch_anomaly_details(
                anomalies,
                time_range_hours=self.window_hours
            )
            
            # Step 5: Analyze patterns if analyzer provided
            patterns = {}
            if pattern_analyzer and entity_events:
                patterns = await pattern_analyzer.analyze(entity_events)
                window.patterns_found = len(patterns)
            
            # Step 6: Update profiles incrementally
            profiles_updated = 0
            for entity_id, events in entity_events.items():
                # Determine entity type
                entity_type = self._determine_entity_type(entity_id)
                
                # Update profile
                profile = self.profile_manager.update_profile_incrementally(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    new_events=events,
                    llm_analysis=patterns.get(entity_id)
                )
                
                profiles_updated += 1
                
                # Check for related entities
                if profile.related_entities:
                    await self._process_related_entities(profile.related_entities)
            
            # Step 7: Handle cross-boundary patterns
            if overlap_entities:
                await self._merge_cross_boundary_patterns(overlap_entities, entity_events)
            
            # Mark window as processed
            window.processed = True
            window.entities_found = len(entity_events)
            self.processed_windows.add(window.window_id)
            self.window_cache[window.window_id] = window
            
            # Save checkpoint
            self._save_checkpoint(window)
            
            result = {
                "status": "processed",
                "window_id": window.window_id,
                "total_events": summary.total_events,
                "entities_analyzed": len(entity_events),
                "profiles_updated": profiles_updated,
                "patterns_found": window.patterns_found,
                "anomalies_detected": len(anomalies)
            }
            
            logger.info(f"Window processing complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing window {window.window_id}: {e}")
            return {
                "status": "error",
                "window_id": window.window_id,
                "error": str(e)
            }
    
    async def _process_overlap(
        self,
        overlap_start: datetime,
        overlap_end: datetime,
        current_anomalies: List[Dict]
    ) -> List[str]:
        """
        Process overlap region between windows to detect cross-boundary patterns.
        """
        logger.info(f"Processing overlap region: {overlap_start} to {overlap_end}")
        
        # Get summary for overlap period
        overlap_summary = await self.mcp_client.get_aggregated_summary(
            overlap_start,
            overlap_end
        )
        
        # Find entities that appear in both windows
        overlap_entities = []
        current_entity_ids = [a["entity_id"] for a in current_anomalies]
        
        for source in overlap_summary.top_sources:
            if source["ip"] in current_entity_ids:
                overlap_entities.append(source["ip"])
        
        logger.info(f"Found {len(overlap_entities)} entities in overlap region")
        return overlap_entities
    
    async def _merge_cross_boundary_patterns(
        self,
        overlap_entities: List[str],
        entity_events: Dict[str, List[Dict]]
    ):
        """
        Merge patterns that span window boundaries.
        """
        for entity_id in overlap_entities:
            if entity_id not in entity_events:
                continue
            
            # Get the profile
            profile = self.profile_manager.get_profile(entity_id)
            if not profile:
                continue
            
            # Check if patterns in current window connect with previous patterns
            current_events = entity_events[entity_id]
            
            # Look for pattern continuation
            for event in current_events:
                event_time = datetime.fromisoformat(event.get("@timestamp", datetime.now().isoformat()))
                
                # Check if this event is close in time to existing patterns
                for pattern in profile.scanning_patterns + profile.exploitation_attempts:
                    time_gap = (event_time - pattern.last_seen).total_seconds() / 60
                    
                    if time_gap < 30:  # Within 30 minutes
                        # This is likely a continuation of the pattern
                        pattern.last_seen = event_time
                        pattern.occurrence_count += 1
                        logger.info(f"Merged cross-boundary pattern for {entity_id}")
            
            # Save updated profile
            self.profile_manager.save_profile(profile)
    
    async def _process_related_entities(self, related_entities: List[str]):
        """
        Process entities related to the current one.
        """
        for entity_id in related_entities[:5]:  # Limit to prevent explosion
            # Check if already processed
            if entity_id in self.processed_windows:
                continue
            
            logger.info(f"Processing related entity: {entity_id}")
            
            # Get minimal data for related entity
            events = await self.mcp_client.get_entity_details(
                entity_id=entity_id,
                entity_type="ip",
                time_range_hours=self.window_hours
            )
            
            if events:
                self.profile_manager.update_profile_incrementally(
                    entity_id=entity_id,
                    entity_type=EntityType.IP,
                    new_events=events[:100]  # Limit events
                )
    
    def _determine_entity_type(self, entity_id: str) -> EntityType:
        """Determine the type of entity from its ID"""
        if entity_id.startswith("ip_"):
            return EntityType.IP
        elif entity_id.startswith("asn_") or entity_id.startswith("AS"):
            return EntityType.ASN
        elif entity_id.startswith("port_"):
            return EntityType.PORT
        elif entity_id.startswith("country_"):
            return EntityType.COUNTRY
        else:
            # Try to infer from format
            if "." in entity_id and all(
                part.isdigit() and 0 <= int(part) <= 255
                for part in entity_id.split(".")
            ):
                return EntityType.IP
            return EntityType.COMPOSITE
    
    def _save_checkpoint(self, window: ProcessingWindow):
        """Save processing checkpoint for resumption"""
        state = {
            "window_id": window.window_id,
            "start_time": window.start_time.isoformat(),
            "end_time": window.end_time.isoformat(),
            "entities_found": window.entities_found,
            "patterns_found": window.patterns_found,
            "processed": window.processed
        }
        
        self.profile_manager.save_analysis_state(
            analysis_id=f"window_{window.window_id}",
            window_start=window.start_time,
            window_end=window.end_time,
            state=state
        )
    
    def load_checkpoint(self, analysis_id: str) -> Optional[Dict]:
        """Load saved checkpoint"""
        return self.profile_manager.get_analysis_state(analysis_id)
    
    async def process_time_range(
        self,
        start_date: datetime,
        end_date: datetime,
        pattern_analyzer=None,
        resume: bool = True
    ) -> Dict:
        """
        Process an entire time range with automatic windowing.
        """
        # Generate windows
        windows = self.generate_time_windows(start_date, end_date)
        
        # Load previous state if resuming
        if resume:
            self._load_processed_windows()
        
        # Process statistics
        total_windows = len(windows)
        processed = 0
        skipped = 0
        errors = 0
        total_entities = 0
        total_patterns = 0
        
        # Process each window
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{total_windows}")
            
            # Check if already processed
            if window.window_id in self.processed_windows:
                skipped += 1
                continue
            
            # Process window
            result = await self.process_window(window, pattern_analyzer)
            
            if result["status"] == "processed":
                processed += 1
                total_entities += result.get("entities_analyzed", 0)
                total_patterns += result.get("patterns_found", 0)
            elif result["status"] == "error":
                errors += 1
            
            # Adaptive delay based on data volume
            if window.summary and window.summary.total_events > 10000:
                await asyncio.sleep(2)  # Longer delay for heavy windows
            else:
                await asyncio.sleep(0.5)
        
        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_windows": total_windows,
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "total_entities_analyzed": total_entities,
            "total_patterns_found": total_patterns
        }
    
    def _load_processed_windows(self):
        """Load previously processed windows from checkpoints"""
        # This would query the analysis_state table to find processed windows
        # For now, we'll just log
        logger.info(f"Loaded {len(self.processed_windows)} previously processed windows")
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about processing"""
        stats = {
            "windows_processed": len(self.processed_windows),
            "windows_cached": len(self.window_cache),
            "total_events_processed": 0,
            "total_entities_found": 0,
            "total_patterns_found": 0
        }
        
        for window in self.window_cache.values():
            if window.summary:
                stats["total_events_processed"] += window.summary.total_events
            stats["total_entities_found"] += window.entities_found
            stats["total_patterns_found"] += window.patterns_found
        
        return stats