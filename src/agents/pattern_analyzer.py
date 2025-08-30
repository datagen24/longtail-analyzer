"""
Pattern recognition agent for identifying attack patterns and anomalies.

This module provides pattern analysis capabilities including:
- Temporal pattern analysis
- Behavioral clustering
- Statistical outlier detection
- TTP classification
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
import statistics

from ..models.pattern import Pattern, PatternType, PatternSeverity

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """
    Analyzes events to identify attack patterns and anomalies.
    
    This class implements various pattern recognition algorithms to identify
    long-tail attack patterns, behavioral anomalies, and TTPs.
    """
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.pattern_templates = self._load_pattern_templates()
        self.ttp_mappings = self._load_ttp_mappings()
        logger.info("PatternAnalyzer initialized")
    
    async def analyze(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze events to identify patterns and anomalies.
        
        Args:
            events: List of events to analyze
            
        Returns:
            Dictionary mapping entity IDs to analysis results
        """
        if not events:
            return {}
        
        logger.debug(f"Analyzing {len(events)} events for patterns")
        
        # Group events by source entity
        entity_events = self._group_events_by_entity(events)
        
        analysis_results = {}
        
        for entity_id, entity_event_list in entity_events.items():
            try:
                # Perform pattern analysis for this entity
                entity_analysis = await self._analyze_entity_patterns(
                    entity_id, entity_event_list
                )
                
                if entity_analysis:
                    analysis_results[entity_id] = entity_analysis
                    
            except Exception as e:
                logger.error(f"Error analyzing patterns for entity {entity_id}: {e}")
                continue
        
        logger.debug(f"Pattern analysis complete: {len(analysis_results)} entities analyzed")
        return analysis_results
    
    async def _analyze_entity_patterns(
        self, 
        entity_id: str, 
        events: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze patterns for a specific entity.
        
        Args:
            entity_id: ID of the entity
            events: Events associated with the entity
            
        Returns:
            Analysis results for the entity
        """
        if not events:
            return None
        
        analysis = {
            "entity_id": entity_id,
            "patterns": [],
            "confidence": 0.0,
            "threat_score": 0.0,
            "ttps": [],
            "anomalies": []
        }
        
        # Analyze different types of patterns
        scanning_patterns = self._detect_scanning_patterns(events)
        exploitation_patterns = self._detect_exploitation_patterns(events)
        persistence_patterns = self._detect_persistence_patterns(events)
        
        # Combine patterns
        all_patterns = scanning_patterns + exploitation_patterns + persistence_patterns
        analysis["patterns"] = all_patterns
        
        # Calculate threat score based on patterns
        analysis["threat_score"] = self._calculate_threat_score(all_patterns)
        
        # Extract TTPs
        analysis["ttps"] = self._extract_ttps(all_patterns)
        
        # Detect anomalies
        analysis["anomalies"] = self._detect_anomalies(events)
        
        # Calculate overall confidence
        analysis["confidence"] = self._calculate_confidence(events, all_patterns)
        
        return analysis
    
    def _detect_scanning_patterns(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """
        Detect port scanning and reconnaissance patterns.
        
        Args:
            events: Events to analyze
            
        Returns:
            List of detected scanning patterns
        """
        patterns = []
        
        # Group events by time windows
        time_windows = self._create_time_windows(events, window_minutes=5)
        
        for window_start, window_events in time_windows.items():
            if len(window_events) < 10:  # Need minimum events for pattern
                continue
            
            # Analyze port distribution
            ports = [e.get("destination.port") for e in window_events if "destination.port" in e]
            if not ports:
                continue
            
            port_counts = Counter(ports)
            unique_ports = len(port_counts)
            total_events = len(ports)
            
            # Detect horizontal scanning (many ports on few hosts)
            if unique_ports > 20 and total_events > 50:
                pattern = Pattern(
                    pattern_id=f"scan_horizontal_{window_start}",
                    pattern_type=PatternType.SCANNING,
                    name="Horizontal Port Scan",
                    description=f"Scanned {unique_ports} unique ports in {total_events} attempts",
                    severity=PatternSeverity.MEDIUM,
                    confidence_score=0.8,
                    first_detected=window_start,
                    last_detected=window_start,
                    indicators=[f"scanned_{unique_ports}_ports"],
                    ttps=["T1046"]  # Network Service Scanning
                )
                patterns.append(pattern)
            
            # Detect vertical scanning (few ports on many hosts)
            hosts = [e.get("destination.ip") for e in window_events if "destination.ip" in e]
            unique_hosts = len(set(hosts))
            
            if unique_hosts > 20 and total_events > 50:
                pattern = Pattern(
                    pattern_id=f"scan_vertical_{window_start}",
                    pattern_type=PatternType.SCANNING,
                    name="Vertical Port Scan",
                    description=f"Scanned {unique_hosts} unique hosts in {total_events} attempts",
                    severity=PatternSeverity.MEDIUM,
                    confidence_score=0.8,
                    first_detected=window_start,
                    last_detected=window_start,
                    indicators=[f"scanned_{unique_hosts}_hosts"],
                    ttps=["T1046"]  # Network Service Scanning
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_exploitation_patterns(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """
        Detect exploitation attempt patterns.
        
        Args:
            events: Events to analyze
            
        Returns:
            List of detected exploitation patterns
        """
        patterns = []
        
        # Look for specific exploitation indicators
        exploitation_actions = ["exploit", "attack", "injection", "buffer_overflow"]
        
        for event in events:
            action = event.get("event.action", "").lower()
            
            for exploit_type in exploitation_actions:
                if exploit_type in action:
                    pattern = Pattern(
                        pattern_id=f"exploit_{event.get('@timestamp', 'unknown')}",
                        pattern_type=PatternType.EXPLOITATION,
                        name=f"Exploitation Attempt: {exploit_type}",
                        description=f"Detected {exploit_type} attempt",
                        severity=PatternSeverity.HIGH,
                        confidence_score=0.9,
                        first_detected=datetime.fromisoformat(event.get("@timestamp", datetime.now().isoformat())),
                        last_detected=datetime.fromisoformat(event.get("@timestamp", datetime.now().isoformat())),
                        indicators=[f"action_{action}"],
                        ttps=["T1059"]  # Command and Scripting Interpreter
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_persistence_patterns(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """
        Detect persistence and lateral movement patterns.
        
        Args:
            events: Events to analyze
            
        Returns:
            List of detected persistence patterns
        """
        patterns = []
        
        # Look for repeated access patterns
        access_events = [e for e in events if e.get("event.action") in ["access", "login", "connect"]]
        
        if len(access_events) > 10:
            # Analyze time distribution
            timestamps = [datetime.fromisoformat(e.get("@timestamp", datetime.now().isoformat())) 
                         for e in access_events]
            
            # Check for regular intervals (potential scheduled access)
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                intervals.append(interval)
            
            if intervals:
                interval_std = statistics.stdev(intervals) if len(intervals) > 1 else 0
                interval_mean = statistics.mean(intervals)
                
                # Low standard deviation suggests regular intervals
                if interval_std < interval_mean * 0.2:  # Less than 20% variation
                    pattern = Pattern(
                        pattern_id=f"persistence_regular_{timestamps[0]}",
                        pattern_type=PatternType.PERSISTENCE,
                        name="Regular Access Pattern",
                        description=f"Regular access pattern with {len(access_events)} events",
                        severity=PatternSeverity.MEDIUM,
                        confidence_score=0.7,
                        first_detected=min(timestamps),
                        last_detected=max(timestamps),
                        indicators=[f"regular_intervals_{interval_mean:.0f}s"],
                        ttps=["T1078"]  # Valid Accounts
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_anomalies(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """
        Detect statistical anomalies in event patterns.
        
        Args:
            events: Events to analyze
            
        Returns:
            List of detected anomalies
        """
        anomalies: List[Pattern] = []
        
        if len(events) < 5:
            return anomalies
        
        # Analyze event frequency
        timestamps = [datetime.fromisoformat(e.get("@timestamp", datetime.now().isoformat())) 
                     for e in events]
        timestamps.sort()
        
        # Calculate intervals between events
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if intervals:
            mean_interval = statistics.mean(intervals)
            std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
            
            # Detect burst activity (very short intervals)
            burst_threshold = mean_interval - (2 * std_interval)
            burst_events = [i for i, interval in enumerate(intervals) if interval < burst_threshold]
            
            if burst_events:
                pattern = Pattern(
                    pattern_id=f"burst_{events[0].get('source.ip', 'unknown')}_{timestamps[0].strftime('%Y%m%d_%H%M%S')}",
                    pattern_type=PatternType.UNKNOWN,
                    name="Burst Activity",
                    description=f"Detected {len(burst_events)} burst events",
                    severity=PatternSeverity.MEDIUM,
                    confidence_score=0.8,
                    first_detected=timestamps[0],
                    last_detected=timestamps[-1]
                )
                pattern.add_source_entity(events[0].get("source.ip", "unknown"))
                anomalies.append(pattern)
        
        # Analyze port distribution anomalies
        ports = [e.get("destination.port") for e in events if "destination.port" in e]
        if ports:
            port_counts = Counter(ports)
            most_common_port = port_counts.most_common(1)[0]
            
            # If one port dominates significantly
            if most_common_port[1] > len(ports) * 0.8:
                pattern = Pattern(
                    pattern_id=f"port_conc_{events[0].get('source.ip', 'unknown')}_{timestamps[0].strftime('%Y%m%d_%H%M%S')}",
                    pattern_type=PatternType.SCANNING,
                    name="Port Concentration",
                    description=f"Port {most_common_port[0]} used in {most_common_port[1]} of {len(ports)} events",
                    severity=PatternSeverity.LOW,
                    confidence_score=0.6,
                    first_detected=timestamps[0],
                    last_detected=timestamps[-1]
                )
                pattern.add_source_entity(events[0].get("source.ip", "unknown"))
                anomalies.append(pattern)
        
        return anomalies
    
    def _calculate_threat_score(self, patterns: List[Pattern]) -> float:
        """
        Calculate overall threat score based on detected patterns.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Threat score between 0.0 and 1.0
        """
        if not patterns:
            return 0.0
        
        # Weight patterns by severity and confidence
        severity_weights = {
            PatternSeverity.LOW: 0.25,
            PatternSeverity.MEDIUM: 0.5,
            PatternSeverity.HIGH: 0.75,
            PatternSeverity.CRITICAL: 1.0
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for pattern in patterns:
            weight = severity_weights.get(pattern.severity, 0.25)
            weighted_score += pattern.confidence_score * weight
            total_weight += weight
        
        return min(1.0, weighted_score / total_weight if total_weight > 0 else 0.0)
    
    def _extract_ttps(self, patterns: List[Pattern]) -> List[str]:
        """
        Extract unique TTPs from patterns.
        
        Args:
            patterns: List of patterns
            
        Returns:
            List of unique TTP identifiers
        """
        ttps = set()
        for pattern in patterns:
            ttps.update(pattern.ttps)
        return list(ttps)
    
    def _calculate_confidence(self, events: List[Dict[str, Any]], patterns: List[Pattern]) -> float:
        """
        Calculate overall confidence in the analysis.
        
        Args:
            events: Original events
            patterns: Detected patterns
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not events:
            return 0.0
        
        # Base confidence on data volume and pattern quality
        data_confidence = min(1.0, len(events) / 100.0)  # More events = higher confidence
        
        if not patterns:
            return data_confidence * 0.5  # Lower confidence if no patterns found
        
        # Average pattern confidence
        pattern_confidence = statistics.mean([p.confidence_score for p in patterns])
        
        # Combine data and pattern confidence
        return (data_confidence + pattern_confidence) / 2.0
    
    def _group_events_by_entity(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group events by source entity.
        
        Args:
            events: Events to group
            
        Returns:
            Dictionary mapping entity IDs to their events
        """
        entity_events = defaultdict(list)
        
        for event in events:
            source_ip = event.get("source.ip")
            if source_ip:
                entity_id = f"ip_{source_ip}"
                entity_events[entity_id].append(event)
        
        return dict(entity_events)
    
    def _create_time_windows(
        self, 
        events: List[Dict[str, Any]], 
        window_minutes: int = 5
    ) -> Dict[datetime, List[Dict[str, Any]]]:
        """
        Create time windows for temporal analysis.
        
        Args:
            events: Events to window
            window_minutes: Size of each window in minutes
            
        Returns:
            Dictionary mapping window start times to events
        """
        windows = defaultdict(list)
        
        for event in events:
            timestamp_str = event.get("@timestamp")
            if not timestamp_str:
                continue
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                # Round down to window boundary
                window_start = timestamp.replace(
                    minute=(timestamp.minute // window_minutes) * window_minutes,
                    second=0,
                    microsecond=0
                )
                windows[window_start].append(event)
            except ValueError:
                continue
        
        return dict(windows)
    
    def _load_pattern_templates(self) -> Dict[str, Any]:
        """Load pattern recognition templates."""
        # This would typically load from a configuration file
        return {
            "scanning": {
                "min_events": 10,
                "min_ports": 20,
                "min_hosts": 20
            },
            "exploitation": {
                "keywords": ["exploit", "attack", "injection", "buffer_overflow"]
            }
        }
    
    def _load_ttp_mappings(self) -> Dict[str, str]:
        """Load TTP mappings for pattern classification."""
        # This would typically load from MITRE ATT&CK mappings
        return {
            "scanning": "T1046",
            "exploitation": "T1059",
            "persistence": "T1078"
        }
