# src/agents/profile_manager.py

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EntityType(Enum):
    IP = "ip"
    ASN = "asn"
    COMPOSITE = "composite"
    PORT = "port"
    COUNTRY = "country"


class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AttackPattern:
    """Represents a specific attack pattern"""

    pattern_type: str  # scanning, exploitation, persistence
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int
    targets: list[str] = field(default_factory=list)
    ports: list[int] = field(default_factory=list)
    techniques: list[str] = field(default_factory=list)  # MITRE ATT&CK
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "occurrence_count": self.occurrence_count,
            "targets": self.targets,
            "ports": self.ports,
            "techniques": self.techniques,
            "confidence": self.confidence,
        }


@dataclass
class AttackerProfile:
    """Comprehensive attacker profile with behavioral patterns"""

    entity_id: str  # e.g., "ip_192.168.1.1" or "asn_AS12345"
    entity_type: EntityType
    entity_value: str  # The actual IP, ASN, etc.

    # Temporal information
    first_seen: datetime
    last_seen: datetime
    active_days: int = 0

    # Attack patterns
    scanning_patterns: list[AttackPattern] = field(default_factory=list)
    exploitation_attempts: list[AttackPattern] = field(default_factory=list)
    persistence_indicators: list[AttackPattern] = field(default_factory=list)

    # Behavioral analysis
    behavioral_embedding: np.ndarray | None = None
    behavior_cluster_id: int | None = None

    # Statistics
    total_events: int = 0
    unique_targets: int = 0
    unique_ports: int = 0
    data_exfiltration_bytes: int = 0

    # Threat assessment
    threat_level: ThreatLevel = ThreatLevel.LOW
    confidence_scores: dict[str, float] = field(default_factory=dict)
    ttps: list[str] = field(default_factory=list)  # MITRE ATT&CK
    iocs: list[str] = field(default_factory=list)  # Indicators of Compromise

    # Infrastructure
    asn: str | None = None
    country: str | None = None
    organization: str | None = None
    related_entities: list[str] = field(default_factory=list)
    known_malware: list[str] = field(default_factory=list)

    # Analysis metadata
    last_analysis: datetime = field(default_factory=datetime.now)
    analysis_depth: str = "shallow"  # shallow, medium, deep, comprehensive
    data_quality_score: float = 0.0
    analysis_version: str = "1.0"

    # Enrichment data
    threat_intel_sources: list[str] = field(default_factory=list)
    reputation_scores: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def calculate_threat_score(self) -> float:
        """Calculate overall threat score based on various factors"""
        score = 0.0

        # Factor in attack patterns
        score += len(self.scanning_patterns) * 0.1
        score += len(self.exploitation_attempts) * 0.3
        score += len(self.persistence_indicators) * 0.5

        # Factor in scale
        score += min(self.total_events / 1000, 1.0) * 0.2
        score += min(self.unique_targets / 100, 1.0) * 0.2

        # Factor in confidence
        avg_confidence = float(
            np.mean(list(self.confidence_scores.values()))
            if self.confidence_scores
            else 0
        )
        score *= 0.5 + avg_confidence * 0.5

        return min(score, 1.0)

    def update_threat_level(self):
        """Update threat level based on current profile data"""
        score = self.calculate_threat_score()

        if score >= 0.8:
            self.threat_level = ThreatLevel.CRITICAL
        elif score >= 0.6:
            self.threat_level = ThreatLevel.HIGH
        elif score >= 0.3:
            self.threat_level = ThreatLevel.MEDIUM
        else:
            self.threat_level = ThreatLevel.LOW

    def to_dict(self) -> dict:
        """Convert profile to dictionary for storage"""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "entity_value": self.entity_value,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "active_days": self.active_days,
            "scanning_patterns": [p.to_dict() for p in self.scanning_patterns],
            "exploitation_attempts": [p.to_dict() for p in self.exploitation_attempts],
            "persistence_indicators": [
                p.to_dict() for p in self.persistence_indicators
            ],
            "behavior_cluster_id": self.behavior_cluster_id,
            "total_events": self.total_events,
            "unique_targets": self.unique_targets,
            "unique_ports": self.unique_ports,
            "data_exfiltration_bytes": self.data_exfiltration_bytes,
            "threat_level": self.threat_level.value,
            "confidence_scores": self.confidence_scores,
            "ttps": self.ttps,
            "iocs": self.iocs,
            "asn": self.asn,
            "country": self.country,
            "organization": self.organization,
            "related_entities": self.related_entities,
            "known_malware": self.known_malware,
            "last_analysis": self.last_analysis.isoformat(),
            "analysis_depth": self.analysis_depth,
            "data_quality_score": self.data_quality_score,
            "analysis_version": self.analysis_version,
            "threat_intel_sources": self.threat_intel_sources,
            "reputation_scores": self.reputation_scores,
            "tags": self.tags,
            "notes": self.notes,
        }


class ProfileManager:
    """Manages attacker profiles with persistent storage and incremental updates"""

    def __init__(self, db_path: str = "data/profiles.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initialize SQLite tables for profile storage"""

        # Main profiles table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_value TEXT NOT NULL,
                profile_data JSON NOT NULL,
                embedding BLOB,
                threat_score REAL,
                threat_level INTEGER,
                last_updated TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version TEXT DEFAULT '1.0'
            )
        """
        )

        # Analysis state table for resumable processing
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_state (
                analysis_id TEXT PRIMARY KEY,
                window_start TIMESTAMP,
                window_end TIMESTAMP,
                state JSON,
                entities_processed INTEGER,
                patterns_found INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """
        )

        # Pattern library for reusable patterns
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_library (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                pattern_data JSON,
                occurrence_count INTEGER,
                entities TEXT,  -- JSON array of entity_ids
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP
            )
        """
        )

        # Relationships between entities
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity1_id TEXT,
                entity2_id TEXT,
                relationship_type TEXT,
                confidence REAL,
                evidence JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entity1_id, entity2_id, relationship_type)
            )
        """
        )

        # Create indices for performance
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_threat_level ON profiles(threat_level)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_type ON profiles(entity_type)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_last_updated ON profiles(last_updated)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships ON entity_relationships(entity1_id)"
        )

        self.conn.commit()

    def save_profile(self, profile: AttackerProfile) -> bool:
        """Save or update an attacker profile"""
        try:
            profile.update_threat_level()
            threat_score = profile.calculate_threat_score()

            # Handle embedding separately
            embedding_bytes = None
            if profile.behavioral_embedding is not None:
                embedding_bytes = profile.behavioral_embedding.tobytes()

            profile_dict = profile.to_dict()

            self.conn.execute(
                """
                INSERT OR REPLACE INTO profiles
                (entity_id, entity_type, entity_value, profile_data, embedding,
                 threat_score, threat_level, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    profile.entity_id,
                    profile.entity_type.value,
                    profile.entity_value,
                    json.dumps(profile_dict),
                    embedding_bytes,
                    threat_score,
                    profile.threat_level.value,
                    datetime.now(),
                ),
            )

            self.conn.commit()
            logger.info(
                f"Saved profile for {profile.entity_id} (threat level: {profile.threat_level.name})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save profile {profile.entity_id}: {e}")
            self.conn.rollback()
            return False

    def get_profile(self, entity_id: str) -> AttackerProfile | None:
        """Retrieve a profile by entity ID"""
        try:
            cursor = self.conn.execute(
                "SELECT * FROM profiles WHERE entity_id = ?", (entity_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            profile_data = json.loads(row["profile_data"])

            # Reconstruct profile
            profile = AttackerProfile(
                entity_id=profile_data["entity_id"],
                entity_type=EntityType(profile_data["entity_type"]),
                entity_value=profile_data["entity_value"],
                first_seen=datetime.fromisoformat(profile_data["first_seen"]),
                last_seen=datetime.fromisoformat(profile_data["last_seen"]),
            )

            # Restore all fields
            for key, value in profile_data.items():
                if key in [
                    "entity_id",
                    "entity_type",
                    "entity_value",
                    "first_seen",
                    "last_seen",
                ]:
                    continue

                if key == "threat_level":
                    profile.threat_level = ThreatLevel(value)
                elif key in [
                    "scanning_patterns",
                    "exploitation_attempts",
                    "persistence_indicators",
                ]:
                    # Reconstruct AttackPattern objects
                    patterns = []
                    for p in value:
                        pattern = AttackPattern(
                            pattern_type=p["pattern_type"],
                            first_seen=datetime.fromisoformat(p["first_seen"]),
                            last_seen=datetime.fromisoformat(p["last_seen"]),
                            occurrence_count=p["occurrence_count"],
                            targets=p["targets"],
                            ports=p["ports"],
                            techniques=p["techniques"],
                            confidence=p["confidence"],
                        )
                        patterns.append(pattern)
                    setattr(profile, key, patterns)
                elif key == "last_analysis":
                    profile.last_analysis = datetime.fromisoformat(value)
                else:
                    setattr(profile, key, value)

            # Restore embedding if present
            if row["embedding"]:
                profile.behavioral_embedding = np.frombuffer(row["embedding"])

            return profile

        except Exception as e:
            logger.error(f"Failed to get profile {entity_id}: {e}")
            return None

    def update_profile_incrementally(
        self,
        entity_id: str,
        entity_type: EntityType,
        new_events: list[dict],
        llm_analysis: dict | None = None,
    ) -> AttackerProfile:
        """
        Incrementally update a profile with new events.
        This is crucial for handling large datasets efficiently.
        """
        profile = self.get_profile(entity_id)

        if not profile:
            # Create new profile
            entity_value = entity_id.split("_", 1)[1] if "_" in entity_id else entity_id
            profile = AttackerProfile(
                entity_id=entity_id,
                entity_type=entity_type,
                entity_value=entity_value,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
            logger.info(f"Creating new profile for {entity_id}")

        # Update with new events
        self._merge_new_events(profile, new_events)

        # Apply LLM insights if provided
        if llm_analysis:
            self._apply_llm_insights(profile, llm_analysis)

        # Update metadata
        profile.last_analysis = datetime.now()
        profile.data_quality_score = self._calculate_data_quality(profile)

        # Save updated profile
        self.save_profile(profile)

        return profile

    def _merge_new_events(self, profile: AttackerProfile, new_events: list[dict]):
        """Merge new events into existing profile"""
        if not new_events:
            return

        # Update temporal information
        for event in new_events:
            event_time = datetime.fromisoformat(
                event.get("@timestamp", datetime.now().isoformat())
            )

            if event_time < profile.first_seen:
                profile.first_seen = event_time
            if event_time > profile.last_seen:
                profile.last_seen = event_time

        # Update statistics
        profile.total_events += len(new_events)

        # Extract unique targets and ports
        targets = set()
        ports = set()

        for event in new_events:
            if "destination.ip" in event:
                targets.add(event["destination.ip"])
            if "destination.port" in event:
                ports.add(event["destination.port"])

        profile.unique_targets = len(targets)
        profile.unique_ports = len(ports)

        # Detect and categorize attack patterns
        self._detect_attack_patterns(profile, new_events)

        # Update active days
        profile.active_days = (profile.last_seen - profile.first_seen).days + 1

    def _detect_attack_patterns(self, profile: AttackerProfile, events: list[dict]):
        """Detect attack patterns from events"""

        # Group events by action/category
        scanning_events = []
        exploitation_events = []
        persistence_events = []

        for event in events:
            action = event.get("event.action", "").lower()

            if any(term in action for term in ["scan", "probe", "discovery"]):
                scanning_events.append(event)
            elif any(term in action for term in ["exploit", "attack", "injection"]):
                exploitation_events.append(event)
            elif any(term in action for term in ["backdoor", "persistence", "c2"]):
                persistence_events.append(event)

        # Create or update patterns
        if scanning_events:
            self._update_pattern_list(
                profile.scanning_patterns, "scanning", scanning_events
            )
        if exploitation_events:
            self._update_pattern_list(
                profile.exploitation_attempts, "exploitation", exploitation_events
            )
        if persistence_events:
            self._update_pattern_list(
                profile.persistence_indicators, "persistence", persistence_events
            )

    def _update_pattern_list(
        self, pattern_list: list[AttackPattern], pattern_type: str, events: list[dict]
    ):
        """Update or create attack patterns"""

        # Look for existing pattern of this type
        existing_pattern = None
        for pattern in pattern_list:
            if pattern.pattern_type == pattern_type:
                existing_pattern = pattern
                break

        if existing_pattern:
            # Update existing pattern
            existing_pattern.occurrence_count += len(events)
            existing_pattern.last_seen = datetime.now()

            # Update targets and ports
            for event in events:
                if (
                    "destination.ip" in event
                    and event["destination.ip"] not in existing_pattern.targets
                ):
                    existing_pattern.targets.append(event["destination.ip"])
                if (
                    "destination.port" in event
                    and event["destination.port"] not in existing_pattern.ports
                ):
                    existing_pattern.ports.append(event["destination.port"])
        else:
            # Create new pattern
            targets = list(
                {e.get("destination.ip", "") for e in events if e.get("destination.ip")}
            )
            ports = list(
                {
                    e.get("destination.port", 0)
                    for e in events
                    if e.get("destination.port")
                }
            )

            new_pattern = AttackPattern(
                pattern_type=pattern_type,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                occurrence_count=len(events),
                targets=targets[:10],  # Limit to top 10
                ports=ports[:10],
                confidence=0.7,
            )

            pattern_list.append(new_pattern)

    def _apply_llm_insights(self, profile: AttackerProfile, llm_analysis: dict):
        """Apply insights from LLM analysis to profile"""

        # Update TTPs if identified
        if "ttps" in llm_analysis:
            for ttp in llm_analysis["ttps"]:
                if ttp not in profile.ttps:
                    profile.ttps.append(ttp)

        # Update confidence scores
        if "confidence_scores" in llm_analysis:
            profile.confidence_scores.update(llm_analysis["confidence_scores"])

        # Update threat assessment
        if "threat_score" in llm_analysis:
            # Weighted average with existing assessment
            current_score = profile.calculate_threat_score()
            new_score = llm_analysis["threat_score"]
            profile.confidence_scores["overall"] = (current_score + new_score) / 2

        # Add tags
        if "tags" in llm_analysis:
            for tag in llm_analysis["tags"]:
                if tag not in profile.tags:
                    profile.tags.append(tag)

        # Update analysis depth
        if "analysis_depth" in llm_analysis:
            profile.analysis_depth = llm_analysis["analysis_depth"]

    def _calculate_data_quality(self, profile: AttackerProfile) -> float:
        """Calculate data quality score for profile"""
        score = 0.0

        # Factor in data completeness
        if profile.asn:
            score += 0.1
        if profile.country:
            score += 0.1
        if profile.organization:
            score += 0.1

        # Factor in analysis depth
        depth_scores = {
            "shallow": 0.2,
            "medium": 0.4,
            "deep": 0.6,
            "comprehensive": 0.8,
        }
        score += depth_scores.get(profile.analysis_depth, 0.2)

        # Factor in confidence
        if profile.confidence_scores:
            avg_confidence = float(np.mean(list(profile.confidence_scores.values())))
            score += avg_confidence * 0.2

        # Factor in data volume
        if profile.total_events > 100:
            score += 0.1
        if profile.total_events > 1000:
            score += 0.1

        return min(score, 1.0)

    def find_similar_profiles(
        self, profile: AttackerProfile, limit: int = 10
    ) -> list[tuple[str, float]]:
        """Find similar profiles based on behavioral patterns"""
        similar = []

        # Get all profiles
        cursor = self.conn.execute(
            "SELECT entity_id, profile_data FROM profiles WHERE entity_id != ?",
            (profile.entity_id,),
        )

        for row in cursor:
            other_data = json.loads(row["profile_data"])

            # Calculate similarity score
            similarity = self._calculate_similarity(profile.to_dict(), other_data)

            if similarity > 0.5:  # Threshold for similarity
                similar.append((row["entity_id"], similarity))

        # Sort by similarity and return top N
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:limit]

    def _calculate_similarity(self, profile1: dict, profile2: dict) -> float:
        """Calculate similarity between two profiles"""
        score = 0.0

        # Compare TTPs
        ttps1 = set(profile1.get("ttps", []))
        ttps2 = set(profile2.get("ttps", []))
        if ttps1 and ttps2:
            ttp_similarity = len(ttps1.intersection(ttps2)) / len(ttps1.union(ttps2))
            score += ttp_similarity * 0.3

        # Compare ports
        ports1 = set(profile1.get("unique_ports", []))
        ports2 = set(profile2.get("unique_ports", []))
        if ports1 and ports2:
            port_similarity = len(ports1.intersection(ports2)) / len(
                ports1.union(ports2)
            )
            score += port_similarity * 0.2

        # Compare threat levels
        if profile1.get("threat_level") == profile2.get("threat_level"):
            score += 0.2

        # Compare tags
        tags1 = set(profile1.get("tags", []))
        tags2 = set(profile2.get("tags", []))
        if tags1 and tags2:
            tag_similarity = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
            score += tag_similarity * 0.2

        # Compare ASN/Country
        if profile1.get("asn") == profile2.get("asn") and profile1.get("asn"):
            score += 0.1

        return score

    def get_high_threat_profiles(self, limit: int = 50) -> list[AttackerProfile]:
        """Get profiles with highest threat levels"""
        cursor = self.conn.execute(
            """SELECT entity_id FROM profiles
               WHERE threat_level >= ?
               ORDER BY threat_score DESC
               LIMIT ?""",
            (ThreatLevel.HIGH.value, limit),
        )

        profiles = []
        for row in cursor:
            profile = self.get_profile(row["entity_id"])
            if profile:
                profiles.append(profile)

        return profiles

    def save_analysis_state(
        self,
        analysis_id: str,
        window_start: datetime,
        window_end: datetime,
        state: dict,
    ):
        """Save analysis state for resumption"""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO analysis_state
            (analysis_id, window_start, window_end, state, entities_processed, patterns_found)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                analysis_id,
                window_start,
                window_end,
                json.dumps(state),
                state.get("entities_processed", 0),
                state.get("patterns_found", 0),
            ),
        )
        self.conn.commit()

    def get_analysis_state(self, analysis_id: str) -> dict | None:
        """Get saved analysis state"""
        cursor = self.conn.execute(
            "SELECT * FROM analysis_state WHERE analysis_id = ?", (analysis_id,)
        )
        row = cursor.fetchone()

        if row:
            return {
                "window_start": datetime.fromisoformat(row["window_start"]),
                "window_end": datetime.fromisoformat(row["window_end"]),
                "state": json.loads(row["state"]),
                "entities_processed": row["entities_processed"],
                "patterns_found": row["patterns_found"],
            }

        return None

    def close(self):
        """Close database connection"""
        self.conn.close()
