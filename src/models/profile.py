"""
Attacker profile data model for the long-tail analysis system.

This module defines the AttackerProfile dataclass which represents
comprehensive attacker entity profiles with behavioral patterns,
confidence scores, and metadata.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass
class AttackerProfile:
    """
    Comprehensive attacker profile for tracking entities over time.

    This dataclass represents a complete profile of an attacker entity
    (IP, ASN, or composite) with behavioral patterns, confidence scores,
    and analysis metadata.
    """

    # Core identification
    entity_id: str  # e.g., "ip_192.168.1.1" or "asn_AS12345"
    entity_type: str  # "ip", "asn", or "composite"

    # Temporal tracking
    first_seen: datetime
    last_seen: datetime

    # Attack patterns and behaviors
    scanning_patterns: dict[str, Any] = field(default_factory=dict)
    exploitation_attempts: dict[str, Any] = field(default_factory=dict)
    persistence_indicators: dict[str, Any] = field(default_factory=dict)

    # Behavioral fingerprint (embedding vector)
    behavioral_embedding: np.ndarray | None = None

    # Confidence and analysis metadata
    confidence_scores: dict[str, float] = field(default_factory=dict)
    ttps: list[str] = field(default_factory=list)  # Tactics, Techniques, Procedures

    # Infrastructure information
    asn: str | None = None
    geo_location: str | None = None
    related_entities: list[str] = field(default_factory=list)

    # Analysis metadata
    last_analysis: datetime = field(default_factory=datetime.now)
    analysis_depth: str = "shallow"  # shallow, deep, comprehensive
    data_quality_score: float = 0.0

    # Activity tracking
    activity_windows: list[dict[str, Any]] = field(default_factory=list)
    total_events: int = 0
    unique_targets: int = 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert profile to dictionary, handling numpy arrays.

        Returns:
            Dictionary representation of the profile
        """
        profile_dict = asdict(self)

        # Handle numpy array serialization
        if self.behavioral_embedding is not None:
            profile_dict["behavioral_embedding"] = self.behavioral_embedding.tolist()

        # Convert datetime objects to ISO strings
        profile_dict["first_seen"] = self.first_seen.isoformat()
        profile_dict["last_seen"] = self.last_seen.isoformat()
        profile_dict["last_analysis"] = self.last_analysis.isoformat()

        return profile_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttackerProfile":
        """
        Create profile from dictionary, handling numpy arrays and datetimes.

        Args:
            data: Dictionary representation of the profile

        Returns:
            AttackerProfile instance
        """
        # Handle numpy array deserialization
        if "behavioral_embedding" in data and data["behavioral_embedding"] is not None:
            data["behavioral_embedding"] = np.array(data["behavioral_embedding"])

        # Convert ISO strings back to datetime objects
        if "first_seen" in data:
            data["first_seen"] = datetime.fromisoformat(data["first_seen"])
        if "last_seen" in data:
            data["last_seen"] = datetime.fromisoformat(data["last_seen"])
        if "last_analysis" in data:
            data["last_analysis"] = datetime.fromisoformat(data["last_analysis"])

        return cls(**data)

    def update_activity(self, events: list[dict[str, Any]]) -> None:
        """
        Update profile with new activity events.

        Args:
            events: List of new events to incorporate
        """
        if not events:
            return

        # Update temporal tracking
        event_times = [
            datetime.fromisoformat(e.get("@timestamp", ""))
            for e in events
            if "@timestamp" in e
        ]
        if event_times:
            self.last_seen = max(self.last_seen, max(event_times))
            if not hasattr(self, "first_seen") or self.first_seen is None:
                self.first_seen = min(event_times)
            else:
                self.first_seen = min(self.first_seen, min(event_times))

        # Update activity counts
        self.total_events += len(events)

        # Extract unique targets
        targets = set()
        for event in events:
            if "destination.ip" in event:
                targets.add(event["destination.ip"])
            if "destination.port" in event:
                targets.add(str(event["destination.port"]))

        self.unique_targets = len(targets)

        # Update last analysis time
        self.last_analysis = datetime.now()

    def get_confidence_score(self, metric: str = "overall") -> float:
        """
        Get confidence score for a specific metric.

        Args:
            metric: Confidence metric to retrieve

        Returns:
            Confidence score (0.0 to 1.0)
        """
        return self.confidence_scores.get(metric, 0.0)

    def set_confidence_score(self, metric: str, score: float) -> None:
        """
        Set confidence score for a specific metric.

        Args:
            metric: Confidence metric to set
            score: Confidence score (0.0 to 1.0)
        """
        self.confidence_scores[metric] = max(0.0, min(1.0, score))

    def add_ttp(self, ttp: str) -> None:
        """
        Add a TTP (Tactic, Technique, Procedure) to the profile.

        Args:
            ttp: TTP identifier to add
        """
        if ttp not in self.ttps:
            self.ttps.append(ttp)

    def add_related_entity(self, entity_id: str) -> None:
        """
        Add a related entity to the profile.

        Args:
            entity_id: ID of related entity
        """
        if entity_id not in self.related_entities:
            self.related_entities.append(entity_id)

    def is_active(self, hours: int = 24) -> bool:
        """
        Check if the profile has been active within the specified hours.

        Args:
            hours: Number of hours to check for activity

        Returns:
            True if active within the time window
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return self.last_seen > cutoff

    def get_activity_summary(self) -> dict[str, Any]:
        """
        Get a summary of the profile's activity.

        Returns:
            Dictionary with activity summary
        """
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "total_events": self.total_events,
            "unique_targets": self.unique_targets,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "confidence_overall": self.get_confidence_score("overall"),
            "analysis_depth": self.analysis_depth,
            "data_quality": self.data_quality_score,
            "ttps_count": len(self.ttps),
            "related_entities_count": len(self.related_entities),
        }
