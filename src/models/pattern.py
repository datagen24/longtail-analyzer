"""
Pattern data model for attack pattern recognition and analysis.

This module defines the Pattern dataclass which represents
identified attack patterns with metadata and confidence scores.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class PatternType(Enum):
    """Types of attack patterns."""

    SCANNING = "scanning"
    EXPLOITATION = "exploitation"
    PERSISTENCE = "persistence"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_EXFILTRATION = "data_exfiltration"
    COMMAND_AND_CONTROL = "command_and_control"
    UNKNOWN = "unknown"


class PatternSeverity(Enum):
    """Severity levels for patterns."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Pattern:
    """
    Represents an identified attack pattern with metadata and analysis.

    This dataclass captures the essential information about attack patterns
    including type, severity, confidence, and associated entities.
    """

    # Core identification
    pattern_id: str
    pattern_type: PatternType
    name: str
    description: str

    # Pattern characteristics
    severity: PatternSeverity
    confidence_score: float  # 0.0 to 1.0

    # Temporal information
    first_detected: datetime
    last_detected: datetime
    detection_count: int = 1

    # Associated entities
    source_entities: list[str] = field(default_factory=list)
    target_entities: list[str] = field(default_factory=list)

    # Pattern details
    indicators: list[str] = field(default_factory=list)
    ttps: list[str] = field(default_factory=list)  # MITRE ATT&CK techniques

    # Analysis metadata
    detection_method: str = "unknown"
    false_positive_probability: float = 0.0
    analyst_notes: str = ""

    # Pattern data
    pattern_data: dict[str, Any] = field(default_factory=dict)
    statistical_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert pattern to dictionary.

        Returns:
            Dictionary representation of the pattern
        """
        pattern_dict = asdict(self)

        # Convert enums to strings
        pattern_dict["pattern_type"] = self.pattern_type.value
        pattern_dict["severity"] = self.severity.value

        # Convert datetime objects to ISO strings
        pattern_dict["first_detected"] = self.first_detected.isoformat()
        pattern_dict["last_detected"] = self.last_detected.isoformat()

        return pattern_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pattern":
        """
        Create pattern from dictionary.

        Args:
            data: Dictionary representation of the pattern

        Returns:
            Pattern instance
        """
        # Convert string enums back to enum objects
        if "pattern_type" in data:
            data["pattern_type"] = PatternType(data["pattern_type"])
        if "severity" in data:
            data["severity"] = PatternSeverity(data["severity"])

        # Convert ISO strings back to datetime objects
        if "first_detected" in data:
            data["first_detected"] = datetime.fromisoformat(data["first_detected"])
        if "last_detected" in data:
            data["last_detected"] = datetime.fromisoformat(data["last_detected"])

        return cls(**data)

    def update_detection(self, timestamp: datetime) -> None:
        """
        Update pattern with a new detection.

        Args:
            timestamp: When the pattern was detected
        """
        self.detection_count += 1
        self.last_detected = timestamp

        # Update first detected if this is earlier
        if timestamp < self.first_detected:
            self.first_detected = timestamp

    def add_indicator(self, indicator: str) -> None:
        """
        Add an indicator to the pattern.

        Args:
            indicator: Indicator to add
        """
        if indicator not in self.indicators:
            self.indicators.append(indicator)

    def add_ttp(self, ttp: str) -> None:
        """
        Add a TTP to the pattern.

        Args:
            ttp: TTP identifier to add
        """
        if ttp not in self.ttps:
            self.ttps.append(ttp)

    def add_source_entity(self, entity_id: str) -> None:
        """
        Add a source entity to the pattern.

        Args:
            entity_id: ID of source entity
        """
        if entity_id not in self.source_entities:
            self.source_entities.append(entity_id)

    def add_target_entity(self, entity_id: str) -> None:
        """
        Add a target entity to the pattern.

        Args:
            entity_id: ID of target entity
        """
        if entity_id not in self.target_entities:
            self.target_entities.append(entity_id)

    def is_recent(self, hours: int = 24) -> bool:
        """
        Check if the pattern was detected recently.

        Args:
            hours: Number of hours to check

        Returns:
            True if pattern was detected within the time window
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return self.last_detected > cutoff

    def get_severity_score(self) -> float:
        """
        Get numeric severity score.

        Returns:
            Numeric severity score (1.0 to 4.0)
        """
        severity_scores = {
            PatternSeverity.LOW: 1.0,
            PatternSeverity.MEDIUM: 2.0,
            PatternSeverity.HIGH: 3.0,
            PatternSeverity.CRITICAL: 4.0,
        }
        return severity_scores.get(self.severity, 1.0)

    def get_risk_score(self) -> float:
        """
        Calculate overall risk score combining severity and confidence.

        Returns:
            Risk score (0.0 to 4.0)
        """
        return self.get_severity_score() * self.confidence_score

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the pattern.

        Returns:
            Dictionary with pattern summary
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "name": self.name,
            "severity": self.severity.value,
            "confidence_score": self.confidence_score,
            "risk_score": self.get_risk_score(),
            "detection_count": self.detection_count,
            "first_detected": self.first_detected.isoformat(),
            "last_detected": self.last_detected.isoformat(),
            "source_entities_count": len(self.source_entities),
            "target_entities_count": len(self.target_entities),
            "indicators_count": len(self.indicators),
            "ttps_count": len(self.ttps),
        }
