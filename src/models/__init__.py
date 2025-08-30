"""
Data models for the long-tail analysis system.

This module defines the core data structures:
- AttackerProfile: Comprehensive attacker entity profiles
- Pattern: Attack pattern representations
"""

from src.models.pattern import Pattern
from src.models.profile import AttackerProfile

__all__ = ["AttackerProfile", "Pattern"]
