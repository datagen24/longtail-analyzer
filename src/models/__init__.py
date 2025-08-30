"""
Data models for the long-tail analysis system.

This module defines the core data structures:
- AttackerProfile: Comprehensive attacker entity profiles
- Pattern: Attack pattern representations
"""

from .profile import AttackerProfile
from .pattern import Pattern

__all__ = ["AttackerProfile", "Pattern"]
