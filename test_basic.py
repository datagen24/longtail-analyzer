#!/usr/bin/env python3
"""
Basic test script for the Long-Tail Analysis Agent.

This script tests the basic functionality of the system components.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.mcp_client import EnhancedMCPClient
from src.models.profile import AttackerProfile
from src.agents.profile_manager import ProfileManager
from src.agents.pattern_analyzer import PatternAnalyzer
from src.utils.config import ConfigManager


async def test_mcp_client():
    """Test MCP client functionality."""
    print("Testing MCP Client...")
    
    client = EnhancedMCPClient("http://localhost:3000")
    
    try:
        # Test health check
        is_healthy = await client.health_check()
        print(f"  MCP Server Health: {'✓' if is_healthy else '✗'}")
        
        if is_healthy:
            # Test data dictionary
            data_dict = await client.get_data_dictionary()
            print(f"  Data Dictionary: {'✓' if data_dict else '✗'}")
            
            # Test small query
            events = await client.query_with_pagination(
                time_range_hours=1,
                page_size=10
            )
            print(f"  Query Test: {'✓' if events is not None else '✗'} ({len(events) if events else 0} events)")
        
    except Exception as e:
        print(f"  MCP Client Error: {e}")
    finally:
        await client.close()


def test_profile_model():
    """Test profile model functionality."""
    print("Testing Profile Model...")
    
    try:
        # Create a test profile
        profile = AttackerProfile(
            entity_id="ip_192.168.1.100",
            entity_type="ip",
            first_seen=datetime.now(),
            last_seen=datetime.now()
        )
        
        # Test serialization
        profile_dict = profile.to_dict()
        print(f"  Profile Serialization: ✓")
        
        # Test deserialization
        restored_profile = AttackerProfile.from_dict(profile_dict)
        print(f"  Profile Deserialization: ✓")
        
        # Test activity update
        test_events = [
            {"@timestamp": "2024-01-01T10:00:00Z", "source.ip": "192.168.1.100"},
            {"@timestamp": "2024-01-01T11:00:00Z", "source.ip": "192.168.1.100"}
        ]
        profile.update_activity(test_events)
        print(f"  Profile Activity Update: ✓")
        
    except Exception as e:
        print(f"  Profile Model Error: {e}")


def test_profile_manager():
    """Test profile manager functionality."""
    print("Testing Profile Manager...")
    
    try:
        # Initialize profile manager
        pm = ProfileManager("data/test_profiles.db")
        
        # Create test profile
        profile = AttackerProfile(
            entity_id="ip_192.168.1.200",
            entity_type="ip",
            first_seen=datetime.now(),
            last_seen=datetime.now()
        )
        
        # Test save
        success = pm.save_profile(profile)
        print(f"  Profile Save: {'✓' if success else '✗'}")
        
        # Test retrieve
        retrieved = pm.get_profile("ip_192.168.1.200")
        print(f"  Profile Retrieve: {'✓' if retrieved else '✗'}")
        
        # Test statistics
        stats = pm.get_profile_statistics()
        print(f"  Profile Statistics: ✓")
        
        # Cleanup
        pm.close()
        Path("data/test_profiles.db").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"  Profile Manager Error: {e}")


def test_pattern_analyzer():
    """Test pattern analyzer functionality."""
    print("Testing Pattern Analyzer...")
    
    try:
        analyzer = PatternAnalyzer()
        
        # Create test events
        test_events = [
            {
                "@timestamp": "2024-01-01T10:00:00Z",
                "source.ip": "192.168.1.100",
                "destination.port": 22,
                "event.action": "ssh_attempt"
            },
            {
                "@timestamp": "2024-01-01T10:01:00Z",
                "source.ip": "192.168.1.100",
                "destination.port": 23,
                "event.action": "telnet_attempt"
            }
        ]
        
        # Test pattern analysis (synchronous for now)
        print(f"  Pattern Analyzer Initialization: ✓")
        
    except Exception as e:
        print(f"  Pattern Analyzer Error: {e}")


def test_config_manager():
    """Test configuration manager functionality."""
    print("Testing Configuration Manager...")
    
    try:
        config_manager = ConfigManager("configs/default.yaml")
        config = config_manager.load_config()
        
        print(f"  Config Load: ✓")
        
        # Test validation
        is_valid = config_manager.validate_config(config)
        print(f"  Config Validation: {'✓' if is_valid else '✗'}")
        
    except Exception as e:
        print(f"  Config Manager Error: {e}")


async def main():
    """Run all tests."""
    print("Long-Tail Analysis Agent - Basic Tests")
    print("=" * 50)
    
    # Test individual components
    await test_mcp_client()
    test_profile_model()
    test_profile_manager()
    test_pattern_analyzer()
    test_config_manager()
    
    print("\n" + "=" * 50)
    print("Basic tests completed!")


if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(main())
