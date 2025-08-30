# test_analyzer.py
"""
Test script to validate the long-tail analyzer implementation.
Run this to test the MCP connection and basic functionality.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.mcp_client import EnhancedMCPClient, MCPQueryOptimizer
from src.agents.profile_manager import ProfileManager, EntityType
from src.agents.data_processor import TimeWindowProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/test_analyzer.log')
    ]
)

logger = logging.getLogger(__name__)


async def test_mcp_connection():
    """Test basic MCP server connection"""
    logger.info("=" * 50)
    logger.info("Testing MCP Server Connection")
    logger.info("=" * 50)
    
    async with EnhancedMCPClient() as client:
        # Test connection
        connected = await client.test_connection()
        if connected:
            logger.info("✓ Successfully connected to MCP server")
        else:
            logger.error("✗ Failed to connect to MCP server")
            return False
        
        # Get data dictionary
        data_dict = await client.get_data_dictionary()
        if data_dict:
            logger.info(f"✓ Retrieved data dictionary with {len(data_dict)} entries")
        else:
            logger.warning("⚠ Could not retrieve data dictionary")
    
    return True


async def test_data_retrieval():
    """Test data retrieval with pagination"""
    logger.info("=" * 50)
    logger.info("Testing Data Retrieval with Pagination")
    logger.info("=" * 50)
    
    async with EnhancedMCPClient() as client:
        # Test pagination - get last 1 hour of data
        total_events = 0
        chunks_retrieved = 0
        
        async for chunk in client.query_with_pagination(
            time_range_hours=1,
            page_size=100,
            fields=["source.ip", "destination.port", "@timestamp", "event.action"]
        ):
            chunks_retrieved += 1
            total_events += len(chunk)
            
            logger.info(f"Retrieved chunk {chunks_retrieved}: {len(chunk)} events")
            
            # Show sample event
            if chunk and chunks_retrieved == 1:
                logger.info(f"Sample event: {chunk[0]}")
            
            # Limit for testing
            if chunks_retrieved >= 3:
                break
        
        logger.info(f"✓ Retrieved {total_events} total events in {chunks_retrieved} chunks")
    
    return total_events > 0


async def test_summary_aggregation():
    """Test summary aggregation for time windows"""
    logger.info("=" * 50)
    logger.info("Testing Summary Aggregation")
    logger.info("=" * 50)
    
    async with EnhancedMCPClient() as client:
        # Get summary for last 6 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=6)
        
        summary = await client.get_aggregated_summary(start_time, end_time)
        
        logger.info(f"Summary for {start_time} to {end_time}:")
        logger.info(f"  Total events: {summary.total_events}")
        logger.info(f"  Unique sources: {summary.unique_sources}")
        logger.info(f"  Top sources: {len(summary.top_sources)}")
        
        # Show top 3 sources
        for i, source in enumerate(summary.top_sources[:3]):
            logger.info(f"    #{i+1}: {source['ip']} ({source['count']} events)")
        
        # Test anomaly detection
        optimizer = MCPQueryOptimizer(client)
        anomalies = await optimizer.identify_anomalies(summary)
        
        logger.info(f"✓ Identified {len(anomalies)} anomalies")
        for anomaly in anomalies[:3]:
            logger.info(f"  Anomaly: {anomaly['entity_id']} "
                       f"(type: {anomaly['entity_type']}, "
                       f"reason: {anomaly['reason']}, "
                       f"score: {anomaly['score']:.2f})")
    
    return summary.total_events > 0


async def test_profile_management():
    """Test profile creation and management"""
    logger.info("=" * 50)
    logger.info("Testing Profile Management")
    logger.info("=" * 50)
    
    # Initialize profile manager
    profile_manager = ProfileManager("data/test_profiles.db")
    
    # Create a test profile
    test_events = [
        {
            "@timestamp": datetime.now().isoformat(),
            "source.ip": "192.168.1.100",
            "destination.ip": "10.0.0.1",
            "destination.port": 22,
            "event.action": "ssh_brute_force"
        },
        {
            "@timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
            "source.ip": "192.168.1.100",
            "destination.ip": "10.0.0.2",
            "destination.port": 3389,
            "event.action": "rdp_scan"
        }
    ]
    
    # Update profile with test events
    profile = profile_manager.update_profile_incrementally(
        entity_id="ip_192.168.1.100",
        entity_type=EntityType.IP,
        new_events=test_events
    )
    
    logger.info(f"✓ Created profile for {profile.entity_id}")
    logger.info(f"  Threat level: {profile.threat_level.name}")
    logger.info(f"  Total events: {profile.total_events}")
    logger.info(f"  Unique targets: {profile.unique_targets}")
    logger.info(f"  Attack patterns: {len(profile.scanning_patterns)} scanning, "
               f"{len(profile.exploitation_attempts)} exploitation")
    
    # Test profile retrieval
    retrieved_profile = profile_manager.get_profile("ip_192.168.1.100")
    if retrieved_profile:
        logger.info(f"✓ Successfully retrieved profile from database")
    
    # Test high threat profiles
    high_threat = profile_manager.get_high_threat_profiles(limit=5)
    logger.info(f"✓ Found {len(high_threat)} high threat profiles")
    
    profile_manager.close()
    return True


async def test_time_window_processing():
    """Test time window processing"""
    logger.info("=" * 50)
    logger.info("Testing Time Window Processing")
    logger.info("=" * 50)
    
    async with EnhancedMCPClient() as client:
        profile_manager = ProfileManager("data/test_profiles.db")
        
        processor = TimeWindowProcessor(
            mcp_client=client,
            profile_manager=profile_manager,
            window_hours=1,  # Small window for testing
            overlap_hours=0.25  # 15 minute overlap
        )
        
        # Generate windows for last 3 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=3)
        
        windows = processor.generate_time_windows(start_time, end_time)
        logger.info(f"✓ Generated {len(windows)} time windows")
        
        # Process first window
        if windows:
            result = await processor.process_window(windows[0])
            
            logger.info(f"✓ Processed window: {result}")
            
            if result["status"] == "processed":
                logger.info(f"  Events in window: {result.get('total_events', 0)}")
                logger.info(f"  Entities analyzed: {result.get('entities_analyzed', 0)}")
                logger.info(f"  Profiles updated: {result.get('profiles_updated', 0)}")
        
        # Get processing stats
        stats = processor.get_processing_stats()
        logger.info(f"✓ Processing statistics: {stats}")
        
        profile_manager.close()
    
    return True


async def test_streaming():
    """Test streaming with session context"""
    logger.info("=" * 50)
    logger.info("Testing Streaming with Session Context")
    logger.info("=" * 50)
    
    async with EnhancedMCPClient() as client:
        # Stream last 1 hour with session grouping
        sessions_processed = 0
        total_events = 0
        
        try:
            async for session_data in client.stream_with_sessions(
                time_range_hours=1,
                chunk_size=100,
                max_session_gap_minutes=15
            ):
                sessions_processed += 1
                
                # Count events in sessions
                for session_id, events in session_data.items():
                    total_events += len(events)
                    logger.info(f"Session {session_id}: {len(events)} events")
                
                # Limit for testing
                if sessions_processed >= 3:
                    break
            
            logger.info(f"✓ Processed {sessions_processed} sessions with {total_events} total events")
        
        except Exception as e:
            logger.warning(f"Streaming test skipped: {e}")
            return False
    
    return sessions_processed > 0


async def run_all_tests():
    """Run all tests in sequence"""
    logger.info("Starting Long-Tail Analyzer Test Suite")
    logger.info("=" * 50)
    
    # Ensure directories exist
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    tests = [
        ("MCP Connection", test_mcp_connection),
        ("Data Retrieval", test_data_retrieval),
        ("Summary Aggregation", test_summary_aggregation),
        ("Profile Management", test_profile_management),
        ("Time Window Processing", test_time_window_processing),
        ("Streaming", test_streaming)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\nRunning: {test_name}")
            result = await test_func()
            results[test_name] = "✓ PASSED" if result else "✗ FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} failed with error: {e}")
            results[test_name] = f"✗ ERROR: {str(e)[:50]}"
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


async def quick_demo():
    """Quick demonstration of the system"""
    logger.info("=" * 50)
    logger.info("QUICK DEMO: Analyzing Last Hour of Data")
    logger.info("=" * 50)
    
    async with EnhancedMCPClient() as client:
        # 1. Get summary
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        logger.info(f"Analyzing data from {start_time} to {end_time}")
        
        summary = await client.get_aggregated_summary(start_time, end_time)
        logger.info(f"Found {summary.total_events} events from {summary.unique_sources} sources")
        
        # 2. Find anomalies
        optimizer = MCPQueryOptimizer(client)
        anomalies = await optimizer.identify_anomalies(summary, threshold_multiplier=1.5)
        logger.info(f"Identified {len(anomalies)} anomalous entities")
        
        # 3. Create profiles for top 3 anomalies
        profile_manager = ProfileManager("data/demo_profiles.db")
        
        for anomaly in anomalies[:3]:
            logger.info(f"\nProcessing anomaly: {anomaly['entity_id']}")
            
            # Get entity details
            events = await client.get_entity_details(
                entity_id=anomaly['entity_id'],
                entity_type=anomaly['entity_type'],
                time_range_hours=1
            )
            
            if events:
                # Update profile
                profile = profile_manager.update_profile_incrementally(
                    entity_id=f"{anomaly['entity_type']}_{anomaly['entity_id']}",
                    entity_type=EntityType.IP,
                    new_events=events[:50]  # Limit for demo
                )
                
                logger.info(f"  Created profile:")
                logger.info(f"    Threat level: {profile.threat_level.name}")
                logger.info(f"    Events: {profile.total_events}")
                logger.info(f"    Threat score: {profile.calculate_threat_score():.2f}")
        
        # 4. Show high threat profiles
        high_threats = profile_manager.get_high_threat_profiles(limit=5)
        
        if high_threats:
            logger.info(f"\nHigh Threat Profiles:")
            for profile in high_threats:
                logger.info(f"  {profile.entity_id}: Level {profile.threat_level.name}, "
                          f"Score {profile.calculate_threat_score():.2f}")
        
        profile_manager.close()
        
        logger.info("\nDemo complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Long-Tail Analyzer")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--demo", action="store_true", help="Run quick demo")
    parser.add_argument("--test", choices=[
        "connection", "retrieval", "summary", "profile", "window", "streaming"
    ], help="Run specific test")
    
    args = parser.parse_args()
    
    if args.full:
        asyncio.run(run_all_tests())
    elif args.demo:
        asyncio.run(quick_demo())
    elif args.test:
        test_map = {
            "connection": test_mcp_connection,
            "retrieval": test_data_retrieval,
            "summary": test_summary_aggregation,
            "profile": test_profile_management,
            "window": test_time_window_processing,
            "streaming": test_streaming
        }
        asyncio.run(test_map[args.test]())
    else:
        # Default to demo
        asyncio.run(quick_demo())