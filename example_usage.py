#!/usr/bin/env python3
"""
Example usage of the Long-Tail Analysis Agent.

This script demonstrates how to use the analysis system for different scenarios.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import LongTailAnalyzer


async def example_historical_analysis():
    """Example: Analyze historical data for the past week."""
    print("Example 1: Historical Analysis (Past Week)")
    print("-" * 50)
    
    # Initialize analyzer
    analyzer = LongTailAnalyzer("configs/default.yaml")
    
    try:
        # Define analysis period (past week)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"Analyzing period: {start_date} to {end_date}")
        
        # Run analysis
        results = await analyzer.analyze_period(
            start_date=start_date,
            end_date=end_date,
            resume=True  # Resume from last checkpoint if available
        )
        
        # Display results
        print(f"\nAnalysis Results:")
        print(f"  - Total time windows: {results['total_windows']}")
        print(f"  - Successfully processed: {results['processed_windows']}")
        print(f"  - Profiles updated: {results['profiles_updated']}")
        print(f"  - Attack patterns found: {results['patterns_found']}")
        print(f"  - Processing time: {results['total_processing_time']:.2f} seconds")
        
        # Show profile statistics
        if 'profile_statistics' in results:
            stats = results['profile_statistics']
            print(f"\nProfile Statistics:")
            print(f"  - Total profiles: {stats.get('total_profiles', 0)}")
            print(f"  - Total events processed: {stats.get('total_events', 0)}")
            
            for entity_type, type_stats in stats.get('by_type', {}).items():
                print(f"  - {entity_type.upper()} profiles: {type_stats['count']}")
        
    except Exception as e:
        print(f"Error in historical analysis: {e}")
    finally:
        await analyzer.close()


async def example_realtime_analysis():
    """Example: Run real-time analysis for 1 hour."""
    print("\nExample 2: Real-time Analysis (1 Hour)")
    print("-" * 50)
    
    # Initialize analyzer
    analyzer = LongTailAnalyzer("configs/default.yaml")
    
    try:
        print("Starting real-time analysis...")
        
        # Run real-time analysis
        results = await analyzer.analyze_realtime(duration_hours=1)
        
        # Display results
        print(f"\nReal-time Analysis Results:")
        print(f"  - Sessions processed: {results['sessions_processed']}")
        print(f"  - Profiles updated: {results['profiles_updated']}")
        print(f"  - High threat sessions: {results['high_threat_sessions']}")
        
        if 'error' in results:
            print(f"  - Errors: {results['error']}")
        
    except Exception as e:
        print(f"Error in real-time analysis: {e}")
    finally:
        await analyzer.close()


async def example_profile_enrichment():
    """Example: Enrich profiles with external intelligence."""
    print("\nExample 3: Profile Enrichment")
    print("-" * 50)
    
    # Initialize analyzer
    analyzer = LongTailAnalyzer("configs/default.yaml")
    
    try:
        print("Starting profile enrichment...")
        
        # Enrich all active profiles
        results = await analyzer.enrich_profiles()
        
        # Display results
        print(f"\nEnrichment Results:")
        print(f"  - Profiles enriched: {results['profiles_enriched']}")
        print(f"  - Enrichment errors: {results['enrichment_errors']}")
        
    except Exception as e:
        print(f"Error in profile enrichment: {e}")
    finally:
        await analyzer.close()


async def example_system_status():
    """Example: Check system status."""
    print("\nExample 4: System Status Check")
    print("-" * 50)
    
    # Initialize analyzer
    analyzer = LongTailAnalyzer("configs/default.yaml")
    
    try:
        # Get system status
        status = analyzer.get_system_status()
        
        print("System Status:")
        print(f"  - MCP Server: {'✓ Healthy' if status['mcp_client_healthy'] else '✗ Unhealthy'}")
        print(f"  - Active Profiles: {status['profile_count']}")
        print(f"  - Processed Windows: {status['processed_windows']}")
        print(f"  - Enrichment Cache: {status['enrichment_cache_size']} items")
        print(f"  - Local LLM: {'✓ Available' if status['local_llm_available'] else '✗ Unavailable'}")
        print(f"  - API LLM: {'✓ Available' if status['api_llm_available'] else '✗ Unavailable'}")
        
    except Exception as e:
        print(f"Error checking system status: {e}")
    finally:
        await analyzer.close()


async def example_custom_analysis():
    """Example: Custom analysis with specific parameters."""
    print("\nExample 5: Custom Analysis")
    print("-" * 50)
    
    # Initialize analyzer with custom config
    analyzer = LongTailAnalyzer("configs/default.yaml")
    
    try:
        # Custom analysis period (last 3 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        print(f"Custom analysis period: {start_date} to {end_date}")
        print("Using custom parameters:")
        print("  - Window size: 4 hours")
        print("  - Overlap: 30 minutes")
        print("  - Max entities per window: 25")
        
        # Note: In a real implementation, you would modify the analyzer's
        # time processor configuration before running analysis
        
        # Run analysis
        results = await analyzer.analyze_period(
            start_date=start_date,
            end_date=end_date,
            resume=False  # Start fresh
        )
        
        print(f"\nCustom Analysis Results:")
        print(f"  - Processing time: {results['total_processing_time']:.2f} seconds")
        print(f"  - Efficiency: {results['processed_windows']}/{results['total_windows']} windows")
        
    except Exception as e:
        print(f"Error in custom analysis: {e}")
    finally:
        await analyzer.close()


async def main():
    """Run all examples."""
    print("Long-Tail Analysis Agent - Usage Examples")
    print("=" * 60)
    
    # Check if MCP server is available
    print("Note: These examples require the DShield MCP server to be running on localhost:3000")
    print("Make sure to start the MCP server before running these examples.\n")
    
    # Run examples
    await example_system_status()
    await example_historical_analysis()
    await example_realtime_analysis()
    await example_profile_enrichment()
    await example_custom_analysis()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nFor more information, see:")
    print("  - README.md for detailed documentation")
    print("  - configs/default.yaml for configuration options")
    print("  - main.py for command-line interface")


if __name__ == "__main__":
    asyncio.run(main())
