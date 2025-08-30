#!/usr/bin/env python3
"""
Main entry point for the Long-Tail Analysis Agent.

This script provides a command-line interface for running the analysis system.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.orchestrator import LongTailAnalyzer
from src.utils.config import ConfigManager


def setup_logging(log_level: str = "INFO", log_file: str = "logs/analyzer.log"):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Long-Tail Analysis Agent")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", default="logs/analyzer.log", help="Log file path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a time period")
    analyze_parser.add_argument("--start-days", type=int, default=7, help="Days back to start analysis")
    analyze_parser.add_argument("--end-days", type=int, default=0, help="Days back to end analysis")
    analyze_parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    
    # Realtime command
    realtime_parser = subparsers.add_parser("realtime", help="Run real-time analysis")
    realtime_parser.add_argument("--duration", type=int, default=1, help="Duration in hours")
    
    # Enrich command
    enrich_parser = subparsers.add_parser("enrich", help="Enrich profiles with external intelligence")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.load_config()
        
        if not config_manager.validate_config(config):
            logger.error("Configuration validation failed")
            return 1
        
        # Initialize analyzer
        logger.info("Initializing Long-Tail Analysis Agent...")
        analyzer = LongTailAnalyzer(args.config)
        
        if args.command == "analyze":
            # Calculate date range
            end_date = datetime.now() - timedelta(days=args.end_days)
            start_date = end_date - timedelta(days=args.start_days)
            
            logger.info(f"Starting analysis from {start_date} to {end_date}")
            
            results = await analyzer.analyze_period(
                start_date=start_date,
                end_date=end_date,
                resume=args.resume
            )
            
            logger.info("Analysis completed successfully")
            print(f"\nAnalysis Results:")
            print(f"  - Total windows: {results['total_windows']}")
            print(f"  - Processed windows: {results['processed_windows']}")
            print(f"  - Skipped windows: {results['skipped_windows']}")
            print(f"  - Error windows: {results['error_windows']}")
            print(f"  - Profiles updated: {results['profiles_updated']}")
            print(f"  - Patterns found: {results['patterns_found']}")
            print(f"  - Processing time: {results['total_processing_time']:.2f} seconds")
            
        elif args.command == "realtime":
            logger.info(f"Starting real-time analysis for {args.duration} hours")
            
            results = await analyzer.analyze_realtime(duration_hours=args.duration)
            
            logger.info("Real-time analysis completed")
            print(f"\nReal-time Analysis Results:")
            print(f"  - Sessions processed: {results['sessions_processed']}")
            print(f"  - Profiles updated: {results['profiles_updated']}")
            print(f"  - High threat sessions: {results['high_threat_sessions']}")
            
        elif args.command == "enrich":
            logger.info("Starting profile enrichment...")
            
            results = await analyzer.enrich_profiles()
            
            logger.info("Profile enrichment completed")
            print(f"\nEnrichment Results:")
            print(f"  - Profiles enriched: {results['profiles_enriched']}")
            print(f"  - Enrichment errors: {results['enrichment_errors']}")
            
        elif args.command == "status":
            status = analyzer.get_system_status()
            
            print(f"\nSystem Status:")
            print(f"  - MCP Client Healthy: {status['mcp_client_healthy']}")
            print(f"  - Active Profiles: {status['profile_count']}")
            print(f"  - Processed Windows: {status['processed_windows']}")
            print(f"  - Enrichment Cache Size: {status['enrichment_cache_size']}")
            print(f"  - Local LLM Available: {status['local_llm_available']}")
            print(f"  - API LLM Available: {status['api_llm_available']}")
            
        else:
            parser.print_help()
            return 1
        
        # Close analyzer
        await analyzer.close()
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
