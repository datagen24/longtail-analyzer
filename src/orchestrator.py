"""
Main orchestrator for the long-tail analysis system.

This module coordinates all components of the analysis system including
data processing, pattern analysis, profile management, and intelligence
enrichment to perform comprehensive long-tail security analysis.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.agents.data_processor import TimeWindowProcessor
from src.agents.enrichment_agent import EnrichmentAgent
from src.agents.pattern_analyzer import PatternAnalyzer
from src.agents.profile_manager import ProfileManager
from src.llm.api_llm import ClaudeAPI, OpenAIAPI
from src.llm.local_llm import OllamaLLM
from src.utils.mcp_client import EnhancedMCPClient

logger = logging.getLogger(__name__)


class LongTailAnalyzer:
    """
    Main orchestrator for long-tail analysis.

    This class coordinates the entire analysis workflow including:
    - Time window processing with intelligent chunking
    - Pattern recognition and analysis
    - Profile management and evolution
    - Intelligence enrichment
    - State management and resumption
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Initialize the long-tail analyzer.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.analysis_state = {}

        # Initialize components
        self.mcp_client = EnhancedMCPClient(
            mcp_url=self.config.get("mcp_url", "http://localhost:3000"),
            timeout=self.config.get("mcp_timeout", 30.0),
        )

        self.profile_manager = ProfileManager(
            db_path=self.config.get("db_path", "data/profiles.db")
        )

        self.time_processor = TimeWindowProcessor(
            mcp_client=self.mcp_client,
            profile_manager=self.profile_manager,
            window_hours=self.config.get("window_hours", 6),
            overlap_hours=self.config.get("overlap_hours", 1),
            max_entities_per_window=self.config.get("max_entities_per_window", 50),
        )

        self.pattern_analyzer = PatternAnalyzer()
        self.enrichment_agent = EnrichmentAgent(self.config.get("enrichment", {}))

        # Initialize LLMs
        self.local_llm: OllamaLLM | None = None
        self.api_llm: ClaudeAPI | OpenAIAPI | None = None
        self._initialize_llms()

        # Load analysis state
        self.analysis_state = self._load_analysis_state()

        logger.info("LongTailAnalyzer initialized successfully")

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file {config_path} not found, using defaults")
                return self._get_default_config()

            with open(config_file) as f:
                config = yaml.safe_load(f)

            logger.info(f"Configuration loaded from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "mcp_url": "http://localhost:3000",
            "mcp_timeout": 30,
            "db_path": "data/profiles.db",
            "window_hours": 6,
            "overlap_hours": 1,
            "max_entities_per_window": 50,
            "anomaly_threshold": 2.0,
        }

    def _initialize_llms(self) -> None:
        """Initialize LLM components based on configuration."""
        try:
            # Initialize local LLM
            local_config = self.config.get("local_llm", {})
            if local_config.get("provider") == "ollama":
                self.local_llm = OllamaLLM(
                    model=local_config.get("model", "mixtral:8x7b"),
                    temperature=local_config.get("temperature", 0.7),
                    max_tokens=local_config.get("max_tokens", 4096),
                )
                logger.info("Local LLM (Ollama) initialized")

            # Initialize API LLM
            api_config = self.config.get("api_llm", {})
            if api_config.get("use_api"):
                providers = api_config.get("providers", {})

                if "claude" in providers:
                    claude_config = providers["claude"]
                    self.api_llm = ClaudeAPI(
                        api_key=claude_config.get("api_key"),
                        model=claude_config.get("model", "claude-3-opus-20240229"),
                    )
                    logger.info("Claude API initialized")

                elif "openai" in providers:
                    openai_config = providers["openai"]
                    self.api_llm = OpenAIAPI(
                        api_key=openai_config.get("api_key"),
                        model=openai_config.get("model", "gpt-4-turbo-preview"),
                    )
                    logger.info("OpenAI API initialized")

        except Exception as e:
            logger.error(f"Error initializing LLMs: {e}")

    def _load_analysis_state(self) -> dict[str, Any]:
        """Load analysis state from database."""
        try:
            # This would load from the profile manager's analysis state table
            return {}
        except Exception as e:
            logger.error(f"Error loading analysis state: {e}")
            return {}

    def _save_analysis_state(self, state: dict[str, Any]) -> None:
        """Save analysis state to database."""
        try:
            # This would save to the profile manager's analysis state table
            self.analysis_state.update(state)
        except Exception as e:
            logger.error(f"Error saving analysis state: {e}")

    async def analyze_period(
        self, start_date: datetime, end_date: datetime, resume: bool = True
    ) -> dict[str, Any]:
        """
        Analyze a time period with automatic chunking and state management.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            resume: Whether to resume from last checkpoint

        Returns:
            Analysis report
        """
        logger.info(f"Starting analysis from {start_date} to {end_date}")

        # Generate time windows
        windows = self.time_processor.generate_time_windows(start_date, end_date)

        # Resume from last checkpoint if requested
        if resume and self.analysis_state:
            windows = self._filter_processed_windows(windows)

        # Process windows in sequence with progress tracking
        results = {
            "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_windows": len(windows),
            "processed_windows": 0,
            "skipped_windows": 0,
            "error_windows": 0,
            "profiles_updated": 0,
            "patterns_found": 0,
            "processing_start": datetime.now().isoformat(),
        }

        for i, window in enumerate(windows):
            logger.info(
                f"Processing window {i+1}/{len(windows)}: {window.start_time} to {window.end_time}"
            )

            try:
                result = await self.time_processor.process_window(
                    window, self.pattern_analyzer
                )

                # Update results
                if result["status"] == "processed":
                    results["processed_windows"] += 1
                    results["profiles_updated"] += result.get("profiles_updated", 0)
                    results["patterns_found"] += result.get("patterns_found", 0)
                elif result["status"] == "skipped":
                    results["skipped_windows"] += 1
                else:
                    results["error_windows"] += 1

                # Save checkpoint
                self._save_checkpoint(window.end_time, result)

                # Adaptive delay based on data volume
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(
                    f"Error processing window {window.start_time} to {window.end_time}: {e}"
                )
                results["error_windows"] += 1
                continue

        # Generate final report
        results["processing_end"] = datetime.now().isoformat()
        results["total_processing_time"] = (
            datetime.fromisoformat(results["processing_end"])
            - datetime.fromisoformat(results["processing_start"])
        ).total_seconds()

        # Get profile statistics
        profile_stats = self.profile_manager.get_profile_statistics()
        results["profile_statistics"] = profile_stats

        logger.info(f"Analysis complete: {results}")
        return results

    async def analyze_realtime(self, duration_hours: int = 1) -> dict[str, Any]:
        """
        Stream and analyze data in real-time using session grouping.

        Args:
            duration_hours: Duration to analyze in real-time

        Returns:
            Real-time analysis results
        """
        logger.info(f"Starting real-time analysis for {duration_hours} hours")

        results = {
            "analysis_id": f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "duration_hours": duration_hours,
            "sessions_processed": 0,
            "profiles_updated": 0,
            "high_threat_sessions": 0,
            "processing_start": datetime.now().isoformat(),
        }

        try:
            async for session_chunk in self.mcp_client.stream_with_sessions(
                time_range_hours=duration_hours, chunk_size=500
            ):
                # Process each session chunk
                for _session_id, events in session_chunk.items():
                    results["sessions_processed"] += 1

                    # Quick local LLM analysis if available
                    quick_analysis = {}
                    if self.local_llm:
                        try:
                            quick_analysis = await self.local_llm.analyze_events(events)
                        except Exception as e:
                            logger.error(f"Error in local LLM analysis: {e}")

                    # Update profile if interesting
                    threat_score = quick_analysis.get("threat_score", 0.0)
                    if threat_score > 0.7:
                        results["high_threat_sessions"] += 1

                        entity_id = self._extract_entity_id(events)
                        if entity_id:
                            # Deep analysis with API if needed
                            if self.api_llm and quick_analysis.get(
                                "needs_deep_analysis"
                            ):
                                try:
                                    deep_analysis = await self.api_llm.analyze_events(
                                        events
                                    )
                                    quick_analysis.update(deep_analysis)
                                except Exception as e:
                                    logger.error(f"Error in API LLM analysis: {e}")

                            # Update profile
                            updated_profile = (
                                self.profile_manager.update_profile_incrementally(
                                    entity_id, events, quick_analysis
                                )
                            )

                            if updated_profile:
                                results["profiles_updated"] += 1

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in real-time analysis: {e}")
            results["error"] = str(e)

        results["processing_end"] = datetime.now().isoformat()
        logger.info(f"Real-time analysis complete: {results}")
        return results

    async def enrich_profiles(
        self, entity_ids: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Enrich profiles with external intelligence.

        Args:
            entity_ids: Specific entity IDs to enrich (None for all active profiles)

        Returns:
            Enrichment results
        """
        logger.info("Starting profile enrichment")

        # Get profiles to enrich
        if entity_ids:
            profiles = [self.profile_manager.get_profile(eid) for eid in entity_ids]
            profiles = [p for p in profiles if p is not None]
        else:
            profiles = self.profile_manager.get_active_profiles(hours=24)

        results = {
            "enrichment_id": f"enrichment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "profiles_enriched": 0,
            "enrichment_errors": 0,
            "processing_start": datetime.now().isoformat(),
        }

        for profile in profiles:
            try:
                enrichment_data = await self.enrichment_agent.enrich_profile(
                    profile.to_dict()
                )

                # Update profile with enrichment data
                profile.enrichment_data = enrichment_data
                self.profile_manager.save_profile(profile)

                results["profiles_enriched"] += 1

            except Exception as e:
                logger.error(f"Error enriching profile {profile.entity_id}: {e}")
                results["enrichment_errors"] += 1

        results["processing_end"] = datetime.now().isoformat()
        logger.info(f"Profile enrichment complete: {results}")
        return results

    def _filter_processed_windows(self, windows: list[tuple]) -> list[tuple]:
        """Filter out already processed windows."""
        filtered_windows = []
        for window_start, window_end in windows:
            if not self.time_processor.is_window_processed(window_start, window_end):
                filtered_windows.append((window_start, window_end))
        return filtered_windows

    def _save_checkpoint(self, window_end: datetime, result: dict[str, Any]) -> None:
        """Save analysis checkpoint."""
        checkpoint = {
            "window_end": window_end.isoformat(),
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
        self._save_analysis_state({"last_checkpoint": checkpoint})

    def _extract_entity_id(self, events: list[dict[str, Any]]) -> str | None:
        """Extract entity ID from events."""
        if not events:
            return None

        # Use the first event's source IP
        first_event = events[0]
        source_ip = first_event.get("source.ip")
        if source_ip:
            return f"ip_{source_ip}"

        return None

    def get_system_status(self) -> dict[str, Any]:
        """Get current system status."""
        return {
            "mcp_client_healthy": asyncio.run(self.mcp_client.health_check()),
            "profile_count": len(self.profile_manager.get_active_profiles()),
            "processed_windows": len(self.time_processor.processed_windows),
            "enrichment_cache_size": len(self.enrichment_agent.cache),
            "local_llm_available": self.local_llm is not None,
            "api_llm_available": self.api_llm is not None,
        }

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        try:
            await self.mcp_client.close()
            await self.enrichment_agent.close()
            self.profile_manager.close()
            logger.info("LongTailAnalyzer closed successfully")
        except Exception as e:
            logger.error(f"Error closing LongTailAnalyzer: {e}")
