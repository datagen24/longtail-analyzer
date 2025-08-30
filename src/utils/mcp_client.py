# src/utils/mcp_client.py

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class QuerySummary:
    """Summary statistics from a query"""

    total_events: int
    unique_sources: int
    unique_destinations: int
    time_range: dict[str, str]
    top_sources: list[dict]
    top_ports: list[dict]
    rare_events: list[dict]


class EnhancedMCPClient:
    """
    Enhanced wrapper for dshield-mcp with proper pagination and streaming support.
    This client properly utilizes the MCP server's capabilities that are being underused.
    """

    def __init__(self, mcp_url: str = "http://localhost:3000", timeout: int = 30):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self.session: httpx.AsyncClient | None = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.aclose()

    async def _call_mcp_function(self, function_name: str, params: dict) -> dict:
        """
        Internal method to call MCP server functions.
        """
        if not self.session:
            self.session = httpx.AsyncClient(timeout=self.timeout)

        try:
            response = await self.session.post(
                f"{self.mcp_url}/tools/{function_name}", json=params
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"MCP call failed for {function_name}: {e}")
            raise

    async def query_with_pagination(
        self,
        time_range_hours: int = 24,
        page_size: int = 500,
        fields: list[str] | None = None,
        filters: dict | None = None,
        indices: list[str] | None = None,
    ) -> AsyncGenerator[list[dict], None]:
        """
        Query DShield events using proper cursor-based pagination.
        Yields chunks of data as they're retrieved.

        Key insight: Use cursor-based pagination for large datasets to avoid timeouts.
        """
        cursor = None
        total_retrieved = 0

        while True:
            params = {
                "time_range_hours": time_range_hours,
                "page_size": min(page_size, 1000),  # MCP max is 1000
                "optimization": "auto",
                "include_summary": True,
                "fallback_strategy": "aggregate",
            }

            if cursor:
                params["cursor"] = cursor
            if fields:
                params["fields"] = fields
            if filters:
                params["filters"] = filters
            if indices:
                params["indices"] = indices

            logger.info(
                f"Querying MCP with cursor: {cursor[:20] if cursor else 'None'}..."
            )

            try:
                response = await self._call_mcp_function(
                    "dshield-mcp:query_dshield_events", params
                )

                if not response or "data" not in response:
                    logger.warning("No data in response, ending pagination")
                    break

                data = response.get("data", [])
                total_retrieved += len(data)

                logger.info(f"Retrieved {len(data)} events (total: {total_retrieved})")

                # Yield this chunk
                yield data

                # Check for next cursor
                cursor = response.get("next_cursor")
                if not cursor:
                    logger.info("No next cursor, pagination complete")
                    break

                # Respect rate limits
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error during pagination: {e}")
                break

    async def stream_with_sessions(
        self,
        time_range_hours: int = 24,
        chunk_size: int = 500,
        session_fields: list[str] | None = None,
        max_session_gap_minutes: int = 30,
        filters: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream events with session grouping for better context.
        This properly uses the stream_dshield_events_with_session_context function.
        """
        if session_fields is None:
            session_fields = ["source.ip", "destination.ip"]

        params = {
            "time_range_hours": time_range_hours,
            "chunk_size": min(chunk_size, 1000),
            "session_fields": session_fields,
            "max_session_gap_minutes": max_session_gap_minutes,
        }

        if filters:
            params["filters"] = filters

        logger.info(f"Starting session stream for {time_range_hours} hours")

        try:
            # Initial stream request
            response = await self._call_mcp_function(
                "dshield-mcp:stream_dshield_events_with_session_context", params
            )

            stream_id = response.get("stream_id")

            while True:
                # Continue streaming with stream_id
                if stream_id:
                    params["stream_id"] = stream_id

                chunk_response = await self._call_mcp_function(
                    "dshield-mcp:stream_dshield_events_with_session_context", params
                )

                sessions = chunk_response.get("sessions", {})

                if not sessions:
                    logger.info("No more sessions in stream")
                    break

                yield sessions

                # Check if stream is complete
                if chunk_response.get("stream_complete", False):
                    break

                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise

    async def get_aggregated_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        aggregations: list[str] | None = None,
    ) -> QuerySummary:
        """
        Get aggregated summary for a time window to reduce data size.
        This is crucial for identifying anomalies without loading all data.
        """
        if aggregations is None:
            aggregations = ["source.ip", "destination.port", "event.action"]

        # Calculate hours between start and end
        time_range_hours = int((end_time - start_time).total_seconds() / 3600)

        params = {
            "time_range_hours": time_range_hours,
            "page_size": 1,  # We only want the summary
            "include_summary": True,
            "optimization": "auto",
            "fields": aggregations,
        }

        # If we have exact timestamps, use them
        if start_time and end_time:
            params["time_range"] = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            }

        logger.info(f"Getting summary for {start_time} to {end_time}")

        response = await self._call_mcp_function(
            "dshield-mcp:query_dshield_events", params
        )

        summary_data = response.get("summary", {})

        # Extract top sources and identify anomalies
        top_sources = []
        if "aggregations" in summary_data:
            source_agg = summary_data["aggregations"].get("source.ip", {})
            for ip, count in source_agg.get("buckets", {}).items():
                top_sources.append({"ip": ip, "count": count})

        # Sort by count
        top_sources.sort(key=lambda x: x["count"], reverse=True)

        return QuerySummary(
            total_events=summary_data.get("total", 0),
            unique_sources=len(top_sources),
            unique_destinations=summary_data.get("unique_destinations", 0),
            time_range={"start": start_time.isoformat(), "end": end_time.isoformat()},
            top_sources=top_sources[:20],  # Top 20 sources
            top_ports=summary_data.get("top_ports", []),
            rare_events=summary_data.get("rare_events", []),
        )

    async def get_entity_details(
        self,
        entity_id: str,
        entity_type: str = "ip",
        time_range_hours: int = 24,
        fields: list[str] | None = None,
    ) -> list[dict]:
        """
        Get detailed events for a specific entity (IP, ASN, etc).
        This is used after anomaly detection to get detailed data.
        """
        if fields is None:
            fields = [
                "source.ip",
                "destination.ip",
                "destination.port",
                "event.action",
                "event.category",
                "@timestamp",
                "source.geo.country_iso_code",
                "destination.geo.country_iso_code",
            ]

        # Build filter based on entity type
        filters = {}
        if entity_type == "ip":
            filters["source.ip"] = entity_id
        elif entity_type == "asn":
            filters["source.as.number"] = entity_id
        elif entity_type == "port":
            filters["destination.port"] = entity_id

        all_events = []

        async for chunk in self.query_with_pagination(
            time_range_hours=time_range_hours,
            page_size=500,
            fields=fields,
            filters=filters,
        ):
            all_events.extend(chunk)

            # Limit to reasonable size for context window
            if len(all_events) >= 5000:
                logger.warning(f"Entity {entity_id} has >5000 events, truncating")
                break

        return all_events

    async def get_data_dictionary(self) -> dict:
        """
        Get the data dictionary from MCP server to understand field structures.
        """
        try:
            response = await self._call_mcp_function(
                "dshield-mcp:get_data_dictionary", {"format": "json"}
            )
            return response
        except Exception as e:
            logger.error(f"Failed to get data dictionary: {e}")
            return {}

    async def test_connection(self) -> bool:
        """
        Test connection to MCP server.
        """
        try:
            # Try to get data dictionary as a simple test
            result = await self.get_data_dictionary()
            return bool(result)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Utility class for managing MCP queries efficiently
class MCPQueryOptimizer:
    """
    Optimizes queries to the MCP server based on data patterns.
    """

    def __init__(self, mcp_client: EnhancedMCPClient):
        self.client = mcp_client
        self.query_cache: dict[str, Any] = {}

    async def identify_anomalies(
        self, summary: QuerySummary, threshold_multiplier: float = 2.0
    ) -> list[dict[str, Any]]:
        """
        Identify entities that warrant detailed analysis based on summary statistics.
        This is KEY to managing context windows efficiently.
        """
        anomalies = []

        # Calculate average events per source
        if summary.unique_sources > 0:
            avg_events_per_source = summary.total_events / summary.unique_sources
        else:
            avg_events_per_source = 0

        # Identify high-volume sources
        for source in summary.top_sources:
            if source["count"] > avg_events_per_source * threshold_multiplier:
                anomalies.append(
                    {
                        "entity_id": source["ip"],
                        "entity_type": "ip",
                        "reason": "high_volume",
                        "score": source["count"] / avg_events_per_source,
                    }
                )

        # Add rare events as anomalies
        for event in summary.rare_events:
            anomalies.append(
                {
                    "entity_id": event.get("source_ip", "unknown"),
                    "entity_type": "ip",
                    "reason": "rare_event",
                    "score": 1.5,
                }
            )

        # Sort by score and limit
        anomalies.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"Identified {len(anomalies)} anomalies from {summary.total_events} events"
        )

        return anomalies[:50]  # Limit to top 50 to manage context

    async def fetch_anomaly_details(
        self, anomalies: list[dict], time_range_hours: int = 24
    ) -> dict[str, list[dict]]:
        """
        Fetch detailed data for identified anomalies.
        Returns a dictionary mapping entity_id to their events.
        """
        entity_events = {}

        for anomaly in anomalies:
            entity_id = anomaly["entity_id"]
            entity_type = anomaly["entity_type"]

            logger.info(f"Fetching details for {entity_type}: {entity_id}")

            events = await self.client.get_entity_details(
                entity_id=entity_id,
                entity_type=entity_type,
                time_range_hours=time_range_hours,
            )

            if events:
                entity_events[entity_id] = events

            # Rate limit between entities
            await asyncio.sleep(0.05)

        return entity_events
