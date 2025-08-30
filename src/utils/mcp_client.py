"""
Enhanced MCP client wrapper for dshield-mcp with proper pagination and streaming support.

This module provides a robust interface to the dshield-mcp server with:
- Cursor-based pagination for large datasets
- Streaming with session context
- Time window summaries
- Error handling and retry logic
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)


class EnhancedMCPClient:
    """
    Wrapper for dshield-mcp with proper pagination and streaming support.
    
    This client properly utilizes the dshield-mcp server's capabilities including:
    - Cursor-based pagination for handling large datasets
    - Streaming with session context for real-time analysis
    - Query optimization and field selection
    - Rate limiting and timeout handling
    """
    
    def __init__(self, mcp_url: str = "http://localhost:3000", timeout: float = 30.0):
        """
        Initialize the MCP client.
        
        Args:
            mcp_url: URL of the dshield-mcp server
            timeout: Request timeout in seconds
        """
        self.mcp_url = mcp_url.rstrip('/')
        self.timeout = timeout
        self.session = httpx.AsyncClient(timeout=timeout)
        self._rate_limit_delay = 0.1  # 100ms between requests
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def close(self):
        """Close the HTTP session."""
        await self.session.aclose()
        
    async def _call_mcp_function(
        self, 
        function_name: str, 
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Call an MCP function with proper error handling.
        
        Args:
            function_name: Name of the MCP function to call
            params: Parameters for the function
            
        Returns:
            Response data or None if error
        """
        try:
            # Construct the MCP JSON-RPC request
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": function_name,
                "params": params
            }
            
            logger.debug(f"Calling MCP function {function_name} with params: {params}")
            
            response = await self.session.post(
                f"{self.mcp_url}/mcp",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "error" in result:
                logger.error(f"MCP error: {result['error']}")
                return None
                
            return result.get("result")
            
        except httpx.TimeoutException:
            logger.error(f"Timeout calling MCP function {function_name}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling MCP function {function_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling MCP function {function_name}: {e}")
            return None
    
    async def query_with_pagination(
        self,
        time_range_hours: int = 24,
        page_size: int = 500,
        fields: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc"
    ) -> List[Dict[str, Any]]:
        """
        Query DShield events using proper cursor-based pagination.
        
        This is the key method for handling large datasets efficiently.
        It uses the dshield-mcp server's cursor-based pagination to avoid
        memory issues with large result sets.
        
        Args:
            time_range_hours: Time window for analysis in hours
            page_size: Number of records per page (max 1000)
            fields: Specific fields to return (reduces data transfer)
            filters: Additional filters to apply
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            List of all records across all pages
        """
        all_results = []
        cursor = None
        page_count = 0
        
        logger.info(f"Starting paginated query for {time_range_hours} hours, page_size={page_size}")
        
        while True:
            params = {
                "time_range_hours": time_range_hours,
                "page_size": min(page_size, 1000),  # Enforce MCP server limit
                "optimization": "auto",
                "include_summary": True
            }
            
            if cursor:
                params["cursor"] = cursor
            if fields:
                params["fields"] = fields
            if filters:
                params["filters"] = filters
            if sort_by:
                params["sort_by"] = sort_by
                params["sort_order"] = sort_order
                
            # Call query_dshield_events with pagination
            response = await self._call_mcp_function("query_dshield_events", params)
            
            if not response or "data" not in response:
                logger.warning("No data returned from MCP server")
                break
                
            page_data = response["data"]
            all_results.extend(page_data)
            page_count += 1
            
            logger.debug(f"Retrieved page {page_count} with {len(page_data)} records")
            
            # Check for next cursor
            cursor = response.get("next_cursor")
            if not cursor:
                logger.info(f"Pagination complete. Retrieved {len(all_results)} total records in {page_count} pages")
                break
                
            # Respect rate limits
            await asyncio.sleep(self._rate_limit_delay)
            
        return all_results
    
    async def stream_with_sessions(
        self,
        time_range_hours: int = 24,
        chunk_size: int = 500,
        session_fields: List[str] = ["source.ip", "destination.ip"],
        max_session_gap_minutes: int = 30
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream events with session grouping for better context.
        
        This method uses the dshield-mcp server's session grouping capabilities
        to provide better context for analysis by grouping related events.
        
        Args:
            time_range_hours: Time window for streaming
            chunk_size: Events per chunk (max 1000)
            session_fields: Fields to use for session correlation
            max_session_gap_minutes: Maximum gap between events in same session
            
        Yields:
            Session chunks with grouped events
        """
        params = {
            "time_range_hours": time_range_hours,
            "chunk_size": min(chunk_size, 1000),  # Enforce MCP server limit
            "session_fields": session_fields,
            "max_session_gap_minutes": max_session_gap_minutes
        }
        
        logger.info(f"Starting session streaming for {time_range_hours} hours")
        
        # Call stream_dshield_events_with_session_context
        response = await self._call_mcp_function(
            "stream_dshield_events_with_session_context", 
            params
        )
        
        if response and "stream_data" in response:
            for chunk in response["stream_data"]:
                yield chunk
                await asyncio.sleep(self._rate_limit_delay)
    
    async def get_time_window_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        aggregations: List[str] = ["source.ip", "destination.port", "event.action"]
    ) -> Dict[str, Any]:
        """
        Get aggregated summary for a time window to reduce data size.
        
        This method is crucial for managing context windows by providing
        statistical summaries before fetching detailed data.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            aggregations: Fields to aggregate on
            
        Returns:
            Summary statistics and aggregations
        """
        time_range_hours = int((end_time - start_time).total_seconds() / 3600)
        
        params = {
            "time_range_hours": time_range_hours,
            "aggregations": aggregations,
            "include_summary": True,
            "optimization": "summary_only"
        }
        
        logger.debug(f"Getting time window summary for {time_range_hours} hours")
        
        response = await self._call_mcp_function("query_dshield_events", params)
        
        if response and "summary" in response:
            return response["summary"]
        
        return {}
    
    async def get_data_dictionary(self) -> Dict[str, Any]:
        """
        Get the data dictionary to understand available fields.
        
        Returns:
            Data dictionary with field descriptions and types
        """
        response = await self._call_mcp_function("get_data_dictionary", {})
        
        if response:
            return response
            
        return {}
    
    async def health_check(self) -> bool:
        """
        Check if the MCP server is healthy and responsive.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Try a simple query with minimal parameters
            response = await self._call_mcp_function(
                "query_dshield_events", 
                {"time_range_hours": 1, "page_size": 1}
            )
            return response is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
