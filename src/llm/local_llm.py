"""
Local LLM integration using Ollama.

This module provides integration with Ollama for local language model
inference, enabling analysis without external API dependencies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import httpx
import json

logger = logging.getLogger(__name__)


class OllamaLLM:
    """
    Local LLM integration using Ollama.
    
    This class provides a simple interface to Ollama for local language
    model inference, enabling analysis without external API dependencies.
    """
    
    def __init__(
        self,
        model: str = "mixtral:8x7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        Initialize the Ollama LLM client.
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = httpx.AsyncClient(timeout=120.0)
        
        logger.info(f"OllamaLLM initialized with model: {model}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP session."""
        await self.session.aclose()
    
    async def analyze_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze security events using the local LLM.
        
        Args:
            events: List of security events to analyze
            
        Returns:
            Analysis results
        """
        if not events:
            return {"threat_score": 0.0, "analysis": "No events to analyze"}
        
        # Prepare prompt for analysis
        prompt = self._create_analysis_prompt(events)
        
        try:
            # Call Ollama API
            response = await self._call_ollama(prompt)
            
            if response:
                # Parse the response
                analysis = self._parse_analysis_response(response)
                return analysis
            else:
                return {"threat_score": 0.0, "analysis": "Failed to get LLM response"}
                
        except Exception as e:
            logger.error(f"Error analyzing events with Ollama: {e}")
            return {"threat_score": 0.0, "analysis": f"Analysis error: {e}"}
    
    async def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        Call the Ollama API with a prompt.
        
        Args:
            prompt: Prompt to send to the model
            
        Returns:
            Model response or None if error
        """
        try:
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = await self.session.post(
                f"{self.base_url}/api/generate",
                json=request_data
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response")
            
        except httpx.TimeoutException:
            logger.error("Timeout calling Ollama API")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling Ollama API: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama API: {e}")
            return None
    
    def _create_analysis_prompt(self, events: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for analyzing security events.
        
        Args:
            events: Events to analyze
            
        Returns:
            Formatted prompt
        """
        # Limit events to prevent context overflow
        max_events = 50
        if len(events) > max_events:
            events = events[:max_events]
        
        # Format events for analysis
        event_summary = self._summarize_events(events)
        
        prompt = f"""
You are a cybersecurity analyst. Analyze the following security events and provide a threat assessment.

Events Summary:
{event_summary}

Please provide your analysis in the following JSON format:
{{
    "threat_score": 0.0-1.0,
    "threat_level": "low|medium|high|critical",
    "attack_patterns": ["pattern1", "pattern2"],
    "ttps": ["T1234", "T5678"],
    "confidence": 0.0-1.0,
    "analysis": "Detailed analysis text",
    "recommendations": ["recommendation1", "recommendation2"],
    "needs_deep_analysis": true/false
}}

Focus on:
1. Identifying attack patterns and techniques
2. Assessing the threat level
3. Mapping to MITRE ATT&CK TTPs
4. Providing actionable recommendations
"""
        
        return prompt
    
    def _summarize_events(self, events: List[Dict[str, Any]]) -> str:
        """
        Create a summary of events for analysis.
        
        Args:
            events: Events to summarize
            
        Returns:
            Event summary string
        """
        if not events:
            return "No events"
        
        # Extract key information
        source_ips = set()
        dest_ips = set()
        dest_ports = set()
        actions = set()
        timestamps = []
        
        for event in events:
            if "source.ip" in event:
                source_ips.add(event["source.ip"])
            if "destination.ip" in event:
                dest_ips.add(event["destination.ip"])
            if "destination.port" in event:
                dest_ports.add(str(event["destination.port"]))
            if "event.action" in event:
                actions.add(event["event.action"])
            if "@timestamp" in event:
                timestamps.append(event["@timestamp"])
        
        summary = f"""
Total Events: {len(events)}
Source IPs: {len(source_ips)} ({', '.join(list(source_ips)[:5])}{'...' if len(source_ips) > 5 else ''})
Destination IPs: {len(dest_ips)} ({', '.join(list(dest_ips)[:5])}{'...' if len(dest_ips) > 5 else ''})
Destination Ports: {len(dest_ports)} ({', '.join(list(dest_ports)[:10])}{'...' if len(dest_ports) > 10 else ''})
Actions: {', '.join(list(actions)[:10])}{'...' if len(actions) > 10 else ''}
Time Range: {min(timestamps) if timestamps else 'unknown'} to {max(timestamps) if timestamps else 'unknown'}
"""
        
        return summary
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured data.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed analysis data
        """
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                # Validate and set defaults
                analysis.setdefault("threat_score", 0.0)
                analysis.setdefault("threat_level", "low")
                analysis.setdefault("attack_patterns", [])
                analysis.setdefault("ttps", [])
                analysis.setdefault("confidence", 0.5)
                analysis.setdefault("analysis", response)
                analysis.setdefault("recommendations", [])
                analysis.setdefault("needs_deep_analysis", False)
                
                return analysis
            else:
                # Fallback if no JSON found
                return {
                    "threat_score": 0.5,
                    "threat_level": "medium",
                    "attack_patterns": [],
                    "ttps": [],
                    "confidence": 0.3,
                    "analysis": response,
                    "recommendations": [],
                    "needs_deep_analysis": True
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response as JSON: {e}")
            return {
                "threat_score": 0.5,
                "threat_level": "medium",
                "attack_patterns": [],
                "ttps": [],
                "confidence": 0.3,
                "analysis": response,
                "recommendations": [],
                "needs_deep_analysis": True
            }
    
    async def health_check(self) -> bool:
        """
        Check if Ollama is healthy and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information
        """
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
