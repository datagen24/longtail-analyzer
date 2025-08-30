"""
API-based LLM integration for Claude and OpenAI.

This module provides integration with external LLM APIs including
Claude and OpenAI for advanced analysis capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import httpx
import json

logger = logging.getLogger(__name__)


class ClaudeAPI:
    """
    Claude API integration for advanced analysis.
    
    This class provides integration with Anthropic's Claude API for
    complex analysis tasks that require advanced reasoning capabilities.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 4096
    ):
        """
        Initialize the Claude API client.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
            base_url: API base URL
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.max_tokens = max_tokens
        self.session = httpx.AsyncClient(timeout=120.0)
        
        logger.info(f"ClaudeAPI initialized with model: {model}")
    
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
        Analyze security events using Claude.
        
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
            # Call Claude API
            response = await self._call_claude(prompt)
            
            if response:
                # Parse the response
                analysis = self._parse_analysis_response(response)
                return analysis
            else:
                return {"threat_score": 0.0, "analysis": "Failed to get Claude response"}
                
        except Exception as e:
            logger.error(f"Error analyzing events with Claude: {e}")
            return {"threat_score": 0.0, "analysis": f"Analysis error: {e}"}
    
    async def _call_claude(self, prompt: str) -> Optional[str]:
        """
        Call the Claude API with a prompt.
        
        Args:
            prompt: Prompt to send to Claude
            
        Returns:
            Claude response or None if error
        """
        try:
            request_data = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            response = await self.session.post(
                f"{self.base_url}/v1/messages",
                json=request_data,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("content", [{}])[0].get("text")
            
        except httpx.TimeoutException:
            logger.error("Timeout calling Claude API")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling Claude API: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Claude API: {e}")
            return None
    
    def _create_analysis_prompt(self, events: List[Dict[str, Any]]) -> str:
        """
        Create a detailed prompt for Claude analysis.
        
        Args:
            events: Events to analyze
            
        Returns:
            Formatted prompt
        """
        # Limit events to prevent context overflow
        max_events = 100
        if len(events) > max_events:
            events = events[:max_events]
        
        # Format events for analysis
        event_summary = self._summarize_events(events)
        
        prompt = f"""
You are an expert cybersecurity analyst with deep knowledge of attack patterns, threat intelligence, and MITRE ATT&CK framework. Analyze the following security events and provide a comprehensive threat assessment.

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
    "needs_deep_analysis": true/false,
    "threat_actor_attribution": "possible threat actor or group",
    "campaign_indicators": ["indicator1", "indicator2"],
    "infrastructure_analysis": "analysis of attacker infrastructure"
}}

Focus on:
1. Identifying sophisticated attack patterns and techniques
2. Assessing the threat level and potential impact
3. Mapping to MITRE ATT&CK TTPs with high precision
4. Providing actionable recommendations for defense
5. Threat actor attribution if possible
6. Campaign correlation and infrastructure analysis
"""
        
        return prompt
    
    def _summarize_events(self, events: List[Dict[str, Any]]) -> str:
        """
        Create a detailed summary of events for analysis.
        
        Args:
            events: Events to summarize
            
        Returns:
            Event summary string
        """
        if not events:
            return "No events"
        
        # Extract comprehensive information
        source_ips = set()
        dest_ips = set()
        dest_ports = set()
        actions = set()
        timestamps = []
        protocols = set()
        user_agents = set()
        
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
            if "network.protocol" in event:
                protocols.add(event["network.protocol"])
            if "user_agent.original" in event:
                user_agents.add(event["user_agent.original"])
        
        summary = f"""
Total Events: {len(events)}
Source IPs: {len(source_ips)} ({', '.join(list(source_ips)[:10])}{'...' if len(source_ips) > 10 else ''})
Destination IPs: {len(dest_ips)} ({', '.join(list(dest_ips)[:10])}{'...' if len(dest_ips) > 10 else ''})
Destination Ports: {len(dest_ports)} ({', '.join(list(dest_ports)[:20])}{'...' if len(dest_ports) > 20 else ''})
Actions: {', '.join(list(actions)[:20])}{'...' if len(actions) > 20 else ''}
Protocols: {', '.join(list(protocols)[:10])}{'...' if len(protocols) > 10 else ''}
User Agents: {len(user_agents)} unique
Time Range: {min(timestamps) if timestamps else 'unknown'} to {max(timestamps) if timestamps else 'unknown'}
"""
        
        return summary
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the Claude response into structured data.
        
        Args:
            response: Raw Claude response
            
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
                analysis.setdefault("threat_actor_attribution", "")
                analysis.setdefault("campaign_indicators", [])
                analysis.setdefault("infrastructure_analysis", "")
                
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
                    "needs_deep_analysis": True,
                    "threat_actor_attribution": "",
                    "campaign_indicators": [],
                    "infrastructure_analysis": ""
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Claude response as JSON: {e}")
            return {
                "threat_score": 0.5,
                "threat_level": "medium",
                "attack_patterns": [],
                "ttps": [],
                "confidence": 0.3,
                "analysis": response,
                "recommendations": [],
                "needs_deep_analysis": True,
                "threat_actor_attribution": "",
                "campaign_indicators": [],
                "infrastructure_analysis": ""
            }


class OpenAIAPI:
    """
    OpenAI API integration for analysis and embeddings.
    
    This class provides integration with OpenAI's API for analysis
    and embedding generation capabilities.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        embedding_model: str = "text-embedding-3-large",
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 4096
    ):
        """
        Initialize the OpenAI API client.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
            embedding_model: Embedding model to use
            base_url: API base URL
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip('/')
        self.max_tokens = max_tokens
        self.session = httpx.AsyncClient(timeout=120.0)
        
        logger.info(f"OpenAIAPI initialized with model: {model}")
    
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
        Analyze security events using OpenAI GPT.
        
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
            # Call OpenAI API
            response = await self._call_openai(prompt)
            
            if response:
                # Parse the response
                analysis = self._parse_analysis_response(response)
                return analysis
            else:
                return {"threat_score": 0.0, "analysis": "Failed to get OpenAI response"}
                
        except Exception as e:
            logger.error(f"Error analyzing events with OpenAI: {e}")
            return {"threat_score": 0.0, "analysis": f"Analysis error: {e}"}
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using OpenAI.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            request_data = {
                "model": self.embedding_model,
                "input": texts
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.session.post(
                f"{self.base_url}/embeddings",
                json=request_data,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    async def _call_openai(self, prompt: str) -> Optional[str]:
        """
        Call the OpenAI API with a prompt.
        
        Args:
            prompt: Prompt to send to OpenAI
            
        Returns:
            OpenAI response or None if error
        """
        try:
            request_data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.7
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.session.post(
                f"{self.base_url}/chat/completions",
                json=request_data,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except httpx.TimeoutException:
            logger.error("Timeout calling OpenAI API")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling OpenAI API: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {e}")
            return None
    
    def _create_analysis_prompt(self, events: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for OpenAI analysis.
        
        Args:
            events: Events to analyze
            
        Returns:
            Formatted prompt
        """
        # Limit events to prevent context overflow
        max_events = 100
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
        Parse the OpenAI response into structured data.
        
        Args:
            response: Raw OpenAI response
            
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
            logger.error(f"Error parsing OpenAI response as JSON: {e}")
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
