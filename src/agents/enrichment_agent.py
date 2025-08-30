"""
Intelligence enrichment agent for external threat intelligence gathering.

This module provides capabilities for enriching attacker profiles with
external intelligence including web research, OSINT correlation, and
threat actor attribution.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class EnrichmentAgent:
    """
    Enriches attacker profiles with external threat intelligence.

    This agent performs web research, OSINT correlation, and threat
    intelligence gathering to enhance attacker profiles with additional
    context and attribution information.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the enrichment agent.

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.session = httpx.AsyncClient(timeout=30.0)
        self.cache: dict[str, Any] = {}  # Simple in-memory cache
        self.cache_ttl = timedelta(hours=24)

        logger.info("EnrichmentAgent initialized")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the HTTP session."""
        await self.session.aclose()

    async def enrich_profile(self, profile_data: dict[str, Any]) -> dict[str, Any]:
        """
        Enrich a profile with external intelligence.

        Args:
            profile_data: Profile data to enrich

        Returns:
            Enriched profile data
        """
        entity_id = profile_data.get("entity_id")
        entity_type = profile_data.get("entity_type")

        if not entity_id or not entity_type:
            logger.warning("Invalid profile data for enrichment")
            return profile_data

        logger.info(f"Enriching profile: {entity_id}")

        enrichment_results = {
            "entity_id": entity_id,
            "enrichment_timestamp": datetime.now().isoformat(),
            "sources": [],
            "intelligence": {},
        }

        try:
            # Extract entity identifier
            if entity_type == "ip":
                ip_address = entity_id.replace("ip_", "")
                await self._enrich_ip_address(ip_address, enrichment_results)
            elif entity_type == "asn":
                asn = entity_id.replace("asn_", "")
                await self._enrich_asn(asn, enrichment_results)

            # Perform web research
            await self._perform_web_research(entity_id, enrichment_results)

            # Correlate with threat intelligence
            await self._correlate_threat_intelligence(entity_id, enrichment_results)

            logger.info(f"Enrichment complete for {entity_id}")

        except Exception as e:
            logger.error(f"Error enriching profile {entity_id}: {e}")
            enrichment_results["error"] = str(e)

        return enrichment_results

    async def _enrich_ip_address(
        self, ip_address: str, enrichment_results: dict[str, Any]
    ) -> None:
        """
        Enrich an IP address with geolocation and reputation data.

        Args:
            ip_address: IP address to enrich
            enrichment_results: Results dictionary to update
        """
        # Check cache first
        cache_key = f"ip_{ip_address}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            enrichment_results["intelligence"]["ip_data"] = cached_data
            enrichment_results["sources"].append("cache")
            return

        try:
            # Get geolocation data
            geo_data = await self._get_geolocation(ip_address)
            if geo_data:
                enrichment_results["intelligence"]["geolocation"] = geo_data
                enrichment_results["sources"].append("geolocation")

            # Get reputation data
            reputation_data = await self._get_ip_reputation(ip_address)
            if reputation_data:
                enrichment_results["intelligence"]["reputation"] = reputation_data
                enrichment_results["sources"].append("reputation")

            # Get ASN information
            asn_data = await self._get_asn_info(ip_address)
            if asn_data:
                enrichment_results["intelligence"]["asn"] = asn_data
                enrichment_results["sources"].append("asn")

            # Cache the results
            ip_data = enrichment_results["intelligence"].get("ip_data", {})
            self._cache_data(cache_key, ip_data)

        except Exception as e:
            logger.error(f"Error enriching IP {ip_address}: {e}")

    async def _enrich_asn(self, asn: str, enrichment_results: dict[str, Any]) -> None:
        """
        Enrich an ASN with organization and reputation data.

        Args:
            asn: ASN to enrich
            enrichment_results: Results dictionary to update
        """
        # Check cache first
        cache_key = f"asn_{asn}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            enrichment_results["intelligence"]["asn_data"] = cached_data
            enrichment_results["sources"].append("cache")
            return

        try:
            # Get ASN organization information
            org_data = await self._get_asn_organization(asn)
            if org_data:
                enrichment_results["intelligence"]["organization"] = org_data
                enrichment_results["sources"].append("asn_org")

            # Get ASN reputation
            asn_reputation = await self._get_asn_reputation(asn)
            if asn_reputation:
                enrichment_results["intelligence"]["asn_reputation"] = asn_reputation
                enrichment_results["sources"].append("asn_reputation")

            # Cache the results
            asn_data = enrichment_results["intelligence"].get("asn_data", {})
            self._cache_data(cache_key, asn_data)

        except Exception as e:
            logger.error(f"Error enriching ASN {asn}: {e}")

    async def _perform_web_research(
        self, entity_id: str, enrichment_results: dict[str, Any]
    ) -> None:
        """
        Perform web research on the entity.

        Args:
            entity_id: Entity to research
            enrichment_results: Results dictionary to update
        """
        try:
            # Search for threat intelligence reports
            threat_reports = await self._search_threat_reports(entity_id)
            if threat_reports:
                enrichment_results["intelligence"]["threat_reports"] = threat_reports
                enrichment_results["sources"].append("threat_reports")

            # Search for malware associations
            malware_info = await self._search_malware_associations(entity_id)
            if malware_info:
                enrichment_results["intelligence"]["malware"] = malware_info
                enrichment_results["sources"].append("malware")

        except Exception as e:
            logger.error(f"Error performing web research for {entity_id}: {e}")

    async def _correlate_threat_intelligence(
        self, entity_id: str, enrichment_results: dict[str, Any]
    ) -> None:
        """
        Correlate with known threat intelligence sources.

        Args:
            entity_id: Entity to correlate
            enrichment_results: Results dictionary to update
        """
        try:
            # Check against known threat actor IOCs
            threat_actors = await self._check_threat_actors(entity_id)
            if threat_actors:
                enrichment_results["intelligence"]["threat_actors"] = threat_actors
                enrichment_results["sources"].append("threat_actors")

            # Check against known malware families
            malware_families = await self._check_malware_families(entity_id)
            if malware_families:
                enrichment_results["intelligence"][
                    "malware_families"
                ] = malware_families
                enrichment_results["sources"].append("malware_families")

        except Exception as e:
            logger.error(f"Error correlating threat intelligence for {entity_id}: {e}")

    async def _get_geolocation(self, ip_address: str) -> dict[str, Any] | None:
        """Get geolocation data for an IP address."""
        try:
            # Use a free geolocation service (in production, use a commercial service)
            response = await self.session.get(f"http://ip-api.com/json/{ip_address}")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return {
                        "country": data.get("country"),
                        "region": data.get("regionName"),
                        "city": data.get("city"),
                        "latitude": data.get("lat"),
                        "longitude": data.get("lon"),
                        "isp": data.get("isp"),
                        "organization": data.get("org"),
                    }
        except Exception as e:
            logger.error(f"Error getting geolocation for {ip_address}: {e}")
        return None

    async def _get_ip_reputation(self, ip_address: str) -> dict[str, Any] | None:
        """Get reputation data for an IP address."""
        # This would integrate with commercial threat intelligence services
        # For now, return a placeholder
        return {
            "reputation_score": 0.5,
            "threat_level": "unknown",
            "last_seen": datetime.now().isoformat(),
        }

    async def _get_asn_info(self, ip_address: str) -> dict[str, Any] | None:
        """Get ASN information for an IP address."""
        try:
            # Use a free ASN lookup service
            response = await self.session.get(f"http://ip-api.com/json/{ip_address}")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return {
                        "asn": data.get("as"),
                        "organization": data.get("org"),
                        "isp": data.get("isp"),
                    }
        except Exception as e:
            logger.error(f"Error getting ASN info for {ip_address}: {e}")
        return None

    async def _get_asn_organization(self, asn: str) -> dict[str, Any] | None:
        """Get organization information for an ASN."""
        # This would integrate with ASN databases
        return {
            "asn": asn,
            "organization": "Unknown Organization",
            "country": "Unknown",
        }

    async def _get_asn_reputation(self, asn: str) -> dict[str, Any] | None:
        """Get reputation data for an ASN."""
        return {
            "reputation_score": 0.5,
            "threat_level": "unknown",
            "last_updated": datetime.now().isoformat(),
        }

    async def _search_threat_reports(self, entity_id: str) -> list[dict[str, Any]]:
        """Search for threat intelligence reports mentioning the entity."""
        # This would integrate with threat intelligence platforms
        return []

    async def _search_malware_associations(
        self, entity_id: str
    ) -> list[dict[str, Any]]:
        """Search for malware associations with the entity."""
        # This would integrate with malware databases
        return []

    async def _check_threat_actors(self, entity_id: str) -> list[dict[str, Any]]:
        """Check if entity is associated with known threat actors."""
        # This would integrate with threat actor databases
        return []

    async def _check_malware_families(self, entity_id: str) -> list[dict[str, Any]]:
        """Check if entity is associated with known malware families."""
        # This would integrate with malware family databases
        return []

    def _get_cached_data(self, cache_key: str) -> dict[str, Any] | None:
        """Get data from cache if not expired."""
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if datetime.now() - cached_item["timestamp"] < self.cache_ttl:
                return cached_item["data"]
            else:
                # Remove expired item
                del self.cache[cache_key]
        return None

    def _cache_data(self, cache_key: str, data: dict[str, Any]) -> None:
        """Cache data with timestamp."""
        self.cache[cache_key] = {"data": data, "timestamp": datetime.now()}

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600,
        }

    def clear_cache(self) -> None:
        """Clear the enrichment cache."""
        self.cache.clear()
        logger.info("Enrichment cache cleared")
