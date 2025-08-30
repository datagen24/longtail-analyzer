"""
Configuration management for the long-tail analysis system.

This module provides configuration loading, validation, and management
capabilities with support for environment variables and YAML files.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MCPConfig:
    """MCP server configuration."""
    url: str = "http://localhost:3000"
    timeout: float = 30.0


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_path: str = "data/profiles.db"
    vector_db_path: str = "data/chroma"
    cache_ttl_hours: int = 24


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    window_hours: int = 6
    overlap_hours: int = 1
    max_entities_per_window: int = 50
    anomaly_threshold: float = 2.0


@dataclass
class LocalLLMConfig:
    """Local LLM configuration."""
    provider: str = "ollama"
    model: str = "mixtral:8x7b"
    temperature: float = 0.7
    max_tokens: int = 4096
    base_url: str = "http://localhost:11434"


@dataclass
class APILLMConfig:
    """API LLM configuration."""
    use_api: bool = False
    claude_api_key: Optional[str] = None
    claude_model: str = "claude-3-opus-20240229"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-large"


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    max_working_memory_mb: int = 1024
    profile_cache_size: int = 1000
    embedding_dimensions: int = 1536


@dataclass
class EnrichmentConfig:
    """Enrichment configuration."""
    enable_web_research: bool = True
    enable_threat_intelligence: bool = True
    cache_ttl_hours: int = 24
    rate_limit_requests_per_minute: int = 60


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/analyzer.log"


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    export_interval_seconds: int = 60
    prometheus_port: int = 9090


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 1


@dataclass
class Config:
    """Main configuration class."""
    mcp: MCPConfig = field(default_factory=MCPConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    local_llm: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    api_llm: APILLMConfig = field(default_factory=APILLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


class ConfigManager:
    """
    Configuration manager for loading and validating configuration.
    
    This class handles loading configuration from YAML files and environment
    variables, with proper validation and type conversion.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[Config] = None
        
    def load_config(self, config_path: Optional[str] = None) -> Config:
        """
        Load configuration from file and environment variables.
        
        Args:
            config_path: Path to configuration file (overrides instance path)
            
        Returns:
            Loaded configuration
        """
        file_path = config_path or self.config_path
        
        # Load from file if it exists
        file_config = {}
        if file_path and Path(file_path).exists():
            file_config = self._load_yaml_config(file_path)
        
        # Load from environment variables
        env_config = self._load_env_config()
        
        # Merge configurations (env overrides file)
        merged_config = self._merge_configs(file_config, env_config)
        
        # Convert to Config object
        self.config = self._dict_to_config(merged_config)
        
        logger.info("Configuration loaded successfully")
        return self.config
    
    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config or {}
            
        except Exception as e:
            logger.error(f"Error loading YAML config from {config_path}: {e}")
            return {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration dictionary from environment variables
        """
        env_config: Dict[str, Any] = {}
        
        # MCP configuration
        if os.getenv("MCP_URL"):
            env_config.setdefault("mcp", {})["url"] = os.getenv("MCP_URL")
        mcp_timeout = os.getenv("MCP_TIMEOUT")
        if mcp_timeout:
            env_config.setdefault("mcp", {})["timeout"] = float(mcp_timeout)
        
        # Database configuration
        if os.getenv("DB_PATH"):
            env_config.setdefault("database", {})["db_path"] = os.getenv("DB_PATH")
        if os.getenv("VECTOR_DB_PATH"):
            env_config.setdefault("database", {})["vector_db_path"] = os.getenv("VECTOR_DB_PATH")
        cache_ttl = os.getenv("CACHE_TTL_HOURS")
        if cache_ttl:
            env_config.setdefault("database", {})["cache_ttl_hours"] = int(cache_ttl)
        
        # Analysis configuration
        window_hours = os.getenv("WINDOW_HOURS")
        if window_hours:
            env_config.setdefault("analysis", {})["window_hours"] = int(window_hours)
        overlap_hours = os.getenv("OVERLAP_HOURS")
        if overlap_hours:
            env_config.setdefault("analysis", {})["overlap_hours"] = int(overlap_hours)
        max_entities = os.getenv("MAX_ENTITIES_PER_WINDOW")
        if max_entities:
            env_config.setdefault("analysis", {})["max_entities_per_window"] = int(max_entities)
        anomaly_threshold = os.getenv("ANOMALY_THRESHOLD")
        if anomaly_threshold:
            env_config.setdefault("analysis", {})["anomaly_threshold"] = float(anomaly_threshold)
        
        # Local LLM configuration
        if os.getenv("OLLAMA_MODEL"):
            env_config.setdefault("local_llm", {})["model"] = os.getenv("OLLAMA_MODEL")
        if os.getenv("OLLAMA_BASE_URL"):
            env_config.setdefault("local_llm", {})["base_url"] = os.getenv("OLLAMA_BASE_URL")
        
        # API LLM configuration
        if os.getenv("CLAUDE_API_KEY"):
            env_config.setdefault("api_llm", {})["claude_api_key"] = os.getenv("CLAUDE_API_KEY")
            env_config.setdefault("api_llm", {})["use_api"] = True
        if os.getenv("OPENAI_API_KEY"):
            env_config.setdefault("api_llm", {})["openai_api_key"] = os.getenv("OPENAI_API_KEY")
            env_config.setdefault("api_llm", {})["use_api"] = True
        
        # Enrichment configuration
        enable_web_research = os.getenv("ENABLE_WEB_RESEARCH")
        if enable_web_research:
            env_config.setdefault("enrichment", {})["enable_web_research"] = enable_web_research.lower() == "true"
        enable_threat_intel = os.getenv("ENABLE_THREAT_INTELLIGENCE")
        if enable_threat_intel:
            env_config.setdefault("enrichment", {})["enable_threat_intelligence"] = enable_threat_intel.lower() == "true"
        
        # Logging configuration
        if os.getenv("LOG_LEVEL"):
            env_config.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FILE"):
            env_config.setdefault("logging", {})["file"] = os.getenv("LOG_FILE")
        
        # Performance configuration
        max_concurrent = os.getenv("MAX_CONCURRENT_REQUESTS")
        if max_concurrent:
            env_config.setdefault("performance", {})["max_concurrent_requests"] = int(max_concurrent)
        request_timeout = os.getenv("REQUEST_TIMEOUT_SECONDS")
        if request_timeout:
            env_config.setdefault("performance", {})["request_timeout_seconds"] = int(request_timeout)
        
        if env_config:
            logger.info("Loaded configuration from environment variables")
        
        return env_config
    
    def _merge_configs(self, file_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge file and environment configurations.
        
        Args:
            file_config: Configuration from file
            env_config: Configuration from environment
            
        Returns:
            Merged configuration
        """
        merged = file_config.copy()
        
        # Deep merge environment config
        for key, value in env_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        
        return merged
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """
        Convert dictionary to Config object.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config object
        """
        # Create sub-config objects
        mcp_config = MCPConfig(**config_dict.get("mcp", {}))
        database_config = DatabaseConfig(**config_dict.get("database", {}))
        analysis_config = AnalysisConfig(**config_dict.get("analysis", {}))
        local_llm_config = LocalLLMConfig(**config_dict.get("local_llm", {}))
        api_llm_config = APILLMConfig(**config_dict.get("api_llm", {}))
        memory_config = MemoryConfig(**config_dict.get("memory", {}))
        enrichment_config = EnrichmentConfig(**config_dict.get("enrichment", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        metrics_config = MetricsConfig(**config_dict.get("metrics", {}))
        performance_config = PerformanceConfig(**config_dict.get("performance", {}))
        
        return Config(
            mcp=mcp_config,
            database=database_config,
            analysis=analysis_config,
            local_llm=local_llm_config,
            api_llm=api_llm_config,
            memory=memory_config,
            enrichment=enrichment_config,
            logging=logging_config,
            metrics=metrics_config,
            performance=performance_config
        )
    
    def validate_config(self, config: Config) -> bool:
        """
        Validate configuration values.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate MCP configuration
            if not config.mcp.url:
                logger.error("MCP URL is required")
                return False
            
            if config.mcp.timeout <= 0:
                logger.error("MCP timeout must be positive")
                return False
            
            # Validate analysis configuration
            if config.analysis.window_hours <= 0:
                logger.error("Window hours must be positive")
                return False
            
            if config.analysis.overlap_hours < 0:
                logger.error("Overlap hours must be non-negative")
                return False
            
            if config.analysis.overlap_hours >= config.analysis.window_hours:
                logger.error("Overlap hours must be less than window hours")
                return False
            
            if config.analysis.max_entities_per_window <= 0:
                logger.error("Max entities per window must be positive")
                return False
            
            if config.analysis.anomaly_threshold <= 0:
                logger.error("Anomaly threshold must be positive")
                return False
            
            # Validate LLM configuration
            if config.api_llm.use_api:
                if not config.api_llm.claude_api_key and not config.api_llm.openai_api_key:
                    logger.warning("API LLM enabled but no API keys provided")
            
            # Validate performance configuration
            if config.performance.max_concurrent_requests <= 0:
                logger.error("Max concurrent requests must be positive")
                return False
            
            if config.performance.request_timeout_seconds <= 0:
                logger.error("Request timeout must be positive")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_config(self) -> Optional[Config]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration or None if not loaded
        """
        return self.config
    
    def save_config(self, config: Config, output_path: str) -> bool:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            output_path: Path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert Config object to dictionary
            config_dict = self._config_to_dict(config)
            
            # Save to YAML file
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {output_path}: {e}")
            return False
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """
        Convert Config object to dictionary.
        
        Args:
            config: Config object to convert
            
        Returns:
            Configuration dictionary
        """
        return {
            "mcp": {
                "url": config.mcp.url,
                "timeout": config.mcp.timeout
            },
            "database": {
                "db_path": config.database.db_path,
                "vector_db_path": config.database.vector_db_path,
                "cache_ttl_hours": config.database.cache_ttl_hours
            },
            "analysis": {
                "window_hours": config.analysis.window_hours,
                "overlap_hours": config.analysis.overlap_hours,
                "max_entities_per_window": config.analysis.max_entities_per_window,
                "anomaly_threshold": config.analysis.anomaly_threshold
            },
            "local_llm": {
                "provider": config.local_llm.provider,
                "model": config.local_llm.model,
                "temperature": config.local_llm.temperature,
                "max_tokens": config.local_llm.max_tokens,
                "base_url": config.local_llm.base_url
            },
            "api_llm": {
                "use_api": config.api_llm.use_api,
                "claude_api_key": config.api_llm.claude_api_key,
                "claude_model": config.api_llm.claude_model,
                "openai_api_key": config.api_llm.openai_api_key,
                "openai_model": config.api_llm.openai_model,
                "openai_embedding_model": config.api_llm.openai_embedding_model
            },
            "memory": {
                "max_working_memory_mb": config.memory.max_working_memory_mb,
                "profile_cache_size": config.memory.profile_cache_size,
                "embedding_dimensions": config.memory.embedding_dimensions
            },
            "enrichment": {
                "enable_web_research": config.enrichment.enable_web_research,
                "enable_threat_intelligence": config.enrichment.enable_threat_intelligence,
                "cache_ttl_hours": config.enrichment.cache_ttl_hours,
                "rate_limit_requests_per_minute": config.enrichment.rate_limit_requests_per_minute
            },
            "logging": {
                "level": config.logging.level,
                "file": config.logging.file
            },
            "metrics": {
                "export_interval_seconds": config.metrics.export_interval_seconds,
                "prometheus_port": config.metrics.prometheus_port
            },
            "performance": {
                "max_concurrent_requests": config.performance.max_concurrent_requests,
                "request_timeout_seconds": config.performance.request_timeout_seconds,
                "retry_attempts": config.performance.retry_attempts,
                "retry_delay_seconds": config.performance.retry_delay_seconds
            }
        }
