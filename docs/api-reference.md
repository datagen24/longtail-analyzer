# Long-Tail Analyzer API Reference

This document provides comprehensive API documentation for the Long-Tail Analyzer project, a sophisticated security analysis system for detecting long-tail attack patterns and anomalies.

## Table of Contents

- [Core Orchestrator](#core-orchestrator)
- [Data Processing](#data-processing)
- [Profile Management](#profile-management)
- [Pattern Analysis](#pattern-analysis)
- [LLM Integration](#llm-integration)
- [Memory and Storage](#memory-and-storage)
- [Configuration Management](#configuration-management)
- [Utility Classes](#utility-classes)
- [Data Models](#data-models)

## Core Orchestrator

### `LongTailAnalyzer`

Main orchestrator class that coordinates all components of the analysis system.

**Location**: `src/orchestrator.py`

#### Constructor

```python
def __init__(self, config_path: str = "configs/default.yaml")
```

Initialize the long-tail analyzer with configuration.

**Parameters**:
- `config_path` (str): Path to configuration file

#### Methods

##### `analyze_period(start_date, end_date, resume=True)`

Analyze a time period with automatic chunking and state management.

**Parameters**:
- `start_date` (datetime): Start of analysis period
- `end_date` (datetime): End of analysis period
- `resume` (bool): Whether to resume from last checkpoint

**Returns**:
- `Dict[str, Any]`: Analysis report with statistics

##### `analyze_realtime(duration_hours=1)`

Stream and analyze data in real-time using session grouping.

**Parameters**:
- `duration_hours` (int): Duration to analyze in real-time

**Returns**:
- `Dict[str, Any]`: Real-time analysis results

##### `enrich_profiles(entity_ids=None)`

Enrich profiles with external intelligence.

**Parameters**:
- `entity_ids` (Optional[List[str]]): Specific entity IDs to enrich

**Returns**:
- `Dict[str, Any]`: Enrichment results

##### `get_system_status()`

Get current system status.

**Returns**:
- `Dict[str, Any]`: System status information

##### `close()`

Close all connections and cleanup resources.

## Data Processing

### `TimeWindowProcessor`

Processes data in overlapping time windows to catch cross-boundary patterns.

**Location**: `src/agents/data_processor.py`

#### Constructor

```python
def __init__(
    self,
    mcp_client: EnhancedMCPClient,
    profile_manager: ProfileManager,
    window_hours: int = 6,
    overlap_hours: int = 1,
    max_entities_per_window: int = 50
)
```

#### Methods

##### `generate_time_windows(start_date, end_date)`

Generate overlapping time windows for analysis.

**Parameters**:
- `start_date` (datetime): Start date for windows
- `end_date` (datetime): End date for windows

**Returns**:
- `List[ProcessingWindow]`: List of processing windows

##### `process_window(window, pattern_analyzer=None)`

Process a single time window with intelligent data retrieval.

**Parameters**:
- `window` (ProcessingWindow): Window to process
- `pattern_analyzer` (Optional): Pattern analyzer instance

**Returns**:
- `Dict`: Processing results

##### `process_time_range(start_date, end_date, pattern_analyzer=None, resume=True)`

Process an entire time range with automatic windowing.

**Parameters**:
- `start_date` (datetime): Start date
- `end_date` (datetime): End date
- `pattern_analyzer` (Optional): Pattern analyzer
- `resume` (bool): Resume from checkpoint

**Returns**:
- `Dict`: Processing statistics

### `ProcessingWindow`

Represents a time window for processing.

**Location**: `src/agents/data_processor.py`

#### Attributes

- `window_id` (str): Unique window identifier
- `start_time` (datetime): Window start time
- `end_time` (datetime): Window end time
- `overlap_start` (Optional[datetime]): Overlap start time
- `overlap_end` (Optional[datetime]): Overlap end time
- `processed` (bool): Whether window has been processed
- `summary` (Optional[QuerySummary]): Window summary
- `entities_found` (int): Number of entities found
- `patterns_found` (int): Number of patterns found

## Profile Management

### `ProfileManager`

Manages attacker profiles with persistent storage and incremental updates.

**Location**: `src/agents/profile_manager.py`

#### Constructor

```python
def __init__(self, db_path: str = "data/profiles.db")
```

#### Methods

##### `save_profile(profile)`

Save or update an attacker profile.

**Parameters**:
- `profile` (AttackerProfile): Profile to save

**Returns**:
- `bool`: Success status

##### `get_profile(entity_id)`

Retrieve a profile by entity ID.

**Parameters**:
- `entity_id` (str): Entity identifier

**Returns**:
- `Optional[AttackerProfile]`: Profile or None

##### `update_profile_incrementally(entity_id, entity_type, new_events, llm_analysis=None)`

Incrementally update a profile with new events.

**Parameters**:
- `entity_id` (str): Entity identifier
- `entity_type` (EntityType): Type of entity
- `new_events` (List[Dict]): New events to incorporate
- `llm_analysis` (Optional[Dict]): LLM analysis results

**Returns**:
- `AttackerProfile`: Updated profile

##### `get_high_threat_profiles(limit=50)`

Get profiles with highest threat levels.

**Parameters**:
- `limit` (int): Maximum number of profiles

**Returns**:
- `List[AttackerProfile]`: High threat profiles

##### `find_similar_profiles(profile, limit=10)`

Find similar profiles based on behavioral patterns.

**Parameters**:
- `profile` (AttackerProfile): Reference profile
- `limit` (int): Maximum number of similar profiles

**Returns**:
- `List[Tuple[str, float]]`: Similar profiles with scores

### `AttackerProfile`

Comprehensive attacker profile with behavioral patterns.

**Location**: `src/agents/profile_manager.py`

#### Key Attributes

- `entity_id` (str): Entity identifier
- `entity_type` (EntityType): Type of entity
- `entity_value` (str): Actual entity value
- `first_seen` (datetime): First observation time
- `last_seen` (datetime): Last observation time
- `active_days` (int): Number of active days
- `scanning_patterns` (List[AttackPattern]): Detected scanning patterns
- `exploitation_attempts` (List[AttackPattern]): Exploitation attempts
- `persistence_indicators` (List[AttackPattern]): Persistence indicators
- `behavioral_embedding` (Optional[np.ndarray]): Behavioral embedding
- `total_events` (int): Total number of events
- `unique_targets` (int): Number of unique targets
- `unique_ports` (int): Number of unique ports
- `threat_level` (ThreatLevel): Current threat level
- `confidence_scores` (Dict[str, float]): Confidence scores
- `ttps` (List[str]): MITRE ATT&CK techniques
- `iocs` (List[str]): Indicators of compromise

#### Methods

##### `calculate_threat_score()`

Calculate overall threat score based on various factors.

**Returns**:
- `float`: Threat score (0.0 to 1.0)

##### `update_threat_level()`

Update threat level based on current profile data.

##### `to_dict()`

Convert profile to dictionary for storage.

**Returns**:
- `Dict`: Dictionary representation

### `AttackPattern`

Represents a specific attack pattern.

**Location**: `src/agents/profile_manager.py`

#### Attributes

- `pattern_type` (str): Type of pattern
- `first_seen` (datetime): First detection time
- `last_seen` (datetime): Last detection time
- `occurrence_count` (int): Number of occurrences
- `targets` (List[str]): Target entities
- `ports` (List[int]): Target ports
- `techniques` (List[str]): MITRE ATT&CK techniques
- `confidence` (float): Pattern confidence

#### Methods

##### `to_dict()`

Convert pattern to dictionary.

**Returns**:
- `Dict`: Dictionary representation

## Pattern Analysis

### `PatternAnalyzer`

Analyzes events to identify attack patterns and anomalies.

**Location**: `src/agents/pattern_analyzer.py`

#### Constructor

```python
def __init__(self)
```

#### Methods

##### `analyze(events)`

Analyze events to identify patterns and anomalies.

**Parameters**:
- `events` (List[Dict[str, Any]]): Events to analyze

**Returns**:
- `Dict[str, Any]`: Analysis results by entity ID

##### `_detect_scanning_patterns(events)`

Detect port scanning and reconnaissance patterns.

**Parameters**:
- `events` (List[Dict[str, Any]]): Events to analyze

**Returns**:
- `List[Pattern]`: Detected scanning patterns

##### `_detect_exploitation_patterns(events)`

Detect exploitation attempt patterns.

**Parameters**:
- `events` (List[Dict[str, Any]]): Events to analyze

**Returns**:
- `List[Pattern]`: Detected exploitation patterns

##### `_detect_persistence_patterns(events)`

Detect persistence and lateral movement patterns.

**Parameters**:
- `events` (List[Dict[str, Any]]): Events to analyze

**Returns**:
- `List[Pattern]`: Detected persistence patterns

##### `_calculate_threat_score(patterns)`

Calculate overall threat score based on detected patterns.

**Parameters**:
- `patterns` (List[Pattern]): Detected patterns

**Returns**:
- `float`: Threat score (0.0 to 1.0)

## LLM Integration

### `OllamaLLM`

Local LLM integration using Ollama.

**Location**: `src/llm/local_llm.py`

#### Constructor

```python
def __init__(
    self,
    model: str = "mixtral:8x7b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int = 4096
)
```

#### Methods

##### `analyze_events(events)`

Analyze security events using the local LLM.

**Parameters**:
- `events` (List[Dict[str, Any]]): Security events to analyze

**Returns**:
- `Dict[str, Any]`: Analysis results

##### `health_check()`

Check if Ollama is healthy and responsive.

**Returns**:
- `bool`: True if healthy

### `ClaudeAPI`

Claude API integration for advanced analysis.

**Location**: `src/llm/api_llm.py`

#### Constructor

```python
def __init__(
    self,
    api_key: str,
    model: str = "claude-3-opus-20240229",
    base_url: str = "https://api.anthropic.com",
    max_tokens: int = 4096
)
```

#### Methods

##### `analyze_events(events)`

Analyze security events using Claude.

**Parameters**:
- `events` (List[Dict[str, Any]]): Security events to analyze

**Returns**:
- `Dict[str, Any]`: Analysis results

### `OpenAIAPI`

OpenAI API integration for analysis and embeddings.

**Location**: `src/llm/api_llm.py`

#### Constructor

```python
def __init__(
    self,
    api_key: str,
    model: str = "gpt-4-turbo-preview",
    embedding_model: str = "text-embedding-3-large",
    base_url: str = "https://api.openai.com/v1",
    max_tokens: int = 4096
)
```

#### Methods

##### `analyze_events(events)`

Analyze security events using OpenAI GPT.

**Parameters**:
- `events` (List[Dict[str, Any]]): Security events to analyze

**Returns**:
- `Dict[str, Any]`: Analysis results

##### `generate_embeddings(texts)`

Generate embeddings for texts using OpenAI.

**Parameters**:
- `texts` (List[str]): Texts to embed

**Returns**:
- `List[List[float]]`: Embedding vectors

## Memory and Storage

### `VectorStore`

Vector store implementation using ChromaDB.

**Location**: `src/memory/vector_store.py`

#### Constructor

```python
def __init__(self, db_path: str = "data/chroma", collection_name: str = "patterns")
```

#### Methods

##### `add_pattern_embedding(pattern_id, embedding, metadata)`

Add a pattern embedding to the vector store.

**Parameters**:
- `pattern_id` (str): Pattern identifier
- `embedding` (np.ndarray): Vector embedding
- `metadata` (Dict[str, Any]): Pattern metadata

**Returns**:
- `bool`: Success status

##### `search_similar_patterns(query_embedding, top_k=10, filter_metadata=None)`

Search for similar patterns using vector similarity.

**Parameters**:
- `query_embedding` (np.ndarray): Query vector
- `top_k` (int): Number of results
- `filter_metadata` (Optional[Dict[str, Any]]): Metadata filters

**Returns**:
- `List[Tuple[str, float, Dict[str, Any]]]`: Similar patterns with scores

##### `get_collection_stats()`

Get statistics about the collection.

**Returns**:
- `Dict[str, Any]`: Collection statistics

## Configuration Management

### `ConfigManager`

Configuration manager for loading and validating configuration.

**Location**: `src/utils/config.py`

#### Constructor

```python
def __init__(self, config_path: Optional[str] = None)
```

#### Methods

##### `load_config(config_path=None)`

Load configuration from file and environment variables.

**Parameters**:
- `config_path` (Optional[str]): Path to configuration file

**Returns**:
- `Config`: Loaded configuration

##### `validate_config(config)`

Validate configuration values.

**Parameters**:
- `config` (Config): Configuration to validate

**Returns**:
- `bool`: True if valid

## Utility Classes

### `EnhancedMCPClient`

Enhanced wrapper for dshield-mcp with proper pagination and streaming support.

**Location**: `src/utils/mcp_client.py`

#### Constructor

```python
def __init__(self, mcp_url: str = "http://localhost:3000", timeout: int = 30)
```

#### Methods

##### `query_with_pagination(time_range_hours=24, page_size=500, fields=None, filters=None, indices=None)`

Query DShield events using proper cursor-based pagination.

**Parameters**:
- `time_range_hours` (int): Time range in hours
- `page_size` (int): Page size
- `fields` (Optional[List[str]]): Fields to retrieve
- `filters` (Optional[Dict]): Query filters
- `indices` (Optional[List[str]]): Indices to query

**Returns**:
- `AsyncGenerator[List[Dict], None]`: Event chunks

##### `stream_with_sessions(time_range_hours=24, chunk_size=500, session_fields=None, max_session_gap_minutes=30, filters=None)`

Stream events with session grouping for better context.

**Parameters**:
- `time_range_hours` (int): Time range in hours
- `chunk_size` (int): Chunk size
- `session_fields` (List[str]): Session grouping fields
- `max_session_gap_minutes` (int): Maximum session gap
- `filters` (Optional[Dict]): Query filters

**Returns**:
- `AsyncGenerator[Dict, None]`: Session chunks

##### `get_aggregated_summary(start_time, end_time, aggregations=None)`

Get aggregated summary for a time window.

**Parameters**:
- `start_time` (datetime): Start time
- `end_time` (datetime): End time
- `aggregations` (Optional[List[str]]): Aggregation fields

**Returns**:
- `QuerySummary`: Summary statistics

##### `test_connection()`

Test connection to MCP server.

**Returns**:
- `bool`: Connection status

## Data Models

### Enums

#### `EntityType`

Entity type enumeration.

**Location**: `src/agents/profile_manager.py`

**Values**:
- `IP`: IP address
- `ASN`: Autonomous system number
- `COMPOSITE`: Composite entity
- `PORT`: Port number
- `COUNTRY`: Country code

#### `ThreatLevel`

Threat level enumeration.

**Location**: `src/agents/profile_manager.py`

**Values**:
- `LOW`: Low threat
- `MEDIUM`: Medium threat
- `HIGH`: High threat
- `CRITICAL`: Critical threat

#### `PatternType`

Pattern type enumeration.

**Location**: `src/models/pattern.py`

**Values**:
- `SCANNING`: Scanning activity
- `EXPLOITATION`: Exploitation attempts
- `PERSISTENCE`: Persistence mechanisms
- `LATERAL_MOVEMENT`: Lateral movement
- `DATA_EXFILTRATION`: Data exfiltration
- `COMMAND_AND_CONTROL`: Command and control
- `UNKNOWN`: Unknown pattern type

#### `PatternSeverity`

Pattern severity enumeration.

**Location**: `src/models/pattern.py`

**Values**:
- `LOW`: Low severity
- `MEDIUM`: Medium severity
- `HIGH`: High severity
- `CRITICAL`: Critical severity

## Usage Examples

### Basic Analysis

```python
import asyncio
from datetime import datetime, timedelta
from src.orchestrator import LongTailAnalyzer

async def main():
    analyzer = LongTailAnalyzer()
    
    # Analyze last 24 hours
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    results = await analyzer.analyze_period(start_date, end_date)
    print(f"Analysis complete: {results}")

asyncio.run(main())
```

### Profile Exploration

```python
from src.agents.profile_manager import ProfileManager

# Get high threat profiles
pm = ProfileManager()
threats = pm.get_high_threat_profiles(limit=10)

for profile in threats:
    print(f"{profile.entity_id}:")
    print(f"  Threat Level: {profile.threat_level.name}")
    print(f"  Score: {profile.calculate_threat_score():.2f}")
    print(f"  Events: {profile.total_events}")
    print(f"  TTPs: {', '.join(profile.ttps)}")
```

### Pattern Analysis

```python
from src.agents.pattern_analyzer import PatternAnalyzer

analyzer = PatternAnalyzer()
patterns = await analyzer.analyze(events)

for entity_id, analysis in patterns.items():
    print(f"Entity {entity_id}:")
    print(f"  Threat Score: {analysis['threat_score']:.2f}")
    print(f"  Patterns: {len(analysis['patterns'])}")
    print(f"  TTPs: {', '.join(analysis['ttps'])}")
```

## Error Handling

All classes implement comprehensive error handling with proper logging. Key error scenarios include:

- **Connection failures**: MCP server connectivity issues
- **Data validation errors**: Invalid input parameters
- **Resource limitations**: Memory and processing constraints
- **External service failures**: LLM API and enrichment service issues

## Performance Considerations

- **Pagination**: Large datasets are processed in chunks to manage memory
- **Caching**: Frequent queries are cached to improve performance
- **Async operations**: All I/O operations are asynchronous for better concurrency
- **Resource limits**: Configurable limits prevent resource exhaustion
- **Checkpointing**: Long-running operations support resumption

## Security Considerations

- **Input validation**: All inputs are validated before processing
- **API key management**: Sensitive credentials are handled securely
- **Data sanitization**: User inputs are sanitized to prevent injection
- **Access control**: Database access is controlled and logged
- **Audit trails**: All operations are logged for security auditing