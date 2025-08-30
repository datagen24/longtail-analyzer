# Cursor AI Development Prompt: Long-Tail Analysis Agent

## Context
I'm building a long-tail analysis agent for DShield honeypot security data. The system needs to efficiently process large datasets (10k+ records/day over month+ periods) while managing context window limitations through intelligent chunking and memory systems.

**Key Requirements:**
- Use existing MCP server (https://github.com/datagen24/dsheild-mcp) with proper pagination/streaming
- Build persistent memory for attacker profiles
- Support both local LLMs (Ollama) and API models (Claude/OpenAI)
- Run on MacBook M4 Max with 64GB RAM

## Task 1: Project Foundation

Create the following project structure:
```
longtail-analyzer/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── data_processor.py      # MCP data retrieval with pagination
│   │   ├── pattern_analyzer.py    # Pattern recognition
│   │   ├── profile_manager.py     # Profile CRUD operations
│   │   └── enrichment_agent.py    # External intelligence
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── vector_store.py        # ChromaDB interface
│   │   ├── state_store.py         # SQLite for profiles
│   │   └── cache_manager.py       # Working memory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── profile.py             # Attacker profile dataclass
│   │   └── pattern.py             # Pattern dataclass
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── local_llm.py          # Ollama integration
│   │   └── api_llm.py            # Claude/OpenAI
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── mcp_client.py         # Enhanced MCP wrapper
│   │   └── config.py             # Configuration management
│   └── orchestrator.py           # Main orchestration logic
├── tests/
├── configs/
│   └── default.yaml
├── requirements.txt
├── .env.example
└── README.md
```

## Task 2: Enhanced MCP Client

Create `src/utils/mcp_client.py` that properly uses the dshield-mcp capabilities:

```python
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import httpx
import json

class EnhancedMCPClient:
    """
    Wrapper for dshield-mcp with proper pagination and streaming support.
    """
    
    def __init__(self, mcp_url: str = "http://localhost:3000"):
        self.mcp_url = mcp_url
        self.session = httpx.AsyncClient(timeout=30.0)
        
    async def query_with_pagination(
        self,
        time_range_hours: int = 24,
        page_size: int = 500,
        fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Query DShield events using proper pagination.
        IMPORTANT: Use cursor-based pagination for large datasets.
        """
        all_results = []
        cursor = None
        
        while True:
            params = {
                "time_range_hours": time_range_hours,
                "page_size": page_size,
                "optimization": "auto",
                "include_summary": True
            }
            
            if cursor:
                params["cursor"] = cursor
            if fields:
                params["fields"] = fields
                
            # Call query_dshield_events with pagination
            response = await self._call_mcp_function(
                "query_dshield_events",
                params
            )
            
            if not response or "data" not in response:
                break
                
            all_results.extend(response["data"])
            
            # Check for next cursor
            cursor = response.get("next_cursor")
            if not cursor:
                break
                
            # Respect rate limits
            await asyncio.sleep(0.1)
            
        return all_results
    
    async def stream_with_sessions(
        self,
        time_range_hours: int = 24,
        chunk_size: int = 500,
        session_fields: List[str] = ["source.ip", "destination.ip"],
        max_session_gap_minutes: int = 30
    ):
        """
        Stream events with session grouping for better context.
        """
        params = {
            "time_range_hours": time_range_hours,
            "chunk_size": chunk_size,
            "session_fields": session_fields,
            "max_session_gap_minutes": max_session_gap_minutes
        }
        
        # Call stream_dshield_events_with_session_context
        async for chunk in self._stream_mcp_function(
            "stream_dshield_events_with_session_context",
            params
        ):
            yield chunk
    
    async def get_time_window_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        aggregations: List[str] = ["source.ip", "destination.port", "event.action"]
    ) -> Dict:
        """
        Get aggregated summary for a time window to reduce data size.
        """
        # Implementation for getting summaries
        pass
```

## Task 3: Profile Manager with Memory

Create `src/agents/profile_manager.py` with persistent storage:

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import sqlite3
from pathlib import Path
import numpy as np

@dataclass
class AttackerProfile:
    entity_id: str  # e.g., "ip_192.168.1.1" or "asn_AS12345"
    entity_type: str  # "ip", "asn", or "composite"
    first_seen: datetime
    last_seen: datetime
    
    # Attack patterns
    scanning_patterns: Dict = field(default_factory=dict)
    exploitation_attempts: Dict = field(default_factory=dict)
    persistence_indicators: Dict = field(default_factory=dict)
    
    # Behavioral fingerprint (embedding)
    behavioral_embedding: Optional[np.ndarray] = None
    
    # Confidence and metadata
    confidence_scores: Dict = field(default_factory=dict)
    ttps: List[str] = field(default_factory=list)
    
    # Infrastructure
    asn: Optional[str] = None
    geo_location: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)
    
    # Analysis metadata
    last_analysis: datetime = field(default_factory=datetime.now)
    analysis_depth: str = "shallow"  # shallow, deep, comprehensive
    data_quality_score: float = 0.0

class ProfileManager:
    def __init__(self, db_path: str = "profiles.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite tables for profile storage."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT,
                profile_data JSON,
                embedding BLOB,
                last_updated TIMESTAMP,
                confidence_score REAL
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_state (
                analysis_id TEXT PRIMARY KEY,
                state JSON,
                timestamp TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def save_profile(self, profile: AttackerProfile):
        """Save or update an attacker profile."""
        # Convert profile to JSON, handle numpy arrays
        profile_dict = asdict(profile)
        
        # Handle embedding separately
        embedding_bytes = None
        if profile.behavioral_embedding is not None:
            embedding_bytes = profile.behavioral_embedding.tobytes()
            profile_dict.pop('behavioral_embedding')
        
        self.conn.execute("""
            INSERT OR REPLACE INTO profiles 
            (entity_id, entity_type, profile_data, embedding, last_updated, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            profile.entity_id,
            profile.entity_type,
            json.dumps(profile_dict, default=str),
            embedding_bytes,
            datetime.now(),
            profile.confidence_scores.get('overall', 0.0)
        ))
        self.conn.commit()
    
    def get_profile(self, entity_id: str) -> Optional[AttackerProfile]:
        """Retrieve a profile by entity ID."""
        # Implementation
        pass
    
    def update_profile_incrementally(
        self,
        entity_id: str,
        new_events: List[Dict],
        llm_analysis: Optional[Dict] = None
    ):
        """
        Incrementally update a profile with new events.
        This is crucial for handling large datasets efficiently.
        """
        profile = self.get_profile(entity_id)
        if not profile:
            # Create new profile
            profile = self._create_profile_from_events(entity_id, new_events)
        else:
            # Update existing profile
            self._merge_new_events(profile, new_events)
            
        if llm_analysis:
            self._apply_llm_insights(profile, llm_analysis)
            
        self.save_profile(profile)
        return profile
```

## Task 4: Time Window Chunking Strategy

Create `src/agents/data_processor.py` with intelligent chunking:

```python
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import hashlib

class TimeWindowProcessor:
    """
    Processes data in overlapping time windows to catch cross-boundary patterns.
    """
    
    def __init__(
        self,
        mcp_client,
        window_hours: int = 6,
        overlap_hours: int = 1
    ):
        self.mcp_client = mcp_client
        self.window_hours = window_hours
        self.overlap_hours = overlap_hours
        self.processed_windows = set()  # Track processed windows
    
    def generate_time_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generate overlapping time windows for analysis.
        """
        windows = []
        current = start_date
        
        while current < end_date:
            window_end = min(
                current + timedelta(hours=self.window_hours),
                end_date
            )
            windows.append((current, window_end))
            
            # Move forward with overlap
            current += timedelta(hours=self.window_hours - self.overlap_hours)
            
        return windows
    
    async def process_window(
        self,
        start_time: datetime,
        end_time: datetime,
        profile_manager,
        pattern_analyzer
    ) -> Dict:
        """
        Process a single time window with proper data retrieval.
        """
        window_id = self._get_window_id(start_time, end_time)
        
        # Skip if already processed
        if window_id in self.processed_windows:
            return {"status": "skipped", "window_id": window_id}
        
        # Step 1: Get summary statistics first
        summary = await self.mcp_client.get_time_window_summary(
            start_time, end_time
        )
        
        # Step 2: Identify interesting entities from summary
        interesting_entities = self._identify_anomalies(summary)
        
        # Step 3: Get detailed data only for interesting entities
        detailed_data = []
        for entity in interesting_entities:
            entity_data = await self.mcp_client.query_with_pagination(
                time_range_hours=self.window_hours,
                page_size=500,
                fields=["source.ip", "destination.port", "event.action", "@timestamp"],
                filters={"source.ip": entity}
            )
            detailed_data.extend(entity_data)
        
        # Step 4: Analyze patterns
        patterns = await pattern_analyzer.analyze(detailed_data)
        
        # Step 5: Update profiles incrementally
        for entity_id, entity_events in self._group_by_entity(detailed_data).items():
            profile_manager.update_profile_incrementally(
                entity_id,
                entity_events,
                patterns.get(entity_id)
            )
        
        self.processed_windows.add(window_id)
        
        return {
            "status": "processed",
            "window_id": window_id,
            "entities_analyzed": len(interesting_entities),
            "patterns_found": len(patterns)
        }
    
    def _identify_anomalies(self, summary: Dict) -> List[str]:
        """
        Identify entities that warrant detailed analysis.
        This is KEY to managing context windows efficiently.
        """
        anomalies = []
        
        # Look for IPs with unusual activity volumes
        if "top_sources" in summary:
            for source in summary["top_sources"]:
                if source["count"] > summary["average_count"] * 2:
                    anomalies.append(source["ip"])
        
        # Look for rare ports
        if "rare_ports" in summary:
            anomalies.extend(summary["rare_ports"]["sources"])
        
        return anomalies[:50]  # Limit to top 50 to manage context
```

## Task 5: Main Orchestrator

Create `src/orchestrator.py` to tie everything together:

```python
import asyncio
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List

class LongTailAnalyzer:
    """
    Main orchestrator for long-tail analysis.
    """
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = self._load_config(config_path)
        self.mcp_client = EnhancedMCPClient(self.config["mcp_url"])
        self.profile_manager = ProfileManager(self.config["db_path"])
        self.time_processor = TimeWindowProcessor(self.mcp_client)
        self.pattern_analyzer = PatternAnalyzer()
        
        # Initialize LLMs
        self.local_llm = OllamaLLM(model="mixtral")
        self.api_llm = ClaudeAPI() if self.config.get("use_api") else None
        
        # State tracking
        self.analysis_state = self._load_analysis_state()
        
    async def analyze_period(
        self,
        start_date: datetime,
        end_date: datetime,
        resume: bool = True
    ):
        """
        Analyze a time period with automatic chunking and state management.
        """
        logging.info(f"Starting analysis from {start_date} to {end_date}")
        
        # Generate time windows
        windows = self.time_processor.generate_time_windows(start_date, end_date)
        
        # Resume from last checkpoint if requested
        if resume and self.analysis_state:
            windows = self._filter_processed_windows(windows)
        
        # Process windows in sequence with progress tracking
        for i, (window_start, window_end) in enumerate(windows):
            logging.info(f"Processing window {i+1}/{len(windows)}")
            
            try:
                result = await self.time_processor.process_window(
                    window_start,
                    window_end,
                    self.profile_manager,
                    self.pattern_analyzer
                )
                
                # Save checkpoint
                self._save_checkpoint(window_end, result)
                
                # Adaptive delay based on data volume
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.error(f"Error processing window: {e}")
                # Continue with next window
                continue
        
        # Generate final report
        report = self._generate_report()
        return report
    
    async def analyze_realtime(self):
        """
        Stream and analyze data in real-time using session grouping.
        """
        async for session_chunk in self.mcp_client.stream_with_sessions(
            time_range_hours=1,
            chunk_size=500
        ):
            # Process each session chunk
            for session_id, events in session_chunk.items():
                # Quick local LLM analysis
                quick_analysis = await self.local_llm.analyze(events)
                
                # Update profile if interesting
                if quick_analysis.get("threat_score", 0) > 0.7:
                    entity_id = self._extract_entity_id(events)
                    
                    # Deep analysis with API if needed
                    if self.api_llm and quick_analysis.get("needs_deep_analysis"):
                        deep_analysis = await self.api_llm.analyze(events)
                        quick_analysis.update(deep_analysis)
                    
                    self.profile_manager.update_profile_incrementally(
                        entity_id,
                        events,
                        quick_analysis
                    )
```

## Task 6: Configuration File

Create `configs/default.yaml`:

```yaml
# MCP Server Configuration
mcp_url: "http://localhost:3000"
mcp_timeout: 30

# Database Configuration  
db_path: "data/profiles.db"
vector_db_path: "data/chroma"
cache_ttl_hours: 24

# Analysis Configuration
window_hours: 6
overlap_hours: 1
max_entities_per_window: 50
anomaly_threshold: 2.0

# LLM Configuration
local_llm:
  provider: "ollama"
  model: "mixtral:8x7b"
  temperature: 0.7
  max_tokens: 4096

api_llm:
  use_api: true
  providers:
    claude:
      api_key: "${CLAUDE_API_KEY}"
      model: "claude-3-opus-20240229"
      max_requests_per_minute: 10
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4-turbo-preview"
      embedding_model: "text-embedding-3-large"

# Memory Configuration
memory:
  max_working_memory_mb: 1024
  profile_cache_size: 1000
  embedding_dimensions: 1536

# Monitoring
logging:
  level: "INFO"
  file: "logs/analyzer.log"
  
metrics:
  export_interval_seconds: 60
  prometheus_port: 9090
```

## Task 7: Requirements File

Create `requirements.txt`:

```txt
# Core dependencies
httpx>=0.24.0
asyncio
pydantic>=2.0
PyYAML>=6.0
python-dotenv

# Database
chromadb>=0.4.0
redis>=4.5.0
sqlalchemy>=2.0

# LLM integrations
ollama-python>=0.1.0
anthropic>=0.7.0
openai>=1.0.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Utilities
tqdm>=4.65.0
rich>=13.0.0
click>=8.1.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-mock>=3.10.0

# Development
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
```

## Implementation Priority:
1. First, set up the project structure and install dependencies
2. Implement the enhanced MCP client with proper pagination
3. Create the profile manager with SQLite storage
4. Build the time window processor with chunking
5. Implement basic pattern analysis
6. Add the orchestrator to tie everything together
7. Test with small dataset first (1 day of data)
8. Add memory systems (vector DB)
9. Integrate LLMs (local first, then API)
10. Add enrichment capabilities

## Key Implementation Notes:

1. **MCP Pagination**: Always use cursor-based pagination for large datasets. The MCP server supports this but current agents aren't using it properly.

2. **Memory Management**: Use incremental profile updates rather than loading all data at once. This is crucial for managing the 64GB RAM constraint.

3. **Context Window Optimization**: 
   - Pre-aggregate data using MCP's summary capabilities
   - Only fetch detailed records for anomalous entities
   - Use embeddings to compress historical patterns

4. **State Persistence**: Save checkpoints after each window to enable resumption of long-running analyses.

5. **Error Handling**: Implement robust error handling for MCP timeouts and API failures.

Please start by creating this project structure and implementing the enhanced MCP client with proper pagination support. Focus on making the pagination work correctly with the dshield-mcp server's `query_dshield_events` function.