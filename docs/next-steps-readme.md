# Long-Tail Analyzer - Implementation Guide

## Quick Start

### 1. Prerequisites
```bash
# Ensure DShield MCP server is running
cd /path/to/dshield-mcp
python mcp_server.py

# Verify MCP is accessible at http://localhost:3000
curl http://localhost:3000/health
```

### 2. Install Dependencies
```bash
cd longtail-analyzer
pip install -r requirements.txt

# Create necessary directories
mkdir -p data logs
```

### 3. Test the Implementation
```bash
# Test MCP connection
python test_analyzer.py --test connection

# Run quick demo (analyzes last hour)
python test_analyzer.py --demo

# Run full test suite
python test_analyzer.py --full
```

## Core Components Implementation Status

### ✅ Completed Components

1. **Enhanced MCP Client** (`src/utils/mcp_client.py`)
   - Proper cursor-based pagination
   - Streaming with session context
   - Aggregated summaries for efficient analysis
   - Anomaly identification from summaries

2. **Profile Manager** (`src/agents/profile_manager.py`)
   - SQLite-based persistent storage
   - Incremental profile updates
   - Attack pattern detection
   - Threat level assessment
   - Profile similarity matching

3. **Time Window Processor** (`src/agents/data_processor.py`)
   - Overlapping window generation
   - Cross-boundary pattern detection
   - Checkpoint/resume capability
   - Efficient anomaly-focused processing

### 🚧 Components to Implement Next

#### Priority 1: Pattern Analyzer
Create `src/agents/pattern_analyzer.py`:

```python
class PatternAnalyzer:
    """Analyzes events to detect attack patterns and TTPs"""
    
    async def analyze(self, entity_events: Dict[str, List[Dict]]) -> Dict:
        """
        Analyze events for each entity and return patterns.
        
        Returns:
            Dict mapping entity_id to analysis results including:
            - ttps: List of MITRE ATT&CK techniques
            - confidence_scores: Dict of confidence values
            - threat_score: Float 0-1
            - tags: List of descriptive tags
            - analysis_depth: shallow/medium/deep
        """
        results = {}
        
        for entity_id, events in entity_events.items():
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal(events)
            
            # Detect attack techniques
            ttps = self._detect_ttps(events)
            
            # Calculate threat score
            threat_score = self._calculate_threat(events, ttps)
            
            results[entity_id] = {
                "ttps": ttps,
                "confidence_scores": {"pattern": 0.8, "ttp": 0.7},
                "threat_score": threat_score,
                "tags": self._generate_tags(events),
                "analysis_depth": "medium"
            }
        
        return results
```

#### Priority 2: Main Orchestrator
Create `src/orchestrator.py`:

```python
class LongTailAnalyzer:
    """Main orchestration class"""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = self._load_config(config_path)
        self.mcp_client = EnhancedMCPClient(self.config["mcp_url"])
        self.profile_manager = ProfileManager(self.config["db_path"])
        self.time_processor = TimeWindowProcessor(
            self.mcp_client,
            self.profile_manager,
            window_hours=self.config["window_hours"],
            overlap_hours=self.config["overlap_hours"]
        )
        self.pattern_analyzer = PatternAnalyzer()
    
    async def analyze_period(
        self,
        start_date: datetime,
        end_date: datetime,
        resume: bool = True
    ) -> Dict:
        """Analyze a time period"""
        return await self.time_processor.process_time_range(
            start_date,
            end_date,
            self.pattern_analyzer,
            resume
        )
```

#### Priority 3: LLM Integration
Create `src/llm/local_llm.py`:

```python
import ollama

class OllamaLLM:
    """Local LLM using Ollama"""
    
    def __init__(self, model: str = "mixtral"):
        self.model = model
        self.client = ollama.Client()
    
    async def analyze_events(self, events: List[Dict]) -> Dict:
        """Analyze events using local LLM"""
        # Format events for LLM
        prompt = self._format_prompt(events)
        
        # Get LLM response
        response = await self.client.generate(
            model=self.model,
            prompt=prompt,
            format="json"
        )
        
        return self._parse_response(response)
```

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

### Real-time Streaming
```python
async def stream_analysis():
    analyzer = LongTailAnalyzer()
    
    # Stream and analyze in real-time
    await analyzer.analyze_realtime(duration_hours=1)

asyncio.run(stream_analysis())
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

## Configuration

### Environment Variables (.env)
```bash
# MCP Server
MCP_URL=http://localhost:3000
MCP_TIMEOUT=30

# Database
DB_PATH=data/profiles.db

# Analysis Parameters
WINDOW_HOURS=6
OVERLAP_HOURS=1
MAX_ENTITIES_PER_WINDOW=50
ANOMALY_THRESHOLD=2.0

# LLM Configuration (optional)
OLLAMA_MODEL=mixtral
CLAUDE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### Analysis Configuration (configs/default.yaml)
```yaml
# Time Window Processing
window_hours: 6
overlap_hours: 1
max_entities_per_window: 50

# Anomaly Detection
anomaly_threshold: 2.0
min_events_for_analysis: 10

# Profile Management
profile_cache_size: 1000
ttl_hours: 24

# Performance
batch_size: 500
max_concurrent_requests: 5
rate_limit_delay: 0.1
```

## Performance Optimization Tips

### 1. MCP Query Optimization
- Always use cursor-based pagination for large datasets
- Set appropriate `page_size` (500-1000 works well)
- Use field filtering to reduce payload size
- Enable query optimization with `optimization: "auto"`

### 2. Memory Management
- Process data in chunks, don't load everything at once
- Use incremental profile updates
- Implement TTL-based cache eviction
- Limit entities per window (50-100 max)

### 3. Context Window Management
- Pre-aggregate with summaries before detailed analysis
- Focus on anomalies only (2x threshold multiplier)
- Limit event details to essential fields
- Use embeddings to compress historical patterns

## Troubleshooting

### MCP Connection Issues
```bash
# Check MCP server is running
curl http://localhost:3000/tools/dshield-mcp:get_data_dictionary

# Test with minimal query
python -c "
import asyncio
from src.utils.mcp_client import EnhancedMCPClient

async def test():
    async with EnhancedMCPClient() as client:
        print(await client.test_connection())

asyncio.run(test())
"
```

### Memory Issues
```python
# Monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

### Performance Profiling
```python
import cProfile
import pstats

# Profile the analysis
cProfile.run('asyncio.run(main())', 'profile_stats')

# View results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Next Development Steps

1. **Implement Pattern Analyzer** (Week 1)
   - Temporal pattern detection
   - TTP identification
   - Threat scoring algorithm

2. **Add LLM Integration** (Week 2)
   - Ollama for local analysis
   - Claude API for complex reasoning
   - Prompt engineering for security analysis

3. **Build Vector Store** (Week 3)
   - ChromaDB integration
   - Embedding generation
   - Similarity search

4. **Create Web Interface** (Week 4)
   - FastAPI backend
   - Real-time monitoring dashboard
   - Profile exploration UI

5. **Add Enrichment Agent** (Week 5)
   - Web research integration
   - OSINT correlation
   - Threat intelligence feeds

## Monitoring & Metrics

### Key Metrics to Track
- Events processed per second
- Memory usage over time
- Profile creation rate
- Pattern detection accuracy
- Analysis latency per window

### Logging
```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analyzer.log'),
        logging.StreamHandler()
    ]
)
```

## Contributing

1. Test your changes:
```bash
pytest tests/ -v
```

2. Format code:
```bash
black src/ tests/
```

3. Type checking:
```bash
mypy src/
```

## Support

For issues or questions:
1. Check the test output: `python test_analyzer.py --full`
2. Review logs in `logs/` directory
3. Open an issue on GitHub with:
   - Error messages
   - MCP server version
   - Python version
   - Sample data that reproduces the issue