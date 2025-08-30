# Import Structure Guide

This document explains the import structure used in the Long-Tail Analyzer project and how to properly import and use the modules.

## Overview

The project uses absolute imports with the `src` package prefix to ensure consistent and reliable imports across all modules and entry points.

## Package Structure

```
longtail-analyzer/
├── src/                    # Main package
│   ├── __init__.py        # Package initialization
│   ├── __main__.py        # Module entry point
│   ├── orchestrator.py    # Main orchestrator
│   ├── agents/            # Analysis agents
│   │   ├── __init__.py
│   │   ├── data_processor.py
│   │   ├── pattern_analyzer.py
│   │   ├── profile_manager.py
│   │   └── enrichment_agent.py
│   ├── llm/               # LLM integrations
│   │   ├── __init__.py
│   │   ├── local_llm.py
│   │   └── api_llm.py
│   ├── memory/            # Memory system
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   ├── state_store.py
│   │   └── cache_manager.py
│   ├── models/            # Data models
│   │   ├── __init__.py
│   │   ├── profile.py
│   │   └── pattern.py
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── config.py
│       └── mcp_client.py
├── main.py                # Main entry point
├── test_basic.py          # Basic tests
└── test_simple.py         # Simple tests
```

## Import Patterns

### Absolute Imports

All modules use absolute imports with the `src` prefix:

```python
# ✅ Correct - Absolute imports
from src.utils.mcp_client import EnhancedMCPClient
from src.agents.data_processor import TimeWindowProcessor
from src.models.profile import AttackerProfile
from src.llm.local_llm import OllamaLLM

# ❌ Incorrect - Relative imports (old pattern)
from ..utils.mcp_client import EnhancedMCPClient
from .agents.data_processor import TimeWindowProcessor
```

### Package Initialization

Each package has an `__init__.py` file that exports the main classes:

```python
# src/agents/__init__.py
from src.agents.data_processor import TimeWindowProcessor
from src.agents.pattern_analyzer import PatternAnalyzer
from src.agents.profile_manager import ProfileManager
from src.agents.enrichment_agent import EnrichmentAgent

__all__ = [
    "TimeWindowProcessor",
    "PatternAnalyzer", 
    "ProfileManager",
    "EnrichmentAgent"
]
```

## Running the Application

### Method 1: Direct Execution (Recommended)

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the main application
python main.py --help
python main.py analyze --start-days 7
```

### Method 2: UV Run

```bash
# Run with UV (no need to activate environment)
uv run python main.py --help
uv run python main.py analyze --start-days 7
```

### Method 3: Module Execution

```bash
# Run as a module
uv run python -m src --help
uv run python -m src analyze --start-days 7
```

## Import Examples

### In Main Application

```python
# main.py
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.orchestrator import LongTailAnalyzer
from src.utils.config import ConfigManager
```

### In Source Modules

```python
# src/orchestrator.py
from src.utils.mcp_client import EnhancedMCPClient
from src.agents.data_processor import TimeWindowProcessor
from src.agents.pattern_analyzer import PatternAnalyzer
from src.agents.profile_manager import ProfileManager
from src.agents.enrichment_agent import EnrichmentAgent
from src.llm.local_llm import OllamaLLM
from src.llm.api_llm import ClaudeAPI, OpenAIAPI
```

### In Test Files

```python
# test_basic.py
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.mcp_client import EnhancedMCPClient
from src.models.profile import AttackerProfile
from src.agents.profile_manager import ProfileManager
from src.agents.pattern_analyzer import PatternAnalyzer
from src.utils.config import ConfigManager
```

## Development Guidelines

### Adding New Modules

1. **Create the module** in the appropriate package directory
2. **Use absolute imports** for all internal dependencies
3. **Update the package's `__init__.py`** to export the new classes/functions
4. **Add to `__all__`** list for explicit exports

Example:
```python
# src/agents/new_agent.py
from src.models.pattern import Pattern
from src.utils.config import ConfigManager

class NewAgent:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.pattern = Pattern()
```

```python
# src/agents/__init__.py
from src.agents.data_processor import TimeWindowProcessor
from src.agents.pattern_analyzer import PatternAnalyzer
from src.agents.profile_manager import ProfileManager
from src.agents.enrichment_agent import EnrichmentAgent
from src.agents.new_agent import NewAgent  # Add new import

__all__ = [
    "TimeWindowProcessor",
    "PatternAnalyzer", 
    "ProfileManager",
    "EnrichmentAgent",
    "NewAgent"  # Add to exports
]
```

### Import Best Practices

1. **Always use absolute imports** with the `src` prefix
2. **Group imports** by type (standard library, third-party, local)
3. **Use specific imports** rather than wildcard imports
4. **Update `__init__.py` files** when adding new exports
5. **Test imports** in your development environment

### Common Import Patterns

```python
# Standard library imports
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

# Third-party imports
import httpx
import pydantic
import yaml

# Local imports (absolute with src prefix)
from src.utils.mcp_client import EnhancedMCPClient
from src.models.profile import AttackerProfile
from src.agents.pattern_analyzer import PatternAnalyzer
```

## Troubleshooting

### Import Errors

If you encounter import errors:

1. **Check the import path**: Ensure you're using the `src` prefix
2. **Verify the module exists**: Check that the file exists in the expected location
3. **Check `__init__.py`**: Ensure the package has proper initialization
4. **Test the import**: Try importing in a Python shell

```bash
# Test imports in Python shell
source .venv/bin/activate
python -c "from src.orchestrator import LongTailAnalyzer; print('Import successful')"
```

### Path Issues

If you have path-related issues:

1. **Use the activation script**: `./activate.sh`
2. **Check virtual environment**: Ensure you're in the correct environment
3. **Verify PYTHONPATH**: The setup adds `src` to the path automatically

### Module Not Found

If you get "Module not found" errors:

1. **Check file structure**: Ensure all `__init__.py` files exist
2. **Verify package structure**: Check the directory structure matches the imports
3. **Test with simple imports**: Start with basic imports and build up

## Migration from Relative Imports

If you're updating code that uses relative imports:

### Before (Relative Imports)
```python
from ..utils.mcp_client import EnhancedMCPClient
from .agents.data_processor import TimeWindowProcessor
from ..models.profile import AttackerProfile
```

### After (Absolute Imports)
```python
from src.utils.mcp_client import EnhancedMCPClient
from src.agents.data_processor import TimeWindowProcessor
from src.models.profile import AttackerProfile
```

## Testing the Import Structure

### Quick Test
```bash
# Test basic imports
source .venv/bin/activate
python test_simple.py
```

### Full Test
```bash
# Test the main application
source .venv/bin/activate
python main.py --help
```

### Module Test
```bash
# Test module execution
uv run python -m src --help
```

## Benefits of This Structure

1. **Consistency**: All imports follow the same pattern
2. **Reliability**: No issues with relative import resolution
3. **Clarity**: Clear indication of where modules are located
4. **Maintainability**: Easy to refactor and move modules
5. **IDE Support**: Better autocomplete and navigation
6. **Testing**: Easier to test individual modules

---

**Status**: ✅ Import Structure Fixed  
**Last Updated**: Current  
**Compatibility**: Python 3.11+ with UV environment
