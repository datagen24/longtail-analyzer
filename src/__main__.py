#!/usr/bin/env python3
"""
Main entry point for the Long-Tail Analysis Agent when run as a module.

Usage: python -m src --help
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run the main function from the root main.py
from main import main

if __name__ == "__main__":
    asyncio.run(main())
