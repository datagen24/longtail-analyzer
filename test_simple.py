#!/usr/bin/env python3
"""
Simple test script for the Long-Tail Analysis Agent.

This script tests basic functionality without complex imports.
"""

import sys
from pathlib import Path

def test_python_version():
    """Test Python version compatibility."""
    print("Testing Python Version...")
    if sys.version_info >= (3, 11):
        print("  ✓ Python 3.11+ is compatible")
        return True
    else:
        print(f"  ✗ Python {sys.version_info.major}.{sys.version_info.minor} is not compatible")
        return False

def test_imports():
    """Test basic imports."""
    print("Testing Basic Imports...")
    
    try:
        import httpx
        print("  ✓ httpx imported successfully")
    except ImportError as e:
        print(f"  ✗ httpx import failed: {e}")
        return False
    
    try:
        import pydantic
        print("  ✓ pydantic imported successfully")
    except ImportError as e:
        print(f"  ✗ pydantic import failed: {e}")
        return False
    
    try:
        import chromadb
        print("  ✓ chromadb imported successfully")
    except ImportError as e:
        print(f"  ✗ chromadb import failed: {e}")
        return False
    
    try:
        import redis
        print("  ✓ redis imported successfully")
    except ImportError as e:
        print(f"  ✗ redis import failed: {e}")
        return False
    
    try:
        import sqlalchemy
        print("  ✓ sqlalchemy imported successfully")
    except ImportError as e:
        print(f"  ✗ sqlalchemy import failed: {e}")
        return False
    
    return True

def test_directories():
    """Test directory structure."""
    print("Testing Directory Structure...")
    
    required_dirs = ["src", "configs", "data", "logs"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ {dir_name}/ directory exists")
        else:
            print(f"  ✗ {dir_name}/ directory missing")
            return False
    
    return True

def test_config_files():
    """Test configuration files."""
    print("Testing Configuration Files...")
    
    config_file = Path("configs/default.yaml")
    if config_file.exists():
        print("  ✓ configs/default.yaml exists")
    else:
        print("  ✗ configs/default.yaml missing")
        return False
    
    pyproject_file = Path("pyproject.toml")
    if pyproject_file.exists():
        print("  ✓ pyproject.toml exists")
    else:
        print("  ✗ pyproject.toml missing")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Long-Tail Analysis Agent - Simple Tests")
    print("=" * 50)
    
    success = True
    
    # Test individual components
    if not test_python_version():
        success = False
    
    if not test_imports():
        success = False
    
    if not test_directories():
        success = False
    
    if not test_config_files():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All simple tests passed!")
    else:
        print("✗ Some tests failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
