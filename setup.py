#!/usr/bin/env python3
"""
Setup script for the Long-Tail Analysis Agent.

This script helps with initial setup and installation of the system.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"  {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"    ✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ✗ {description} failed: {e}")
        if e.stdout:
            print(f"    stdout: {e.stdout}")
        if e.stderr:
            print(f"    stderr: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 11):
        print("  ✗ Python 3.11+ is required")
        return False
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    return run_command("pip install -r requirements.txt", "Installing requirements")


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = [
        "data",
        "logs",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ Created directory: {directory}")
    
    return True


def setup_environment():
    """Set up environment file."""
    print("Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("  ✓ Created .env file from .env.example")
        print("  ⚠  Please edit .env file with your API keys and configuration")
    elif env_file.exists():
        print("  ✓ .env file already exists")
    else:
        print("  ⚠  No .env.example file found, please create .env manually")
    
    return True


def check_ollama():
    """Check if Ollama is installed and running."""
    print("Checking Ollama installation...")
    
    # Check if ollama command exists
    try:
        result = subprocess.run("ollama --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Ollama is installed: {result.stdout.strip()}")
            
            # Check if Ollama server is running
            try:
                result = subprocess.run("curl -s http://localhost:11434/api/tags", shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print("  ✓ Ollama server is running")
                else:
                    print("  ⚠  Ollama server is not running. Start it with: ollama serve")
            except:
                print("  ⚠  Could not check Ollama server status")
            
            return True
        else:
            print("  ✗ Ollama is not installed")
            print("    Install from: https://ollama.ai/")
            return False
    except:
        print("  ✗ Ollama is not installed")
        print("    Install from: https://ollama.ai/")
        return False


def check_mcp_server():
    """Check if MCP server is available."""
    print("Checking MCP server...")
    
    try:
        result = subprocess.run("curl -s http://localhost:3000/health", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ MCP server is running on localhost:3000")
            return True
        else:
            print("  ⚠  MCP server is not running on localhost:3000")
            print("    Make sure the DShield MCP server is started")
            return False
    except:
        print("  ⚠  Could not check MCP server status")
        return False


def run_basic_test():
    """Run basic functionality test."""
    print("Running basic tests...")
    return run_command("python test_basic.py", "Basic functionality test")


def main():
    """Main setup function."""
    print("Long-Tail Analysis Agent - Setup")
    print("=" * 40)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Create directories
    if not create_directories():
        success = False
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Setup environment
    if not setup_environment():
        success = False
    
    # Check Ollama
    if not check_ollama():
        success = False
    
    # Check MCP server
    if not check_mcp_server():
        success = False
    
    # Run basic test
    if not run_basic_test():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys")
        print("2. Start the MCP server if not already running")
        print("3. Pull a model in Ollama: ollama pull mixtral:8x7b")
        print("4. Run analysis: python main.py analyze --start-days 7")
    else:
        print("✗ Setup completed with errors")
        print("Please fix the errors above before proceeding")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
