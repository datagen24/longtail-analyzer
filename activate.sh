#!/bin/bash
# Activation script for Long-Tail Analyzer UV environment

echo "Long-Tail Analyzer - UV Environment Setup"
echo "=========================================="

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Creating UV virtual environment..."
    uv venv --python 3.11
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        exit 1
    fi
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
if ! python -c "import httpx, pydantic, chromadb" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install -e ".[dev,test,docs]"
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies"
        exit 1
    fi
fi

echo "✓ Environment ready!"
echo ""
echo "To activate manually in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "Available commands:"
echo "  python main.py --help"
echo "  python setup.py"
echo "  pytest"
echo "  black src/"
echo "  ruff check src/"
