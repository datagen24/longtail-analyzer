# UV Setup Guide for Long-Tail Analyzer

This guide explains how to use UV (a fast Python package installer) with the Long-Tail Analyzer project.

## What is UV?

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust. It's designed to be a drop-in replacement for pip with significant performance improvements.

## Benefits of Using UV

- **Speed**: 10-100x faster than pip for dependency resolution and installation
- **Reliability**: Better dependency resolution with lockfile support
- **Modern Python**: Built for modern Python packaging standards
- **Compatibility**: Works with existing pip workflows and requirements.txt
- **MCP Support**: Better support for newer Python features used by MCP modules

## Quick Start

### 1. Install UV (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew (macOS)
brew install uv

# Or via pip
pip install uv
```

### 2. Set up the project environment

```bash
# Option 1: Use the activation script (recommended)
./activate.sh

# Option 2: Manual setup
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,test,docs]"
```

### 3. Verify the setup

```bash
python setup.py
```

## Common UV Commands

### Environment Management

```bash
# Create a new virtual environment
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate

# Deactivate the environment
deactivate
```

### Package Management

```bash
# Install the project in development mode
uv pip install -e .

# Install with optional dependencies
uv pip install -e ".[dev,test,docs]"

# Install a specific package
uv pip install package-name

# Install from requirements.txt
uv pip install -r requirements.txt

# Update all packages
uv pip install --upgrade -e ".[dev,test,docs]"

# Uninstall a package
uv pip uninstall package-name
```

### Running Commands

```bash
# Run Python scripts with UV
uv run python script.py

# Run tests
uv run pytest tests/ -v

# Run code quality tools
uv run black src/ tests/
uv run ruff check src/ tests/
uv run mypy src/

# Run the main application
uv run python main.py --help
```

## Project Structure

The project uses a modern Python packaging structure:

```
longtail-analyzer/
├── pyproject.toml          # Project configuration and dependencies
├── requirements.txt        # Legacy requirements (for compatibility)
├── .venv/                  # UV virtual environment
├── activate.sh             # Convenience activation script
├── src/                    # Source code
├── tests/                  # Test suite
├── docs/                   # Documentation
└── configs/                # Configuration files
```

## Dependencies

The project dependencies are defined in `pyproject.toml`:

- **Core dependencies**: httpx, pydantic, PyYAML, python-dotenv
- **Database**: chromadb, redis, sqlalchemy
- **LLM integrations**: ollama-python, anthropic, openai
- **Data processing**: numpy, pandas, scikit-learn
- **Utilities**: tqdm, rich, click

Optional dependency groups:
- **dev**: Development tools (black, ruff, mypy, pre-commit)
- **test**: Testing tools (pytest, pytest-asyncio, pytest-mock, pytest-cov)
- **docs**: Documentation tools (pdoc, pydoc-markdown, mkdocs)

## Troubleshooting

### NumPy Compatibility Issues

If you encounter NumPy 2.0 compatibility issues with ChromaDB:

```bash
# The project pins NumPy to <2.0.0 to avoid compatibility issues
uv pip install "numpy>=1.24.0,<2.0.0"
```

### Virtual Environment Issues

If the virtual environment becomes corrupted:

```bash
# Remove the existing environment
rm -rf .venv

# Recreate it
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,test,docs]"
```

### Import Issues

If you encounter import issues when running tests:

```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Or use uv run
uv run python test_simple.py
```

## Migration from pip

If you're migrating from a pip-based setup:

1. **Backup your current environment** (optional):
   ```bash
   pip freeze > requirements-backup.txt
   ```

2. **Create a new UV environment**:
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e ".[dev,test,docs]"
   ```

4. **Verify the setup**:
   ```bash
   python setup.py
   ```

## Best Practices

1. **Always use the virtual environment**: Either activate it with `source .venv/bin/activate` or use `uv run`
2. **Use pyproject.toml**: Add new dependencies to `pyproject.toml` rather than requirements.txt
3. **Pin versions**: Use specific version ranges for critical dependencies
4. **Test regularly**: Run `python setup.py` to verify the environment
5. **Keep dependencies updated**: Regularly update dependencies for security and compatibility

## Integration with IDEs

### VS Code

Add this to your `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

### PyCharm

1. Go to Settings → Project → Python Interpreter
2. Add New Interpreter → Existing Environment
3. Select `.venv/bin/python`

## Performance Comparison

Typical performance improvements with UV:

- **Dependency resolution**: 10-50x faster
- **Package installation**: 5-20x faster
- **Environment creation**: 3-10x faster
- **Overall setup time**: 5-15x faster

## Support

For UV-specific issues:
- [UV Documentation](https://github.com/astral-sh/uv)
- [UV GitHub Issues](https://github.com/astral-sh/uv/issues)

For project-specific issues:
- Check the main README.md
- Run `python setup.py` for environment verification
- Review the logs in the `logs/` directory
