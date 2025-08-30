"""
LLM integration modules for local and API-based language models.

This module provides:
- Local LLM integration via Ollama
- API-based LLM integration (Claude, OpenAI)
"""

from .local_llm import OllamaLLM
from .api_llm import ClaudeAPI, OpenAIAPI

__all__ = ["OllamaLLM", "ClaudeAPI", "OpenAIAPI"]
