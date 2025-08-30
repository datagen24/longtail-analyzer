"""
LLM integration modules for local and API-based language models.

This module provides:
- Local LLM integration via Ollama
- API-based LLM integration (Claude, OpenAI)
"""

from src.llm.api_llm import ClaudeAPI, OpenAIAPI
from src.llm.local_llm import OllamaLLM

__all__ = ["OllamaLLM", "ClaudeAPI", "OpenAIAPI"]
