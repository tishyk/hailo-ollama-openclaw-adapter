"""Hailo-Ollama to OpenClaw adapter.

A FastAPI adapter that bridges Hailo-Ollama (running on Hailo AI
accelerators) with OpenClaw by exposing OpenAI- and Ollama-compatible HTTP
endpoints and working around Hailo 5.3.0 prompt-renderer quirks.
"""

from hailo_ollama_adapter.adapter import app

__version__ = "2.0.0"
__all__ = ["app"]
