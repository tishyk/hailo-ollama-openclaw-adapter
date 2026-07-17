"""Shared fixtures for adapter tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from hailo_ollama_adapter import adapter


@pytest.fixture(autouse=True)
def reset_model_cache() -> Generator[None, None, None]:
    """Keep model-cache state isolated between tests."""
    adapter._model_cache.clear()
    yield
    adapter._model_cache.clear()
