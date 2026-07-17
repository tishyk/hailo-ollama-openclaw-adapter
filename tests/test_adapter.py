"""Correctness tests for model discovery and upstream HTTP errors."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from hailo_ollama_adapter import adapter


class StubResponse:
    """Provide the minimal successful response interface used by list probes."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class RecordingAsyncClient:
    """Record model-list URLs and return a configurable native catalog."""

    requested_urls: list[str] = []
    response_payload: Any = {
        "models": [
            {"name": "qwen3:1.7b"},
            {"name": "qwen2.5-coder:1.5b"},
        ]
    }

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> RecordingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def get(self, url: str, **_kwargs: Any) -> StubResponse:
        type(self).requested_urls.append(url)
        return StubResponse(type(self).response_payload)


class ErrorPostingAsyncClient:
    """Return a native non-success response for chat error propagation tests."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> ErrorPostingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def post(self, url: str, **_kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            404,
            request=request,
            json={"error": "model 'missing' not found"},
        )


def upstream_status_error(
    status_code: int,
    payload: dict[str, str],
) -> httpx.HTTPStatusError:
    """Build a realistic Hailo status exception for endpoint tests."""
    request = httpx.Request("POST", adapter.HAILO_URL)
    response = httpx.Response(status_code, request=request, json=payload)
    return httpx.HTTPStatusError(
        f"upstream returned {status_code}",
        request=request,
        response=response,
    )


@pytest.fixture(autouse=True)
def reset_recording_client() -> None:
    """Clear request history before each test."""
    RecordingAsyncClient.requested_urls.clear()


def test_flatten_newlines_preserves_words_without_literal_line_breaks() -> None:
    assert adapter._flatten_newlines("alpha\r\nbeta\ngamma\rdelta") == (
        "alpha beta gamma delta"
    )


@pytest.mark.asyncio
async def test_refresh_uses_native_api_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter.httpx, "AsyncClient", RecordingAsyncClient)

    models = await adapter._refresh_models_from_hailo()

    assert RecordingAsyncClient.requested_urls == ["http://127.0.0.1:8000/api/tags"]
    assert [model["name"] for model in models] == [
        "qwen3:1.7b",
        "qwen2.5-coder:1.5b",
    ]


@pytest.mark.asyncio
async def test_post_hailo_rejects_non_success_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter.httpx, "AsyncClient", ErrorPostingAsyncClient)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await adapter._post_hailo(b"{}")

    assert exc_info.value.response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
async def test_chat_endpoint_preserves_upstream_status(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_post(_body: bytes) -> dict[str, Any]:
        """Simulate a non-streaming upstream status failure."""
        raise upstream_status_error(404, {"error": "model 'missing' not found"})

    monkeypatch.setattr(adapter, "_post_hailo", fail_post)
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 404
    assert response.json() == {"error": "model 'missing' not found"}
