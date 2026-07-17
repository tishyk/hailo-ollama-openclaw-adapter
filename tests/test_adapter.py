"""Correctness and lifecycle tests for the Hailo adapter."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
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


class ConcurrencyRecordingAsyncClient:
    """Measure concurrent non-streaming calls to assert single-flight behavior."""

    active = 0
    max_active = 0

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> ConcurrencyRecordingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def post(self, url: str, **_kwargs: Any) -> httpx.Response:
        type(self).active += 1
        type(self).max_active = max(type(self).max_active, type(self).active)
        try:
            await asyncio.sleep(0.02)
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"message": {"role": "assistant", "content": "ok"}},
            )
        finally:
            type(self).active -= 1


class CancellationRecordingAsyncClient:
    """Model backend work that continues after downstream cancellation."""

    active = 0
    max_active = 0
    calls = 0
    started = asyncio.Event()
    backend_tasks: set[asyncio.Task[None]] = set()

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> CancellationRecordingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def post(self, url: str, **_kwargs: Any) -> httpx.Response:
        type(self).calls += 1
        type(self).active += 1
        type(self).max_active = max(type(self).max_active, type(self).active)
        type(self).started.set()
        done = asyncio.Event()

        async def finish_backend() -> None:
            await asyncio.sleep(0.05)
            type(self).active -= 1
            done.set()

        task = asyncio.create_task(finish_backend())
        type(self).backend_tasks.add(task)
        task.add_done_callback(type(self).backend_tasks.discard)
        await done.wait()
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"message": {"role": "assistant", "content": "ok"}},
        )


class TimeoutThenSuccessAsyncClient:
    """Raise one ambiguous read timeout, then expose an unsafe retry."""

    calls = 0

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> TimeoutThenSuccessAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def post(self, url: str, **_kwargs: Any) -> httpx.Response:
        type(self).calls += 1
        request = httpx.Request("POST", url)
        if type(self).calls == 1:
            raise httpx.ReadTimeout("ambiguous timeout", request=request)
        return httpx.Response(
            200,
            request=request,
            json={"message": {"role": "assistant", "content": "unsafe overlap"}},
        )


class ConnectThenSuccessAsyncClient:
    """Raise one pre-accept connect failure, then recover successfully."""

    calls = 0

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> ConnectThenSuccessAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def post(self, url: str, **_kwargs: Any) -> httpx.Response:
        type(self).calls += 1
        request = httpx.Request("POST", url)
        if type(self).calls == 1:
            raise httpx.ConnectError("not listening", request=request)
        return httpx.Response(
            200,
            request=request,
            json={"message": {"role": "assistant", "content": "recovered"}},
        )


class CancellationResistantStreamResponse:
    """Keep upstream generation active after the downstream consumer stops."""

    def __init__(self, owner: type[CancellationStreamingAsyncClient]) -> None:
        self.owner = owner
        self.done = asyncio.Event()

    async def __aenter__(self) -> CancellationResistantStreamResponse:
        self.owner.active += 1
        self.owner.max_active = max(self.owner.max_active, self.owner.active)
        self.owner.started.set()

        async def finish_backend() -> None:
            await asyncio.sleep(0.05)
            self.owner.active -= 1
            self.done.set()

        task = asyncio.create_task(finish_backend())
        self.owner.backend_tasks.add(task)
        task.add_done_callback(self.owner.backend_tasks.discard)
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> Any:
        await self.done.wait()
        yield json.dumps(
            {"message": {"role": "assistant", "content": "stream ok"}, "done": True}
        )


class CancellationStreamingAsyncClient:
    """Return cancellation-resistant streams while recording overlap."""

    active = 0
    max_active = 0
    calls = 0
    started = asyncio.Event()
    backend_tasks: set[asyncio.Task[None]] = set()

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> CancellationStreamingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def stream(
        self, *_args: Any, **_kwargs: Any
    ) -> CancellationResistantStreamResponse:
        type(self).calls += 1
        return CancellationResistantStreamResponse(type(self))


class BurstStreamResponse:
    """Emit a fast finite stream to exercise bounded downstream buffering."""

    def __init__(self, owner: type[BurstStreamingAsyncClient]) -> None:
        self.owner = owner

    async def __aenter__(self) -> BurstStreamResponse:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        self.owner.finished.set()

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> Any:
        for index in range(100):
            await asyncio.sleep(0)
            yield json.dumps(
                {
                    "message": {"role": "assistant", "content": str(index)},
                    "done": index == 99,
                }
            )


class BurstStreamingAsyncClient:
    """Return burst streams and signal when upstream resources close."""

    finished = asyncio.Event()

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> BurstStreamingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def stream(self, *_args: Any, **_kwargs: Any) -> BurstStreamResponse:
        return BurstStreamResponse(type(self))


class ErrorStreamContext:
    """Expose an upstream status error during stream startup."""

    def __init__(self, status_code: int) -> None:
        request = httpx.Request("POST", adapter.HAILO_URL)
        self.response = httpx.Response(
            status_code,
            request=request,
            json={"error": "stream model unavailable"},
        )

    async def __aenter__(self) -> httpx.Response:
        return self.response

    async def __aexit__(self, *_args: Any) -> None:
        return None


class ErrorStreamingAsyncClient:
    """Return a stream whose status fails before downstream HTTP commit."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> ErrorStreamingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def stream(self, *_args: Any, **_kwargs: Any) -> ErrorStreamContext:
        return ErrorStreamContext(429)


class MidStreamFailureResponse:
    """Yield one partial chunk, then raise a transport read error."""

    async def __aenter__(self) -> MidStreamFailureResponse:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> Any:
        yield json.dumps(
            {
                "message": {"role": "assistant", "content": "partial"},
                "done": False,
            }
        )
        request = httpx.Request("POST", adapter.HAILO_URL)
        raise httpx.ReadError("truncated stream", request=request)


class MidStreamFailureAsyncClient:
    """Return a stream that fails after yielding partial content."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> MidStreamFailureAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def stream(self, *_args: Any, **_kwargs: Any) -> MidStreamFailureResponse:
        return MidStreamFailureResponse()


class DoneThenFailureStreamResponse:
    """Complete logically, then fail during iterator or context teardown."""

    async def __aenter__(self) -> DoneThenFailureStreamResponse:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        raise RuntimeError("teardown after logical completion")

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> Any:
        yield json.dumps(
            {
                "message": {"role": "assistant", "content": "complete"},
                "done": True,
            }
        )
        request = httpx.Request("POST", adapter.HAILO_URL)
        raise httpx.ReadError("teardown after done", request=request)


class DoneThenFailureStreamingAsyncClient:
    """Return a stream whose teardown fails after authoritative completion."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> DoneThenFailureStreamingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def stream(self, *_args: Any, **_kwargs: Any) -> DoneThenFailureStreamResponse:
        return DoneThenFailureStreamResponse()


class EofBeforeDoneStreamResponse:
    """End cleanly after a partial chunk without an authoritative done marker."""

    async def __aenter__(self) -> EofBeforeDoneStreamResponse:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> Any:
        yield json.dumps(
            {
                "message": {"role": "assistant", "content": "partial"},
                "done": False,
            }
        )


class EofBeforeDoneStreamingAsyncClient:
    """Return a stream that reaches EOF before authoritative completion."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> EofBeforeDoneStreamingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def stream(self, *_args: Any, **_kwargs: Any) -> EofBeforeDoneStreamResponse:
        return EofBeforeDoneStreamResponse()


class MalformedDoneStreamResponse:
    """Emit a caller-selected non-Boolean completion value."""

    def __init__(self, done_value: Any) -> None:
        self.done_value = done_value

    async def __aenter__(self) -> MalformedDoneStreamResponse:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> Any:
        yield json.dumps(
            {
                "message": {"role": "assistant", "content": "partial"},
                "done": self.done_value,
            }
        )


class MalformedDoneStreamingAsyncClient:
    """Return streams with configurable malformed completion markers."""

    done_value: Any = None

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> MalformedDoneStreamingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def stream(self, *_args: Any, **_kwargs: Any) -> MalformedDoneStreamResponse:
        return MalformedDoneStreamResponse(type(self).done_value)


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
    """Reset fake-client counters, events, and adapter lifecycle state."""
    RecordingAsyncClient.requested_urls.clear()
    ConcurrencyRecordingAsyncClient.active = 0
    ConcurrencyRecordingAsyncClient.max_active = 0
    CancellationRecordingAsyncClient.active = 0
    CancellationRecordingAsyncClient.max_active = 0
    CancellationRecordingAsyncClient.calls = 0
    CancellationRecordingAsyncClient.started = asyncio.Event()
    CancellationRecordingAsyncClient.backend_tasks.clear()
    TimeoutThenSuccessAsyncClient.calls = 0
    ConnectThenSuccessAsyncClient.calls = 0
    CancellationStreamingAsyncClient.active = 0
    CancellationStreamingAsyncClient.max_active = 0
    CancellationStreamingAsyncClient.calls = 0
    CancellationStreamingAsyncClient.started = asyncio.Event()
    CancellationStreamingAsyncClient.backend_tasks.clear()
    BurstStreamingAsyncClient.finished = asyncio.Event()
    adapter._hailo_semaphore = asyncio.Semaphore(adapter.MAX_CONCURRENT_HAILO_CALLS)
    adapter._hailo_quarantined = False
    adapter._background_hailo_tasks = set()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
@pytest.mark.parametrize("stream", [False, True])
async def test_quarantine_returns_service_unavailable(path: str, stream: bool) -> None:
    adapter._hailo_quarantined = True
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "qwen3:1.7b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )

    assert response.status_code == 503
    assert "quarantined after an ambiguous" in response.json()["error"]


@pytest.mark.asyncio
async def test_hailo_generation_is_single_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        ConcurrencyRecordingAsyncClient,
    )
    monkeypatch.setattr(
        adapter,
        "_hailo_semaphore",
        asyncio.Semaphore(adapter.MAX_CONCURRENT_HAILO_CALLS),
    )

    await asyncio.gather(adapter._post_hailo(b"{}"), adapter._post_hailo(b"{}"))

    assert ConcurrencyRecordingAsyncClient.max_active == 1


@pytest.mark.asyncio
async def test_cancelled_post_holds_slot_until_backend_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        CancellationRecordingAsyncClient,
    )

    first = asyncio.create_task(adapter._post_hailo(b"{}"))
    await CancellationRecordingAsyncClient.started.wait()
    first.cancel()
    with suppress(asyncio.CancelledError):
        await first

    await adapter._post_hailo(b"{}")

    assert CancellationRecordingAsyncClient.max_active == 1


@pytest.mark.asyncio
async def test_cancelled_queued_post_never_reaches_hailo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        CancellationRecordingAsyncClient,
    )

    first = asyncio.create_task(adapter._post_hailo(b"first"))
    await CancellationRecordingAsyncClient.started.wait()
    queued = asyncio.create_task(adapter._post_hailo(b"queued"))
    await asyncio.sleep(0)
    queued.cancel()
    with suppress(asyncio.CancelledError):
        await queued
    await first
    await asyncio.sleep(0.01)

    assert CancellationRecordingAsyncClient.calls == 1


@pytest.mark.asyncio
async def test_ambiguous_timeout_quarantines_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        TimeoutThenSuccessAsyncClient,
    )

    with pytest.raises(httpx.ReadTimeout):
        await adapter._post_hailo(b"{}")
    with pytest.raises(RuntimeError, match="quarantined after an ambiguous"):
        await adapter._post_hailo(b"{}")

    assert TimeoutThenSuccessAsyncClient.calls == 1


@pytest.mark.asyncio
async def test_connect_failure_does_not_quarantine_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        ConnectThenSuccessAsyncClient,
    )

    with pytest.raises(httpx.ConnectError):
        await adapter._post_hailo(b"{}")
    response = await adapter._post_hailo(b"{}")

    assert response["message"]["content"] == "recovered"
    assert ConnectThenSuccessAsyncClient.calls == 2


@pytest.mark.asyncio
async def test_cancelled_stream_holds_slot_until_backend_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        CancellationStreamingAsyncClient,
    )

    async def consume_first_chunk() -> dict[str, Any]:
        async for chunk in adapter._stream_hailo_lines(b"{}"):
            return chunk
        raise AssertionError("stream ended without a chunk")

    first = asyncio.create_task(consume_first_chunk())
    await CancellationStreamingAsyncClient.started.wait()
    first.cancel()
    with suppress(asyncio.CancelledError):
        await first

    second_chunks = [chunk async for chunk in adapter._stream_hailo_lines(b"{}")]

    assert second_chunks[0]["message"]["content"] == "stream ok"
    assert CancellationStreamingAsyncClient.max_active == 1


@pytest.mark.asyncio
async def test_cancelled_queued_stream_never_reaches_hailo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        CancellationStreamingAsyncClient,
    )

    async def consume(body: bytes) -> list[dict[str, Any]]:
        return [chunk async for chunk in adapter._stream_hailo_lines(body)]

    first = asyncio.create_task(consume(b"first"))
    await CancellationStreamingAsyncClient.started.wait()
    queued = asyncio.create_task(consume(b"queued"))
    await asyncio.sleep(0)
    queued.cancel()
    with suppress(asyncio.CancelledError):
        await queued
    await first
    await asyncio.sleep(0.01)

    assert CancellationStreamingAsyncClient.calls == 1


@pytest.mark.asyncio
async def test_disconnected_stream_uses_bounded_queue_and_finishes_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter.httpx, "AsyncClient", BurstStreamingAsyncClient)
    queues: list[asyncio.Queue[object]] = []
    max_observed = 0

    class RecordingQueue(asyncio.Queue[object]):
        async def put(self, item: object) -> None:
            nonlocal max_observed
            await super().put(item)
            max_observed = max(max_observed, self.qsize())

        def put_nowait(self, item: object) -> None:
            nonlocal max_observed
            super().put_nowait(item)
            max_observed = max(max_observed, self.qsize())

    def queue_factory(maxsize: int = 0) -> asyncio.Queue[object]:
        queue = RecordingQueue(maxsize)
        queues.append(queue)
        return queue

    monkeypatch.setattr(adapter.asyncio, "Queue", queue_factory)
    stream: Any = adapter._stream_hailo_lines(b"{}")
    await anext(stream)
    await stream.aclose()
    await asyncio.wait_for(BurstStreamingAsyncClient.finished.wait(), timeout=1)

    assert queues[0].maxsize == adapter.MAX_STREAM_QUEUE_CHUNKS
    assert max_observed <= adapter.MAX_STREAM_QUEUE_CHUNKS


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
async def test_streaming_status_is_known_before_response_commit(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(adapter.httpx, "AsyncClient", ErrorStreamingAsyncClient)
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with real_async_client(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 429
    assert response.json() == {"error": "stream model unavailable"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "normal_terminal"),
    [
        ("/v1/chat/completions", "data: [DONE]"),
        ("/api/chat", '"done": true'),
    ],
)
async def test_midstream_failure_emits_error_without_successful_completion(
    path: str,
    normal_terminal: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(adapter.httpx, "AsyncClient", MidStreamFailureAsyncClient)
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with real_async_client(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "qwen3:1.7b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "Hailo upstream stream failed" in response.text
    assert normal_terminal not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "normal_terminal"),
    [
        ("/v1/chat/completions", "data: [DONE]"),
        ("/api/chat", '"done": true'),
    ],
)
async def test_done_marker_is_authoritative_during_transport_teardown(
    path: str,
    normal_terminal: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        DoneThenFailureStreamingAsyncClient,
    )
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with real_async_client(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "qwen3:1.7b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert normal_terminal in response.text
    assert "Hailo upstream stream failed" not in response.text
    assert adapter._hailo_quarantined is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "normal_terminal"),
    [
        ("/v1/chat/completions", "data: [DONE]"),
        ("/api/chat", '"done": true'),
    ],
)
async def test_eof_before_done_emits_error_without_successful_completion(
    path: str,
    normal_terminal: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        EofBeforeDoneStreamingAsyncClient,
    )
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with real_async_client(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "qwen3:1.7b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "Hailo upstream stream failed" in response.text
    assert normal_terminal not in response.text
    assert adapter._hailo_quarantined is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "successful_terminal"),
    [
        ("/v1/chat/completions", '"finish_reason": "stop"'),
        ("/api/chat", '"done": true'),
    ],
)
@pytest.mark.parametrize("malformed_done", [1, "false"])
async def test_non_boolean_done_never_emits_successful_completion(
    path: str,
    successful_terminal: str,
    malformed_done: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_async_client = httpx.AsyncClient
    MalformedDoneStreamingAsyncClient.done_value = malformed_done
    monkeypatch.setattr(
        adapter.httpx,
        "AsyncClient",
        MalformedDoneStreamingAsyncClient,
    )
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with real_async_client(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "qwen3:1.7b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "Hailo upstream stream failed" in response.text
    assert successful_terminal not in response.text
    assert adapter._hailo_quarantined is True


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
async def test_queued_stream_observes_quarantine_before_response_commit(
    path: str,
) -> None:
    await adapter._hailo_semaphore.acquire()
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        request = asyncio.create_task(
            client.post(
                path,
                json={
                    "model": "qwen3:1.7b",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
        )
        await asyncio.sleep(0.01)
        adapter._hailo_quarantined = True
        adapter._hailo_semaphore.release()
        response = await request

    assert response.status_code == 503
