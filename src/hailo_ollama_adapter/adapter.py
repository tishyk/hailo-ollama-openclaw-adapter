"""Hailo-to-OpenAI/Ollama API adapter.

A FastAPI adapter that exposes OpenAI- and Ollama-compatible HTTP endpoints
while proxying requests to a local Hailo 5.3.0 inference server.

Works around Hailo 5.3.0 prompt-renderer quirks: control-character rejection,
newline-in-content rejection, and system-role-on-continuation rejection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HAILO_DEFAULT_MODEL = "qwen3:1.7b"
HAILO_URL = "http://127.0.0.1:8000/api/chat"
HAILO_LIST_URL = "http://127.0.0.1:8000/api/tags"
HAILO_HEADERS = {"Content-Type": "application/json"}

REQUEST_TIMEOUT = 180.0
LIST_TIMEOUT = 5.0
STARTUP_RETRY_ATTEMPTS = 10
STARTUP_RETRY_DELAY = 2.0
MAX_USER_CONTENT_CHARS = 2000
MAX_EXTRACTED_INTENT_CHARS = 500
MAX_HISTORY_TURNS = 12
# Hailo-Ollama exposes one generation slot; queue chat calls in the adapter.
MAX_CONCURRENT_HAILO_CALLS = 1
MAX_UPSTREAM_ERROR_CHARS = 500
MAX_STREAM_QUEUE_CHUNKS = 16

FULL_TOOLING = """Tools available for this request:
- read: Read file contents
- write: Create or overwrite files
- exec: Run shell commands (PTY available)
- web_search: Search the web (Brave API)
- You need to add your tools here"""

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_OPENCLAW_INTENT_RE = re.compile(r"\]\s*([^\n\r]+?)\s*$")

logger = logging.getLogger(__name__)
_hailo_semaphore = asyncio.Semaphore(MAX_CONCURRENT_HAILO_CALLS)
_hailo_quarantined = False
_background_hailo_tasks: set[asyncio.Task[Any]] = set()
_STREAM_END = object()
_STREAM_FAILED = object()


class HailoQuarantinedError(RuntimeError):
    """Raised when chat is rejected because Hailo execution state is uncertain."""


def _track_hailo_task(task: asyncio.Task[Any]) -> None:
    """Keep detached hardware workers alive and consume unobserved exceptions."""
    _background_hailo_tasks.add(task)

    def finished(completed: asyncio.Task[Any]) -> None:
        """Forget a terminal worker after retrieving any stored exception."""
        _background_hailo_tasks.discard(completed)
        if not completed.cancelled():
            completed.exception()

    task.add_done_callback(finished)


def _ensure_hailo_available() -> None:
    """Reject new chat work while the backend state is quarantined."""
    if _hailo_quarantined:
        raise HailoQuarantinedError(
            "Hailo adapter is quarantined after an ambiguous in-flight request "
            "failure; confirm the backend is idle, then restart the adapter"
        )


def _is_ambiguous_transport_error(exc: httpx.RequestError) -> bool:
    """Return whether Hailo may have accepted work before the failure."""
    return not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))


def _quarantine_hailo(reason: str) -> None:
    """Block later chat work after an ambiguous in-flight failure."""
    global _hailo_quarantined
    if _hailo_quarantined:
        return
    _hailo_quarantined = True
    logger.error(
        "Quarantining Hailo chat traffic after ambiguous in-flight failure: reason=%s",
        reason,
    )


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Kick off a background retry loop that populates the model cache.

    The adapter starts serving immediately; while the first request may
    briefly see the fallback model list, a background task keeps retrying
    until Hailo becomes reachable and the real list is loaded.
    """
    task = asyncio.create_task(_preload_models_with_retry())
    try:
        yield
    finally:
        task.cancel()


async def _preload_models_with_retry() -> None:
    """Populate the model cache after bounded startup retries."""
    for attempt in range(1, STARTUP_RETRY_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(HAILO_LIST_URL, timeout=LIST_TIMEOUT)
                response.raise_for_status()
            models = _extract_model_list(response.json())
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError):
            if attempt < STARTUP_RETRY_ATTEMPTS:
                logger.info(
                    "Hailo not ready yet (attempt %d/%d), retrying in %.1fs",
                    attempt, STARTUP_RETRY_ATTEMPTS, STARTUP_RETRY_DELAY,
                )
                await asyncio.sleep(STARTUP_RETRY_DELAY)
                continue
            logger.warning(
                "Hailo-Ollama still unreachable after %d attempts. "
                "Check that the server is running; default %s will be used",
                STARTUP_RETRY_ATTEMPTS,
                HAILO_DEFAULT_MODEL,
            )
            models = [_DEFAULT_MODEL_INFO]
        else:
            models = models or [_DEFAULT_MODEL_INFO]

        _model_cache.clear()
        _model_cache.extend(models)
        logger.info("Loaded %d Hailo model(s)", len(_model_cache))
        return


app = FastAPI(title="Hailo Adapter", version="1.0.0", lifespan=_lifespan)


# --------------------------------------------------------------------------- #
# Text sanitization
# --------------------------------------------------------------------------- #

def _sanitize(text: str) -> str:
    """Strip ASCII control chars that Hailo's parser rejects."""
    return _CONTROL_CHAR_RE.sub("", text)


def _flatten_newlines(text: str) -> str:
    """Collapse newlines to spaces.

    Hailo 5.3.0's prompt renderer re-encodes content through an internal
    template that doesn't escape newlines, so any literal newline -- even
    when the outer JSON correctly escapes it -- causes a parse error.
    """
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _deep_sanitize(obj: Any) -> Any:
    if isinstance(obj, str):
        return _sanitize(obj)
    if isinstance(obj, dict):
        return {k: _deep_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_sanitize(item) for item in obj]
    return obj


def _encode_for_hailo(payload: dict) -> bytes:
    return json.dumps(_deep_sanitize(payload), ensure_ascii=True).encode("utf-8")


# --------------------------------------------------------------------------- #
# Message helpers
# --------------------------------------------------------------------------- #

def build_system_message(request_requires_tools: bool = False) -> dict:
    base = "You are a personal assistant running inside OpenClaw. Use short answers"
    if request_requires_tools:
        base += "\n" + FULL_TOOLING
    return {"role": "system", "content": _sanitize(base)}


def _extract_text(content: Any) -> str:
    """Coerce OpenAI-style content (string or list of parts) to plain text."""
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return content if isinstance(content, str) else str(content)


def normalize_messages(messages: list[dict]) -> list[dict]:
    return [
        {
            "role": m.get("role", "user"),
            "content": _sanitize(_extract_text(m.get("content", ""))),
        }
        for m in messages
    ]


def _extract_user_intent(raw: str) -> str:
    """Pull the trailing user intent from an OpenClaw envelope, else flatten."""
    match = _OPENCLAW_INTENT_RE.search(raw)
    if match:
        intent = match.group(1).strip()
        if intent and len(intent) < MAX_EXTRACTED_INTENT_CHARS:
            return intent
    return _flatten_newlines(raw)


def assemble_messages_for_hailo(incoming: list[dict]) -> list[dict]:
    """Prepare messages for Hailo, working around known 5.3.0 quirks.

    Keeps up to ``MAX_HISTORY_TURNS`` recent user/assistant turns so the
    model has conversational context. Drops empty placeholders, flattens
    newlines, extracts the actual user intent from the latest user turn,
    and never emits a system role (Hailo rejects that on continuations).
    """
    turns = [
        m for m in incoming
        if m.get("role") in ("user", "assistant")
        and m.get("content", "").strip()
    ]
    if not turns:
        return [{"role": "user", "content": "hello"}]

    recent = turns[-MAX_HISTORY_TURNS:]

    # Conversations should start on a user turn
    while recent and recent[0].get("role") != "user":
        recent = recent[1:]
    if not recent:
        recent = [turns[-1]]

    assembled = []
    last_index = len(recent) - 1
    for i, message in enumerate(recent):
        raw = message.get("content", "")
        is_latest_user = i == last_index and message.get("role") == "user"
        content = (
            _extract_user_intent(raw) if is_latest_user else _flatten_newlines(raw)
        )
        assembled.append({
            "role": message["role"],
            "content": content[:MAX_USER_CONTENT_CHARS],
        })
    return assembled


# --------------------------------------------------------------------------- #
# Response formatting
# --------------------------------------------------------------------------- #

def to_openai_chunk(
    content: str,
    model: str,
    finish_reason: str | None = None,
    is_meta: bool = False,
) -> str:
    now = int(time.time())
    delta = (
        {"role": "assistant", "content": content}
        if is_meta
        else {"content": content}
    )
    chunk = {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _openai_full_response(content: str, model: str) -> dict:
    now = int(time.time())
    return {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


def _ollama_full_response(content: str, model: str) -> dict:
    return {
        "model": model,
        "created_at": f"{int(time.time())}",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


# --------------------------------------------------------------------------- #
# Hailo client
# --------------------------------------------------------------------------- #


def _upstream_error_detail(response: httpx.Response) -> str:
    """Extract a bounded explicit error without reflecting arbitrary bodies."""
    detail: Any = None
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("error") or payload.get("detail")
        if isinstance(detail, dict):
            detail = detail.get("message")

    if not isinstance(detail, str) or not detail.strip():
        return f"Hailo upstream returned HTTP {response.status_code}"

    return _flatten_newlines(_sanitize(detail)).strip()[:MAX_UPSTREAM_ERROR_CHARS]


def _upstream_error_response(exc: httpx.HTTPStatusError) -> JSONResponse:
    """Preserve Hailo's status with a bounded downstream error response."""
    status = exc.response.status_code
    logger.warning("Hailo upstream rejected chat request: status=%d", status)
    return JSONResponse(
        status_code=status,
        content={"error": _upstream_error_detail(exc.response)},
    )


def _build_payload(
    request_data: dict,
    default_stream: bool,
) -> tuple[bytes, bool, str]:
    is_stream = request_data.get("stream", default_stream)
    model_name = request_data.get("model") or HAILO_DEFAULT_MODEL
    messages = assemble_messages_for_hailo(
        normalize_messages(request_data.get("messages", []))
    )
    body = _encode_for_hailo({
        "model": model_name,
        "messages": messages,
        "stream": is_stream,
    })
    return body, is_stream, model_name


async def _acquire_hailo_slot() -> None:
    """Acquire the chat slot and recheck quarantine after queueing."""
    _ensure_hailo_available()
    await _hailo_semaphore.acquire()
    try:
        _ensure_hailo_available()
    except BaseException:
        _hailo_semaphore.release()
        raise


async def _post_hailo_request(body: bytes) -> dict:
    """Send one non-streaming request inside an already-owned chat slot."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            HAILO_URL,
            content=body,
            headers=HAILO_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    return response.json()


async def _post_hailo_worker(body: bytes) -> dict:
    """Run the local request, quarantining uncertainty before releasing the slot."""
    try:
        try:
            return await asyncio.wait_for(
                _post_hailo_request(body),
                timeout=REQUEST_TIMEOUT,
            )
        except httpx.RequestError as exc:
            if _is_ambiguous_transport_error(exc):
                _quarantine_hailo(type(exc).__name__)
            raise
        except asyncio.TimeoutError:
            _quarantine_hailo("request_deadline")
            raise
        except asyncio.CancelledError:
            _quarantine_hailo("request_worker_cancelled")
            raise
    finally:
        _hailo_semaphore.release()


async def _post_hailo(body: bytes) -> dict:
    """Acquire the chat slot and shield its worker from caller cancellation."""
    await _acquire_hailo_slot()
    task = asyncio.create_task(_post_hailo_worker(body))
    _track_hailo_task(task)
    return await asyncio.shield(task)


async def _publish_stream_item(
    queue: asyncio.Queue[object],
    consumer_done: asyncio.Event,
    item: object,
) -> bool:
    """Publish with bounded backpressure, or stop after downstream disconnect."""
    if consumer_done.is_set():
        return False

    put_task = asyncio.create_task(queue.put(item))
    done_task = asyncio.create_task(consumer_done.wait())
    try:
        done, _pending = await asyncio.wait(
            {put_task, done_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if done_task in done:
            return False
        await put_task
        return True
    finally:
        for task in (put_task, done_task):
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def _drain_hailo_lines(
    response: Any,
    queue: asyncio.Queue[object],
    consumer_done: asyncio.Event,
) -> bool:
    """Drain until authoritative completion or EOF, discarding after disconnect."""
    deliver = True
    async for line in response.aiter_lines():
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if deliver:
            deliver = await _publish_stream_item(queue, consumer_done, item)
        if isinstance(item, dict) and item.get("done") is True:
            return True
    return False


async def _pump_hailo_lines(
    response: Any,
    resources: AsyncExitStack,
    queue: asyncio.Queue[object],
    consumer_done: asyncio.Event,
) -> None:
    """Drain one accepted Hailo stream while retaining the hardware slot."""
    failed = False
    completed = False
    try:
        completed = await asyncio.wait_for(
            _drain_hailo_lines(response, queue, consumer_done),
            timeout=REQUEST_TIMEOUT,
        )
        if not completed:
            failed = True
            _quarantine_hailo("stream_eof_before_done")
            logger.warning("Hailo stream ended before its completion marker")
    except httpx.RequestError as exc:
        failed = True
        if _is_ambiguous_transport_error(exc):
            _quarantine_hailo(type(exc).__name__)
        logger.warning(
            "Hailo stream ended early: error_class=%s",
            type(exc).__name__,
        )
    except asyncio.TimeoutError:
        failed = True
        _quarantine_hailo("stream_deadline")
        logger.warning("Hailo stream exceeded its total deadline")
    except asyncio.CancelledError:
        failed = True
        _quarantine_hailo("stream_worker_cancelled")
        raise
    except Exception:
        failed = True
        _quarantine_hailo("stream_worker_failure")
        logger.exception("Unexpected error streaming from Hailo")
    finally:
        try:
            await _close_hailo_stream_resources(
                resources,
                quarantine_on_failure=not completed,
            )
        finally:
            _hailo_semaphore.release()
        if not consumer_done.is_set():
            terminal = _STREAM_FAILED if failed else _STREAM_END
            await _publish_stream_item(queue, consumer_done, terminal)


async def _consume_hailo_queue(
    queue: asyncio.Queue[object],
    consumer_done: asyncio.Event,
) -> AsyncIterator[object]:
    """Yield queued stream items and signal when the downstream consumer stops."""
    try:
        while True:
            item = await queue.get()
            if item is _STREAM_END:
                return
            yield item
            if item is _STREAM_FAILED:
                return
    finally:
        consumer_done.set()


async def _close_hailo_stream_resources(
    resources: AsyncExitStack,
    *,
    quarantine_on_failure: bool = True,
) -> None:
    """Close upstream resources and quarantine ambiguous cleanup failures."""
    try:
        await resources.aclose()
    except Exception as exc:
        if quarantine_on_failure:
            _quarantine_hailo("stream_close_failure")
        logger.error(
            "Failed to close Hailo stream resources: error_class=%s",
            type(exc).__name__,
        )


async def _start_hailo_stream(body: bytes) -> AsyncIterator[object]:
    """Acquire the slot and validate upstream status before committing HTTP 200."""
    await _acquire_hailo_slot()
    resources = AsyncExitStack()
    try:
        client = await resources.enter_async_context(httpx.AsyncClient())
        response = await resources.enter_async_context(
            client.stream(
                "POST",
                HAILO_URL,
                content=body,
                headers=HAILO_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
        )
        response.raise_for_status()
    except httpx.RequestError as exc:
        if _is_ambiguous_transport_error(exc):
            _quarantine_hailo(type(exc).__name__)
        try:
            await _close_hailo_stream_resources(resources)
        finally:
            _hailo_semaphore.release()
        raise
    except asyncio.CancelledError:
        _quarantine_hailo("stream_start_cancelled")
        try:
            await _close_hailo_stream_resources(resources)
        finally:
            _hailo_semaphore.release()
        raise
    except BaseException:
        try:
            await _close_hailo_stream_resources(resources)
        finally:
            _hailo_semaphore.release()
        raise

    queue: asyncio.Queue[object] = asyncio.Queue(MAX_STREAM_QUEUE_CHUNKS)
    consumer_done = asyncio.Event()
    task = asyncio.create_task(
        _pump_hailo_lines(response, resources, queue, consumer_done)
    )
    _track_hailo_task(task)
    return _consume_hailo_queue(queue, consumer_done)


async def _stream_hailo_lines(body: bytes) -> AsyncIterator[object]:
    """Stream parsed Hailo objects while a detached worker owns the slot.

    If the downstream client disconnects, the worker keeps draining Hailo so a
    second request cannot overlap generation on the single hardware slot.
    """
    stream = await _start_hailo_stream(body)
    try:
        async for item in stream:
            yield item
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


def _extract_content(hailo_json: dict) -> str:
    return hailo_json.get("message", {}).get("content", "")


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #

_DEFAULT_MODEL_INFO = {
    "name": HAILO_DEFAULT_MODEL,
    "model": HAILO_DEFAULT_MODEL,
    "modified_at": "2026-04-22T00:00:00.000Z",
    "size": 0,
    "digest": "",
    "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "qwen3",
        "families": ["qwen3"],
        "parameter_size": "1.7B",
        "quantization_level": "",
    },
}

_model_cache: list[dict] = []


def _infer_family(name: str) -> str:
    return name.split(":", 1)[0] if ":" in name else name


def _normalize_model_info(entry: Any) -> dict | None:
    """Convert a Hailo list entry into an Ollama-style model dict."""
    if isinstance(entry, str):
        name = entry
        details: dict[str, Any] = {}
    elif isinstance(entry, dict):
        name = entry.get("name") or entry.get("model") or entry.get("id")
        details = entry
    else:
        return None

    if not name:
        return None

    family = details.get("family") or _infer_family(name)
    return {
        "name": name,
        "model": name,
        "modified_at": details.get("modified_at", "2026-01-01T00:00:00.000Z"),
        "size": details.get("size", 0),
        "digest": details.get("digest", ""),
        "details": {
            "parent_model": details.get("parent_model", ""),
            "format": details.get("format", "gguf"),
            "family": family,
            "families": details.get("families", [family]),
            "parameter_size": details.get("parameter_size", ""),
            "quantization_level": details.get("quantization_level", ""),
        },
    }


def _extract_model_list(raw: Any) -> list[dict]:
    """Accept several response shapes from Hailo's list endpoint."""
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = (
            raw.get("models")
            or raw.get("available")
            or raw.get("pulled")
            or raw.get("items")
            or []
        )
    else:
        entries = []

    models = [_normalize_model_info(e) for e in entries]
    return [m for m in models if m]


async def _refresh_models_from_hailo() -> list[dict]:
    """Query Hailo for its current model list; fall back on failure."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(HAILO_LIST_URL, timeout=LIST_TIMEOUT)
            response.raise_for_status()
        models = _extract_model_list(response.json())
    except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
        logger.warning("Hailo model list unavailable, using fallback: %s", exc)
        return [_DEFAULT_MODEL_INFO]

    return models or [_DEFAULT_MODEL_INFO]


async def _get_models() -> list[dict]:
    """Return the cached model list, populating it on first call."""
    if not _model_cache:
        _model_cache.extend(await _refresh_models_from_hailo())
    return _model_cache


async def _get_model_details(name: str) -> dict:
    for model in await _get_models():
        if model["name"] == name:
            return model
    return _DEFAULT_MODEL_INFO


# --------------------------------------------------------------------------- #
# Streaming generators
# --------------------------------------------------------------------------- #

async def _stream_openai(
    chunks: AsyncIterator[object], model: str
) -> AsyncIterator[str]:
    """Translate Hailo stream items into OpenAI-compatible SSE framing."""
    try:
        yield to_openai_chunk("", model, is_meta=True)
        async for chunk in chunks:
            if chunk is _STREAM_FAILED:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "error": {
                                "message": "Hailo upstream stream failed",
                                "type": "upstream_error",
                                "code": "hailo_stream_failed",
                            }
                        }
                    )
                    + "\n\n"
                )
                return
            if not isinstance(chunk, dict):
                continue
            content = _extract_content(chunk)
            if content:
                yield to_openai_chunk(content, model)
            if chunk.get("done") is True:
                yield to_openai_chunk("", model, finish_reason="stop")
        yield "data: [DONE]\n\n"
    finally:
        aclose = getattr(chunks, "aclose", None)
        if aclose is not None:
            await aclose()


async def _stream_ollama(
    chunks: AsyncIterator[object], model: str
) -> AsyncIterator[str]:
    """Translate Hailo stream items into Ollama-compatible NDJSON framing."""
    try:
        async for chunk in chunks:
            if chunk is _STREAM_FAILED:
                yield json.dumps({"error": "Hailo upstream stream failed"}) + "\n"
                return
            if not isinstance(chunk, dict):
                continue
            yield json.dumps({
                "model": model,
                "created_at": f"{int(time.time())}",
                "message": {"role": "assistant", "content": _extract_content(chunk)},
                "done": chunk.get("done") is True,
            }) + "\n"
    finally:
        aclose = getattr(chunks, "aclose", None)
        if aclose is not None:
            await aclose()


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoints
# --------------------------------------------------------------------------- #

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/api/chat/completions")
async def chat_completions(request: Request) -> Any:
    """Serve OpenAI chat completions, defaulting requests to non-streaming."""
    try:
        body, is_stream, model = _build_payload(
            await request.json(), default_stream=False,
        )
        if is_stream:
            chunks = await _start_hailo_stream(body)
            return StreamingResponse(
                _stream_openai(chunks, model), media_type="text/event-stream",
            )
        hailo_response = await _post_hailo(body)
        return _openai_full_response(_extract_content(hailo_response), model)
    except httpx.HTTPStatusError as exc:
        return _upstream_error_response(exc)
    except HailoQuarantinedError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    except Exception as exc:
        logger.exception("Error in chat adapter")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# --------------------------------------------------------------------------- #
# Ollama-compatible endpoints
# --------------------------------------------------------------------------- #

@app.get("/api/tags")
async def api_tags() -> dict:
    return {"models": await _get_models()}


@app.post("/api/tags/refresh")
async def api_tags_refresh() -> dict:
    """Force a refresh of the cached Hailo model list."""
    _model_cache.clear()
    _model_cache.extend(await _refresh_models_from_hailo())
    return {"models": _model_cache, "refreshed": True}


@app.post("/api/show")
async def api_show(request: Request) -> dict:
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        body = {}
    name = body.get("model") or body.get("name") or HAILO_DEFAULT_MODEL
    model = await _get_model_details(name)
    return {
        "details": model["details"],
        "capabilities": ["completion"],
    }


@app.post("/api/chat")
async def api_chat(request: Request) -> Any:
    """Serve Ollama chat requests, defaulting to NDJSON streaming."""
    body, is_stream, model = _build_payload(
        await request.json(), default_stream=True,
    )
    try:
        if is_stream:
            chunks = await _start_hailo_stream(body)
            return StreamingResponse(
                _stream_ollama(chunks, model), media_type="application/x-ndjson"
            )
        hailo_response = await _post_hailo(body)
    except httpx.HTTPStatusError as exc:
        return _upstream_error_response(exc)
    except HailoQuarantinedError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return _ollama_full_response(_extract_content(hailo_response), model)
