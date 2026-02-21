from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import time
import json

app = FastAPI()

# Point to  Hailo-Ollama service
HAILO_URL = "http://127.0.0.1:8000/api/chat"

MAX_HISTORY = 3 # Increased slightly to give the model more context

FULL_TOOLING = """Tools available for this request:
- read: Read file contents
- write: Create or overwrite files
- exec: Run shell commands (PTY available)
- web_search: Search the web (Brave API)
- canvas: Present/eval/snapshot the Canvas""" # Paste your tooling string here if needed

def build_system_message(request_requires_tools=False):
    base = "You are a personal assistant running inside OpenClaw. Use short answers"
    if request_requires_tools:
        base += "\n" + FULL_TOOLING
    return {"role": "system", "content": base}

def normalize_messages(messages):
    normalized = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = " ".join(text_parts)
        if not isinstance(content, str):
            content = str(content)
        normalized.append({"role": role, "content": content})
    return normalized

def to_openai_chunk(content, model, finish_reason=None, is_meta=False):
    """Wraps content into the OpenAI SSE format that the OpenClaw Dashboard expects."""
    chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content} if is_meta else {"content": content},
                "finish_reason": finish_reason
            }
        ]
    }
    return f"data: {json.dumps(chunk)}\n\n"

async def handle_chat(req: Request):
    try:
        data = await req.json()
        model_name = data.get("model", "qwen2:1.5b")
        is_stream = data.get("stream", False)
        
        # 1. Prepare messages for the Hailo NPU
        system_message = build_system_message(data.get("use_tools", False))
        user_messages = normalize_messages(data["messages"])
        messages_to_send = [system_message] + user_messages[-MAX_HISTORY:]

        payload = {
            "model": model_name,
            "messages": messages_to_send,
            "stream": is_stream
        }

        # 2. Handle Streaming (Required for the Dashboard to show text)
        if is_stream:
            async def stream_generator():
                async with httpx.AsyncClient() as client:
                    async with client.stream("POST", HAILO_URL, json=payload, timeout=180.0) as r:
                        # Send initial metadata chunk to start the bubble
                        yield to_openai_chunk("", model_name, is_meta=True)
                        
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            
                            hailo_json = json.loads(line)
                            content = hailo_json.get("message", {}).get("content", "")
                            done = hailo_json.get("done", False)
                            
                            if content:
                                yield to_openai_chunk(content, model_name)
                            
                            if done:
                                yield to_openai_chunk("", model_name, finish_reason="stop")
                                
                        # Crucial termination signal for OpenClaw gateway
                        yield "data: [DONE]\n\n"
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        
        # 3. Handle Non-Streaming (Unary)
        else:
            async with httpx.AsyncClient() as client:
                r = await client.post(HAILO_URL, json=payload, timeout=180.0)
                hailo_res = r.json()
                assistant_message = hailo_res.get("message", {}).get("content", "")
                
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": assistant_message},
                        "finish_reason": "stop"
                    }]
                }

    except Exception as e:
        print(f"Error in adapter: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Combined Routes for broad compatibility
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/api/chat/completions")
async def chat_handler(req: Request):
    return await handle_chat(req)
