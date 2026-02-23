# Hailo-Ollama + OpenClaw Adapter for Hailo-10 chip

![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![Supports Hailo Model Zoo GenAI 5.1.1 & 5.2.0](https://img.shields.io/badge/Hailo%20Model%20Zoo%20GenAI-5.1.1%20%26%205.2.0-success)

This repository provides a simple **FastAPI-based adapter** that bridges **Hailo-Ollama** (running on Hailo AI accelerators like AI HAT+2 or Hailo-10) with **OpenClaw**, enabling fast, local, privacy-focused LLM inference on Raspberry Pi 5.

**Tested and supported with Hailo Model Zoo GenAI (official name) versions 5.1.1 and 5.2.0** — the standard releases for Hailo-10H / Hailo-10 compatible setups, including HailoRT 5.1.1 & 5.2.0.

## Why this adapter is needed

Hailo-Ollama does **not** natively support the exact OpenAI `/v1/chat/completions` endpoint and response format that OpenClaw expects by default.

This proxy:
- Listens on port 11435 (recommended)
- Forwards trimmed requests to Hailo-Ollama /api/chat (port 8000)
- Keeps only the last user message → avoids context overflow
- Converts Hailo-Ollama response to OpenAI-compatible format

Works with OpenClaw dashboard and messengers (Telegram, etc.).

## Prerequisites

- Raspberry Pi 5 (64-bit OS recommended)
- Hailo drivers & platform installed
- Hailo-Ollama running on default port 8000
  - Compatible with Hailo Model Zoo GenAI 5.1.1 and 5.2.0
  - At least one model pulled (example: qwen:1.5b, llama-3.2:1b, …)
- OpenClaw installed and dashboard working
- Python 3.13 (Trixie default)

## Setup Steps

### 1. Clone the repository

``` bash
git clone https://github.com/tishyk/hailo-ollama-openclaw-adapter.git
cd hailo-ollama-openclaw-adapter
```

### 2. Create & activate virtual environment

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
# or:
# pip install fastapi uvicorn requests
```

### 4. (Optional) Review adapter.py

Open adapter.py in your preferred text editor (nano, vim, VS Code, etc.) and check or adjust the following parts if needed:

- The Hailo-Ollama endpoint URL:

``` python
hailo_url = "http://localhost:8000/api/chat"
```

Change the port (8000) if your Hailo-Ollama server is running on a different one.

- Context trimming logic (currently keeps only the latest user message):

``` python
if "messages" in data and isinstance(data["messages"], list):
    user_messages = [msg for msg in data["messages"] if msg.get("role") == "user"]
    if user_messages:
        data["messages"] = [user_messages[-1]]  # just the latest user input
```

- Response formatting section (makes Hailo-Ollama output look like a standard OpenAI response):

``` python
content = hailo_data.get("response", hailo_data.get("content", "No response from model"))

formatted = {
    "id": f"chatcmpl-{int(time.time())}",
    "object": "chat.completion",
    "created": int(time.time()),
    "model": data.get("model", "hailo-model"),
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # dummy values
}

```

Save the file if you made any changes.

This step is optional — the default code already works for most users.

### 5. Update openclaw.json

Find your openclaw.json (usually ~/.openclaw/ or in install folder).

**Caution!**
Add openclaw.json file backup first!

Add or modify a provider:

``` json
"models": {
    "providers": {
      "hailo-ollama": {
        "baseUrl": "http://127.0.0.1:11435",
        "apiKey": "ollama-local",
        "api": "openai-completions",
	"models": [
          {
            "id": "qwen2:1.5b",
            "name": "Hailo qwen2:1.5b",
            "reasoning": false,
            "input": [
              "text"
            ],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 131072,
            "maxTokens": 16394
          }
        ]
      }
    }

...

  "agents": {
    "defaults": {
      "model": {
        "primary": "hailo-ollama/qwen2:1.5b"
      },
```

Important: Use port 11435

### 6. Restart OpenClaw gateway
#### Example – adjust to your setup
``` bash
openclaw gateway restart
```

Check logs for errors related to config or port 11435.

### 7. Start the adapter

(Inside the venv folder)

``` bash
source venv/bin/activate
uvicorn adapter:app --host 0.0.0.0 --port 11435 --timeout-keep-alive 240 --limit-concurrency 2
```

Recommended: run in tmux, screen, or as systemd service.

### 8. Start / confirm Hailo-Ollama

In a separate terminal:

``` bash
hailo-ollama
```

Make sure it's listening on http://localhost:8000

### 9. Test in OpenClaw Dashboard

1. Open dashboard
``` bash
openclaw dashboard
```
3. Go to Agents / Models / Providers
4. Look for hailo-ollama-local — it should appear and be available
5. Go to Chat
6. Select the Hailo provider
7. Type: Hello! Tell me a short joke about AI hats.

You should get a quick response using Hailo acceleration.

## Troubleshooting

- Provider missing → check JSON syntax, restart gateway
- Adapter receives nothing → wrong baseUrl (must point to 11435)
- Hailo timeouts → smaller model, check Hailo-Ollama logs
- Context problems → adapter already limits to the last message

## Optional: systemd service

File: /etc/systemd/system/hailo-openclaw-adapter.service

``` ini
[Unit]
Description=Hailo-Ollama → OpenClaw Adapter
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/hailo-ollama-openclaw-adapter
ExecStart=/home/pi/hailo-ollama-openclaw-adapter/venv/bin/uvicorn adapter:app --host 0.0.0.0 --port 11435 --timeout-keep-alive 240 --limit-concurrency 2
Restart=always

[Install]
WantedBy=multi-user.target
```

Then run:

``` bash
sudo systemctl daemon-reload
sudo systemctl enable --now hailo-openclaw-adapter.service
sudo systemctl status hailo-openclaw-adapter
```

MIT License

Happy local accelerated AI on Raspberry Pi! 🚀
