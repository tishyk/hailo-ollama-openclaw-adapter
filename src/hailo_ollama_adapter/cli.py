"""Command-line entry point for the adapter."""

from __future__ import annotations

import argparse

import uvicorn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hailo-ollama-adapter",
        description="Hailo-Ollama to OpenClaw adapter server.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11435,
        help="Port to listen on (default: 11435).",
    )
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=240,
        help="HTTP keep-alive timeout in seconds (default: 240).",
    )
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=2,
        help="Max concurrent HTTP connections (default: 2).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on source changes (development only).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    uvicorn.run(
        "hailo_ollama_adapter.adapter:app",
        host=args.host,
        port=args.port,
        timeout_keep_alive=args.timeout_keep_alive,
        limit_concurrency=args.limit_concurrency,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
