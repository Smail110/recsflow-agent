"""Проверить два отдельных HTTP-процесса из чистой копии проекта."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _get_json(url: str, *, timeout: float = 3) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _wait_for_health(url: str, process: subprocess.Popen, *, seconds: float = 20) -> dict:
    deadline = time.monotonic() + seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Процесс завершился до health: {process.returncode}")
        try:
            return _get_json(url)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"Health не ответил: {type(last_error).__name__}")


def _launch(module: str, port: int, environment: dict[str, str]) -> subprocess.Popen:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:8b")
    args = parser.parse_args()

    platform_port = _free_port()
    api_port = _free_port()
    environment = os.environ.copy()
    environment.update(
        RECAGENT_PROVIDER="recsflow",
        RECAGENT_PROVIDER_URL=f"http://127.0.0.1:{platform_port}",
        RECAGENT_PROVIDER_VERIFIED_TITLE_LOOKUP="true",
        RECAGENT_MODE="ollama",
        OLLAMA_URL="http://127.0.0.1:11434",
        OLLAMA_MODEL=args.model,
    )
    platform = _launch("recagent.mock_platform:app", platform_port, environment)
    api = _launch("recagent.api:app", api_port, environment)
    try:
        platform_health = _wait_for_health(f"http://127.0.0.1:{platform_port}/health", platform)
        api_health = _wait_for_health(f"http://127.0.0.1:{api_port}/health", api)
        readiness = _get_json(f"http://127.0.0.1:{api_port}/ready")
        request = urllib.request.Request(
            f"http://127.0.0.1:{api_port}/v1/chat",
            data=json.dumps({"user_id": "public-tree-smoke", "message": "Хочу курс Python с практикой."}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            http_status = response.status
            chat = json.load(response)
        result = {
            "platform_health": platform_health.get("status"),
            "api_health": api_health.get("status"),
            "api_readiness": readiness.get("status"),
            "http_status": http_status,
            "state": chat.get("state"),
            "mode": chat.get("mode"),
            "recommendations": len(chat.get("recommendations", [])),
            "llm_calls": chat.get("llm_calls"),
            "fallback_reason": (chat.get("telemetry") or {}).get("fallback_reason"),
        }
        print(json.dumps(result, ensure_ascii=False))
        passed = (
            http_status == 200
            and result["state"] == "recommend"
            and result["mode"] == "ollama"
            and result["recommendations"] > 0
            and not result["fallback_reason"]
        )
        return int(not passed)
    finally:
        for process in (api, platform):
            process.terminate()
        for process in (api, platform):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
