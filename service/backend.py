"""Backend clients: llama-server (real) and a deterministic fake (tests/CI).

The fake backend answers the same synthetic cases the real matrix uses
(math "12", "PING", solid-colour vision) so the service, clients and
benchmarks can be exercised offline with no model download and no network.
It never claims to be the real model.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Config


class BackendError(RuntimeError):
    pass


class BackendTimeout(BackendError):
    pass


def _text_of(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    parts.append("[image]")
    return "\n".join(parts)


def _fake_reply(messages: list[dict]) -> str:
    """Deterministic canned replies for the synthetic cases only."""
    text = _text_of(messages)
    lower = text.lower()
    if "seven plus five" in lower:
        return "12"
    if re.search(r"\breply with exactly one uppercase word: ping\b", lower):
        return "PING"
    if "ping" in lower and "exactly" in lower:
        return "PING"
    if "[image]" in text and "color" in lower:
        # Synthetic solid-colour probe images used by the bench harness.
        if "blue" in lower:
            return "Blue"
        return "Red"
    if "beacon keyword" in lower:
        return "The beacon keyword named in the middle entry is NEBULA."
    if "what is a cpu" in lower:
        return "A CPU is the processor that executes instructions."
    return "Fake reply: " + text.strip().split("\n")[-1][:80]


class FakeBackend:
    """In-process stand-in for llama-server. kind='fake', never 'real'."""

    kind = "fake"

    def __init__(self) -> None:
        self.calls = 0
        self.healthy = True

    def health(self) -> dict:
        return {"status": "ok" if self.healthy else "error", "kind": "fake"}

    def is_ready(self) -> bool:
        return self.healthy

    def chat(self, messages: list[dict], **kwargs) -> dict:
        self.calls += 1
        if not self.healthy:
            raise BackendError("fake backend is down")
        reply = _fake_reply(messages)
        return {
            "choices": [{"message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": max(1, len(_text_of(messages)) // 4),
                      "completion_tokens": max(1, len(reply) // 4)},
            "model": "fake-omni",
        }

    def chat_stream(self, messages: list[dict], **kwargs):
        """Yield SSE-style chunks splitting the reply into words."""
        result = self.chat(messages, **kwargs)
        reply = result["choices"][0]["message"]["content"]
        for word in reply.split(" "):
            yield {"choices": [{"delta": {"content": word + " "}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


class LlamaServerBackend:
    """Thin urllib client for a real llama-server /v1/chat/completions."""

    kind = "real"

    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.calls = 0

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        body = json.dumps(payload).encode()
        req = Request(self.base_url + path, data=body,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=timeout or self.timeout_s) as resp:
                raw = resp.read(5_000_001)
        except HTTPError as exc:
            raise BackendError(f"backend HTTP {exc.code}: {exc.read(500)[:200]!r}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise BackendError(f"backend unreachable: {type(exc).__name__}: {exc}") from exc
        if len(raw) > 5_000_000:
            raise BackendError("backend response too large")
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise BackendError("backend returned non-JSON") from exc
        if not isinstance(data, dict):
            raise BackendError("backend response is not a JSON object")
        return data

    def health(self) -> dict:
        try:
            return self._post("/health", {}) if False else self._get("/health")
        except BackendError:
            return {"status": "error", "kind": "real"}

    def _get(self, path: str) -> dict:
        try:
            with urlopen(self.base_url + path, timeout=5) as resp:
                data = json.loads(resp.read(100_001))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise BackendError(f"backend health check failed: {exc}") from exc
        if not isinstance(data, dict):
            raise BackendError("backend health is not a JSON object")
        return data

    def is_ready(self) -> bool:
        try:
            return self._get("/health").get("status") == "ok"
        except BackendError:
            return False

    def chat(self, messages: list[dict], **kwargs) -> dict:
        self.calls += 1
        payload = {"model": "omni", "messages": messages,
                   "temperature": kwargs.get("temperature", 0),
                   "max_tokens": kwargs.get("max_tokens", 512),
                   "stream": False}
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
        return self._post("/v1/chat/completions", payload)

    def chat_stream(self, messages: list[dict], **kwargs):
        """Yield parsed SSE delta dicts from a streaming completion."""
        self.calls += 1
        payload = {"model": "omni", "messages": messages,
                   "temperature": kwargs.get("temperature", 0),
                   "max_tokens": kwargs.get("max_tokens", 512),
                   "stream": True}
        body = json.dumps(payload).encode()
        req = Request(self.base_url + "/v1/chat/completions", data=body,
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        try:
            resp = urlopen(req, timeout=self.timeout_s)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise BackendError(f"backend stream failed: {exc}") from exc
        with resp:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    return
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except ValueError:
                    continue


class BackendManager:
    """Owns a llama-server subprocess when configured; otherwise reports
    'not_configured' and never pretends a backend exists."""

    def __init__(self, config: Config, backend=None) -> None:
        self.config = config
        self.backend = backend  # injected backend (tests) or None
        self.process: subprocess.Popen | None = None
        self.restarts = 0
        self.last_error: str | None = None
        self._lock = threading.Lock()
        if backend is None:
            self.backend = LlamaServerBackend(config.backend_url, config.request_timeout_s)

    @property
    def managed(self) -> bool:
        return bool(self.config.backend_command and self.config.model_path)

    def state(self) -> dict:
        ready = False
        try:
            ready = self.backend.is_ready()
        except Exception:  # noqa: BLE001 - readiness must never raise
            ready = False
        return {
            "kind": getattr(self.backend, "kind", "unknown"),
            "managed": self.managed,
            "running": self.process is not None and self.process.poll() is None,
            "ready": ready,
            "restarts": self.restarts,
            "last_error": self.last_error,
            "backend_url": self.config.backend_url,
        }

    def start(self) -> bool:
        """Start the managed backend process. Returns False when nothing
        is configured to manage (caller must check state(), not assume)."""
        if not self.managed:
            return False
        with self._lock:
            return self._start_locked()

    def _start_locked(self) -> bool:
        binary = Path(self.config.backend_command)
        model = Path(self.config.model_path)
        if not binary.is_file() or not model.is_file():
            self.last_error = "backend binary or model file missing"
            return False
        from urllib.parse import urlparse

        port = urlparse(self.config.backend_url).port or 8767
        command = [str(binary.resolve()), "--model", str(model.resolve()),
                   "--host", "127.0.0.1", "--port", str(port),
                   "--ctx-size", str(self.config.max_context_tokens),
                   "--threads", str(self.config.threads),
                   "--parallel", "1", "--n-gpu-layers", "0", "--temp", "0"]
        if self.config.mmproj_path:
            command += ["--mmproj", str(Path(self.config.mmproj_path).resolve())]
        log_path = Path("runtime/omni-backend.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()
        deadline = time.monotonic() + self.config.backend_start_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.last_error = f"backend exited during startup (code {self.process.returncode})"
                return False
            if self.backend.is_ready():
                return True
            time.sleep(1.0)
        self.last_error = "backend did not become ready in time"
        return False

    def ensure_ready(self) -> bool:
        """Restart a crashed managed backend up to the configured limit."""
        if not self.managed:
            return self.backend.is_ready()
        if self.process is not None and self.process.poll() is None and self.backend.is_ready():
            return True
        if self.restarts >= self.config.backend_max_restarts:
            self.last_error = "backend restart limit reached"
            return False
        self.restarts += 1
        self.stop()
        return self.start()

    def stop(self) -> None:
        proc, self.process = self.process, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
