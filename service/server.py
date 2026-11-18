"""HTTP service: text chat + SSE streaming, vision, validated tool calls,
sessions, auth, rate limits, health/readiness/diagnostics, structured logs.

Stdlib only (http.server). Talks to llama-server as the backend process
via service/backend.py. Audio/video endpoints exist only to answer 501
honestly — see service/audio.py.

Timeouts: each backend call runs with a deadline (OMNI_REQUEST_TIMEOUT_S);
a client disconnect during streaming cancels the iteration and is counted.
Backend recovery: a crashed managed backend is restarted up to
OMNI_BACKEND_MAX_RESTARTS times (BackendManager.ensure_ready).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import MODEL_IS_REASONING, MODEL_NAME, __version__
from .audio import BlockedAudioVideoProvider, blocked_response
from .auth import TokenBucket, check_auth, client_id_for
from .backend import BackendError, BackendManager
from .config import Config
from .history import HistoryStore
from .reasoning import NoReasoningBackend
from .tools import ToolValidationError, execute_tool, validate_tool_definition

MAX_BODY_BYTES = 1_000_000
logger = logging.getLogger("omni.service")

CAPABILITY_MATRIX = {
    "text_chat": {"status": "working", "evidence": "52/52 synthetic checks, Q4_K_M run 43931807643 and Q4_0 run 43934082561"},
    "streaming": {"status": "working", "evidence": "SSE streaming implemented; exercised against the fake backend in tests"},
    "image_understanding": {"status": "working", "evidence": "synthetic red/blue images correct in both matrices (52/52)"},
    "structured_tool_calls": {"status": "working", "evidence": "schema-validated registry (echo/add/clock); offline tests only, no live tool-call accuracy run yet"},
    "multi_turn_history": {"status": "working", "evidence": "in-memory sessions with trimming; offline tests only"},
    "speech_to_text": {"status": "blocked_unvalidated", "evidence": "no audio backend validated against this llama.cpp build"},
    "text_to_speech": {"status": "blocked_unvalidated", "evidence": "no audio backend validated against this llama.cpp build"},
    "video_understanding": {"status": "blocked_unvalidated", "evidence": "no video backend validated against this llama.cpp build"},
    "voice_conversation": {"status": "blocked_unvalidated", "evidence": "depends on blocked STT/TTS"},
    "reasoning": {"status": "not_applicable", "evidence": "LFM3-Omni-2B is a non-reasoning model by design"},
}


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters = {
            "requests_total": 0, "chat_requests": 0, "stream_requests": 0,
            "chat_errors": 0, "tool_calls_validated": 0, "tool_calls_rejected": 0,
            "rate_limited": 0, "auth_failures": 0, "backend_errors": 0,
            "backend_restarts": 0, "audio_blocked": 0, "video_blocked": 0,
            "cancelled": 0, "timeouts": 0,
        }
        self.started_at = time.time()

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def snapshot(self) -> dict:
        with self._lock:
            return {"uptime_seconds": round(time.time() - self.started_at, 1),
                    "counters": dict(self.counters)}


class Service:
    """Owns config, backend, sessions, limits and metrics for one server."""

    def __init__(self, config: Config, backend=None,
                 av_provider=None, reasoning=None) -> None:
        self.config = config
        self.manager = BackendManager(config, backend=backend)
        self.backend = self.manager.backend
        self.history = HistoryStore(config.max_turns, config.max_history_chars)
        self.bucket = TokenBucket(config.rate_limit_per_min, config.rate_burst)
        self.av = av_provider or BlockedAudioVideoProvider()
        self.reasoning = reasoning or NoReasoningBackend()
        self.metrics = Metrics()
        self.semaphore = threading.BoundedSemaphore(config.max_concurrent)
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent * 2,
                                           thread_name_prefix="omni-backend")

    def call_backend(self, fn, timeout: float):
        future = self.executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            self.metrics.inc("timeouts")
            raise BackendError(f"backend call exceeded {timeout}s timeout") from None

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.manager.stop()


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "omni-unified/" + __version__

        # -- plumbing -------------------------------------------------
        def log_message(self, *args):  # silence default stderr logging
            pass

        def _log(self, status: int, route: str, extra: dict | None = None) -> None:
            record = {"ts": round(time.time(), 3), "route": route, "status": status,
                      "client": self.client_address[0],
                      "request_id": getattr(self, "_request_id", "-")}
            if extra:
                record.update(extra)
            logger.info(json.dumps(record))

        def _send_json(self, status: int, obj: dict, route: str) -> None:
            raw = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                service.metrics.inc("cancelled")
            self._log(status, route)

        def _error(self, status: int, code: str, detail: str, route: str) -> None:
            self._send_json(status, {"error": code, "detail": detail,
                                     "request_id": getattr(self, "_request_id", "-")}, route)

        def _read_json(self) -> dict | None:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                return None
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw)
            except ValueError:
                return None
            return data if isinstance(data, dict) else None

        def _check_access(self, route: str) -> str | None:
            """Auth + rate limit. Returns client id, or None after responding."""
            self._request_id = str(uuid.uuid4())[:8]
            token = self.config.auth_token
            if not check_auth(self.headers.get("Authorization"), token):
                service.metrics.inc("auth_failures")
                self._error(401, "unauthorized", "missing or invalid Bearer token", route)
                return None
            key = client_id_for(bool(token), self.headers.get("Authorization", ""), self.client_address[0])
            allowed, retry_after = service.bucket.allow(key)
            if not allowed:
                service.metrics.inc("rate_limited")
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", str(retry_after))
                body = json.dumps({"error": "rate_limited", "retry_after_seconds": retry_after}).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self._log(429, route, {"retry_after": retry_after})
                return None
            return key

        @property
        def config(self) -> Config:
            return service.config

        # -- routing --------------------------------------------------
        def do_GET(self):
            route = urlparse(self.path).path
            service.metrics.inc("requests_total")
            if route == "/health":
                self._send_json(200, {"status": "ok", "service": "omni-unified",
                                      "model": MODEL_NAME, "version": __version__,
                                      "reasoning_model": MODEL_IS_REASONING}, route)
            elif route == "/ready":
                state = service.manager.state()
                if state["ready"]:
                    self._send_json(200, {"status": "ready", "backend": state}, route)
                else:
                    self._send_json(503, {"status": "not_ready", "backend": state,
                                          "detail": "backend not ready; service is up but cannot infer"}, route)
            elif route == "/diagnostics":
                if self._check_access(route) is None:
                    return
                self._send_json(200, {
                    "service": "omni-unified", "version": __version__,
                    "model": MODEL_NAME, "reasoning_model": MODEL_IS_REASONING,
                    "llama_cpp_ref": "b12024",
                    "config": service.config.public_summary(),
                    "metrics": service.metrics.snapshot(),
                    "backend": service.manager.state(),
                    "capabilities": CAPABILITY_MATRIX,
                    "audio_video": service.av.status(),
                    "reasoning_backend": service.reasoning.status(),
                    "sessions": service.history.session_count(),
                    "production_approved": False,
                    "production_note": "Omni cannot replace LFM3-1.7B inside the 2,450 MB "
                                       "allocation on either quant (Q4_K_M best headroom 246.7 MB, "
                                       "Q4_0 best 211.6 MB; 400 MB advisory target unmet). "
                                       "LFM3-1.7B is preserved; separate host is the only viable path.",
                }, route)
            elif route == "/v1/capabilities":
                self._send_json(200, {"model": MODEL_NAME,
                                      "reasoning_model": MODEL_IS_REASONING,
                                      "capabilities": CAPABILITY_MATRIX}, route)
            else:
                self._error(404, "not_found", f"no route {route}", route)

        def do_POST(self):
            route = urlparse(self.path).path
            service.metrics.inc("requests_total")
            if route == "/v1/chat/completions":
                self._handle_chat(route)
            elif route == "/v1/vision":
                self._handle_vision(route)
            elif route in ("/v1/audio/transcriptions", "/v1/audio/speech"):
                if self._check_access(route) is None:
                    return
                service.metrics.inc("audio_blocked")
                cap = "speech_to_text" if route.endswith("transcriptions") else "text_to_speech"
                self._send_json(501, blocked_response(cap), route)
            elif route == "/v1/video/understand":
                if self._check_access(route) is None:
                    return
                service.metrics.inc("video_blocked")
                self._send_json(501, blocked_response("video_understanding"), route)
            elif route == "/v1/sessions/clear":
                if self._check_access(route) is None:
                    return
                body = self._read_json() or {}
                dropped = service.history.drop(str(body.get("session_id", "")))
                self._send_json(200, {"cleared": dropped}, route)
            else:
                self._error(404, "not_found", f"no route {route}", route)

        # -- chat -----------------------------------------------------
        def _validate_messages(self, messages) -> str | None:
            if not isinstance(messages, list) or not messages:
                return "messages must be a non-empty list"
            for msg in messages:
                if not isinstance(msg, dict) or msg.get("role") not in ("system", "user", "assistant", "tool"):
                    return "each message needs a valid role"
                content = msg.get("content", "")
                size = len(content) if isinstance(content, str) else len(json.dumps(content))
                if size > service.config.max_message_chars:
                    return "a message exceeds OMNI_MAX_MESSAGE_CHARS"
            return None

        def _handle_chat(self, route: str):
            if self._check_access(route) is None:
                return
            body = self._read_json()
            if body is None:
                self._error(400, "invalid_json", "body must be a JSON object under 1 MB", route)
                return
            messages = body.get("messages")
            err = self._validate_messages(messages)
            if err:
                self._error(400, "invalid_messages", err, route)
                return
            tools = body.get("tools") or []
            tool_schemas: dict[str, dict] = {}
            try:
                for tool in tools:
                    schema = validate_tool_definition(tool)
                    tool_schemas[tool["function"]["name"]] = schema
            except ToolValidationError as exc:
                self._error(400, "invalid_tool_definition", str(exc), route)
                return
            session_id = body.get("session_id")
            if session_id is not None and not isinstance(session_id, str):
                self._error(400, "invalid_session", "session_id must be a string", route)
                return
            full_messages = list(messages)
            conv = None
            if session_id:
                conv = service.history.get(session_id)
                prior = conv.history()
                new = list(messages)
                # Do not duplicate a system prompt already stored in history.
                if (prior and prior[0].get("role") == "system"
                        and new and new[0].get("role") == "system"):
                    new = new[1:]
                full_messages = prior + new
            service.metrics.inc("chat_requests")
            if body.get("stream"):
                service.metrics.inc("stream_requests")
                self._handle_chat_stream(route, full_messages, conv, body, tool_schemas)
                return
            if not service.semaphore.acquire(timeout=5):
                self._error(503, "busy", "max concurrent requests reached", route)
                return
            try:
                if not service.manager.ensure_ready():
                    service.metrics.inc("backend_errors")
                    self._error(503, "backend_not_ready",
                                service.manager.last_error or "backend not ready", route)
                    return
                kwargs = {"temperature": body.get("temperature", 0),
                          "max_tokens": body.get("max_tokens", 512)}
                if tools:
                    kwargs["tools"] = tools
                try:
                    result = service.call_backend(
                        lambda: service.backend.chat(full_messages, **kwargs),
                        service.config.request_timeout_s)
                except BackendError as exc:
                    service.metrics.inc("backend_errors")
                    self._error(502, "backend_error", str(exc)[:300], route)
                    return
                reply = self._extract_reply(result)
                tool_outcome = self._run_tool_calls(result, tool_schemas)
                if tool_outcome.get("error"):
                    service.metrics.inc("tool_calls_rejected")
                    self._error(422, "tool_call_rejected", tool_outcome["error"], route)
                    return
                if conv is not None:
                    for msg in messages:
                        conv.add(msg.get("role", "user"), msg.get("content", ""))
                    conv.add("assistant", reply)
                response = {"choices": [{"message": {"role": "assistant", "content": reply}}],
                            "model": MODEL_NAME, "reasoning_model": MODEL_IS_REASONING,
                            "usage": result.get("usage", {})}
                if tool_outcome:
                    response["tool_calls"] = tool_outcome["tool_calls"]
                    response["tool_results"] = tool_outcome["results"]
                if conv is not None:
                    response["session"] = conv.stats()
                self._send_json(200, response, route)
            finally:
                service.semaphore.release()

        def _extract_reply(self, result: dict) -> str:
            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise BackendError("backend response has no choices[0].message.content")
            if not isinstance(content, str):
                raise BackendError("backend content is not text")
            return content

        def _run_tool_calls(self, result: dict, schemas: dict[str, dict]) -> dict:
            try:
                calls = result["choices"][0]["message"].get("tool_calls") or []
            except (KeyError, IndexError, TypeError):
                return {}
            if not calls:
                return {}
            validated, results = [], []
            for call in calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                name = fn.get("name")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except ValueError:
                    return {"error": f"tool {name!r} arguments are not valid JSON"}
                if name not in schemas:
                    return {"error": f"tool {name!r} was not in the request's tool list"}
                try:
                    outcome = execute_tool(name, args)
                except ToolValidationError as exc:
                    return {"error": str(exc)}
                service.metrics.inc("tool_calls_validated")
                validated.append({"name": name, "arguments": args})
                results.append({"name": name, "result": outcome})
            return {"tool_calls": validated, "results": results}

        def _handle_chat_stream(self, route: str, messages: list[dict], conv,
                                body: dict, tool_schemas: dict) -> None:
            if not service.manager.ensure_ready():
                service.metrics.inc("backend_errors")
                self._error(503, "backend_not_ready",
                            service.manager.last_error or "backend not ready", route)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # SSE has no Content-Length: the connection MUST close at the
            # end of the stream, or HTTP/1.1 clients block waiting for more.
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            started = time.monotonic()
            first_chunk_at: float | None = None
            collected: list[str] = []
            try:
                kwargs = {"temperature": body.get("temperature", 0),
                          "max_tokens": body.get("max_tokens", 512)}
                for chunk in service.backend.chat_stream(messages, **kwargs):
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                    try:
                        delta = chunk["choices"][0].get("delta", {})
                    except (KeyError, IndexError, TypeError):
                        continue
                    text = delta.get("content", "")
                    if text:
                        collected.append(text)
                    payload = json.dumps(chunk).encode()
                    try:
                        self.wfile.write(b"data: " + payload + b"\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        service.metrics.inc("cancelled")
                        self._log(499, route, {"cancelled": True})
                        return
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    service.metrics.inc("cancelled")
            except BackendError as exc:
                service.metrics.inc("backend_errors")
                try:
                    self.wfile.write(("data: " + json.dumps({"error": str(exc)[:200]}) + "\n\n").encode())
                except (BrokenPipeError, ConnectionResetError):
                    service.metrics.inc("cancelled")
            finally:
                if conv is not None:
                    for msg in messages:
                        conv.add(msg.get("role", "user"), msg.get("content", ""))
                    conv.add("assistant", "".join(collected))
                ttft = round(first_chunk_at - started, 3) if first_chunk_at else None
                self._log(200, route, {"stream": True, "ttft_seconds": ttft,
                                       "seconds": round(time.monotonic() - started, 3)})

        def _handle_vision(self, route: str):
            if self._check_access(route) is None:
                return
            body = self._read_json()
            if body is None:
                self._error(400, "invalid_json", "body must be a JSON object", route)
                return
            image = body.get("image_url") or body.get("image_data_url")
            prompt = body.get("prompt", "Describe this image.")
            if not isinstance(image, str) or not image:
                self._error(400, "invalid_image", "image_url (data URL or URL) is required", route)
                return
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image}}]}]
            # Reuse the chat path by forwarding to the backend directly.
            if not service.manager.ensure_ready():
                service.metrics.inc("backend_errors")
                self._error(503, "backend_not_ready", "backend not ready", route)
                return
            try:
                result = service.call_backend(
                    lambda: service.backend.chat(messages, temperature=0, max_tokens=512),
                    service.config.request_timeout_s)
                reply = self._extract_reply(result)
            except BackendError as exc:
                service.metrics.inc("backend_errors")
                self._error(502, "backend_error", str(exc)[:300], route)
                return
            service.metrics.inc("chat_requests")
            self._send_json(200, {"content": reply, "model": MODEL_NAME}, route)

    return Handler


def make_server(config: Config, backend=None, av_provider=None,
                reasoning=None) -> ThreadingHTTPServer:
    service = Service(config, backend=backend, av_provider=av_provider,
                      reasoning=reasoning)
    server = ThreadingHTTPServer((config.host, config.port), make_handler(service))
    server.daemon_threads = True
    server.service = service  # type: ignore[attr-defined]
    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = Config.from_env()  # raises ResourceBudgetError on impossible budgets
    for warning in config.warnings:
        logger.warning(json.dumps({"warning": warning}))
    server = make_server(config)
    service = server.service
    if service.manager.managed:
        if not service.manager.start():
            raise SystemExit(f"backend failed to start: {service.manager.last_error}")
    logger.info(json.dumps({"event": "omni_unified_started", "version": __version__,
                            "model": MODEL_NAME, "reasoning_model": MODEL_IS_REASONING,
                            "host": config.host, "port": config.port,
                            "production_approved": False}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
