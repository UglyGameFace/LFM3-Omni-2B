"""Shared client for the unified Omni service (JARVIS, RivaL, Watch Yo Mouf).

Sync, stdlib urllib only. No model logic lives here — the client only
transports requests and surfaces the service's honest capability errors
(501 for blocked audio/video, 429 for rate limits, 401 for auth).
"""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OmniError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, payload: dict | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


class OmniAuthError(OmniError):
    pass


class OmniRateLimitError(OmniError):
    pass


class OmniCapabilityBlocked(OmniError):
    pass


class OmniBackendNotReady(OmniError):
    pass


def _raise_for_status(status: int, payload: dict) -> None:
    detail = payload.get("detail") or payload.get("error") or f"HTTP {status}"
    if status == 401:
        raise OmniAuthError(detail, status, payload)
    if status == 429:
        raise OmniRateLimitError(detail, status, payload)
    if status == 501:
        raise OmniCapabilityBlocked(payload.get("reason") or detail, status, payload)
    if status == 503:
        raise OmniBackendNotReady(detail, status, payload)
    raise OmniError(detail, status, payload)


class OmniClient:
    def __init__(self, base_url: str, token: str = "", timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read(5_000_001)
        except HTTPError as exc:
            try:
                data = json.loads(exc.read(100_001))
            except ValueError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            _raise_for_status(exc.code, data)
            raise  # unreachable
        except (URLError, TimeoutError, OSError) as exc:
            raise OmniError(f"service unreachable: {exc}") from exc
        if len(raw) > 5_000_000:
            raise OmniError("service response too large")
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise OmniError("service returned non-JSON") from exc
        if not isinstance(data, dict):
            raise OmniError("service response is not a JSON object")
        return data

    def health(self) -> dict:
        return self._request("GET", "/health")

    def ready(self) -> dict:
        try:
            return self._request("GET", "/ready")
        except OmniBackendNotReady:
            return {"status": "not_ready"}

    def diagnostics(self) -> dict:
        return self._request("GET", "/diagnostics")

    def capabilities(self) -> dict:
        return self._request("GET", "/v1/capabilities")

    def chat(self, messages: list[dict], session_id: str | None = None,
             tools: list[dict] | None = None, temperature: float = 0,
             max_tokens: int = 512) -> dict:
        payload: dict = {"messages": messages, "temperature": temperature,
                         "max_tokens": max_tokens}
        if session_id:
            payload["session_id"] = session_id
        if tools:
            payload["tools"] = tools
        return self._request("POST", "/v1/chat/completions", payload)

    def chat_text(self, messages: list[dict], **kwargs) -> str:
        result = self.chat(messages, **kwargs)
        return result["choices"][0]["message"]["content"]

    def stream_chat(self, messages: list[dict], session_id: str | None = None,
                    temperature: float = 0, max_tokens: int = 512):
        """Yield text deltas from the SSE stream. First yield is the first
        chunk received (time-to-first-token is measured by the caller)."""
        payload: dict = {"messages": messages, "stream": True,
                         "temperature": temperature, "max_tokens": max_tokens}
        if session_id:
            payload["session_id"] = session_id
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = Request(self.base_url + "/v1/chat/completions",
                      data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            resp = urlopen(req, timeout=self.timeout_s)
        except HTTPError as exc:
            try:
                data = json.loads(exc.read(100_001))
            except ValueError:
                data = {}
            _raise_for_status(exc.code, data if isinstance(data, dict) else {})
            raise
        except (URLError, TimeoutError, OSError) as exc:
            raise OmniError(f"service unreachable: {exc}") from exc
        with resp:
            # Read line-by-line: SSE events are newline-delimited, and a
            # blocking read(n) would wait for n bytes that may never come.
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    item = json.loads(data)
                except ValueError:
                    continue
                if "error" in item:
                    raise OmniError(str(item["error"]))
                try:
                    text = item["choices"][0].get("delta", {}).get("content", "")
                except (KeyError, IndexError, TypeError):
                    continue
                if text:
                    yield text

    def vision(self, image_url: str, prompt: str = "Describe this image.") -> str:
        result = self._request("POST", "/v1/vision",
                               {"image_url": image_url, "prompt": prompt})
        return result["content"]

    def transcribe(self, audio: bytes):
        """STT is BLOCKED on the service (501). This raises OmniCapabilityBlocked
        rather than faking a transcript."""
        raise OmniCapabilityBlocked(
            "speech_to_text is blocked_unvalidated on this service: no audio "
            "backend has been validated against this llama.cpp build")

    def synthesize(self, text: str):
        """TTS is BLOCKED on the service (501). Raises, never fakes audio."""
        raise OmniCapabilityBlocked(
            "text_to_speech is blocked_unvalidated on this service: no audio "
            "backend has been validated against this llama.cpp build")
