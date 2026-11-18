"""Bearer-token auth (env) and per-client token-bucket rate limiting.

Auth: if OMNI_AUTH_TOKEN is set, requests must present
`Authorization: Bearer <token>` (constant-time compare). If unset, auth
is disabled and /diagnostics says so — local dev only, never production.
The token value is never logged, returned, or stored anywhere else.
"""
from __future__ import annotations

import hashlib
import hmac
import threading
import time


def client_id_for(token_present: bool, token: str, remote: str) -> str:
    """Stable per-client key that never exposes the token itself."""
    if token_present and token:
        return "token:" + hashlib.sha256(token.encode()).hexdigest()[:16]
    return "ip:" + remote


def check_auth(auth_header: str | None, expected_token: str) -> bool:
    if not expected_token:
        return True  # auth disabled by configuration (see module docstring)
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    presented = auth_header[len("Bearer "):].strip()
    return hmac.compare_digest(presented, expected_token)


class TokenBucket:
    """Per-client bucket: `burst` tokens, refilled at `per_min` per minute."""

    def __init__(self, per_min: int, burst: int) -> None:
        self.per_min = per_min
        self.burst = burst
        self._tokens: dict[str, float] = {}
        self._updated: dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens = self._tokens.get(key, float(self.burst))
            last = self._updated.get(key, now)
            tokens = min(float(self.burst), tokens + (now - last) * self.per_min / 60.0)
            if tokens >= 1.0:
                self._tokens[key] = tokens - 1.0
                self._updated[key] = now
                return True, 0.0
            deficit = 1.0 - tokens
            retry_after = deficit * 60.0 / self.per_min
            self._tokens[key] = tokens
            self._updated[key] = now
            return False, round(retry_after, 2)
