import sys
import threading
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients"))

from clients.omni_client import (OmniAuthError, OmniBackendNotReady,
                                 OmniCapabilityBlocked, OmniClient, OmniError,
                                 OmniRateLimitError)
from service.backend import FakeBackend
from service.config import Config
from service.server import make_server


class StubToolBackend(FakeBackend):
    """Returns a backend-proposed tool call so the server's validation path runs."""

    def __init__(self, tool_name="add", arguments='{"a": 2, "b": 3}'):
        super().__init__()
        self.tool_name = tool_name
        self.arguments = arguments

    def chat(self, messages, **kwargs):
        self.calls += 1
        return {"choices": [{"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": self.tool_name,
                                         "arguments": self.arguments}}]}}],
                "usage": {}}


def start_server(**env_overrides):
    env = {"OMNI_AUTH_TOKEN": "", "OMNI_ALLOW_NO_AUTH": "1", "OMNI_PORT": "0",
           "OMNI_RATE_BURST": "100"}  # rate limiting has its own dedicated test
    env.update(env_overrides)
    config = Config.from_env(env)
    # Default backend is the in-process fake: tests run offline with no
    # llama-server. Pass _backend explicitly to override (e.g. unhealthy).
    backend = env_overrides.pop("_backend", None)
    if backend is None:
        backend = FakeBackend()
    server = make_server(config, backend=backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, OmniClient(f"http://127.0.0.1:{port}")


class ServerTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.client = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.server.service.shutdown()
        cls.server.shutdown()
        cls.server.server_close()

    def test_health_ready_diagnostics(self):
        health = self.client.health()
        self.assertEqual(health["status"], "ok")
        self.assertFalse(health["reasoning_model"])
        ready = self.client.ready()
        self.assertEqual(ready["status"], "ready")
        diag = self.client.diagnostics()
        self.assertFalse(diag["production_approved"])
        self.assertEqual(diag["backend"]["kind"], "fake")
        self.assertFalse(diag["config"]["reasoning_model_loaded"])
        self.assertEqual(diag["capabilities"]["speech_to_text"]["status"],
                         "blocked_unvalidated")

    def test_chat_text_and_vision(self):
        text = self.client.chat_text(
            [{"role": "user", "content": "What is seven plus five? Reply only with the number."}])
        self.assertEqual(text, "12")
        colour = self.client.vision("data:image/png;base64,AAAA", "What color is this image?")
        self.assertEqual(colour, "Red")

    def test_streaming_returns_deltas_and_done(self):
        parts = list(self.client.stream_chat(
            [{"role": "user", "content": "Reply with exactly one uppercase word: PING"}]))
        self.assertEqual("".join(parts).strip(), "PING")

    def test_audio_video_endpoints_are_501(self):
        for path in ("/v1/audio/transcriptions", "/v1/audio/speech",
                     "/v1/video/understand"):
            with self.assertRaises(OmniCapabilityBlocked, msg=path):
                self.client._request("POST", path, {"x": 1})

    def test_sessions_accumulate_and_clear(self):
        self.client.chat([{"role": "user", "content": "What is seven plus five?"}],
                         session_id="test-session")
        second = self.client.chat([{"role": "user", "content": "Reply with exactly one uppercase word: PING"}],
                                  session_id="test-session")
        self.assertEqual(second["session"]["turns"], 4)
        cleared = self.client._request("POST", "/v1/sessions/clear",
                                       {"session_id": "test-session"})
        self.assertTrue(cleared["cleared"])

    def test_invalid_messages_rejected(self):
        with self.assertRaises(OmniError):
            self.client.chat([])


class ToolCallServerTests(TestCase):
    def test_registered_tool_call_executes(self):
        server, client = start_server(_backend=StubToolBackend())
        try:
            result = client.chat(
                [{"role": "user", "content": "add 2 and 3"}],
                tools=[{"type": "function",
                        "function": {"name": "add",
                                     "parameters": {"type": "object",
                                                    "properties": {"a": {"type": "integer"},
                                                                   "b": {"type": "integer"}},
                                                    "required": ["a", "b"]}}}])
            self.assertEqual(result["tool_results"], [{"name": "add", "result": {"result": 5}}])
        finally:
            server.service.shutdown()
            server.shutdown()
            server.server_close()

    def test_unregistered_tool_call_is_refused(self):
        server, client = start_server(_backend=StubToolBackend(tool_name="launch_missiles"))
        try:
            with self.assertRaises(OmniError):
                client.chat(
                    [{"role": "user", "content": "do it"}],
                    tools=[{"type": "function",
                            "function": {"name": "launch_missiles",
                                         "parameters": {"type": "object"}}}])
        finally:
            server.service.shutdown()
            server.shutdown()
            server.server_close()


class AuthRateLimitServerTests(TestCase):
    def test_auth_enforced_when_token_set(self):
        server, _ = start_server(OMNI_AUTH_TOKEN="test-token-123",
                                 OMNI_ALLOW_NO_AUTH="1")
        try:
            anon = OmniClient(f"http://127.0.0.1:{server.server_address[1]}")
            with self.assertRaises(OmniAuthError):
                anon.diagnostics()
            authed = OmniClient(f"http://127.0.0.1:{server.server_address[1]}",
                                token="test-token-123")
            self.assertEqual(authed.diagnostics()["service"], "omni-unified")
        finally:
            server.service.shutdown()
            server.shutdown()
            server.server_close()

    def test_rate_limit_returns_429(self):
        server, _ = start_server(OMNI_RATE_LIMIT_PER_MIN="60", OMNI_RATE_BURST="2")
        try:
            client = OmniClient(f"http://127.0.0.1:{server.server_address[1]}")
            client.diagnostics()
            client.diagnostics()
            with self.assertRaises(OmniRateLimitError):
                client.diagnostics()
        finally:
            server.service.shutdown()
            server.shutdown()
            server.server_close()

    def test_backend_down_reports_not_ready(self):
        fake = FakeBackend()
        fake.healthy = False
        server, client = start_server(_backend=fake)
        try:
            self.assertEqual(client.ready(), {"status": "not_ready"})
            with self.assertRaises(OmniBackendNotReady):
                client._request("GET", "/ready")
        finally:
            server.service.shutdown()
            server.shutdown()
            server.server_close()
