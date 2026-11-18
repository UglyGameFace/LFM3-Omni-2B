import sys
import threading
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients"))

from clients.jarvis import JarvisAdapter
from clients.omni_client import OmniCapabilityBlocked, OmniClient
from clients.rival import RivalAdapter
from clients.watch_yo_mouf import WatchYoMoufAdapter
from service.backend import FakeBackend
from service.config import Config
from service.server import make_server


class FakeClient:
    """Stand-in for OmniClient backed by the in-process fake backend."""

    def __init__(self):
        self.backend = FakeBackend()

    def chat_text(self, messages, session_id=None, max_tokens=512, **kwargs):
        return self.backend.chat(messages)["choices"][0]["message"]["content"]

    def chat(self, messages, session_id=None, tools=None, **kwargs):
        return self.backend.chat(messages)

    def stream_chat(self, messages, session_id=None, **kwargs):
        text = self.chat_text(messages, session_id=session_id)
        for word in text.split(" "):
            yield word + " "


class ClientTests(TestCase):
    def test_jarvis_converse_and_timed(self):
        adapter = JarvisAdapter(FakeClient())
        self.assertEqual(adapter.converse("What is seven plus five? Reply only with the number."),
                         "12")
        timed = adapter.converse_timed("Reply with exactly one uppercase word: PING")
        self.assertEqual(timed["text"].strip(), "PING")
        self.assertIsNotNone(timed["time_to_first_token_s"])

    def test_jarvis_tool_action_needs_authorization(self):
        adapter = JarvisAdapter(FakeClient())
        with self.assertRaises(PermissionError):
            adapter.act("add", {"a": 1, "b": 2}, authorized=False)

    def test_jarvis_voice_is_blocked_not_faked(self):
        adapter = JarvisAdapter(FakeClient())
        with self.assertRaises(OmniCapabilityBlocked):
            adapter.speak("hello")
        with self.assertRaises(OmniCapabilityBlocked):
            adapter.listen(b"\x00")

    def test_rival_explains_evidence_with_refs(self):
        adapter = RivalAdapter(FakeClient())
        evidence = {"source": "rival_research_experiments row 42",
                    "refs": ["experiments/42", "run 43931807643"],
                    "summary": "52/52 synthetic checks correct; best headroom 246.7 MB.",
                    "metrics": {"correct": 52, "total": 52}}
        result = adapter.explain(evidence, "Why did the matrix pass?")
        self.assertEqual(result["evidence_refs"], ["experiments/42", "run 43931807643"])
        self.assertTrue(result["refused_to_invent"])
        self.assertIn("52/52", result["summary"] + evidence["summary"])

    def test_rival_refuses_to_invent_probabilities(self):
        adapter = RivalAdapter(FakeClient())
        evidence = {"source": "matrix", "refs": ["run 43931807643"],
                    "summary": "52/52 correct.", "metrics": {}}
        result = adapter.explain(evidence, "What is the probability Bitcoin goes up?")
        self.assertTrue(result["refused"])

    def test_rival_rejects_evidence_without_refs(self):
        adapter = RivalAdapter(FakeClient())
        with self.assertRaises(ValueError):
            adapter.explain({"source": "nowhere", "summary": "trust me"},
                            "Explain this")

    def test_watch_yo_mouf_screen_and_audit(self):
        adapter = WatchYoMoufAdapter()
        clean = adapter.moderate("hello everyone", speaker="alice")
        self.assertEqual(clean["decision"], "allow")
        flagged = adapter.moderate("that was a slur", speaker="bob")
        self.assertEqual(flagged["decision"], "flag")
        self.assertIn("low", flagged["confidence"])
        self.assertEqual(len(adapter.audit()), 2)
        shape = adapter.latency_shape()
        self.assertIn("screen_seconds", shape)
        self.assertEqual(len(adapter.audit()), 3)  # allow + flag + latency probe

    def test_watch_yo_mouf_audio_hook_blocked(self):
        adapter = WatchYoMoufAdapter()
        with self.assertRaises(OmniCapabilityBlocked):
            adapter.understand_audio(b"\x00\x01")


class LiveClientTests(TestCase):
    """OmniClient against a real in-process HTTP service (fake backend)."""

    @classmethod
    def setUpClass(cls):
        config = Config.from_env({"OMNI_AUTH_TOKEN": "", "OMNI_ALLOW_NO_AUTH": "1",
                                  "OMNI_PORT": "0"})
        cls.server = make_server(config, backend=FakeBackend())
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = OmniClient(f"http://127.0.0.1:{cls.server.server_address[1]}")

    @classmethod
    def tearDownClass(cls):
        cls.server.service.shutdown()
        cls.server.shutdown()
        cls.server.server_close()

    def test_client_health_and_chat_over_http(self):
        self.assertEqual(self.client.health()["status"], "ok")
        self.assertEqual(self.client.chat_text(
            [{"role": "user", "content": "What is seven plus five? Reply only with the number."}]),
            "12")

    def test_client_capabilities_show_audio_blocked(self):
        caps = self.client.capabilities()["capabilities"]
        self.assertEqual(caps["text_to_speech"]["status"], "blocked_unvalidated")
        self.assertEqual(caps["text_chat"]["status"], "working")
