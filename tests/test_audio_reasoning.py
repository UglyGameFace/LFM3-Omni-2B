import sys
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.audio import (BlockedAudioVideoProvider, CapabilityBlocked,
                           blocked_response)
from service.reasoning import (COMPENSATIONS, NoReasoningBackend,
                               ReasoningUnavailable, check_case, run_bench)
from service.backend import FakeBackend


class AudioGatingTests(TestCase):
    def test_all_capabilities_blocked_unvalidated(self):
        provider = BlockedAudioVideoProvider()
        status = provider.status()
        for name in ("speech_to_text", "text_to_speech", "video_understanding",
                     "voice_conversation"):
            self.assertEqual(status[name]["status"], "blocked_unvalidated")

    def test_calls_raise_501_not_fake_success(self):
        provider = BlockedAudioVideoProvider()
        for fn in (lambda: provider.transcribe(b"\x00"),
                   lambda: provider.synthesize("hi"),
                   lambda: provider.understand_video(b"\x00", "what?")):
            with self.assertRaises(CapabilityBlocked) as ctx:
                fn()
            self.assertEqual(ctx.exception.status_code, 501)

    def test_blocked_response_shape(self):
        body = blocked_response("text_to_speech")
        self.assertEqual(body["error"], "capability_blocked")
        self.assertFalse(body["production_ready"])


class ReasoningTests(TestCase):
    def test_no_reasoning_backend_holds_nothing_in_ram(self):
        backend = NoReasoningBackend()
        self.assertFalse(backend.loaded)
        self.assertEqual(backend.ram_mb, 0)
        self.assertEqual(backend.status()["name"], "none")
        with self.assertRaises(ReasoningUnavailable):
            backend.reason("think hard")

    def test_compensations_cover_the_known_limits(self):
        for key in ("arithmetic", "factual_claims", "multi_step_planning",
                    "tool_selection", "probabilities", "long_context"):
            self.assertIn(key, COMPENSATIONS)

    def test_fake_backend_bench_documents_expected_failures(self):
        fake = FakeBackend()
        chat_fn = lambda messages: fake.chat(messages)["choices"][0]["message"]["content"]  # noqa: E731
        result = run_bench(chat_fn)
        self.assertEqual(result["total"], 8)
        self.assertFalse(result["model_is_reasoning"])
        by_id = {c["id"]: c for c in result["cases"]}
        # The fake (like Omni on these prompts) passes grounded cases...
        self.assertTrue(by_id["simple_arithmetic"]["passed"])
        self.assertTrue(by_id["long_context_canary"]["passed"])
        self.assertTrue(by_id["complex_instruction"]["passed"])
        # ...and fails the two cases that require refusing to invent facts.
        self.assertFalse(by_id["factual_consistency_refusal"]["passed"])
        self.assertFalse(by_id["no_invented_probability"]["passed"])

    def test_check_case_modes(self):
        self.assertTrue(check_case({"expect": "PING", "mode": "exact"}, "PING"))
        self.assertFalse(check_case({"expect": "PING", "mode": "exact"}, "PONG"))
        self.assertTrue(check_case({"expect": "12", "mode": "contains"}, "The answer is 12"))
