"""Capability-gated audio/video adapters.

Honesty rule: STT, TTS and video are BLOCKED / UNVALIDATED by default.
No audio/video backend has been validated against this llama.cpp b12024
CPU build — the llama-server chat endpoint used here exposes text +
vision only. These adapters therefore return a plain 501 with the reason
and never fake success. No paid dependency, no second model loaded in RAM.

A real provider can be added later behind this same interface, but only
after an end-to-end validation run records it as working.
"""
from __future__ import annotations

REASON_NO_BACKEND = (
    "No audio/video backend has been validated against this llama.cpp "
    "b12024 CPU build. The llama-server chat endpoint exposes text and "
    "vision only; speech recognition, speech output and video understanding "
    "have never been exercised end-to-end here. No paid service and no "
    "second model are loaded."
)

CAPABILITIES = (
    "speech_to_text",
    "text_to_speech",
    "video_understanding",
    "voice_conversation",
)


class CapabilityBlocked(RuntimeError):
    def __init__(self, capability: str) -> None:
        super().__init__(f"{capability} is blocked: {REASON_NO_BACKEND}")
        self.capability = capability
        self.status_code = 501


class AudioVideoProvider:
    """Interface a real provider must implement. Default is blocked."""

    def status(self) -> dict:
        return {name: {"status": "blocked_unvalidated", "reason": REASON_NO_BACKEND}
                for name in CAPABILITIES}

    def transcribe(self, audio: bytes, **kwargs):
        raise CapabilityBlocked("speech_to_text")

    def synthesize(self, text: str, **kwargs):
        raise CapabilityBlocked("text_to_speech")

    def understand_video(self, video: bytes, prompt: str, **kwargs):
        raise CapabilityBlocked("video_understanding")


class BlockedAudioVideoProvider(AudioVideoProvider):
    """The only provider shipped: every audio/video call is a 501."""


def blocked_response(capability: str) -> dict:
    return {
        "error": "capability_blocked",
        "capability": capability,
        "status": "blocked_unvalidated",
        "reason": REASON_NO_BACKEND,
        "production_ready": False,
    }
