"""Watch Yo Mouf adapter: context-aware moderation assistance.

Shape and honesty rules:

- Speech/audio understanding is a HOOK, not a working path: audio input
  raises OmniCapabilityBlocked because STT is blocked_unvalidated on the
  service. Text transcripts can be screened now.
- Text screening is a deterministic keyword screen ONLY, labelled low
  confidence. It is not a model verdict and never claims to be one.
- An optional model-assisted explanation (via the shared client) is
  clearly marked model-assisted and never overrides the deterministic
  decision.
- Every decision is written to an in-memory audit log with timestamp,
  speaker, decision, reasons and confidence.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from .omni_client import OmniCapabilityBlocked

# Deterministic screen only. This list is deliberately narrow: it flags
# slurs/harassment patterns for human review, it does not decide guilt.
FLAG_WORDS = ("slur", "hate", "threat", "kill yourself")
REVIEW_WORDS = ("idiot", "stupid", "shut up")


class WatchYoMoufAdapter:
    def __init__(self, client=None, session_id: str = "watch-yo-mouf") -> None:
        self.client = client
        self.session_id = session_id
        self.audit_log: list[dict] = []

    def understand_audio(self, audio: bytes):
        """Audio hook: blocked until an STT backend is validated."""
        raise OmniCapabilityBlocked(
            "speech_to_text is blocked_unvalidated: Watch Yo Mouf audio "
            "understanding is not available until an STT backend is validated "
            "end-to-end against this llama.cpp build")

    def moderate(self, transcript: str, speaker: str = "unknown",
                 context: str = "", model_assist: bool = False) -> dict:
        lowered = transcript.lower()
        if any(word in lowered for word in FLAG_WORDS):
            decision, reasons = "flag", ["matched flag-list term (deterministic screen)"]
        elif any(word in lowered for word in REVIEW_WORDS):
            decision, reasons = "review", ["matched review-list term (deterministic screen)"]
        else:
            decision, reasons = "allow", []
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "speaker": speaker,
            "transcript_preview": transcript[:120],
            "context": context[:200],
            "decision": decision,
            "reasons": reasons,
            "confidence": "low — deterministic keyword screen only, not a model verdict",
            "model_assisted": False,
            "explanation": None,
        }
        if model_assist and self.client is not None and decision != "allow":
            entry["explanation"] = self.client.chat_text(
                [{"role": "user", "content":
                  "Explain in one sentence why this transcript may need moderation. "
                  "Do not invent context beyond the transcript.\nTranscript: " + transcript}],
                session_id=self.session_id, max_tokens=128)
            entry["model_assisted"] = True
            entry["confidence"] = ("low — deterministic screen + model-assisted explanation; "
                                   "human review still required")
        self.audit_log.append(entry)
        return dict(entry)

    def audit(self) -> list[dict]:
        return [dict(entry) for entry in self.audit_log]

    def latency_shape(self) -> dict:
        """Low-latency shape: the deterministic screen is synchronous and
        local; only the optional model assist adds a model round-trip."""
        started = time.monotonic()
        self.moderate("hello there", speaker="latency-probe")
        return {"screen_seconds": round(time.monotonic() - started, 4),
                "note": "deterministic screen only; model assist adds one chat round-trip"}
