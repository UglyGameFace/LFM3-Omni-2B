"""JARVIS adapter: conversational voice assistant on the shared Omni backend.

- Streamed low-latency replies (caller measures time-to-first-token).
- Image / screen understanding via the vision endpoint.
- Tool actions ONLY when the caller passes an explicit authorization flag
  for that tool; unauthorized tool calls raise PermissionError and are
  never sent to the model.
- Voice input/output are NOT available: STT/TTS are blocked on the
  service (501) and this adapter surfaces that instead of faking audio.
"""
from __future__ import annotations

import time

from .omni_client import OmniCapabilityBlocked, OmniClient


class JarvisAdapter:
    def __init__(self, client: OmniClient, session_id: str = "jarvis") -> None:
        self.client = client
        self.session_id = session_id

    def converse(self, text: str, stream: bool = False, max_tokens: int = 256):
        """Plain conversational turn. Returns str, or an iterator of deltas
        when stream=True. Low-latency shape: small max_tokens by default."""
        messages = [{"role": "user", "content": text}]
        if stream:
            return self.client.stream_chat(messages, session_id=self.session_id,
                                           max_tokens=max_tokens)
        return self.client.chat_text(messages, session_id=self.session_id,
                                     max_tokens=max_tokens)

    def converse_timed(self, text: str) -> dict:
        """One streamed turn with measured time-to-first-token."""
        started = time.monotonic()
        first_at: float | None = None
        parts: list[str] = []
        for delta in self.converse(text, stream=True):
            if first_at is None:
                first_at = time.monotonic()
            parts.append(delta)
        total = time.monotonic() - started
        return {"text": "".join(parts),
                "time_to_first_token_s": round(first_at - started, 3) if first_at else None,
                "total_s": round(total, 3)}

    def describe_image(self, image_url: str, prompt: str = "Describe this image or screen.") -> str:
        return self.client.vision(image_url, prompt)

    def act(self, tool_name: str, args: dict, authorized: bool = False) -> dict:
        """Run a tool action. `authorized` must be True for this specific
        tool, otherwise the call is refused before anything happens."""
        if not authorized:
            raise PermissionError(
                f"tool {tool_name!r} is not authorized for this JARVIS action; "
                "pass authorized=True only after an explicit user approval for it")
        result = self.client.chat(
            [{"role": "user", "content": f"Call the {tool_name} tool."}],
            tools=[{"type": "function",
                    "function": {"name": tool_name, "parameters": {"type": "object"}}}])
        return result.get("tool_results", [{}])[0].get("result", {}) if result.get("tool_results") else {}

    def speak(self, text: str):
        """Voice output is blocked on the service; raise honestly."""
        raise OmniCapabilityBlocked(
            "text_to_speech is blocked_unvalidated: JARVIS voice output is not "
            "available until a TTS backend is validated end-to-end")

    def listen(self, audio: bytes):
        """Voice input is blocked on the service; raise honestly."""
        raise OmniCapabilityBlocked(
            "speech_to_text is blocked_unvalidated: JARVIS voice input is not "
            "available until an STT backend is validated end-to-end")
