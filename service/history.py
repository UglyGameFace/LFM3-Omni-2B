"""Multi-turn conversation history with configurable limits and trimming.

In-memory only; nothing is persisted. Trimming keeps the system prompt
(if any) plus the most recent turns that fit the char budget, and reports
exactly what was dropped so callers can see the loss.
"""
from __future__ import annotations

import threading


class Conversation:
    def __init__(self, session_id: str, max_turns: int = 20, max_chars: int = 12000) -> None:
        self.session_id = session_id
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.messages: list[dict] = []
        self.dropped_turns = 0
        self.dropped_chars = 0

    def _content_len(self, message: dict) -> int:
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, dict):
                    total += len(str(item.get("text", ""))) + len(str(item.get("image_url", "")))
            return total
        return 0

    def add(self, role: str, content) -> None:
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self) -> None:
        # Keep a leading system message, then the newest turns.
        system = [m for m in self.messages[:1] if m.get("role") == "system"]
        rest = self.messages[len(system):]
        max_rest = max(0, self.max_turns * 2)
        while len(rest) > max_rest:
            dropped = rest.pop(0)
            self.dropped_turns += 1
            self.dropped_chars += self._content_len(dropped)
        total = sum(self._content_len(m) for m in system + rest)
        while rest and total > self.max_chars:
            dropped = rest.pop(0)
            self.dropped_turns += 1
            dropped_len = self._content_len(dropped)
            self.dropped_chars += dropped_len
            total -= dropped_len
        self.messages = system + rest

    def history(self) -> list[dict]:
        return [dict(m) for m in self.messages]

    def stats(self) -> dict:
        return {"session_id": self.session_id, "turns": len(self.messages),
                "chars": sum(self._content_len(m) for m in self.messages),
                "dropped_turns": self.dropped_turns,
                "dropped_chars": self.dropped_chars,
                "estimated_tokens": sum(self._content_len(m) for m in self.messages) // 4}


class HistoryStore:
    def __init__(self, max_turns: int = 20, max_chars: int = 12000) -> None:
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._sessions: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Conversation:
        with self._lock:
            conv = self._sessions.get(session_id)
            if conv is None:
                conv = Conversation(session_id, self.max_turns, self.max_chars)
                self._sessions[session_id] = conv
            return conv

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
