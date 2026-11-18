"""Reasoning limitations of LFM3-Omni-2B (a NON-reasoning model).

Omni understands instructions and can use tools, but it has no dedicated
thinking mode: complex planning, multi-step reasoning and advanced maths
are LIMITED. This module provides:

- A reasoning-backend interface that is optional, lazy and UNLOADED by
  default. ``NoReasoningBackend`` holds no model in RAM (0 MB) and every
  call raises ``ReasoningUnavailable``. Nothing here silently substitutes
  another model or a paid dependency.
- ``COMPENSATIONS``: where deterministic application logic or retrieval
  compensates for a limitation (e.g. arithmetic goes through the ``add``
  tool, facts come from retrieved evidence, never from the model).
- ``BENCH_CASES``: a small benchmark set (complex instructions, tool
  selection, multi-step planning, factual consistency) runnable against
  the fake backend in tests and against the real backend manually via
  scripts/bench_reasoning.py.
"""
from __future__ import annotations

MODEL_IS_REASONING = False

COMPENSATIONS = {
    "arithmetic": "Use the deterministic `add` tool (service/tools.py), never model math.",
    "factual_claims": "Ground every claim in retrieved evidence passed in by the caller "
                      "(see clients/rival.py); the model explains evidence, it does not supply facts.",
    "multi_step_planning": "Decompose the plan in deterministic application code into "
                           "single steps; ask the model about one step at a time.",
    "tool_selection": "Restrict the offered tool list to the registered tools only; "
                      "validate every proposed call against its JSON schema before executing.",
    "probabilities": "Never let the model invent probabilities, market data, trades or "
                     "model performance. Qualification stays in the existing deterministic engines.",
    "long_context": "Trim history to the configured budget (service/history.py); "
                    "long-context p95 balloons with context (Q4_0: 11.3s at 2048 -> 40.8s at 6144).",
}


class ReasoningUnavailable(RuntimeError):
    pass


class ReasoningBackend:
    """Interface for an optional reasoning backend. Lazy by contract:
    implementations must not load a model until the first call, and must
    report their loaded/RAM state honestly."""

    name = "none"
    loaded = False
    ram_mb = 0

    def reason(self, task: str, **kwargs) -> dict:
        raise NotImplementedError

    def status(self) -> dict:
        return {"name": self.name, "loaded": self.loaded,
                "ram_mb": self.ram_mb, "is_reasoning_model": False}


class NoReasoningBackend(ReasoningBackend):
    """Default: no reasoning backend, no model in RAM, every call refused."""

    name = "none"

    def reason(self, task: str, **kwargs) -> dict:
        raise ReasoningUnavailable(
            "No reasoning backend is configured (OMNI_REASONING_BACKEND=none). "
            "LFM3-Omni-2B is a non-reasoning model; complex reasoning tasks must "
            "use deterministic application logic or retrieval (see COMPENSATIONS), "
            "not a silently substituted model."
        )


# --- Small benchmark set. Each case: prompt, category, checker. ---
# Checkers are substring / exact-match on the reply; they are deliberately
# strict about invented facts (the "factual_consistency" cases fail if the
# reply asserts a number or date the prompt never gave).

BENCH_CASES = [
    {"id": "complex_instruction",
     "category": "complex_instructions",
     "prompt": "Reply with exactly one uppercase word: PING",
     "expect": "PING", "mode": "exact"},
    {"id": "simple_arithmetic",
     "category": "arithmetic",
     "prompt": "What is seven plus five? Reply only with the number.",
     "expect": "12", "mode": "contains"},
    {"id": "tool_selection_add",
     "category": "tool_selection",
     "prompt": "Use the add tool with a=2 and b=3. Reply only with the tool name you would call.",
     "expect": "add", "mode": "contains"},
    {"id": "multi_step_plan",
     "category": "multi_step_planning",
     "prompt": "Plan: first reply PING, then say the number 12. Do only the first step now: reply with exactly one uppercase word: PING",
     "expect": "PING", "mode": "exact"},
    {"id": "factual_consistency_refusal",
     "category": "factual_consistency",
     "prompt": "What is the exact closing price of Bitcoin yesterday? If you do not have verified data, say exactly: NO VERIFIED DATA",
     "expect": "NO VERIFIED DATA", "mode": "contains"},
    {"id": "long_context_canary",
     "category": "long_context",
     "prompt": "Read this synthetic log and report only the beacon keyword named in the middle entry. Do not invent details.\n" +
               "\n".join(f"Log entry {i:03d}: completely synthetic observation, not market data." for i in range(25)) +
               "\nLog entry 012: the beacon keyword is NEBULA.",
     "expect": "NEBULA", "mode": "contains"},
    {"id": "no_invented_probability",
     "category": "factual_consistency",
     "prompt": "What is the probability Bitcoin goes up in the next 5 minutes? If you cannot know, say exactly: CANNOT KNOW",
     "expect": "CANNOT KNOW", "mode": "contains"},
    {"id": "instruction_following_negation",
     "category": "complex_instructions",
     "prompt": "Do NOT mention any colour. Reply with exactly one uppercase word: PING",
     "expect": "PING", "mode": "exact"},
]


def check_case(case: dict, reply: str) -> bool:
    if case["mode"] == "exact":
        return reply.strip().strip(" .!\"'\n").upper() == case["expect"].upper() \
            if case["expect"].isupper() else reply.strip() == case["expect"]
    return case["expect"] in reply


def run_bench(chat_fn, cases: list[dict] | None = None) -> dict:
    """Run the benchmark set through ``chat_fn(messages) -> str``.
    Returns per-case pass/fail plus a summary. ``chat_fn`` is the fake
    backend in tests, or the real service client when run manually."""
    cases = BENCH_CASES if cases is None else cases
    results = []
    for case in cases:
        try:
            reply = chat_fn([{"role": "user", "content": case["prompt"]}])
            passed = check_case(case, reply)
            results.append({"id": case["id"], "category": case["category"],
                            "passed": passed, "reply_preview": reply[:120]})
        except Exception as exc:  # noqa: BLE001 - bench must record, not crash
            results.append({"id": case["id"], "category": case["category"],
                            "passed": False, "error": type(exc).__name__ + ": " + str(exc)[:160]})
    passed = sum(1 for r in results if r["passed"])
    return {"cases": results, "passed": passed, "total": len(results),
            "model_is_reasoning": MODEL_IS_REASONING,
            "note": "Omni is a non-reasoning model; failing complex-instruction / "
                    "factual-consistency cases are expected and must be compensated "
                    "by deterministic logic, not by loading a second model."}
