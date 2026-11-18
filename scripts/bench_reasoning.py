"""Reasoning benchmark runner (manual, real backend) + fake-backend check.

Usage:
  python scripts/bench_reasoning.py                      # fake backend (offline)
  python scripts/bench_reasoning.py --base-url http://127.0.0.1:8770   # real service

Omni is a NON-reasoning model: failing complex-instruction and
factual-consistency cases are EXPECTED. The point of this bench is to
document where deterministic logic must compensate, not to pretend the
model reasons. See service/reasoning.py COMPENSATIONS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients"))

from clients.omni_client import OmniClient  # noqa: E402
from service.backend import FakeBackend  # noqa: E402
from service.reasoning import BENCH_CASES, run_bench  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="")
    parser.add_argument("--output", type=Path, default=Path("reports/omni-reasoning-bench.json"))
    args = parser.parse_args()
    if args.base_url:
        client = OmniClient(args.base_url)
        chat_fn = lambda messages: client.chat_text(messages)  # noqa: E731
        mode = "real"
    else:
        fake = FakeBackend()
        chat_fn = lambda messages: fake.chat(messages)["choices"][0]["message"]["content"]  # noqa: E731
        mode = "fake"
    result = run_bench(chat_fn)
    result["mode"] = mode
    result["cases_total"] = len(BENCH_CASES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nReasoning bench ({mode}): {result['passed']}/{result['total']} passed. "
          "Omni is non-reasoning; failures here are documented limits, not bugs.")


if __name__ == "__main__":
    main()
