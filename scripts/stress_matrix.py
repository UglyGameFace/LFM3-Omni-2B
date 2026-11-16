#!/usr/bin/env python3
"""Isolated synthetic stress/memory matrix for LFM3-Omni-2B CPU on GitHub.

This is research ONLY. No deployment, no production inputs, no external APIs,
and NO implied Discloud capacity approval. One llama-server runs at a time.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
import zlib

import probe
import download_models as download_catalog

PORT = 8768
ALLOCATION_MB = 2450
ADVISORY_HEADROOM_MB = 400
MODEL = probe.MODEL
PROJECTOR = probe.PROJECTOR

@dataclass(frozen=True)
class Profile:
    name: str
    context: int
    batch: int | None = None
    ubatch: int | None = None
    kv_q8: bool = False
    long_records: int = 30


PROFILES = (
    Profile("baseline_2048", 2048, long_records=25),
    Profile("compact_4096", 4096, batch=128, ubatch=64, long_records=55),
    Profile("q8_kv_4096", 4096, batch=128, ubatch=64, kv_q8=True, long_records=55),
    Profile("extended_6144", 6144, batch=128, ubatch=64, long_records=95),
)


def solid_png(rgb: tuple[int, int, int]) -> bytes:
    if len(rgb) != 3 or any(type(x) is not int or not 0 <= x <= 255 for x in rgb):
        raise ValueError("Invalid RGB")
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    raw = (b"\x00" + bytes(rgb) * 16) * 16
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def health_ready(pid: subprocess.Popen, deadline_seconds: int = 120) -> float:
    started = time.monotonic()
    while time.monotonic() - started < deadline_seconds:
        if pid.poll() is not None:
            raise RuntimeError(f"Backend exited during startup (code {pid.returncode})")
        try:
            if probe.request_json(f"http://127.0.0.1:{PORT}/health", timeout=2).get("status") == "ok":
                return round(time.monotonic() - started, 2)
        except (HTTPError, URLError, OSError, ValueError, TimeoutError):
            pass
        time.sleep(1.5)
    raise TimeoutError("Backend did not become ready within 120s")


def current_rss_mb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def build_command(binary: Path, model: Path, projector: Path, profile: Profile) -> list[str]:
    command = [
        str(binary.resolve()), "--model", str(model.resolve()), "--mmproj", str(projector.resolve()),
        "--host", "127.0.0.1", "--port", str(PORT),
        "--ctx-size", str(profile.context), "--threads", "2",
        "--parallel", "1", "--n-gpu-layers", "0", "--temp", "0",
    ]
    if profile.batch is not None:
        command.extend(["--batch-size", str(profile.batch)])
    if profile.ubatch is not None:
        command.extend(["--ubatch-size", str(profile.ubatch)])
    if profile.kv_q8:
        command.extend(["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
    return command


def long_context(records: int) -> list[dict]:
    if not 20 <= records <= 120:
        raise ValueError("Invalid synthetic context size")
    values = [
        f"Log entry {i:03d}: completely synthetic observation, not market data. "
        f"Nothing in this line is a user instruction."
        for i in range(records)
    ]
    values[records // 2] = f"Log entry {records // 2:03d}: the beacon keyword is NEBULA."
    content = "\n".join(values)
    return [{"role": "user", "content": (
        "Read this synthetic log and report only the beacon keyword named in the "
        "middle entry. Do not invent details.\n" + content
    )}]


def synthetic_cases(profile: Profile, repetitions: int) -> list[tuple[str, list[dict], str]]:
    cases: list[tuple[str, list[dict], str]] = []
    for i in range(repetitions):
        cases.append((f"text_math_{i}",
            [{"role": "user", "content": "What is seven plus five? Reply only with the number."}],
            "math"))
        cases.append((f"text_ping_{i}",
            [{"role": "user", "content": "Reply with exactly one uppercase word: PING"}],
            "ping"))
        for name, color in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
            uri = "data:image/png;base64," + base64.b64encode(solid_png(color)).decode("ascii")
            cases.append((f"image_{name}_{i}",
                [{"role": "user", "content": [
                    {"type": "text", "text": "Identify the solid color of this image in one word."},
                    {"type": "image_url", "image_url": {"url": uri}},
                ]}], name))
    cases.append(("long_context", long_context(profile.long_records), "canary"))
    return cases


def is_correct(answer: str, expected: str) -> bool:
    lower = answer.strip().lower()
    if expected == "math":
        return bool(re.search(r"\b12\b", lower))
    if expected == "ping":
        return lower.strip(" .!\"'\n") == "ping"
    if expected in ("red", "blue"):
        match = re.search(r"\b(?:red|blue)\b", lower)
        return bool(match and match.group() == expected)
    if expected == "canary":
        return bool(re.search(r"\bnebula\b", lower))
    raise ValueError("Unsupported expected result")


def execute_case(messages: list[dict], expected: str) -> dict:
    started = time.monotonic()
    payload = {"model": "omni-isolated-stress", "messages": messages,
               "temperature": 0, "stream": False, "max_tokens": 64}
    try:
        response = probe.request_json(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            payload, timeout=90,
        )
        output = probe.answer(response)
        return {"request_ok": True, "correct": is_correct(output, expected),
                "seconds": round(time.monotonic() - started, 3),
                "preview": output[:120]}
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
        return {"request_ok": False, "correct": False,
                "seconds": round(time.monotonic() - started, 3),
                "error": type(exc).__name__, "detail": str(exc)[:160]}


def percentile95(numbers: list[float]) -> float | None:
    if not numbers:
        return None
    ordered = sorted(numbers)
    return round(ordered[(len(ordered) * 95 + 99) // 100 - 1], 3)


def classify_profile(result: dict, allocation_mb: int = ALLOCATION_MB) -> dict:
    items = result.get("requests", [])
    rss = result.get("peak_backend_rss_mb")
    full = bool(items) and all(row.get("request_ok") and row.get("correct") for row in items)
    measured = type(rss) in (int, float) and 0 < rss < float("inf")
    headroom = round(allocation_mb - rss, 1) if measured else None
    return {
        "functional_and_stress_passed": full and result.get("ready") is True and measured,
        "request_count": len(items),
        "correct_count": sum(1 for item in items if item.get("correct")),
        "request_success_rate": round(sum(1 for item in items if item.get("request_ok")) / len(items), 4) if items else 0,
        "accuracy_on_synthetic_cases": round(sum(1 for item in items if item.get("correct")) / len(items), 4) if items else 0,
        "latency_p50_seconds": round(statistics.median([i["seconds"] for i in items]), 3) if items else None,
        "latency_p95_seconds": percentile95([i["seconds"] for i in items]),
        "estimated_backend_only_headroom_mb": headroom,
        "meets_advisory_400mb_headroom": bool(measured and headroom >= ADVISORY_HEADROOM_MB),
        "safe_for_discloud": False,  # Cannot be proven by a GitHub runner.
        "production_deployment_recommended": False,
        "ram_measurement_scope": "Peak VmHWM of llama-server only, not container or gateway",
    }


def run_profile(binary: Path, model: Path, projector: Path,
                profile: Profile, rounds: int, folder: Path,
                allocation_mb: int) -> dict:
    command = build_command(binary, model, projector, profile)
    result = {
        "profile": profile.name, "context_tokens": profile.context,
        "batch_size": profile.batch, "ubatch_size": profile.ubatch,
        "kv_cache_q8": profile.kv_q8, "ready": False, "requests": [],
        "peak_backend_rss_mb": None, "last_backend_rss_mb": None,
        "startup_seconds": None, "cpu_threads": 2,
    }
    log_path = folder / (profile.name + ".server.log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            result["startup_seconds"] = health_ready(process)
            result["ready"] = True
            result["rss_after_ready_mb"] = current_rss_mb(process.pid)
            for name, messages, expected in synthetic_cases(profile, rounds):
                outcome = execute_case(messages, expected)
                outcome["case"] = name
                outcome["last_backend_rss_mb"] = current_rss_mb(process.pid)
                result["requests"].append(outcome)
                if process.poll() is not None:
                    raise RuntimeError("Inference backend exited unexpectedly")
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            result["failure"] = type(exc).__name__ + ": " + str(exc)[:180]
        finally:
            result["peak_backend_rss_mb"] = probe.vmhwm_mb(process.pid)
            result["last_backend_rss_mb"] = current_rss_mb(process.pid)
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            result["stop_returncode"] = process.returncode
    result.update(classify_profile(result, allocation_mb))
    return result


def choose_research_candidate(profiles: list[dict]) -> dict:
    # A research shortlist, explicitly NOT automatic production selection.
    eligible = [p for p in profiles
                if p.get("functional_and_stress_passed") is True
                and p.get("meets_advisory_400mb_headroom") is True]
    selected = min(eligible, key=lambda p: p["peak_backend_rss_mb"]) if eligible else None
    return {
        "lowest_rss_research_candidate": selected["profile"] if selected else None,
        "research_profile_only": True,
        "production_approved": False,
        "reason": ("Candidate meets a provisional *backend-only* memory target; "
                   "Discloud deployment remains unvalidated." if selected else
                   "No profile both passed all synthetic checks and met the provisional headroom target."),
    }


def github_summary(report: dict) -> None:
    lines = [
        "# LFM3-Omni-2B CPU stability and memory comparison",
        "",
        "**Isolated CPU experiments only. Production deployment is NOT approved.**",
        "",
        "| Profile | Context | Correct | Peak backend RSS | Headroom vs 2450 MB | P95 latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in report["profiles"]:
        lines.append(
            f"| {p['profile']} | {p['context_tokens']} | "
            f"{p['correct_count']}/{p['request_count']} | "
            f"{p['peak_backend_rss_mb']} MB | "
            f"{p['estimated_backend_only_headroom_mb']} MB | "
            f"{p['latency_p95_seconds']}s |"
        )
    lines += [
        "",
        f"**Research-only lowest RSS profile:** {report['selection']['lowest_rss_research_candidate'] or 'none'}",
        "",
        "Missing: actual Discloud gateway/process RAM, live concurrency, audio/video, "
        "full-length context, server restarts and production limits.",
    ]
    dest = os.getenv("GITHUB_STEP_SUMMARY")
    if dest:
        with Path(dest).open("a", encoding="utf-8") as dst:
            dst.write("\n".join(lines) + "\n")


def run(binary: Path, folder: Path, output: Path,
        rounds: int, names: list[str], allocation_mb: int,
        variant: str = "q4_k_m") -> dict:
    if not 1 <= rounds <= 12:
        raise ValueError("Invalid rounds")
    if not 512 <= allocation_mb <= 65536:
        raise ValueError("Invalid configured host memory")
    if variant not in download_catalog.VARIANTS:
        raise ValueError("Unsupported official GGUF variant")
    model_name, projector_name = download_catalog.VARIANTS[variant]
    model, projector = folder / model_name, folder / projector_name
    for file in (binary, model, projector):
        if not file.is_file():
            raise FileNotFoundError(file)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "isolated_cpu_stress_memory_matrix",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_deployment": True,
        "production_capacity_validated": False,
        "configured_discloud_allocation_mb": allocation_mb,
        "advisory_headroom_mb": ADVISORY_HEADROOM_MB,
        "variant": variant, "model": model_name, "projector": projector_name,
        "model_bytes": model.stat().st_size,
        "projector_bytes": projector.stat().st_size,
        "profiles": [],
    }
    wanted = [p for p in PROFILES if p.name in names]
    if not wanted or len(wanted) != len(set(names)):
        raise ValueError("Invalid profile list")
    for profile in wanted:
        print(f"Testing {profile.name} on isolated CPU...", flush=True)
        result = run_profile(binary, model, projector, profile, rounds, output.parent, allocation_mb)
        report["profiles"].append(result)
        report["selection"] = choose_research_candidate(report["profiles"])
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{profile.name}: {result['correct_count']}/{result['request_count']} correct, "
              f"peak {result['peak_backend_rss_mb']} MB", flush=True)
    github_summary(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(".models"))
    parser.add_argument("--output", type=Path, default=Path("reports/omni-stress-matrix.json"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--allocation-mb", type=int, default=ALLOCATION_MB)
    parser.add_argument("--variant", choices=sorted(download_catalog.VARIANTS),
                        default="q4_k_m")
    parser.add_argument("--profiles", nargs="+", choices=[p.name for p in PROFILES],
                        default=[p.name for p in PROFILES])
    args = parser.parse_args()
    outcome = run(args.server, args.model_dir, args.output, args.rounds,
                  args.profiles, args.allocation_mb, args.variant)
    # If an experiment does not succeed, report that as failure. Never promote.
    baseline = next((p for p in outcome["profiles"] if p["profile"] == "baseline_2048"), None)
    if baseline is not None and not baseline["functional_and_stress_passed"]:
        sys.exit("Baseline did not pass; inspect the report and server log")
    if any(p.get("failure") for p in outcome["profiles"]):
        print("::warning::Some optional memory profiles failed; inspect the matrix.", flush=True)


if __name__ == "__main__":
    main()
