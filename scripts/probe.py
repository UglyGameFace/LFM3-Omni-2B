#!/usr/bin/env python3
"""Single local CPU text/vision probe. Synthetic inputs only, no deployment."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zlib

PORT = 8767
MODEL = "LFM3-Omni-2B-Q4_K_M.gguf"
PROJECTOR = "mmproj-LFM3-Omni-2B-Q4_K_M.gguf"
MEMORY_LIMIT_MB = 2100  # caution target only, not a functional pass condition
DISCLOUD_ALLOCATION_MB = 2450  # config value, not live host measurement


def png() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    raw = (b"\x00" + b"\xff\x00\x00" * 16) * 16
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def vmhwm_mb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def request_json(url: str, payload: dict | None = None, timeout: int = 5) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    with urlopen(Request(url, data=body, headers=headers), timeout=timeout) as response:
        raw = response.read(200_001)
    if len(raw) > 200_000:
        raise ValueError("Response too large")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Response is not a JSON object")
    return result


def answer(reply: dict) -> str:
    choices = reply.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("No response choices")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty model response")
    return content.strip()


def run_case(messages: list[dict]) -> dict:
    payload = {"model": "omni-isolated", "messages": messages, "temperature": 0, "max_tokens": 96, "stream": False}
    started = time.monotonic()
    try:
        reply = request_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", payload, timeout=300)
        content = answer(reply)
        return {"success": True, "latency_seconds": round(time.monotonic() - started, 2),
                "answer_preview": content[:200]}
    except (HTTPError, URLError, OSError, ValueError, KeyError, TimeoutError) as exc:
        return {"success": False, "latency_seconds": round(time.monotonic() - started, 2),
                "error": type(exc).__name__, "detail": str(exc)[:200]}



def classify_probe(result: dict) -> dict:
    """Do not confuse functional inference with production RAM suitability."""
    peak = result.get("peak_rss_mb")
    measured = type(peak) in (int, float) and 0 < peak < float("inf")
    functional = (
        result.get("text_success") is True
        and result.get("vision_success") is True
        and result.get("vision_color_correct") is True
    )
    headroom = round(DISCLOUD_ALLOCATION_MB - peak, 1) if measured else None
    within_target = bool(measured and peak <= MEMORY_LIMIT_MB)
    if not measured:
        capacity = "measurement_unavailable"
    elif peak >= DISCLOUD_ALLOCATION_MB:
        capacity = "exceeds_configured_allocation"
    elif not within_target:
        capacity = "above_provisional_headroom_target"
    else:
        capacity = "within_provisional_headroom_target"
    return {
        "functional_inference_passed": functional,
        "memory_measurement_complete": measured,
        "within_provisional_memory_budget": within_target,
        "estimated_allocation_headroom_mb": headroom,
        "capacity_status": capacity,
        # Green only means text + accurate synthetic vision + RSS collection.
        "passed_minimal_probe": functional and measured,
        "production_capacity_validated": False,
        "discloud_production_validated": False,
        "production_deployment_recommended": False,
        "memory_measurement_scope": "llama-server VmHWM only, not total app/container use",
    }


def publish_action_summary(result: dict) -> None:
    """Keep marginal RAM visibly flagged even on a successful research run."""
    peak = result.get("peak_rss_mb")
    status = "PASS" if result.get("passed_minimal_probe") else "FAIL"
    lines = [
        "## Omni CPU test: " + status,
        "",
        f"- Text request: {'pass' if result.get('text_success') else 'fail'}",
        f"- Synthetic red image: {'pass' if result.get('vision_color_correct') else 'fail'}",
        f"- Peak llama-server RSS: {peak} MB",
        f"- Provisional RAM headroom target: {MEMORY_LIMIT_MB} MB",
        f"- Existing Discloud configured allocation: {DISCLOUD_ALLOCATION_MB} MB",
        f"- Capacity: **{result.get('capacity_status')}**",
        f"- Theoretical remaining RAM: {result.get('estimated_allocation_headroom_mb')} MB",
        "",
        "**NOT APPROVED for production.** GitHub Actions cannot prove Discloud RAM",
        "headroom, gateway overhead, stability, or long-context performance.",
    ]
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
    if result.get("capacity_status") != "within_provisional_headroom_target":
        print(
            "::warning title=Omni capacity not validated::"
            f"Measured {peak} MB; {result.get('capacity_status')}. "
            "Experimental inference result is not production approval.",
            flush=True,
        )


def run(binary: Path, model_dir: Path, report: Path, timeout_s: int = 240) -> dict:
    model, projector = model_dir / MODEL, model_dir / PROJECTOR
    for path in (binary, model, projector):
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [str(binary.resolve()), "--model", str(model.resolve()), "--mmproj", str(projector.resolve()),
               "--host", "127.0.0.1", "--port", str(PORT), "--ctx-size", "2048",
               "--threads", "2", "--parallel", "1", "--n-gpu-layers", "0", "--temp", "0"]
    result = {"type": "isolated_cpu_probe", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "model": MODEL, "projector": PROJECTOR, "model_file_bytes": model.stat().st_size,
              "projector_file_bytes": projector.stat().st_size, "no_production_deployment": True,
              "discloud_allocation_mb": DISCLOUD_ALLOCATION_MB, "provisional_rss_budget_mb": MEMORY_LIMIT_MB,
              "text_success": False, "vision_success": False, "vision_color_correct": False,
              "peak_rss_mb": None}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.with_suffix(".server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=report.parent.resolve(), stdout=log, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            while time.monotonic() - started < timeout_s:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with {process.returncode}")
                try:
                    if request_json(f"http://127.0.0.1:{PORT}/health", timeout=3).get("status") == "ok":
                        break
                except (HTTPError, URLError, OSError, ValueError, TimeoutError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Model backend readiness timed out")
            result["ready_seconds"] = round(time.monotonic() - started, 2)
            result["memory_after_ready_mb"] = vmhwm_mb(process.pid)
            result["text"] = run_case([{"role": "user", "content": "In one sentence, what is a CPU?"}])
            result["text_success"] = result["text"]["success"]
            image_url = "data:image/png;base64," + base64.b64encode(png()).decode()
            result["vision"] = run_case([{"role": "user", "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}])
            result["vision_success"] = result["vision"]["success"]
            result["vision_color_correct"] = bool(
                result["vision_success"] and re.search(
                    r"\bred\b", result["vision"].get("answer_preview", ""), flags=re.IGNORECASE
                )
            )
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            result["fatal_error"] = type(exc).__name__ + ": " + str(exc)[:160]
        finally:
            result["peak_rss_mb"] = vmhwm_mb(process.pid)
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            result["server_exit_code"] = process.returncode
    result.update(classify_probe(result))
    result["audio_video_validated"] = False
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    publish_action_summary(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(".models"))
    parser.add_argument("--report", type=Path, default=Path("reports/omni-probe.json"))
    args = parser.parse_args()
    result = run(args.server, args.model_dir, args.report)
    print(json.dumps(result, indent=2))
    if not result["passed_minimal_probe"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
