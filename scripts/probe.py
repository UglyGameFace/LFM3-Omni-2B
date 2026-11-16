#!/usr/bin/env python3
"""Single local CPU text/vision probe. Synthetic inputs only, no deployment."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
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
MEMORY_LIMIT_MB = 2100  # provisional headroom below 2450 MB Discloud allocation


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
              "discloud_allocation_mb": 2450, "provisional_rss_budget_mb": MEMORY_LIMIT_MB,
              "text_success": False, "vision_success": False, "peak_rss_mb": None}
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
    result["within_provisional_memory_budget"] = (result["peak_rss_mb"] is not None and result["peak_rss_mb"] <= MEMORY_LIMIT_MB)
    result["passed_minimal_probe"] = (result["text_success"] and result["vision_success"] and result["within_provisional_memory_budget"])
    result["audio_video_validated"] = False
    result["discloud_production_validated"] = False
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
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
