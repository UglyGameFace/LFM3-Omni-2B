#!/usr/bin/env python3
"""Download the exact LiquidAI GGUF model and multimodal projector, with LFS SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "LiquidAI/LFM3-Omni-2B-GGUF"
REVISION = "main"
FILES = ("LFM3-Omni-2B-Q4_K_M.gguf", "mmproj-LFM3-Omni-2B-Q4_K_M.gguf")
BOUNDS = ((900_000_000, 1_500_000_000), (200_000_000, 500_000_000))
SHA = re.compile(r"^[a-f0-9]{64}$")


def metadata(rows: list[dict]) -> dict[str, tuple[int, str]]:
    if not isinstance(rows, list):
        raise ValueError("Invalid upstream manifest")
    output = {}
    for index, name in enumerate(FILES):
        matches = [row for row in rows if isinstance(row, dict)
                   and row.get("type") == "file" and row.get("path") == name]
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate official file: {name}")
        lfs = matches[0].get("lfs")
        if not isinstance(lfs, dict):
            raise ValueError(f"Unverified LFS metadata: {name}")
        size, digest = lfs.get("size"), lfs.get("oid")
        low, high = BOUNDS[index]
        if type(size) is not int or not low <= size <= high or not isinstance(digest, str) or not SHA.fullmatch(digest):
            raise ValueError(f"Invalid size or digest: {name}")
        output[name] = (size, digest)
    return output


def manifest() -> dict[str, tuple[int, str]]:
    url = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=false"
    with urlopen(Request(url, headers={"Accept": "application/json", "User-Agent": "omni-cpu-probe/1"}), timeout=30) as src:
        raw = src.read(600_001)
    if len(raw) > 600_000:
        raise ValueError("Manifest exceeds size bound")
    return metadata(json.loads(raw))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(folder: Path, name: str, size: int, digest: str) -> Path:
    path = folder / name
    if path.exists():
        if path.is_file() and path.stat().st_size == size and sha256_file(path) == digest:
            return path
        raise ValueError(f"Refusing to overwrite unverified file: {name}")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{quote(name)}"
    part = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".omni-", suffix=".part", dir=folder, delete=False) as target:
            part = Path(target.name)
            actual = hashlib.sha256()
            total = 0
            with urlopen(Request(url, headers={"User-Agent": "omni-cpu-probe/1"}), timeout=180) as src:
                while block := src.read(1024 * 1024):
                    total += len(block)
                    if total > size:
                        raise ValueError("Download exceeded manifest size")
                    target.write(block)
                    actual.update(block)
            target.flush()
            os.fsync(target.fileno())
        if total != size or actual.hexdigest() != digest:
            raise ValueError("Downloaded model failed SHA-256/length verification")
        part.replace(path)
        return path
    finally:
        if part is not None and part.exists():
            part.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(".models"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, (size, digest) in manifest().items():
        path = download(args.output_dir, name, size, digest)
        result[name] = {"bytes": size, "sha256": digest, "path": str(path)}
    print(json.dumps({"repository": REPO, "files": result}, indent=2))


if __name__ == "__main__":
    main()
