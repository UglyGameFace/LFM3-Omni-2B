#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKOUT="$ROOT/build/llama.cpp"
REF=b12024
EXPECTED_SHA=ddf48f85c933c1f438212a02eb21e7b001d2d37d
for tool in git cmake c++; do
  command -v "$tool" >/dev/null || { echo "Missing $tool" >&2; exit 1; }
done
mkdir -p "$ROOT/build"
if [[ ! -d "$CHECKOUT/.git" ]]; then
  git clone --depth 1 --branch "$REF" https://github.com/ggml-org/llama.cpp.git "$CHECKOUT"
fi
[[ "$(git -C "$CHECKOUT" rev-parse HEAD)" == "$EXPECTED_SHA" ]] || { echo "Unexpected llama.cpp source SHA" >&2; exit 1; }
cmake -S "$CHECKOUT" -B "$CHECKOUT/build" -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$CHECKOUT/build" --target llama-server --parallel 2
"$CHECKOUT/build/bin/llama-server" --help >/dev/null
