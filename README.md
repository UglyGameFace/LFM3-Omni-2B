# LFM3-Omni-2B isolated evaluation

This public repository benchmarks [Liquid AI's LFM3-Omni-2B GGUF](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) on **temporary GitHub Actions CPU runners**, without running a second model on Discloud.

**No connection to RivaL, Jarvis, Discloud, databases, trading systems, or production credentials.** Existing RivaL Intelligence continues using its independently deployed LFM3-1.7B model.

## Run it

1. Run the lightweight **Source validation** workflow on `main` (also runs on pushes).
2. In **Actions → Omni isolated CPU probe → Run workflow**, start an optional, manual benchmark.
3. Inspect the small `omni-isolated-report` artifact for text/vision outcomes, latency, and maximum RSS. The workflow can fail if the model, runner or network is incompatible; it does **not** deploy anything.

Standard GitHub-hosted Actions on public repositories do not consume private-repo minutes. This repo keeps the heavyweight workflow manual, never on a schedule.

### Reproducible local commands (Linux x86_64)

```bash
python -m unittest discover -s tests -v
bash scripts/build_llama.sh
python scripts/download_models.py --output-dir .models
python scripts/probe.py --server build/llama.cpp/build/bin/llama-server --model-dir .models --report reports/omni-probe.json
```

The runtime is pinned to llama.cpp `b12024`, newer than Liquid AI's documented `b11960` minimum. The download script validates official upstream LFS metadata and the full SHA-256 of both `Q4_K_M` weights and projector.

**Important limitations:** A successful GitHub CPU test is **not** evidence that Omni fits into RivaL Intelligence's configured 2,450 MB Discloud allocation. It also does not validate voice, video, speech output, production authentication, or actual model usefulness. Peak resident memory is a provisional measurement, not a production guarantee. The first experimental workflow only exercises text and a synthetic image.

Never add production secrets, model weights, or private user data to this public repository. See [SECURITY.md](SECURITY.md). Review the upstream LFM license before using the model commercially.
