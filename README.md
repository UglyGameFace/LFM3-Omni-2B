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

## November 16, 2026: first actual GitHub CPU probe

The model completed text (1.13 seconds) and synthetic red-image recognition (2.44 seconds) correctly. Its measured peak llama-server process RSS was **2,188.2 MB**. The first workflow failed solely because the code treated a **2,100 MB advisory memory target** as a mandatory functional test.

The corrected benchmark separates **functional inference** from **capacity advisories**. Crossing the soft target still produces a visible GitHub Actions warning, a structured capacity status, and **no production approval**. Theoretically, 2,450 MB minus 2,188.2 MB is **261.8 MB**, which does not include other app processes, longer contexts, load spikes, or Discloud-specific overhead. Production replacement remains blocked pending real host measurements.

The updated workflow has not been rerun on the production hardware; use the manual GitHub Actions workflow for a repeat isolated measurement.


## Extended isolation: memory and stability matrix

The manual **Omni CPU memory and stability matrix** workflow tests four
sequential CPU-only configurations using the exact same official Q4 model and
multimodal projector:

| Profile | Context | Memory technique |
| --- | ---: | --- |
| baseline_2048 | 2,048 | Reproduces first benchmark's baseline arguments |
| compact_4096 | 4,096 | Smaller 128-token batch and 64-token microbatch |
| q8_kv_4096 | 4,096 | Reduced batches plus q8_0 key/value caches |
| extended_6144 | 6,144 | Larger context, smaller batches |

Each profile launches a fresh, loopback-only llama-server process, waits for
readiness, then runs 3 rounds of arithmetic, exact text, synthetic red-image
and blue-image recognition, plus one fixed long-context canary-retrieval test.
Results record correctness, request failures, median/P95 response latency,
startup latency, and backend-process peak resident memory. The raw structured
JSON artifact records **which** profile fails rather than silently choosing
another model. The workflow is not scheduled. It also runs automatically
on the first main-branch push that adds the test harness.

The report uses an **advisory, not validated** 400 MB headroom target against the
**configured** 2,450 MB Discloud allocation. Other host processes, request
concurrency, gateway overhead, actual Discloud cgroup memory metrics, audio and
video performance are still unverified. No result is allowed to authorize
production replacement automatically. The RivaL service and Discloud are not
contacted.

The previous minimal text-and-image smoke test remains separate and can be
run independently. For local isolated testing:

    python -m unittest discover -s tests -v
    python scripts/stress_matrix.py --server build/llama.cpp/build/bin/llama-server --model-dir .models --output reports/omni-stress-matrix.json

The matrix reports an experimental research candidate, not a production
recommendation. Do not deploy unless full application memory, stability,
and rollback tests pass in a suitable isolated host.
