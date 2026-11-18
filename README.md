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


## Additional official low-memory quantization comparison

The initial 4-profile Q4_K_M stress run [#43931807643](https://github.com/UglyGameFace/LFM3-Omni-2B/actions/runs/43931807643) finished successfully: all 52 checks passed.

| Q4_K_M profile | Correct | Peak backend RSS | Backend-only headroom (2450 MB) |
|---|---:|---:|---:|
| Baseline 2048 | 13/13 | 2263.0 MB | 187.0 MB |
| Compact 4096 | 13/13 | 2239.9 MB | 210.1 MB |
| Q8 KV 4096 | 13/13 | 2203.3 MB | 246.7 MB |
| Extended 6144 | 13/13 | 2264.4 MB | 185.6 MB |

No Q4_K_M profile reached the provisional 400 MB backend-only headroom target.
The *official* [Liquid AI GGUF tree](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/tree/main)
also includes a smaller Q4_0 base model and projector. A distinct **Omni official
Q4_0 low-memory comparison** workflow downloads those exact two files, checks
their upstream LFS SHA-256 and compares the same four CPU profiles. This uses
no Discloud RAM and changes no RivaL files.

Q4_0 may reduce memory, but quality and reliability must be tested rather than
assumed equivalent to Q4_K_M. The older Q4_K_M stress workflow remains
available manually. Neither workflow can approve production deployment.

## Q4_0 comparison: results (run 43934082561)

The official Q4_0 comparison [#43934082561](https://github.com/UglyGameFace/LFM3-Omni-2B/actions/runs/43934082561)
finished successfully: all 52/52 synthetic checks correct (13/13 per profile),
`functional_and_stress_passed` true on all four profiles,
`production_approved` false, and the research-candidate selection stayed
`lowest_rss_research_candidate: null` (no profile met the headroom target).

| Q4_0 profile | Correct | Peak backend RSS | Backend-only headroom (2450 MB) | P50 | P95 | Startup |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 2048 | 13/13 | 2297.1 MB | 152.9 MB | 2.582s | 11.334s | 3.01s |
| Compact 4096 | 13/13 | 2268.4 MB | 181.6 MB | 2.583s | 23.784s | 3.0s |
| Q8 KV 4096 | 13/13 | 2238.4 MB | 211.6 MB | 2.603s | 26.139s | 3.0s |
| Extended 6144 | 13/13 | 2300.2 MB | 149.8 MB | 2.567s | 40.755s | 3.0s |

On-disk sizes: Q4_0 model 1,049,617,920 B vs Q4_K_M 1,099,630,080 B; Q4_0
projector 270,576,544 B vs Q4_K_M 291,586,720 B (~71 MB smaller total).

**Verdict: Q4_0 is WORSE than Q4_K_M on memory.** Every Q4_0 profile peaks
28–36 MB HIGHER than its Q4_K_M counterpart (Q4_K_M best 2203.3 MB / 246.7 MB
headroom; Q4_0 best 2238.4 MB / 211.6 MB headroom), despite the smaller files
on disk. Neither quant meets the 400 MB advisory headroom target.

**Conclusion: Omni cannot replace LFM3-1.7B inside the 2,450 MB allocation on
either quant.** Keep LFM3-1.7B running. The only viable production path for
Omni is a genuinely separate host (see `deploy/`).

Long-context latency warning: P95 latency balloons with context — 11.334s at
2048 → 40.755s at 6144 (Q4_0) — driven by the long-context canary case. Short
text stays ~0.54s and synthetic images ~2.6s in both quants. Any acceptance
criteria for long sessions must use the 6144 numbers, not the 2048 ones.

## Unified Omni service (local build, NOT deployed)

`service/` is a stdlib-only HTTP service in front of llama-server, built so
JARVIS, RivaL and Watch Yo Mouf can eventually share one Omni backend.
Omni is a **non-reasoning model**; the service configures and reports it as
such and never claims reasoning ability.

- `service/server.py` — text chat + SSE streaming, vision/VQA, structured
  tool calls with JSON-schema validation (registered tools only: `echo`,
  `add`, `clock`), multi-turn history with trimming, per-request timeout,
  client cancellation, managed-backend restart/recovery, Bearer-token auth
  from env, per-client token-bucket rate limits, `/health`, `/ready`,
  `/diagnostics` with structured JSON logs and metrics counters.
  Resource-aware startup **refuses to start** below a 2303 MB budget
  (lowest measured backend peak 2203.3 MB + 100 MB estimated service
  overhead — the overhead is an estimate; no full host measurement exists).
- `clients/omni_client.py` — shared sync client (stdlib urllib).
- `clients/jarvis.py` — streamed conversational turns, image/screen
  understanding, tool actions only with an explicit authorization flag.
- `clients/rival.py` — evidence-grounded explanations ONLY: takes verified
  metrics/market evidence with traceable refs, refuses to invent
  probabilities, market data, trades or model performance. RivaL's
  prediction/qualification engines are untouched.
- `clients/watch_yo_mouf.py` — deterministic keyword screen + audit log
  (low confidence, labelled as such); audio hook is blocked (see matrix).
- `service/audio.py` — STT/TTS/video are capability-gated adapters that
  return **501 blocked_unvalidated** by default. No audio/video backend has
  been validated against this llama.cpp build; nothing fakes success, no
  paid dependency, no second model in RAM.
- `service/reasoning.py` — optional reasoning-backend interface, lazy and
  **unloaded by default (0 MB in RAM)**, plus the compensation map
  (deterministic tools, retrieval, decomposition) and a small benchmark set
  (`scripts/bench_reasoning.py`).
- `scripts/bench_service.py` — reproducible harness: tokens/sec,
  time-to-first-token, image latency, memory vs context, sequential +
  concurrent load, long sessions, tool-call accuracy/failure handling,
  restart/OOM recovery. Fake backend offline; real backend manually.
- `deploy/discloud.config.example` + `deploy/ROLLBACK.md` — config example
  and rollback procedure for a **separate host only**. Nothing is deployed.

### Honest capability matrix

| Capability | Status | Evidence |
|---|---|---|
| Text chat | Working | 52/52 synthetic checks in both matrices (runs 43931807643, 43934082561) |
| Streaming (SSE) | Working | Implemented; exercised against the fake backend in offline tests |
| Image understanding / VQA | Working | Synthetic red/blue images correct in both matrices (52/52) |
| Structured tool calls | Working (offline tests only) | Schema-validated registry; no live tool-call accuracy run yet |
| Multi-turn history | Working (offline tests only) | In-memory sessions with trimming |
| Speech-to-text | **Blocked / unvalidated** | No audio backend validated against this llama.cpp build; endpoint returns 501 |
| Text-to-speech | **Blocked / unvalidated** | Same; endpoint returns 501 |
| Video understanding | **Blocked / unvalidated** | Same; endpoint returns 501 |
| Voice conversation | **Blocked / unvalidated** | Depends on blocked STT/TTS |
| Reasoning | Not applicable | LFM3-Omni-2B is a non-reasoning model by design |

### Acceptance criteria (measured numbers only)

- Short-text P50 reference: ~0.54s; synthetic-image P50: ~2.6s (both quants).
- Long-context P95: 11.334s (2048) → 40.755s (6144) on Q4_0 — long-session
  acceptance must use the 6144 figure.
- Tool calls: 100% of registered-tool calls must validate and execute; 100%
  of unregistered-tool calls must be refused (verified offline in tests).
- Voice first-audio < 2s is **aspirational and not applicable** while TTS is
  blocked; there is no first-audio output to measure.
- Memory: service refuses budgets below 2303 MB; neither quant fits the
  2,450 MB allocation with the 400 MB advisory headroom.

Run the service locally (needs a real llama-server + models for inference;
offline tests use the fake backend):

```bash
python -m unittest discover -s tests -v        # offline, no model download
python scripts/bench_service.py --mode fake    # harness self-check
OMNI_AUTH_TOKEN=<token> python -m service      # real service (backend via env)
```

**Production status: NOT approved.** LFM3-1.7B is preserved and still the
production model. Omni's only viable production path is a separate host.
