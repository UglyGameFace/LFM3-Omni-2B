# ACTIVE_TASK: Omni validation + unified service build

**Updated:** 2026-11-17
**Repository:** UglyGameFace/LFM3-Omni-2B
**Branch:** fix/omni-unified-service-20261117 (local only — not pushed, no PR)
**Production boundary:** Do not modify RivaL, RivaL-Intelligence, Jarvis,
Watch Yo Mouf, Discloud allocations, DNS, or credentials. Production
replacement of LFM3-1.7B is NOT approved; LFM3-1.7B is preserved.

## Objective
Establish whether official LFM3-Omni-2B text/image/longer-context CPU inference
can be stable and fit the existing 2,450 MB Discloud LLM allocation **without
interruption to RivaL**, or else preserve the existing production LFM3-1.7B —
and build the unified Omni inference service so JARVIS, RivaL and Watch Yo
Mouf can eventually share one backend (on a separate host, if anywhere).

## Verified evidence
- Original synthetic text/image smoke test [#43927795421](https://github.com/UglyGameFace/LFM3-Omni-2B/actions/runs/43927795421) passed, peak backend RSS 2188.2 MB; provisional 2100 MB target exceeded.
- Official Q4_K_M CPU matrix [#43931807643](https://github.com/UglyGameFace/LFM3-Omni-2B/actions/runs/43931807643) completed successfully, with all 52/52 synthetic cases correct.
- Peak backend RSS per profile: baseline_2048 2263.0 MB, compact_4096 2239.9 MB, q8_kv_4096 2203.3 MB, extended_6144 2264.4 MB.
- The minimum peak, 2203.3 MB, leaves just 246.7 MB *theoretical* backend-only headroom under the configured 2450 MB allocation.
- No tested Q4_K_M configuration meets the provisional 400 MB headroom target.
- No deployment or full host memory measurement has been performed.
- Official Q4_0 CPU matrix [#43934082561](https://github.com/UglyGameFace/LFM3-Omni-2B/actions/runs/43934082561) completed successfully, all 52/52 synthetic checks correct (13/13 per profile), functional_and_stress_passed true on all four profiles, production_approved false, selection lowest_rss_research_candidate null.
- Q4_0 peak backend RSS per profile: baseline_2048 2297.1 MB (p50 2.582s, p95 11.334s, startup 3.01s, headroom 152.9 MB); compact_4096 2268.4 MB (p50 2.583s, p95 23.784s, startup 3.0s, headroom 181.6 MB); q8_kv_4096 2238.4 MB (p50 2.603s, p95 26.139s, startup 3.0s, headroom 211.6 MB); extended_6144 2300.2 MB (p50 2.567s, p95 40.755s, startup 3.0s, headroom 149.8 MB).
- Q4_0 on-disk sizes: model 1,049,617,920 B vs Q4_K_M 1,099,630,080 B; projector 270,576,544 B vs Q4_K_M 291,586,720 B (~71 MB smaller total).

## Verdict (2026-11-17)
- **Q4_0 is WORSE than Q4_K_M on memory**: every Q4_0 profile peaks 28–36 MB
  HIGHER than its Q4_K_M counterpart (Q4_K_M best 2203.3 MB / 246.7 MB
  headroom; Q4_0 best 2238.4 MB / 211.6 MB headroom), despite smaller files
  on disk. Neither quant meets the 400 MB advisory headroom target.
- **Conclusion: Omni cannot replace LFM3-1.7B inside the 2,450 MB allocation
  on either quant.** Keep LFM3-1.7B running. The only viable production path
  for Omni is a genuinely separate host (see deploy/).
- Long-context p95 latency balloons with context (Q4_0: 11.334s at 2048 →
  40.755s at 6144), driven by the long-context case. Long-session acceptance
  criteria must use the 6144 figure.

## Unified service build (local, this branch)
- `service/` — stdlib-only HTTP service over llama-server: text chat + SSE
  streaming, vision/VQA, schema-validated tool calls (registry: echo/add/
  clock; unregistered tools never execute), multi-turn history with
  trimming, per-request timeout, client cancellation, managed-backend
  restart/recovery, Bearer auth from env, per-client rate limits,
  /health /ready /diagnostics, structured logs + metrics. Startup refuses
  budgets below 2303 MB (2203.3 MB lowest measured peak + 100 MB estimated
  overhead; the overhead figure is an estimate, not a measurement).
- `service/audio.py` — STT/TTS/video capability-gated: 501
  blocked_unvalidated by default. No audio/video backend has been validated
  against this llama.cpp build; nothing fakes success.
- `service/reasoning.py` — Omni is a NON-reasoning model. Optional reasoning
  backend interface is lazy and unloaded by default (0 MB RAM); deterministic
  compensation map + 8-case benchmark set included.
- `clients/` — shared OmniClient + JARVIS / RivaL / Watch Yo Mouf adapters
  with simulated integration tests against a fake backend. RivaL adapter is
  evidence-grounded explanations only and refuses to invent probabilities,
  market data, trades or performance. The three projects are NOT modified.
- `scripts/bench_service.py`, `scripts/bench_reasoning.py` — reproducible
  harnesses (fake backend offline; real backend manual).
- `deploy/discloud.config.example`, `deploy/ROLLBACK.md` — separate-host
  config example + rollback procedure. Nothing deployed; Discloud untouched.

## Capability status
- Working: text chat, streaming, image understanding (both matrices 52/52).
- Working (offline tests only): tool calls, multi-turn history.
- Blocked/unvalidated: speech-to-text, text-to-speech, video, voice
  conversation (no validated audio/video backend; endpoints return 501).
- Not applicable: reasoning (non-reasoning model by design).

## Remaining blockers
- No actual memory/cgroup measurement on any real host with the gateway,
  same workloads, crashes/restarts; **production migration is NOT
  authorized**. A separate host is the only viable production path.
- Real voice, speech output, video and live tool-call accuracy still need
  end-to-end validation against a real backend (never done).
- Deployment rollback must be tested on the separate host before any switch.
- Long sessions on CPU are slow at large contexts (p95 40.755s at 6144).

## Current implementation
- main + local branch fix/omni-unified-service-20261117 (not pushed).
- CI: Python compilation and unit tests, plus an offline service-test job.
  No scheduled/recurrent testing, no Discloud deploy.
- All model artifacts download from official LiquidAI only with SHA-256
  validation and remain excluded from Git.

## Next engineering decision
Felix reviews the local diff. If Omni is wanted in production, provision a
separate host (≥2450 MB + real headroom), run the real-backend benchmarks
there, and test rollback — do not force Omni onto the 2450 MB allocation.
