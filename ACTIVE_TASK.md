# ACTIVE_TASK: Isolated Omni CPU validation

**Updated:** 2026-11-16
**Repository:** UglyGameFace/LFM3-Omni-2B
**Production boundary:** Do not modify RivaL, RivaL-Intelligence, Jarvis,
Discloud allocations, DNS, or credentials.

## Objective
Establish whether official LFM3-Omni-2B text/image/longer-context CPU inference
can be stable and fit the existing 2,450 MB Discloud LLM allocation **without
interruption to RivaL**, or else preserve the existing production LFM3-1.7B.

## Verified evidence
- Original synthetic text/image smoke test [#43927795421](https://github.com/UglyGameFace/LFM3-Omni-2B/actions/runs/43927795421) passed, peak backend RSS 2188.2 MB; provisional 2100 MB target exceeded.
- Official Q4_K_M CPU matrix [#43931807643](https://github.com/UglyGameFace/LFM3-Omni-2B/actions/runs/43931807643) completed successfully, with all 52/52 synthetic cases correct.
- Peak backend RSS per profile: baseline_2048 2263.0 MB, compact_4096 2239.9 MB, q8_kv_4096 2203.3 MB, extended_6144 2264.4 MB.
- The minimum peak, 2203.3 MB, leaves just 246.7 MB *theoretical* backend-only headroom under the configured 2450 MB allocation.
- No tested Q4_K_M configuration meets the provisional 400 MB headroom target.
- No deployment or full host memory measurement has been performed.

## Current work
- Run the isolated official Q4_0 model + projector comparison via GitHub Actions.
- Keep initial Q4_K_M proof intact, and do not trade away model quality silently.

## Remaining blockers
- No actual memory/cgroup measurement on Discloud with the gateway, same
  workloads, crashes/restarts and private VLAN interaction; **production
  migration is NOT authorized by these CPU benchmarks**.
- Longer multi-turn conversations, continuous load, real voice, speech output,
  video, tool-call correctness and production stability still need coverage.
- Deployment rollback must be tested before any switch.

## Current implementation
- Single canonical main branch.
- CI: Python compilation and unit tests.
- Workflows: original manual Q4_K_M CPU stress and manually runnable/first-push
  Q4_0 memory research. No scheduled/recurrent testing, no Discloud deploy.
- All model artifacts download from official LiquidAI only with SHA-256
  validation and remain excluded from Git.

## Next engineering decision
Use completed Q4_0 evidence to compare correctness and real peak RSS with
Q4_K_M. If adequate memory margin cannot be demonstrated, keep LFM3-1.7B
running and consider a genuinely separate host rather than forcing Omni onto
the existing 2450 MB allocation.
