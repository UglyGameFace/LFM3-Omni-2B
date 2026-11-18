# Rollback procedure — unified Omni service

> Nothing here has ever been deployed. This procedure is written BEFORE
> any deploy so the rollback is rehearsed, not improvised. Production
> replacement of LFM3-1.7B is NOT approved; Omni needs a separate host.

## Rollback SHAs (fill in at deploy time — do NOT guess)

| Item | Value |
|---|---|
| Omni service commit deployed | _TO FILL AT DEPLOY TIME_ |
| Previous known-good commit | _TO FILL AT DEPLOY TIME_ |
| LFM3-1.7B service commit / deploy | _TO FILL — the service Omni must never disturb_ |
| Discloud app ID / host | _TO FILL AT DEPLOY TIME_ |

## Rollback triggers (any one is enough)

- `/ready` returns 503 for more than 5 minutes after a deploy.
- Backend RSS exceeds the configured budget, or the process is OOM-killed.
- Chat error rate rises, or tool calls execute unregistered tools.
- Any audio/video endpoint returns anything other than 501.
- LFM3-1.7B (the preserved production service) shows any degradation.

## Rollback steps (separate host)

1. Stop the Omni service on the separate host (do NOT touch the LFM3-1.7B app).
2. Restore the previous known-good commit from the table above
   (`git checkout <previous known-good commit>` or redeploy its zip).
3. Restart the service and wait for `/health` → `{"status": "ok"}`.
4. Confirm `/ready` → 200 and `/diagnostics` shows the previous version
   and the expected variant (`q4_k_m` unless Q4_0 was the deployed one).
5. Run one text probe and one synthetic-image probe; both must succeed.
6. Confirm LFM3-1.7B is still serving independently (its own health check).
7. Record the rollback (commit SHAs, time, reason) in ACTIVE_TASK.md.

## Rollback steps (if Omni was ever placed on the 2,450 MB allocation — NOT approved)

Do not do this. The allocation cannot hold Omni on either quant
(Q4_K_M best headroom 246.7 MB, Q4_0 best 211.6 MB, 400 MB target unmet).
If it ever happens by mistake, the rollback is: stop Omni immediately,
restart/redeploy LFM3-1.7B from its own known-good commit, and verify
RivaL Intelligence is serving before anything else.

## What rollback does NOT fix

- A bad model file (re-download with `scripts/download_models.py`, which
  SHA-256-verifies against official LiquidAI LFS metadata).
- A wrong budget (the service refuses to start below 2303 MB by design;
  that refusal IS the safety mechanism — do not bypass it).
