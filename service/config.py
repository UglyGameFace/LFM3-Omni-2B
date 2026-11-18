"""Environment configuration for the unified Omni service.

All values come from the environment; no secrets are ever logged or
returned (only whether a token is configured). Resource budgets are
checked against MEASURED evidence, not guesses:

- Q4_K_M matrix, run 43931807643, 52/52 correct. Peak backend RSS:
  baseline_2048 2263.0 MB, compact_4096 2239.9 MB, q8_kv_4096 2203.3 MB,
  extended_6144 2264.4 MB. Best headroom 246.7 MB under 2450 MB.
- Q4_0 matrix, run 43934082561, 52/52 correct. Peak backend RSS:
  baseline_2048 2297.1 MB, compact_4096 2268.4 MB, q8_kv_4096 2238.4 MB,
  extended_6144 2300.2 MB. Best headroom 211.6 MB under 2450 MB.
  Q4_0 peaks 28-36 MB HIGHER per profile than Q4_K_M despite smaller
  files on disk (Q4_0 model 1,049,617,920 B vs Q4_K_M 1,099,630,080 B;
  Q4_0 projector 270,576,544 B vs Q4_K_M 291,586,720 B).
- Neither quant meets the 400 MB advisory headroom target. Omni cannot
  replace LFM3-1.7B inside the 2,450 MB allocation on either quant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os

# --- Verified evidence (verbatim, runs 43931807643 and 43934082561) ---
ALLOCATION_MB = 2450
ADVISORY_HEADROOM_MB = 400
MEASURED_MIN_PEAK_RSS_MB = 2203.3  # Q4_K_M q8_kv_4096, the lowest peak ever measured
MEASURED_MAX_PEAK_RSS_MB = 2300.2  # Q4_0 extended_6144, the highest peak measured
Q4_K_M_BEST_PEAK_MB = 2203.3
Q4_K_M_BEST_HEADROOM_MB = 246.7
Q4_0_BEST_PEAK_MB = 2238.4
Q4_0_BEST_HEADROOM_MB = 211.6
# Gateway + service process overhead is an ESTIMATE, not a measurement:
# no full host/container measurement has ever been performed.
SERVICE_OVERHEAD_MB_ESTIMATE = 100
# Minimum total budget the service will accept: lowest measured backend
# peak plus the (estimated) service overhead. Below this, startup refuses.
MIN_TOTAL_BUDGET_MB = int(MEASURED_MIN_PEAK_RSS_MB) + SERVICE_OVERHEAD_MB_ESTIMATE  # 2303

Q4_K_M_PEAKS_MB = {
    "baseline_2048": 2263.0,
    "compact_4096": 2239.9,
    "q8_kv_4096": 2203.3,
    "extended_6144": 2264.4,
}
Q4_0_PEAKS_MB = {
    "baseline_2048": 2297.1,
    "compact_4096": 2268.4,
    "q8_kv_4096": 2238.4,
    "extended_6144": 2300.2,
}
# Long-context p95 latency balloons with context (Q4_0, driven by the
# long-context canary case): 11.334s at 2048 -> 40.755s at 6144.
Q4_0_P95_SECONDS = {
    "baseline_2048": 11.334,
    "compact_4096": 23.784,
    "q8_kv_4096": 26.139,
    "extended_6144": 40.755,
}


class ResourceBudgetError(ValueError):
    """Configured budget is below the measured minimum; refuse to start."""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8770
    backend_url: str = "http://127.0.0.1:8767"
    backend_command: str = ""  # llama-server binary; empty = do not manage a backend process
    model_path: str = ""
    mmproj_path: str = ""
    variant: str = "q4_k_m"
    memory_budget_mb: int = ALLOCATION_MB
    max_context_tokens: int = 4096
    max_turns: int = 20
    max_history_chars: int = 12000
    max_message_chars: int = 20000
    request_timeout_s: float = 120.0
    backend_start_timeout_s: float = 120.0
    max_concurrent: int = 4
    rate_limit_per_min: int = 60
    rate_burst: int = 10
    backend_max_restarts: int = 3
    auth_token: str = ""  # empty = auth disabled (local dev only); never log this
    allow_no_auth: bool = True
    threads: int = 2
    reasoning_backend: str = "none"  # "none" = unloaded by default; no model held in RAM
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Config":
        src = os.environ if env is None else env

        def get(name: str, default: str = "") -> str:
            value = src.get(name)
            return default if value is None else value

        def get_int(name: str, default: int) -> int:
            raw = src.get(name)
            if raw is None or raw == "":
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

        def get_float(name: str, default: float) -> float:
            raw = src.get(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number, got {raw!r}") from exc

        cfg = cls(
            host=get("OMNI_HOST", "127.0.0.1"),
            port=get_int("OMNI_PORT", 8770),
            backend_url=get("OMNI_BACKEND_URL", "http://127.0.0.1:8767"),
            backend_command=get("OMNI_BACKEND_COMMAND", ""),
            model_path=get("OMNI_MODEL_PATH", ""),
            mmproj_path=get("OMNI_MMPROJ_PATH", ""),
            variant=get("OMNI_VARIANT", "q4_k_m"),
            memory_budget_mb=get_int("OMNI_MEMORY_BUDGET_MB", ALLOCATION_MB),
            max_context_tokens=get_int("OMNI_MAX_CONTEXT_TOKENS", 4096),
            max_turns=get_int("OMNI_MAX_TURNS", 20),
            max_history_chars=get_int("OMNI_MAX_HISTORY_CHARS", 12000),
            max_message_chars=get_int("OMNI_MAX_MESSAGE_CHARS", 20000),
            request_timeout_s=get_float("OMNI_REQUEST_TIMEOUT_S", 120.0),
            backend_start_timeout_s=get_float("OMNI_BACKEND_START_TIMEOUT_S", 120.0),
            max_concurrent=get_int("OMNI_MAX_CONCURRENT", 4),
            rate_limit_per_min=get_int("OMNI_RATE_LIMIT_PER_MIN", 60),
            rate_burst=get_int("OMNI_RATE_BURST", 10),
            backend_max_restarts=get_int("OMNI_BACKEND_MAX_RESTARTS", 3),
            auth_token=get("OMNI_AUTH_TOKEN", ""),
            allow_no_auth=get("OMNI_ALLOW_NO_AUTH", "1") != "0",
            threads=get_int("OMNI_THREADS", 2),
            reasoning_backend=get("OMNI_REASONING_BACKEND", "none"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Refuse impossible configs; warn about unvalidated capacity."""
        errors: list[str] = []
        if self.variant not in ("q4_k_m", "q4_0"):
            errors.append(f"unsupported variant {self.variant!r}; use q4_k_m or q4_0")
        if not 512 <= self.memory_budget_mb <= 65536:
            errors.append(f"memory budget {self.memory_budget_mb} MB out of range 512-65536")
        if self.memory_budget_mb < MIN_TOTAL_BUDGET_MB:
            errors.append(
                f"memory budget {self.memory_budget_mb} MB is below the measured minimum "
                f"{MIN_TOTAL_BUDGET_MB} MB (lowest measured backend peak "
                f"{MEASURED_MIN_PEAK_RSS_MB} MB + {SERVICE_OVERHEAD_MB_ESTIMATE} MB "
                f"estimated service overhead). Refusing to start."
            )
        if not 512 <= self.max_context_tokens <= 32768:
            errors.append("max context tokens out of range 512-32768")
        if self.max_turns < 1 or self.max_history_chars < 100:
            errors.append("history limits too small")
        if self.request_timeout_s <= 0 or self.max_concurrent < 1:
            errors.append("timeout / concurrency out of range")
        if self.rate_limit_per_min < 1 or self.rate_burst < 1:
            errors.append("rate limits must be >= 1")
        if self.reasoning_backend != "none":
            errors.append(
                f"reasoning backend {self.reasoning_backend!r} is not a validated option; "
                "only 'none' (unloaded) is supported — no second model is held in RAM"
            )
        if errors:
            raise ResourceBudgetError("; ".join(errors))
        # Advisory warnings (do not block startup; capacity is unvalidated).
        headroom = self.memory_budget_mb - MEASURED_MIN_PEAK_RSS_MB
        if headroom < ADVISORY_HEADROOM_MB:
            self.warnings.append(
                f"budget headroom vs lowest measured backend peak is {headroom:.1f} MB, "
                f"below the {ADVISORY_HEADROOM_MB} MB advisory target; production "
                "replacement of LFM3-1.7B is NOT approved on this evidence"
            )
        if self.variant == "q4_0":
            self.warnings.append(
                "Q4_0 peaks 28-36 MB HIGHER per profile than Q4_K_M (best 2238.4 MB "
                "vs 2203.3 MB) despite smaller files on disk; Q4_0 is not the "
                "low-memory win it looked like"
            )
        if not self.auth_token and self.allow_no_auth:
            self.warnings.append("auth disabled: OMNI_AUTH_TOKEN is unset (local dev only)")

    def public_summary(self) -> dict:
        """Config safe to expose in /diagnostics: no secrets, ever."""
        return {
            "host": self.host,
            "port": self.port,
            "backend_url": self.backend_url,
            "backend_managed": bool(self.backend_command),
            "variant": self.variant,
            "memory_budget_mb": self.memory_budget_mb,
            "min_total_budget_mb": MIN_TOTAL_BUDGET_MB,
            "measured_min_peak_rss_mb": MEASURED_MIN_PEAK_RSS_MB,
            "measured_max_peak_rss_mb": MEASURED_MAX_PEAK_RSS_MB,
            "q4_k_m_best_peak_mb": Q4_K_M_BEST_PEAK_MB,
            "q4_0_best_peak_mb": Q4_0_BEST_PEAK_MB,
            "advisory_headroom_mb": ADVISORY_HEADROOM_MB,
            "max_context_tokens": self.max_context_tokens,
            "max_turns": self.max_turns,
            "max_history_chars": self.max_history_chars,
            "request_timeout_s": self.request_timeout_s,
            "max_concurrent": self.max_concurrent,
            "rate_limit_per_min": self.rate_limit_per_min,
            "rate_burst": self.rate_burst,
            "backend_max_restarts": self.backend_max_restarts,
            "auth_enabled": bool(self.auth_token),
            "reasoning_backend": self.reasoning_backend,
            "reasoning_model_loaded": False,
            "warnings": list(self.warnings),
        }
