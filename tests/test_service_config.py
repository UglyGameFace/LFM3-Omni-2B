import json
import sys
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.config import (ADVISORY_HEADROOM_MB, ALLOCATION_MB, Config,
                            MIN_TOTAL_BUDGET_MB, Q4_0_BEST_PEAK_MB,
                            Q4_K_M_BEST_PEAK_MB, ResourceBudgetError)


def env(**overrides):
    base = {"OMNI_AUTH_TOKEN": "", "OMNI_ALLOW_NO_AUTH": "1"}
    base.update(overrides)
    return base


class ConfigTests(TestCase):
    def test_defaults_pass_with_headroom_warning(self):
        cfg = Config.from_env(env())
        self.assertEqual(cfg.memory_budget_mb, ALLOCATION_MB)
        self.assertTrue(any("400 MB" in w for w in cfg.warnings))

    def test_budget_below_measured_minimum_refuses(self):
        with self.assertRaises(ResourceBudgetError):
            Config.from_env(env(OMNI_MEMORY_BUDGET_MB="2000"))
        with self.assertRaises(ResourceBudgetError):
            Config.from_env(env(OMNI_MEMORY_BUDGET_MB=str(MIN_TOTAL_BUDGET_MB - 1)))

    def test_minimum_budget_accepted(self):
        cfg = Config.from_env(env(OMNI_MEMORY_BUDGET_MB=str(MIN_TOTAL_BUDGET_MB)))
        self.assertEqual(cfg.memory_budget_mb, MIN_TOTAL_BUDGET_MB)

    def test_q4_0_warns_it_is_worse_on_memory(self):
        cfg = Config.from_env(env(OMNI_VARIANT="q4_0"))
        self.assertTrue(any("28-36 MB HIGHER" in w for w in cfg.warnings))

    def test_q4_0_best_peak_is_higher_than_q4_k_m(self):
        # Verified: Q4_0 best 2238.4 MB vs Q4_K_M best 2203.3 MB.
        self.assertEqual(Q4_0_BEST_PEAK_MB, 2238.4)
        self.assertEqual(Q4_K_M_BEST_PEAK_MB, 2203.3)
        self.assertGreater(Q4_0_BEST_PEAK_MB, Q4_K_M_BEST_PEAK_MB)
        self.assertEqual(round(ALLOCATION_MB - Q4_0_BEST_PEAK_MB, 1), 211.6)
        self.assertEqual(round(ALLOCATION_MB - Q4_K_M_BEST_PEAK_MB, 1), 246.7)

    def test_neither_quant_meets_advisory_headroom(self):
        self.assertLess(ALLOCATION_MB - Q4_0_BEST_PEAK_MB, ADVISORY_HEADROOM_MB)
        self.assertLess(ALLOCATION_MB - Q4_K_M_BEST_PEAK_MB, ADVISORY_HEADROOM_MB)

    def test_unknown_variant_and_reasoning_backend_refused(self):
        with self.assertRaises(ResourceBudgetError):
            Config.from_env(env(OMNI_VARIANT="q8_0"))
        with self.assertRaises(ResourceBudgetError):
            Config.from_env(env(OMNI_REASONING_BACKEND="some-model"))

    def test_public_summary_never_exposes_the_token(self):
        cfg = Config.from_env(env(OMNI_AUTH_TOKEN="super-secret-token"))
        dumped = json.dumps(cfg.public_summary())
        self.assertNotIn("super-secret-token", dumped)
        self.assertTrue(cfg.public_summary()["auth_enabled"])

    def test_bad_env_values_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_env(env(OMNI_PORT="not-a-port"))
