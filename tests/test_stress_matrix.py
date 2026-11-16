from pathlib import Path
import struct
import sys
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stress_matrix as m


class StressMatrixTests(TestCase):
    def test_profiles_have_unique_names(self):
        self.assertEqual(len({p.name for p in m.PROFILES}), len(m.PROFILES))
        self.assertEqual(m.PROFILES[0].name, "baseline_2048")
        self.assertEqual(m.PROFILES[1].context, 4096)

    def test_no_public_listening_and_no_gpu(self):
        argv = m.build_command(Path("/bin/echo"), Path("model"), Path("projector"), m.PROFILES[1])
        self.assertIn("127.0.0.1", argv)
        self.assertIn("--n-gpu-layers", argv)
        self.assertEqual(argv[argv.index("--n-gpu-layers") + 1], "0")
        self.assertEqual(argv[argv.index("--batch-size") + 1], "128")

    def test_kv_flags_only_on_quantized_profile(self):
        plain = m.build_command(Path("/bin/echo"), Path("a"), Path("b"), m.PROFILES[1])
        quant = m.build_command(Path("/bin/echo"), Path("a"), Path("b"), m.PROFILES[2])
        self.assertNotIn("--cache-type-k", plain)
        self.assertIn("--cache-type-k", quant)

    def test_generated_image_is_png(self):
        red = m.solid_png((255, 0, 0))
        blue = m.solid_png((0, 0, 255))
        self.assertTrue(red.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertNotEqual(red, blue)
        self.assertEqual(struct.unpack(">II", red[16:24]), (16, 16))

    def test_bad_color_rejected(self):
        with self.assertRaises(ValueError):
            m.solid_png((400, 0, 0))

    def test_deterministic_scoring(self):
        self.assertTrue(m.is_correct("12", "math"))
        self.assertFalse(m.is_correct("21", "math"))
        self.assertTrue(m.is_correct("PING", "ping"))
        self.assertFalse(m.is_correct("PONG", "ping"))
        self.assertTrue(m.is_correct("The image is blue.", "blue"))
        self.assertFalse(m.is_correct("The image is blue.", "red"))
        self.assertTrue(m.is_correct("NEBULA", "canary"))

    def test_synthetic_case_count(self):
        cases = m.synthetic_cases(m.PROFILES[0], repetitions=3)
        self.assertEqual(len(cases), 13)
        self.assertTrue(all(len(c) == 3 for c in cases))

    def test_full_success_still_not_production_approved(self):
        data = {"ready": True, "peak_backend_rss_mb": 1940,
                "requests": [{"request_ok": True, "correct": True, "seconds": 1.2}]}
        result = m.classify_profile(data)
        self.assertTrue(result["functional_and_stress_passed"])
        self.assertTrue(result["meets_advisory_400mb_headroom"])
        self.assertFalse(result["safe_for_discloud"])
        self.assertFalse(result["production_deployment_recommended"])

    def test_healthy_function_but_insufficient_headroom(self):
        data = {"ready": True, "peak_backend_rss_mb": 2188.2,
                "requests": [{"request_ok": True, "correct": True, "seconds": 2.1}]}
        result = m.classify_profile(data)
        self.assertTrue(result["functional_and_stress_passed"])
        self.assertFalse(result["meets_advisory_400mb_headroom"])
        self.assertEqual(result["estimated_backend_only_headroom_mb"], 261.8)

    def test_failed_request_disqualifies(self):
        data = {"ready": True, "peak_backend_rss_mb": 1700,
                "requests": [{"request_ok": True, "correct": False, "seconds": 1.2}]}
        self.assertFalse(m.classify_profile(data)["functional_and_stress_passed"])

    def test_no_measurement_disqualifies(self):
        data = {"ready": True, "peak_backend_rss_mb": None,
                "requests": [{"request_ok": True, "correct": True, "seconds": 1.2}]}
        self.assertFalse(m.classify_profile(data)["functional_and_stress_passed"])

    def test_candidate_selection_remains_research_only(self):
        profile = {"profile": "compact_4096", "peak_backend_rss_mb": 1800,
                   "functional_and_stress_passed": True,
                   "meets_advisory_400mb_headroom": True}
        result = m.choose_research_candidate([profile])
        self.assertEqual(result["lowest_rss_research_candidate"], "compact_4096")
        self.assertFalse(result["production_approved"])

    def test_p95(self):
        self.assertIsNone(m.percentile95([]))
        self.assertEqual(m.percentile95([1, 2, 3, 4, 5]), 5)
