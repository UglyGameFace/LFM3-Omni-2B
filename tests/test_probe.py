from pathlib import Path
import struct
import sys
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe


class ProbeTests(TestCase):
    def test_synthetic_image(self):
        data = probe.png()
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(struct.unpack(">II", data[16:24]), (16, 16))

    def test_response_must_have_content(self):
        self.assertEqual(probe.answer({"choices": [{"message": {"content": "ok"}}]}), "ok")
        with self.assertRaises(ValueError):
            probe.answer({"choices": [{"message": {"content": ""}}]})

    def test_reserved_host_ram(self):
        self.assertLess(probe.MEMORY_LIMIT_MB, 2450)
        self.assertEqual(probe.PORT, 8767)

    def test_real_november_16_cpu_result_is_functional_pass_with_warning(self):
        result = probe.classify_probe({
            "text_success": True, "vision_success": True,
            "vision_color_correct": True, "peak_rss_mb": 2188.2,
        })
        self.assertTrue(result["passed_minimal_probe"])
        self.assertFalse(result["within_provisional_memory_budget"])
        self.assertEqual(result["estimated_allocation_headroom_mb"], 261.8)
        self.assertEqual(result["capacity_status"], "above_provisional_headroom_target")
        self.assertFalse(result["production_deployment_recommended"])

    def test_missing_memory_measurement_fails(self):
        result = probe.classify_probe({
            "text_success": True, "vision_success": True,
            "vision_color_correct": True, "peak_rss_mb": None,
        })
        self.assertFalse(result["passed_minimal_probe"])
        self.assertEqual(result["capacity_status"], "measurement_unavailable")

    def test_bad_vision_response_fails_even_with_ram_headroom(self):
        result = probe.classify_probe({
            "text_success": True, "vision_success": True,
            "vision_color_correct": False, "peak_rss_mb": 1800,
        })
        self.assertFalse(result["passed_minimal_probe"])

    def test_overallocation_warns_even_when_inference_works(self):
        result = probe.classify_probe({
            "text_success": True, "vision_success": True,
            "vision_color_correct": True, "peak_rss_mb": 2600,
        })
        self.assertTrue(result["passed_minimal_probe"])
        self.assertEqual(result["capacity_status"], "exceeds_configured_allocation")
        self.assertFalse(result["discloud_production_validated"])

    def test_below_advisory_target_is_still_not_production_validated(self):
        result = probe.classify_probe({
            "text_success": True, "vision_success": True,
            "vision_color_correct": True, "peak_rss_mb": 1900,
        })
        self.assertTrue(result["passed_minimal_probe"])
        self.assertEqual(result["capacity_status"], "within_provisional_headroom_target")
        self.assertFalse(result["production_capacity_validated"])
