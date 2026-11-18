import sys
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_service


class BenchTests(TestCase):
    def test_fake_mode_bench_runs_offline(self):
        bench = bench_service.Bench("fake")
        result = bench.run()
        self.assertEqual(result["mode"], "fake")
        self.assertEqual(result["latency"]["short_text"]["correct"], "3/3")
        self.assertEqual(result["latency"]["image"]["correct"], "3/3")
        self.assertEqual(result["latency"]["long_context"]["correct"], "3/3")
        self.assertTrue(all(c["passed"] for c in result["tool_call_accuracy"]))
        self.assertTrue(all(c["passed"] for c in result["audio_video_blocked"]))
        self.assertEqual(result["restart_recovery"]["status"], "skipped")
        self.assertEqual(result["reasoning_bench"]["total"], 8)

    def test_acceptance_criteria_record_voice_as_not_applicable(self):
        self.assertIn("NOT APPLICABLE", bench_service.ACCEPTANCE["voice_first_audio_s"])
        self.assertIn("40.755", bench_service.ACCEPTANCE["long_context_p95_note"])
