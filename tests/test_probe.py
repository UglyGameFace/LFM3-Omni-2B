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
