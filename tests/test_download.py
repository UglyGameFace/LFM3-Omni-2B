import hashlib
from pathlib import Path
import sys
import tempfile
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_models as d


class DownloadTests(TestCase):
    def test_official_manifest(self):
        rows = [{"path": filename, "type": "file", "lfs": {
            "oid": "a" * 64, "size": [1_100_000_000, 292_000_000][i]}}
            for i, filename in enumerate(d.FILES)]
        self.assertEqual(len(d.metadata(rows)), 2)
        with self.assertRaises(ValueError):
            d.metadata(rows + [rows[0]])

    def test_rejects_invalid_manifest(self):
        with self.assertRaises(ValueError):
            d.metadata([])
        with self.assertRaises(ValueError):
            d.metadata([{"path": d.FILES[0], "type": "file", "lfs": {"oid": "invalid", "size": 1_100_000_000}}])

    def test_corrupt_existing_file_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / d.FILES[0]
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                d.download(Path(folder), d.FILES[0], 1_100_000_000, "a" * 64)
            self.assertEqual(path.read_bytes(), b"bad")

    def test_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture"
            path.write_bytes(b"hello")
            self.assertEqual(d.sha256_file(path), hashlib.sha256(b"hello").hexdigest())
