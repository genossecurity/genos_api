import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from genos.huggingface import DEFAULT_BACKBONE, resolve_backbone


class HuggingFacePathTests(unittest.TestCase):
    def test_explicit_local_directory_is_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            source, local_only = resolve_backbone(temp)
            self.assertEqual(source, temp)
            self.assertTrue(local_only)

    def test_missing_explicit_path_can_be_forced_offline(self):
        missing = "/tmp/genos-codebert-that-does-not-exist"
        with patch.dict(os.environ, {"GENOS_HF_LOCAL_ONLY": "1"}, clear=False):
            source, local_only = resolve_backbone(missing)
        self.assertEqual(source, missing)
        self.assertTrue(local_only)

    def test_cache_snapshot_is_preferred_to_network(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "hub" / "models--microsoft--codebert-base"
            snapshot = repo / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
            with patch.dict(os.environ, {"HF_HOME": temp}, clear=False), patch(
                "genos.huggingface.DEFAULT_LOCAL_BACKBONE", Path(temp) / "missing"
            ):
                source, local_only = resolve_backbone()
        self.assertEqual(source, str(snapshot))
        self.assertTrue(local_only)

    def test_without_local_artifacts_preserves_hf_default(self):
        with patch("genos.huggingface.DEFAULT_LOCAL_BACKBONE", Path("/tmp/missing-codebert")), patch(
            "genos.huggingface._cached_snapshot", return_value=None
        ), patch.dict(os.environ, {"GENOS_HF_LOCAL_ONLY": "0"}, clear=False):
            source, local_only = resolve_backbone()
        self.assertEqual(source, DEFAULT_BACKBONE)
        self.assertFalse(local_only)


if __name__ == "__main__":
    unittest.main()
