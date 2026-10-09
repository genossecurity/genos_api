"""Startup contract: weights are loaded before Flask serves pages."""
import importlib.util
import runpy
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class AppStartupTests(unittest.TestCase):
    def load_app(self):
        spec = importlib.util.spec_from_file_location("genos_startup_test_app", ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        engine = Mock()
        engine.family_labels = ["Execution"]
        engine.specialist_mode = "family"
        engine.device.type = "cpu"
        with patch("genos.engine.GenosEngine", return_value=engine):
            spec.loader.exec_module(module)
        return module, engine

    def test_weights_and_warmup_complete_during_import(self):
        module, engine = self.load_app()
        engine.scan.assert_called_once_with("warmup")
        self.assertIs(module.engine, engine)
        client = module.app.test_client()
        self.assertEqual(client.get("/api").status_code, 200)
        self.assertEqual(client.get("/health").status_code, 200)

    def test_engine_construction_failure_prevents_app_import(self):
        spec = importlib.util.spec_from_file_location("genos_startup_failure_app", ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        with patch("genos.engine.GenosEngine", side_effect=RuntimeError("weights unavailable")):
            with self.assertRaises(RuntimeError):
                spec.loader.exec_module(module)

    def test_gunicorn_uses_one_model_worker_with_request_threads(self):
        config = runpy.run_path(str(ROOT / "gunicorn.conf.py"))
        self.assertEqual(config["workers"], 1)
        self.assertEqual(config["worker_class"], "gthread")
        self.assertGreaterEqual(config["threads"], 2)


if __name__ == "__main__":
    unittest.main()
