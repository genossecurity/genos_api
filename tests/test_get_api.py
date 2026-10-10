"""GET API contract and optional scan work, without loading model weights."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genos.deobfuscator import Deobfuscator
from genos.engine import GenosEngine


def sample_result():
    return {
        "label": "Suspicious", "label_confidence": 99.0,
        "deobfuscated_cmd": None, "should_run_specialist": True,
        "behavior": {"stage": "Credential Access", "stage_confidence": 98.4},
        "evidence": {
            "urls": ["https://example.org/payload"], "domains": ["example.org"],
            "ips": ["192.0.2.1"], "ports": [443],
            "file_paths": ["/etc/shadow"], "registry_paths": [],
        },
    }


class GetApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Mock()
        spec = importlib.util.spec_from_file_location("genos_api_test_app", ROOT / "app.py")
        cls.web = importlib.util.module_from_spec(spec)
        with patch("genos.engine.GenosEngine", return_value=cls.engine):
            spec.loader.exec_module(cls.web)
        cls.web.engine = cls.engine
        cls.web.app.config["TESTING"] = True

    def setUp(self):
        self.engine.reset_mock()
        self.engine.scan.side_effect = None
        self.engine.scan.return_value = sample_result()
        self.client = self.web.app.test_client()

    def test_default_only_returns_tier1_and_decoded_command(self):
        response = self.client.get("/api/scan", query_string={"command": "cat /etc/shadow"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {
            "label": "Context_Dependent", "label_confidence": 99.0, "deobfuscated_cmd": None,
        })
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.engine.scan.assert_called_once_with(
            "cat /etc/shadow", run_specialist=False, collect_iocs=False, use_baseline=False,
        )

    def test_tier2_is_independent_of_iocs(self):
        response = self.client.get("/api/scan", query_string={"command": "cat /etc/shadow", "tier2": "true"})
        self.assertEqual(response.json["tier2"], {
            "status": "completed", "stage": "Credential Access", "stage_confidence": 98.4,
        })
        self.assertNotIn("iocs", response.json)
        self.engine.scan.assert_called_once_with(
            "cat /etc/shadow", run_specialist=True, collect_iocs=False, use_baseline=False,
        )

    def test_selected_iocs_only(self):
        response = self.client.get("/api/scan", query_string={"command": "whoami", "iocs": "urls,ports,files"})
        self.assertEqual(response.json["iocs"], {
            "urls": ["https://example.org/payload"], "ports": [443], "files": ["/etc/shadow"],
        })
        self.assertNotIn("tier2", response.json)
        self.engine.scan.assert_called_once_with(
            "whoami", run_specialist=False, collect_iocs=True, use_baseline=False,
        )

    def test_all_iocs_with_tier2(self):
        response = self.client.get("/api/scan", query_string={"command": "whoami", "tier2": "1", "iocs": "all"})
        self.assertEqual(set(response.json["iocs"]), set(self.web.IOC_FIELDS))
        self.assertIn("tier2", response.json)

    def test_benign_does_not_report_a_mitre_stage(self):
        self.engine.scan.return_value.update({"label": "Benign", "should_run_specialist": False})
        response = self.client.get("/api/scan", query_string={"command": "whoami", "tier2": "true"})
        self.assertEqual(response.json["tier2"], {
            "status": "skipped_benign", "stage": None, "stage_confidence": None,
        })

    def test_command_survives_query_encoding(self):
        command = "printf '%s' 'quotes & + % ? # $(whoami) `id`'\n# café"
        response = self.client.get("/api/scan", query_string={"command": command, "tier2": "false"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.engine.scan.call_args.args[0], command)

    def test_invalid_parameters_rejected_before_inference(self):
        queries = [
            {}, {"command": "  "}, {"command": "whoami", "tier2": "yes"},
            {"command": "whoami", "iocs": "passwords"}, {"command": "whoami", "iocs": ""},
            {"command": "whoami", "iocs": "all,files"}, {"command": "whoami", "tier": "2"},
            [("command", "whoami"), ("command", "id")],
            [("command", "whoami"), ("tier2", "true"), ("tier2", "false")],
        ]
        for query in queries:
            with self.subTest(query=query):
                response = self.client.get("/api/scan", query_string=query)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.engine.scan.assert_not_called()

    def test_oversized_command_is_rejected_before_inference(self):
        response = self.client.get("/api/scan", query_string={"command": "x" * 65537})
        self.assertEqual(response.status_code, 400)
        self.engine.scan.assert_not_called()

    def test_builder_and_scanner_navigation(self):
        response = self.client.get("/api")
        self.assertEqual(response.status_code, 200)
        self.assertIn('data-endpoint="/api/scan"', response.text)
        for field in self.web.IOC_FIELDS:
            self.assertIn('data-ioc="' + field + '"', response.text)
        self.assertIn('href="/api"', self.client.get("/").text)

    def test_existing_post_routes_still_use_full_inference(self):
        for url in ("/scan", "/api/scan"):
            with self.subTest(url=url), patch.object(self.web, "_run_inference", return_value={"label": "Benign"}) as run:
                response = self.client.post(url, json={"command": "whoami"})
                self.assertEqual(response.status_code, 200)
                run.assert_called_once_with("whoami", include_flags=None)

    def test_failure_is_json_and_not_cached(self):
        self.engine.scan.side_effect = RuntimeError("internal detail")
        with self.assertLogs(self.web.app.logger, level="ERROR"):
            response = self.client.get("/api/scan", query_string={"command": "whoami"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {"error": "Inference failed"})
        self.assertEqual(response.headers["Cache-Control"], "no-store")


class EngineOptionsTests(unittest.TestCase):
    def setUp(self):
        self.engine = GenosEngine.__new__(GenosEngine)
        engine = self.engine
        engine.device = torch.device("cpu")
        engine.gatekeeper_backend = "tfidf"
        engine.deobfuscator = Deobfuscator()
        engine.tokenizer = Mock()
        engine.tokenizer.encode.return_value = []
        engine.max_length = 256
        engine.provenance = {}
        engine.baseline_store = Mock()
        engine._gate_labels = ["Benign", "Suspicious"]
        engine.gatekeeper_meta_path = None
        engine.view_policy = "decoded"
        engine.specialist_mode = "family"
        engine.behavior_model = None
        engine._score_status = Mock(return_value="uncalibrated_model_estimate")
        engine._gate_probs = Mock(return_value=torch.tensor([[.01, .99]]))
        engine._select_gate_summary = Mock(return_value={
            "label": "Suspicious", "public_label": "Suspicious", "label_conf": .99,
            "second_label": "Benign", "second_public_label": "Benign", "second_conf": .01,
            "benign_prob": .01, "suspicious_prob": .99, "decision_margin": .98,
            "class_probabilities": {"Benign": .01, "Suspicious": .99},
        })
        engine._extract_routing_features = Mock(return_value={})
        engine._route_gatekeeper = Mock(return_value={
            "label": "Suspicious", "label_confidence": .99, "model_confidence": .99,
            "confidence_driver": "model_aligned", "reason": "test", "routing_policy": "model_top_class",
            "triggered_features": [], "should_run_specialist": True,
        })
        engine._predict_family_specialist = Mock(return_value={"all_family_scores": []})
        engine._predict_behavior = Mock(return_value=({"stage": "Credential Access", "stage_confidence": 98.4}, {}))
        engine._collect_indicator_evidence = Mock(return_value=sample_result()["evidence"])
        engine._predict_mitre_codes = Mock(return_value=[])
        engine._mitre_distribution = Mock(return_value=[1.0])

    def test_tier1_only_still_deobfuscates_and_ignores_baseline(self):
        result = self.engine.scan("d2hvYW1p", run_specialist=False, collect_iocs=False, use_baseline=False)
        self.assertEqual(result["deobfuscated_cmd"], "whoami")
        self.engine._gate_probs.assert_any_call("whoami")
        self.assertFalse(result["should_run_specialist"])
        self.assertNotIn("attack_stage", result)
        self.engine._predict_family_specialist.assert_not_called()
        self.engine._predict_behavior.assert_not_called()
        self.engine._collect_indicator_evidence.assert_not_called()
        self.engine.baseline_store.evaluate.assert_not_called()
        self.engine.baseline_store.observe_and_admit.assert_not_called()

    def test_iocs_can_run_without_tier2(self):
        result = self.engine.scan("cat /etc/shadow", run_specialist=False, collect_iocs=True, use_baseline=False)
        self.assertEqual(result["evidence"]["file_paths"], ["/etc/shadow"])
        self.engine._predict_family_specialist.assert_not_called()
        self.engine._predict_behavior.assert_not_called()

    def test_tier2_receives_decoded_command_without_iocs(self):
        result = self.engine.scan("d2hvYW1p", run_specialist=True, collect_iocs=False, use_baseline=False)
        self.assertTrue(result["should_run_specialist"])
        self.engine._predict_family_specialist.assert_called_once_with("whoami")
        self.assertEqual(self.engine._predict_behavior.call_args.args[0], "whoami")
        self.engine._collect_indicator_evidence.assert_not_called()

    def test_benign_skips_tier2_but_can_return_iocs(self):
        self.engine._route_gatekeeper.return_value["label"] = "Benign"
        result = self.engine.scan("whoami", run_specialist=True, collect_iocs=True, use_baseline=False)
        self.assertFalse(result["should_run_specialist"])
        self.engine._predict_family_specialist.assert_not_called()
        self.engine._predict_behavior.assert_not_called()
        self.engine._collect_indicator_evidence.assert_called_once()

    def test_evaluation_cannot_force_disabled_specialist(self):
        self.engine.specialist_mode = "mitre"
        result = self.engine.scan("whoami", include_evaluation=True, run_specialist=False, collect_iocs=False, use_baseline=False)
        self.assertIsNone(result["_evaluation"]["mitre"])
        self.engine._mitre_distribution.assert_not_called()
        self.engine._predict_mitre_codes.assert_not_called()
        self.engine._predict_behavior.assert_not_called()

    def test_existing_scan_defaults_still_include_specialist_and_evidence(self):
        self.engine.scan("cat /etc/shadow", use_baseline=False)
        self.engine._predict_family_specialist.assert_called_once()
        self.engine._collect_indicator_evidence.assert_called_once()


if __name__ == "__main__":
    unittest.main()
