import base64
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parser"))

from genos.baseline import BaselineMode, BaselineStore, ExecutionContext, SignatureExtractor
from genos.deobfuscator import Deobfuscator, deobfuscate_with_trace
from genos.engine import GenosEngine, summarize_triage_consistency
from genos.evidence import collect_indicator_evidence
from genos.specialist import Specialist
from parser import parse_command


class TriagePipelineTests(unittest.TestCase):
    def test_triage_consistency_flags_benign_family_without_overriding_gate(self):
        family = {"predicted_families": [{"family": "Benign Admin", "probability": 99.42}]}
        self.assertEqual(summarize_triage_consistency("Suspicious", family)["status"], "gate_family_disagreement")
        self.assertEqual(summarize_triage_consistency("Benign", family)["status"], "benign_family_candidate")
        self.assertEqual(summarize_triage_consistency("Benign", None)["status"], "not_evaluated")

    def test_nested_decoding_keeps_each_layer(self):
        command = "curl https://example.org/a"
        nested = base64.b64encode(base64.b64encode(command.encode())).decode()
        trace = deobfuscate_with_trace(nested)
        self.assertEqual(trace["final"], command)
        self.assertGreaterEqual(len(trace["steps"]), 2)
        self.assertEqual(trace["steps"][-1]["text"], command)

    def test_decoding_trace_has_a_size_limit(self):
        encoded = base64.b64encode(b"curl https://example.org/a").decode()
        trace = deobfuscate_with_trace(encoded, max_trace_chars=4)
        self.assertTrue(trace["limited"])
        self.assertEqual(trace["stop_reason"], "trace_limit")

    def test_decoded_ioc_has_provenance_and_legacy_list(self):
        command = "curl https://example.org/a"
        encoded = base64.b64encode(command.encode()).decode()
        trace = deobfuscate_with_trace(encoded)
        evidence = collect_indicator_evidence(encoded, trace["final"], True, decoding_trace=trace)
        self.assertIn("https://example.org/a", evidence["urls"])
        records = [row for row in evidence["indicator_records"] if row["value"] == "https://example.org/a"]
        self.assertTrue(any(row["source_view"] == "decoded" and row["decoder"] == "bare_base64" for row in records))

    def test_extended_iocs(self):
        evidence = collect_indicator_evidence(
            "curl hxxps://example[.]org/a; echo 2001:db8::1; echo " + "a" * 64,
            "curl hxxps://example[.]org/a; echo 2001:db8::1; echo " + "a" * 64,
        )
        self.assertIn("https://example.org/a", evidence["defanged_urls"])
        self.assertIn("2001:db8::1", evidence["ipv6"])
        self.assertIn("a" * 64, evidence["hashes"])

    def test_behavior_input_reuses_parsed_view(self):
        specialist = Specialist.__new__(Specialist)
        cache = {}
        with patch("genos.specialist._parse_command", wraps=parse_command) as parse_spy:
            first = specialist._build_behavior_input("whoami", cache)
            second = specialist._build_behavior_input("whoami", cache)
        self.assertEqual(first, second)
        self.assertEqual(parse_spy.call_count, 1)

    def test_baseline_bypass_has_no_model_score(self):
        store = BaselineStore(mode=BaselineMode.ENFORCE, min_sightings=1, min_distinct_days=1)
        context = ExecutionContext("host-a", "shell", "user", datetime.now(timezone.utc), role="ops")
        signature = SignatureExtractor.extract(parse_command("whoami"))
        store.observe_and_admit(signature, context, {"class_probabilities": {"Benign": 97, "Malicious": 1}})
        engine = GenosEngine.__new__(GenosEngine)
        engine.provenance = {}
        engine.deobfuscator = Deobfuscator()
        engine.baseline_store = store
        result = engine.scan("whoami", context=context)
        self.assertIsNone(result["label_confidence"])
        self.assertEqual(result["score_type"], "baseline_policy_no_model_score")
        self.assertEqual(result["class_probabilities"], {})


if __name__ == "__main__":
    unittest.main()
