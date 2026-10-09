import sys
import unittest
from datetime import datetime, timezone

sys.path.append("/home/sam/genos/genos_api")
sys.path.append("/home/sam/genos/genos_api/parser")

from genos.baseline import (
    BaselineMode,
    BaselineStatus,
    BaselineStore,
    ExecutionContext,
    SignatureExtractor,
)
from parser import parse_command


def _ctx(host_id="host-1", role="ops", timestamp=None, parent_process="powershell.exe", in_maintenance_window=False):
    if timestamp is None:
        timestamp = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
    return ExecutionContext(
        host_id=host_id,
        parent_process=parent_process,
        user="svc",
        timestamp=timestamp,
        role=role,
        fleet="fleet-a",
        in_maintenance_window=in_maintenance_window,
    )


class BaselineRegressionTests(unittest.TestCase):
    def test_signature_retains_executionpolicy_and_inline_payload_content(self):
        parsed = parse_command("powershell -ExecutionPolicy Bypass -EncodedCommand SGVsbG8=")
        sig = SignatureExtractor.extract(parsed)

        self.assertTrue(any("executionpolicy" in v.lower() for v in sig.flag_set))
        self.assertTrue(any("bypass" in v.lower() for v in sig.flag_set))
        self.assertTrue(any("encodedcommand" in v.lower() for v in sig.flag_set))

    def test_fleet_scope_is_prior_only_not_allowlist(self):
        store = BaselineStore(mode=BaselineMode.ENFORCE, min_sightings=1, min_distinct_days=1)
        sig = SignatureExtractor.extract(parse_command("powershell -NoProfile -EncodedCommand SGVsbG8="))

        ctx1 = _ctx(host_id="host-a")
        store.observe_and_admit(sig, ctx1, {"class_probabilities": {"Malicious": 2.0, "Context_Dependent": 5.0, "Benign": 93.0}})

        eval_on_other_host = store.evaluate(sig, _ctx(host_id="host-b"))
        self.assertIsNot(eval_on_other_host.status, BaselineStatus.KNOWN_STABLE)
        self.assertTrue(eval_on_other_host.should_run_genos)

    def test_maintenance_window_does_not_bypass_genos(self):
        store = BaselineStore(mode=BaselineMode.ENFORCE)
        sig = SignatureExtractor.extract(parse_command("cmd /c whoami"))
        eval_result = store.evaluate(sig, _ctx(in_maintenance_window=True))

        self.assertIsNot(eval_result.status, BaselineStatus.KNOWN_STABLE)
        self.assertTrue(eval_result.should_run_genos)

    def test_flagged_sighting_blocks_promotion(self):
        store = BaselineStore(mode=BaselineMode.ENFORCE, min_sightings=2, min_distinct_days=1)
        sig = SignatureExtractor.extract(parse_command("vssadmin delete shadows /all /quiet"))

        first = store.observe_and_admit(
            sig,
            _ctx(host_id="host-a", parent_process="vssadmin.exe"),
            {"class_probabilities": {"Malicious": 91.0, "Context_Dependent": 55.0, "Benign": 9.0}, "triggered_features": ["defense_impairment"]},
        )
        second = store.observe_and_admit(
            sig,
            _ctx(host_id="host-a", parent_process="vssadmin.exe", timestamp=datetime(2026, 10, 7, 13, 0, tzinfo=timezone.utc)),
            {"class_probabilities": {"Malicious": 5.0, "Context_Dependent": 10.0, "Benign": 85.0}, "triggered_features": []},
        )

        self.assertFalse(first)
        self.assertFalse(second)

    def test_first_seen_uses_context_timestamp(self):
        store = BaselineStore(mode=BaselineMode.LEARN, min_sightings=1, min_distinct_days=1)
        sig = SignatureExtractor.extract(parse_command("whoami"))
        ctx = _ctx(timestamp=datetime(2026, 10, 8, 9, 30, tzinfo=timezone.utc))

        store.observe_and_admit(sig, ctx, {"class_probabilities": {"Malicious": 2.0, "Context_Dependent": 8.0, "Benign": 90.0}})
        stats = store._stores["host"][f"host:{ctx.host_id}:{sig.level3_full}"]

        self.assertEqual(stats.first_seen, ctx.timestamp)


if __name__ == "__main__":
    unittest.main()
