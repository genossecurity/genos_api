"""Regression coverage for shared binary names and OS-specific evidence."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'parser'))
from parser import parse_command

class PlatformTests(unittest.TestCase):
    def test_windows_identity_sample(self):
        self.assertEqual(parse_command('whoami /all && net localgroup administrators')['platform'], 'windows')

    def test_shared_binary_does_not_imply_linux(self):
        self.assertEqual(parse_command('whoami')['platform'], 'unknown')

    def test_windows_exe(self):
        self.assertEqual(parse_command('whoami.exe /all')['platform'], 'windows')

    def test_unix_absolute_path_is_not_windows_switch_evidence(self):
        for command in ['curl https://example.org/a -o /tmp', 'cat /etc/passwd', 'ls /home']:
            with self.subTest(command=command):
                self.assertEqual(parse_command(command)['platform'], 'linux')

    def test_windows_registry(self):
        self.assertEqual(parse_command(r'reg query HKLM\Software')['platform'], 'windows')

if __name__ == '__main__':
    unittest.main()
