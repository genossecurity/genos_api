"""GPU telemetry regressions without downloading models or requiring CUDA.

Run from the repository root: venv/bin/python scripts/benchmark/test_gpu_memory.py
"""
import importlib
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

SCAN_RESULT = {'label': 'Benign', 'label_confidence': 99.0, 'MITRE_codes': []}
bootstrap = SimpleNamespace(device=torch.device('cpu'), scan=Mock(return_value=SCAN_RESULT))
with patch('engine.GenosEngine', return_value=bootstrap), patch.dict(os.environ, {'MONGO_URI': ''}):
    api = importlib.import_module('app')


class GpuMemoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = SimpleNamespace(device=torch.device('cuda:0'), scan=Mock(return_value=SCAN_RESULT))
        self.cuda = Mock()
        self.cuda.memory_allocated.return_value = 100 * 1048576
        self.cuda.max_memory_allocated.return_value = 132 * 1048576
        self.cuda.max_memory_reserved.return_value = 160 * 1048576
        self.cuda.get_device_name.return_value = 'Test GPU'
        self.engine_patch = patch.object(api, 'engine', self.engine)
        self.cuda_patch = patch.object(api.torch, 'cuda', self.cuda)
        self.engine_patch.start()
        self.cuda_patch.start()
        self.addCleanup(self.engine_patch.stop)
        self.addCleanup(self.cuda_patch.stop)

    def test_peak_above_resident_baseline(self):
        result, memory = api._scan_with_gpu_memory('whoami')
        self.assertEqual(result, SCAN_RESULT)
        self.assertEqual(memory['status'], 'measured')
        self.assertEqual(memory['device_name'], 'Test GPU')
        self.assertEqual(memory['command_peak_bytes'], 32 * 1048576)
        self.assertEqual(memory['peak_allocated_bytes'], 132 * 1048576)
        self.assertEqual(memory['peak_reserved_bytes'], 160 * 1048576)
        self.cuda.reset_peak_memory_stats.assert_called_once_with(self.engine.device)
        self.assertEqual(self.cuda.synchronize.call_count, 2)

    def test_each_command_resets_peak(self):
        self.cuda.max_memory_allocated.side_effect = [132 * 1048576, 104 * 1048576]
        first = api._scan_with_gpu_memory('first')[1]
        second = api._scan_with_gpu_memory('second')[1]
        self.assertEqual(first['command_peak_bytes'], 32 * 1048576)
        self.assertEqual(second['command_peak_bytes'], 4 * 1048576)
        self.assertEqual(self.cuda.reset_peak_memory_stats.call_count, 2)

    def test_cpu_does_not_report_fake_zero(self):
        self.engine.device = torch.device('cpu')
        memory = api._scan_with_gpu_memory('whoami')[1]
        self.assertEqual(memory['status'], 'cpu')
        self.assertIsNone(memory['command_peak_bytes'])
        self.assertEqual(self.cuda.mock_calls, [])

    def test_telemetry_failure_preserves_scan(self):
        for failing_method in ['synchronize', 'reset_peak_memory_stats', 'max_memory_allocated', 'get_device_name']:
            with self.subTest(failing_method=failing_method):
                method = getattr(self.cuda, failing_method)
                method.side_effect = RuntimeError('Stats unavailable')
                result, memory = api._scan_with_gpu_memory('whoami')
                self.assertEqual(result, SCAN_RESULT)
                self.assertEqual(memory['status'], 'unavailable')
                self.assertIsNone(memory['command_peak_bytes'])
                method.side_effect = None
        self.assertEqual(self.engine.scan.call_count, 4)

    def test_scan_failure_releases_lock(self):
        self.engine.scan.side_effect = [ValueError('Inference failed'), SCAN_RESULT]
        with self.assertRaisesRegex(ValueError, 'Inference failed'):
            api._scan_with_gpu_memory('bad')
        self.assertFalse(api._inference_lock.locked())
        self.assertEqual(api._scan_with_gpu_memory('good')[1]['status'], 'measured')

    def test_overlapping_requests_are_isolated(self):
        running = 0
        peak_running = 0
        guard = threading.Lock()
        def scan(command):
            nonlocal running, peak_running
            with guard:
                running += 1
                peak_running = max(peak_running, running)
            time.sleep(.02)
            with guard:
                running -= 1
            return dict(SCAN_RESULT, command=command)
        self.engine.scan.side_effect = scan
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(api._scan_with_gpu_memory, ['one','two','three']))
        self.assertEqual(peak_running, 1)
        self.assertEqual([r[0]['command'] for r in results], ['one','two','three'])
        self.assertTrue(all(r[1]['status'] == 'measured' for r in results))

    def test_api_meta_flag_controls_telemetry(self):
        default = api._run_inference('whoami')
        self.assertEqual(default['gpu_memory']['command_peak_bytes'], 32 * 1048576)
        without_meta = api._run_inference('whoami', {'meta': False})
        self.assertNotIn('gpu_memory', without_meta)
        self.assertNotIn('elapsed_ms', without_meta)


if __name__ == '__main__':
    unittest.main()
