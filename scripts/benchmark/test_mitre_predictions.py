"""Regression tests for ranking the saved MITRE classifier's predictions."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine import GenosEngine


class MitrePredictionTests(unittest.TestCase):
    def setUp(self):
        self.engine = GenosEngine.__new__(GenosEngine)
        self.engine.use_residual_format = False
        self.engine._tfidf_idx_to_label = {7: 'T1007', 2: 'T1002', 9: 'T1009', 1: 'T1001', 5: 'T1005', 3: 'T1003'}
        self.engine.t2 = SimpleNamespace(classes_=[7, 2, 9, 1, 5, 3], predict_proba=Mock(return_value=[[.1, .2, .3, .15, .05, .2]]))

    def test_top_five_use_classifier_indices_and_percentage_confidence(self):
        result = self.engine._predict_mitre_codes('whoami')
        self.assertEqual([item['code'] for item in result], ['T1009', 'T1002', 'T1003', 'T1001', 'T1007'])
        self.assertEqual([item['confidence'] for item in result], [30, 20, 20, 15, 10])
        self.engine.t2.predict_proba.assert_called_once_with(['whoami'])

    def test_decoded_view_merges_per_technique_maximum(self):
        self.engine.t2.predict_proba.return_value = [[.1, .2, .3, .15, .05, .2], [.6, .1, .02, .08, .15, .05]]
        result = self.engine._predict_mitre_codes('encoded', 'decoded')
        self.assertEqual([item['code'] for item in result], ['T1007', 'T1009', 'T1002', 'T1003', 'T1001'])
        self.assertEqual(result[0]['confidence'], 60)
        self.engine.t2.predict_proba.assert_called_once_with(['encoded', 'decoded'])

    def test_identical_decoded_view_is_not_run_twice(self):
        self.engine._predict_mitre_codes('whoami', 'whoami')
        self.engine.t2.predict_proba.assert_called_once_with(['whoami'])

    def test_structured_input_matches_training_format(self):
        self.engine.use_residual_format = True
        self.engine._build_variant_a_text = Mock(side_effect=lambda cmd: (f'RAW: {cmd}\nRESIDUAL: parsed', {}))
        self.engine._predict_mitre_codes('encoded', 'decoded')
        self.engine.t2.predict_proba.assert_called_once_with(['RAW: encoded\nRESIDUAL: parsed', 'RAW: decoded\nRESIDUAL: parsed'])


if __name__ == '__main__':
    unittest.main()
