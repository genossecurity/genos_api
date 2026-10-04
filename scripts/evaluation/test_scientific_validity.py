"""Scientific-validity regressions; no downloads or live services required."""
import importlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine import GenosEngine, _resolve_behavior_model_path
from scientific_validation import split_audit, require_disjoint, temperature_scale, probability_metrics
from scripts.data.prepare_scientific_splits import prepare
from scripts.evaluation.calibrate_scores import fit_temperature, fit_export
from scripts.evaluation.compare_ablations import compare
from scripts.evaluation.benchmark_training_runs import summarize as summarize_training_runs
from scripts.evaluation.audit_annotations import reconcile
from scripts.evaluation.select_action_thresholds import select_thresholds, fit_export as fit_action_export
from scripts.evaluation.evaluate_mitre_relevance import evaluate as evaluate_relevance
from scripts.ops import model_training_orchestrator as training_orchestrator
from scripts.training.train_tfidf_gatekeeper import LABELS as GATE_TFIDF_LABELS
from scripts.training.train_tfidf_gatekeeper import WORD_TOKEN_PATTERN, build_model as build_tfidf_gatekeeper
from sklearn.model_selection import StratifiedGroupKFold
import hashlib


class SplitTests(unittest.TestCase):
    def test_case_normalized_overlap_and_conflicts(self):
        splits={'train':[{'command':' WHOAMI ', 'label':'Benign'}], 'test':[{'command':'whoami','label':'Context_Dependent'}]}
        result=split_audit(splits)
        self.assertEqual(result['overlap']['train:test']['conflicting_commands'],1)
        with self.assertRaises(ValueError): require_disjoint(splits)

    def test_family_and_source_overlap_rejected(self):
        for axis in ['family_id','source_group','holdout_group']:
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                require_disjoint({'train':[dict(command='a',label='Benign',**{axis:'shared'})], 'test':[dict(command='b',label='Benign',**{axis:'shared'})]})

    def test_conflicts_and_development_cases_quarantined(self):
        data={'train':[{'command':'x','label':'Benign'}, {'command':'y','label':'Benign'}, {'command':'z','label':'Benign'}], 'test':[{'command':'X','label':'Malicious'}]}
        clean, removed=prepare(data, {'y'})
        self.assertEqual({r['quarantine_reason'] for r in removed},{'conflicting_labels','benchmark_exposure'})
        self.assertEqual(sum(map(len,clean.values())),1)
        self.assertTrue(require_disjoint(clean)['passed'])
        self.assertEqual(next(r for rows in clean.values() for r in rows)['family_basis'],'inferred_first_executable')

    def test_group_connections_survive_duplicate_provenance(self):
        data={'train':[{'command':'one','label':'Benign','source_group':'A'}, {'command':'two','label':'Benign','source_group':'B'}], 'test':[{'command':'one','label':'Benign','source_group':'B'}]}
        clean,_=prepare(data,set())
        self.assertEqual(sorted(len(rows) for rows in clean.values()),[0,0,2])
        self.assertTrue(require_disjoint(clean)['passed'])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.engine=GenosEngine.__new__(GenosEngine)
        self.engine._gate_labels=list(GenosEngine._GATE_LABELS)
        self.engine.specialist_mode='mitre'
        self.engine.view_policy='mean'
        self.engine.calibration=None
        self.engine.use_residual_format=False
        self.engine._tfidf_idx_to_label={7:'T7',2:'T2',9:'T9'}
        self.engine.t2=SimpleNamespace(classes_=[7,2,9], predict_proba=Mock(return_value=np.array([[.1,.2,.7],[.6,.3,.1]])))

    def test_mitre_fusion_normalizes_and_uses_actual_class_indices(self):
        scores=self.engine._mitre_distribution('raw','decoded')
        np.testing.assert_allclose(scores,[.35,.25,.4])
        self.assertAlmostEqual(sum(scores),1)
        self.assertEqual(self.engine._predict_mitre_codes('raw','decoded')[0]['code'],'T9')
        self.assertEqual(self.engine._predict_mitre_codes('raw','decoded')[0]['score_type'],'uncalibrated_model_estimate')

    def test_identical_views_are_not_duplicated(self):
        self.engine.t2.predict_proba.return_value=np.array([[.2,.3,.5]])
        self.engine._predict_mitre_codes('raw','raw')
        self.engine.t2.predict_proba.assert_called_once_with(['raw'])

    def test_context_is_preserved_and_benign_always_gets_behavior(self):
        for probabilities, expected in [([.05,.05,.9],'Context_Dependent'),([.99,.005,.005],'Benign')]:
            summary=self.engine._select_gate_summary(torch.tensor([probabilities]))
            route=self.engine._route_gatekeeper(summary,{})
            self.assertEqual(route['label'],expected)
            self.assertTrue(route['should_run_specialist'])
            self.assertEqual(summary['model_view'],'raw')

    def test_gate_mean_and_raw_policies(self):
        decoded=torch.tensor([[.1,.8,.1]]);raw=torch.tensor([[.9,.05,.05]])
        summary=self.engine._select_gate_summary(decoded,raw)
        self.assertEqual(summary['label'],'Benign')
        self.assertAlmostEqual(summary['benign_prob'],.5,places=6)
        self.engine.view_policy='raw'
        self.assertAlmostEqual(self.engine._select_gate_summary(decoded,raw)['benign_prob'],.9,places=6)

    def test_behavior_reports_the_actual_raw_input(self):
        self.engine.view_policy='raw'
        self.engine.behavior_input_format='raw'
        self.engine._predict_behavior_with_model=Mock(return_value={'stage':'Execution'})
        result,_=self.engine._predict_behavior('pwd','Benign',{},raw_cmd='echo original')
        self.assertEqual(result['input_text'],'echo original')
        self.assertEqual(result['input_views'],['echo original'])
        self.engine._predict_behavior_with_model.assert_called_once_with(['echo original'])

    def test_calibration_is_applied_after_fusion(self):
        self.engine.calibration={'temperatures':{'mitre':2}}
        np.testing.assert_allclose(self.engine._mitre_distribution('raw','decoded'),temperature_scale([[.35,.25,.4]],2)[0])

    def test_fallback_requires_explicit_opt_in(self):
        self.engine.behavior_model_path='/definitely/missing/model.pt'
        self.engine.allow_behavior_fallback=False
        with self.assertRaisesRegex(RuntimeError,'explicitly set'):
            self.engine._load_behavior_model()
        self.engine.allow_behavior_fallback=True
        with self.assertWarns(RuntimeWarning): self.engine._load_behavior_model()
        self.assertIn('FileNotFoundError',self.engine.behavior_load_error)

    def test_explicit_missing_checkpoint_is_not_substituted(self):
        self.assertEqual(_resolve_behavior_model_path('/missing/behavior.pt'),'/missing/behavior.pt')

    def test_corrupt_metadata_does_not_silently_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'behavior.pt';path.touch();path.with_suffix('.json').write_text('{bad')
            self.engine.behavior_model_path=str(path);self.engine.allow_behavior_fallback=False
            with self.assertRaises(RuntimeError):self.engine._load_behavior_model()

    def test_bare_base64_decoded_in_engine(self):
        self.assertEqual(self.engine._decode_bare_base64('d2hvYW1p'),'whoami')
        self.assertTrue(self.engine.is_obfuscated('d2hvYW1p'))
        self.assertEqual(self.engine._decode_bare_base64('hostname'),'hostname')

    def test_gatekeeper_verdict_is_not_overridden_by_command_rules(self):
        summary=self.engine._select_gate_summary(torch.tensor([[.01,.98,.01]]))
        route=self.engine._route_gatekeeper(summary, {})
        self.assertEqual(route['label'], 'Malicious')
        self.assertEqual(route['routing_policy'], 'model_top_class')

    def test_tfidf_gate_probs_use_runtime_class_order_and_normalization(self):
        self.engine.gatekeeper_backend='tfidf'
        self.engine.gatekeeper_model=SimpleNamespace(
            classes_=np.array([2,0,1]),
            predict_proba=Mock(return_value=np.array([[.3,.2,.5]])),
        )
        probabilities=self.engine._gate_probs(' WHOAMI ')
        np.testing.assert_allclose(probabilities.numpy(), [[.2,.5,.3]])
        self.engine.gatekeeper_model.predict_proba.assert_called_once_with(['whoami'])
        self.assertEqual(self.engine._score_status('gatekeeper'), 'cross_validated_calibrated_model_estimate')


class TrainingBenchmarkTests(unittest.TestCase):
    def test_candidate_selection_uses_validation_not_test(self):
        candidates = []
        for seed, val_score, test_score in [(42, 0.7, 0.95), (43, 0.8, 0.2)]:
            candidates.append({
                'component': 'gatekeeper', 'representation': 'raw', 'model': 'codebert',
                'seed': seed, 'validation': {'macro_f1': val_score},
                'test': {'macro_f1': test_score}, 'selection_metric': 'macro_f1',
                'selection_score': val_score, 'checkpoint_sha256': 'hash',
                'metadata_path': str(seed), 'evaluation_scope': 'weak_labels',
                'independence_audit_passed': True, 'prediction_dir': None,
            })
        report = summarize_training_runs(candidates)
        self.assertEqual(report['best_by_validation']['gatekeeper']['seed'], 43)
        self.assertEqual(report['summaries'][0]['test_metrics_by_seed'][0]['macro_f1'], 0.95)


class TfidfGatekeeperTests(unittest.TestCase):
    def test_word_features_keep_shell_flags(self):
        tokens = TfidfVectorizer(token_pattern=WORD_TOKEN_PATTERN).build_analyzer()(
            'powershell -enc AbCd== /c whoami --no-profile'
        )
        self.assertIn('-enc', tokens)
        self.assertIn('/c', tokens)
        self.assertIn('--no-profile', tokens)

    def test_group_calibrated_linear_gate_returns_three_ordered_probabilities(self):
        phrases = ['pwd harmless', 'curl payload', 'maybe context']
        texts = [f'{phrase} --option{group} /c {group}' for phrase in phrases for group in range(6)]
        targets = np.repeat(np.arange(len(GATE_TFIDF_LABELS)), 6)
        groups = np.asarray([f'{label}:{group}' for label in range(len(GATE_TFIDF_LABELS)) for group in range(6)])
        folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(np.zeros(len(targets)), targets, groups))
        model = build_tfidf_gatekeeper(42, folds, char_features=5000, word_features=5000)
        model.fit(texts, targets)
        probabilities = model.predict_proba(['pwd harmless', 'curl payload'])
        self.assertEqual(model.classes_.tolist(), [0, 1, 2])
        self.assertEqual(probabilities.shape, (2, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1], atol=1e-6)


class TrainingOrchestratorTests(unittest.TestCase):
    def test_training_output_is_streamed_and_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / 'training.log'
            output = StringIO()
            with redirect_stdout(output):
                status = training_orchestrator.execute(
                    [sys.executable, '-c', "print('progress marker', flush=True)"],
                    ROOT,
                    log_path,
                )
            self.assertEqual(status, 0)
            self.assertIn('progress marker', output.getvalue())
            self.assertIn('progress marker', log_path.read_text())

    def test_expired_budget_stops_before_starting_another_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(run_dir=root, max_runtime_hours=1e-12)
            command = ([sys.executable], root / 'model.pt', root / 'model.json')
            with patch.object(training_orchestrator, 'training_commands', return_value=[command]), \
                    patch.object(training_orchestrator, 'execute') as execute:
                completed, stopped = training_orchestrator.run_training(args, root, dry_run=False)
            self.assertEqual((completed, stopped), (0, True))
            execute.assert_not_called()

    def test_partial_checkpoint_is_preserved_and_retried_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / 'gatekeeper' / 'seed_44'
            output_dir.mkdir(parents=True)
            checkpoint = output_dir / 'gatekeeper.pt'
            checkpoint.touch()
            metadata = output_dir / 'gatekeeper_meta.json'
            command = [sys.executable, 'trainer1.py', '--output-dir', str(output_dir)]
            args = SimpleNamespace(run_dir=root, max_runtime_hours=1)

            def finish_retry(retry_command, _cwd, _log_path=None):
                retry_dir = Path(retry_command[retry_command.index('--output-dir') + 1])
                retry_dir.mkdir(parents=True)
                (retry_dir / checkpoint.name).touch()
                (retry_dir / metadata.name).touch()
                return 0

            with patch.object(training_orchestrator, 'training_commands', return_value=[(command, checkpoint, metadata)]), \
                    patch.object(training_orchestrator, 'execute', side_effect=finish_retry):
                completed, stopped = training_orchestrator.run_training(args, root, dry_run=False)
            self.assertEqual((completed, stopped), (1, False))
            self.assertTrue(checkpoint.exists())
            self.assertTrue((output_dir / 'retry_1' / checkpoint.name).exists())


class CalibrationTests(unittest.TestCase):
    def test_temperature_fit_improves_validation_nll_and_preserves_argmax(self):
        p=np.array([[.99,.01]]*8+[[.01,.99]]*2);y=np.array([0]*10)
        temp=fit_temperature(p,y);scaled=temperature_scale(p,temp)
        self.assertGreater(temp,1)
        self.assertLess(probability_metrics(scaled,y)['nll'],probability_metrics(p,y)['nll'])
        np.testing.assert_array_equal(scaled.argmax(axis=1),p.argmax(axis=1))

    def test_test_data_and_unaudited_exports_cannot_fit(self):
        for data in [{'split':'test','independence_audit':{'passed':True}}, {'split':'validation','independence_audit':{'passed':False}}]:
            with self.assertRaises(ValueError):fit_export(data)

    def test_invalid_distribution_and_temperature_rejected(self):
        with self.assertRaises(ValueError):temperature_scale([[.9,.9]],1)
        with self.assertRaises(ValueError):temperature_scale([[.5,.5]],0)

    def test_paired_cluster_comparison_requires_same_cases(self):
        a=[{'command':'a','target':1,'prediction':0,'family_id':'x'}, {'command':'b','target':1,'prediction':1,'family_id':'y'}]
        b=[dict(r,prediction=1) for r in a]
        self.assertEqual(compare(a,b)['accuracy_delta_right_minus_left'],.5)
        with self.assertRaises(ValueError):compare(a,b[:1])


    def test_bootstrap_uses_connected_holdout_groups(self):
        a=[{'command':'a','target':1,'prediction':0,'family_id':'x','holdout_group':'shared'},
           {'command':'b','target':1,'prediction':1,'family_id':'y','holdout_group':'shared'}]
        b=[dict(r,prediction=1) for r in a]
        self.assertEqual(compare(a,b)['groups'],1)
        with self.assertRaises(ValueError): compare(a,[dict(r,holdout_group='different') for r in b])
        with self.assertRaises(ValueError): compare([],[])


class AnnotationTests(unittest.TestCase):
    def row(self, reviewer, verdict='Benign'):
        return {'id':hashlib.sha256(b'pwd').hexdigest(),'command':'pwd','verdict':verdict,
                'annotator_id':reviewer,'rationale':'Reviewed visible command behavior',
                'authorization':'unknown','observable_behavior':['print_working_directory'],'mitre_codes':[]}

    def test_disagreement_stays_unresolved(self):
        accepted, conflicts, report=reconcile([self.row('A')],[self.row('B','Context_Dependent')])
        self.assertEqual(len(accepted),0)
        self.assertEqual(len(conflicts),1)
        self.assertEqual(report['unresolved'],1)

    def test_same_reviewer_and_self_adjudication_rejected(self):
        with self.assertRaises(ValueError):reconcile([self.row('A')],[self.row('A')])
        with self.assertRaises(ValueError):reconcile([self.row('A')],[self.row('B','Context_Dependent')],[self.row('A')])

    def test_third_reviewer_can_adjudicate(self):
        accepted, conflicts, _=reconcile([self.row('A')],[self.row('B','Context_Dependent')],[self.row('C')])
        self.assertTrue(accepted[0]['adjudicated'])
        self.assertEqual(conflicts,[])


class MitreRelevanceTests(unittest.TestCase):
    def test_no_technique_examples_count_false_candidates(self):
        truth=[{'command':'pwd','mitre_codes':[]},{'command':'whoami','mitre_codes':['T1033','T1087']}]
        predictions=[{'command':'pwd','MITRE_codes':[{'code':'T1059','confidence':10}]},{'command':'whoami','MITRE_codes':[{'code':'T1033','confidence':80}]}]
        report=evaluate_relevance(truth,predictions)
        self.assertEqual(report['no_technique_candidate_rate'],1)
        self.assertEqual(report['micro_precision'],.5)
        self.assertEqual(report['micro_recall'],.5)

    def test_missing_annotation_is_not_treated_as_no_technique(self):
        with self.assertRaises(ValueError):evaluate_relevance([{'command':'pwd'}],[{'command':'pwd','MITRE_codes':[]}])


class ActionThresholdTests(unittest.TestCase):
    def test_validation_threshold_can_differ_from_default(self):
        thresholds,diagnostics=select_thresholds([[.1],[.3],[.4],[.45]],[[0],[0],[1],[1]],['download'])
        self.assertAlmostEqual(thresholds['download'],.4)
        self.assertEqual(diagnostics['download']['validation_f1'],1)

    def test_missing_positive_examples_keep_declared_default(self):
        thresholds,diagnostics=select_thresholds([[.1],[.7]],[[0],[0]],['download'])
        self.assertEqual(thresholds['download'],.5)
        self.assertEqual(diagnostics['download']['status'],'insufficient_class_coverage')

    def test_test_export_cannot_select_action_threshold(self):
        with self.assertRaises(ValueError):fit_action_export({'split':'test','independence_audit':{'passed':True}})


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('engine.GenosEngine',return_value=SimpleNamespace(device=torch.device('cpu'), scan=Mock(return_value={}))),patch.dict('os.environ',{'MONGO_URI':''}):
            cls.api=importlib.import_module('app')

    def test_raw_input_and_score_semantics_survive_api(self):
        payload={'label':'Context_Dependent','label_confidence':70,'label_probabilities':{'context_dependent':70},'MITRE_codes':[{'code':'T1059','confidence':20,'score_type':'uncalibrated_model_estimate'}], 'score_type':'uncalibrated_model_estimate','provenance':{'view_policy':'mean'},'action':'requires_context'}
        with patch.object(self.api,'_scan_with_gpu_memory',return_value=(payload,{})) as scan:
            result=self.api._run_inference('d2hvYW1p')
        scan.assert_called_once_with('d2hvYW1p')
        self.assertEqual(result['label'],'Context_Dependent')
        self.assertNotIn('suspicious',result['label_probabilities'])
        self.assertEqual(result['MITRE_codes'][0]['score_type'],'uncalibrated_model_estimate')
        self.assertEqual(result['provenance']['view_policy'],'mean')

    def test_family_specialist_api_does_not_add_mitre_codes(self):
        family_result={
            'label':'Malicious', 'label_confidence':80, 'specialist_mode':'family',
            'attack_families':{'predicted_families':[{'family':'Execution','probability':82.0}], 'all_family_scores':[]},
            'mitre_scope':'disabled_family_specialist_mode',
        }
        with patch.object(self.api,'_scan_with_gpu_memory',return_value=(family_result,{})):
            result=self.api._run_inference('bash -c whoami')
        self.assertEqual(result['attack_families'],family_result['attack_families'])
        self.assertEqual(result['specialist_mode'],'family')
        self.assertNotIn('MITRE_codes',result)


if __name__=='__main__':unittest.main()
