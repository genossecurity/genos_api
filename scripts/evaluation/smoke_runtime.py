#!/usr/bin/env python3
"""Exercise real local checkpoints and API structure without claiming accuracy."""
import argparse
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gatekeeper', default='models/gatekeeper.pt')
    parser.add_argument('--gatekeeper-meta', default='config/gatekeeper_meta.json')
    parser.add_argument('--behavior', default='models/behavior_encoder.pt')
    parser.add_argument('--specialist-mode', choices=['family', 'mitre'], default=None)
    parser.add_argument('--view-policy', choices=['raw', 'decoded', 'mean'], default='mean')
    parser.add_argument('--api', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output path; keep prior evidence')
    from genos.engine import GenosEngine
    engine = GenosEngine(t1_path=args.gatekeeper, t2_path=args.behavior,
                         gatekeeper_meta_path=args.gatekeeper_meta, view_policy=args.view_policy,
                         specialist_mode=args.specialist_mode)
    cases = []
    for command in ['pwd', 'whoami', 'd2hvYW1p', 'curl https://example.org/tool.sh -o /tmp/tool.sh']:
        result = engine.scan(command, include_evaluation=True)
        stage2_ran = result['should_run_specialist']
        components = ['gatekeeper'] + ((['behavior', 'mitre'] if engine.specialist_mode == 'mitre' else
                                       ['behavior', 'family_specialist']) if stage2_ran else [])
        for component in components:
            scores = np.asarray(result['_evaluation'][component], dtype=float)
            if scores.ndim != 1 or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise AssertionError(f'Invalid {component} distribution')
            if component != 'family_specialist' and not np.isclose(scores.sum(), 1, atol=1e-5):
                raise AssertionError(f'Invalid normalized {component} distribution')
        if stage2_ran and result['behavior']['model_type'] != 'behavior_encoder':
            raise AssertionError('Learned behavior must run when specialist is routed')
        if not stage2_ran and result['label'] != 'Benign':
            raise AssertionError('Only benign commands may bypass the specialist')
        if engine.specialist_mode == 'family':
            if 'MITRE_codes' in result or (stage2_ran and len(result['attack_families']['all_family_scores']) != 11):
                raise AssertionError('Family mode must return 11 family scores and no MITRE codes')
        elif stage2_ran and 'MITRE_codes' not in result:
            raise AssertionError('MITRE mode did not return technique candidates')
        if command == 'd2hvYW1p' and result['deobfuscated_cmd'] != 'whoami':
            raise AssertionError('Bare Base64 decoding failed')
        cases.append({'command': command, 'label': result['label'], 'specialist_ran': stage2_ran,
                      'class_scores': result['class_probabilities'],
                      'specialist_mode': engine.specialist_mode,
                      'family_predictions': result.get('attack_families', {}).get('predicted_families'),
                      'behavior': result['attack_stage'],
                      'behavior_model_type': result.get('behavior', {}).get('model_type'),
                      'view': result['gatekeeper']['model_view'],
                      'score_type': result['score_type'],
                      'input_truncated': result['input_truncated']})
    api_status = 'not_requested'
    if args.api:
        # Exercise the real inference and Flask normalization with this exact engine.
        with patch('genos.engine.GenosEngine', return_value=engine), patch.dict('os.environ', {'MONGO_URI': ''}):
            api = importlib.import_module('app')
        client = api.app.test_client()
        for route in ['/', '/api', '/health']:
            if client.get(route).status_code != 200:
                raise AssertionError(f'Route failed: {route}')
        result = api._run_inference('d2hvYW1p')
        if result['label'] != cases[2]['label'] or result['score_type'] != cases[2]['score_type']:
            raise AssertionError('API changed verdict or score semantics')
        if result.get('deobfuscated_cmd') != 'whoami' or result['provenance'] != engine.provenance:
            raise AssertionError('API lost decoding or provenance')
        if engine.specialist_mode == 'family' and ('MITRE_codes' in result or
                                                   (cases[2]['specialist_ran'] and 'attack_families' not in result)):
            raise AssertionError('API lost family output or returned technique codes in family mode')
        api_status = 'passed'
    report = {'provenance': engine.provenance, 'cases': cases, 'api_and_templates': api_status,
              'scope': 'Structural integration checks; examples are development cases, not an independent accuracy estimate'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cases': cases, 'api_and_templates': api_status}, indent=2))


if __name__ == '__main__':
    main()
