#!/usr/bin/env python3
"""Export deployed pipeline scores with explicit exposure checks and provenance."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from genos.scientific_validation import read_rows, command_of, split_audit, dataset_manifest, probability_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--training-data', type=Path, action='append', required=True, help='Every training source, including patches; repeat for each file')
    p.add_argument('--split', choices=['validation', 'test', 'development'], required=True)
    p.add_argument('--component', choices=['gatekeeper', 'mitre', 'behavior'], default='gatekeeper')
    p.add_argument('--view-policy', choices=['raw', 'decoded', 'mean'], default='mean')
    p.add_argument('--gatekeeper', type=Path, default=ROOT / 'models/gatekeeper.pt')
    p.add_argument('--gatekeeper-meta', type=Path, default=ROOT / 'config/gatekeeper_meta.json')
    p.add_argument('--behavior', type=Path, default=ROOT / 'models/behavior_encoder.pt')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output path')
    rows = read_rows(args.dataset)
    if not rows:
        raise ValueError("Evaluation data is empty")
    training = [r for path in args.training_data for r in read_rows(path)]
    audit = split_audit({'training': training, args.split: rows})
    if args.split != 'development' and not audit['passed']:
        raise ValueError('Exposure check failed: ' + json.dumps(audit))
    from genos.engine import GenosEngine
    engine = GenosEngine(t1_path=str(args.gatekeeper), t2_path=str(args.behavior), gatekeeper_meta_path=str(args.gatekeeper_meta), view_policy=args.view_policy)
    names = {'gatekeeper': engine._GATE_LABELS, 'mitre': [engine._tfidf_idx_to_label[int(i)] for i in engine.t2.classes_], 'behavior': [engine.behavior_stage_labels[i] for i in sorted(engine.behavior_stage_labels)]}[args.component]
    outputs = []
    for index, row in enumerate(rows):
        result = engine.scan(command_of(row), include_evaluation=True)
        expected = row.get('stage_label') if args.component == 'behavior' else row.get('verdict') or row.get('label')
        if expected not in names:
            raise ValueError(f'Unknown or missing target {expected!r}; add explicit no-technique/multilabel evaluation for such cases')
        scores = result['_evaluation'][args.component]
        outputs.append({'command': command_of(row), 'target': names.index(expected), 'label': expected, 'family_id': row.get('family_id'), 'source_group': row.get('source_group'), 'holdout_group': row.get('holdout_group'), 'probabilities': scores, 'prediction': max(range(len(scores)), key=lambda i: scores[i]), 'verdict': result['label'], 'action_probabilities': result['_evaluation']['behavior_actions'], 'action_targets': [int(label in row['action_tags']) for label in [engine.behavior_action_labels[i] for i in sorted(engine.behavior_action_labels)]] if 'action_tags' in row else None, 'behavior': result['behavior'], 'MITRE_codes': result['MITRE_codes'], 'input_truncated': result['input_truncated']})
        if index % 50 == 0:
            print(f'{index + 1}/{len(rows)}', flush=True)
    export = {'split': args.split, 'component': args.component, 'class_names': names, 'action_class_names': [engine.behavior_action_labels[i] for i in sorted(engine.behavior_action_labels)], 'runtime_signature': engine.provenance, 'calibration_applied': bool(engine.calibration), 'training_manifest': dataset_manifest({str(i): path for i,path in enumerate(args.training_data)}), 'dataset_manifest': dataset_manifest({'evaluation': args.dataset}), 'independence_audit': audit, 'label_basis': sorted({r.get('label_basis', 'legacy_unverified') for r in rows}), 'metrics': probability_metrics([r['probabilities'] for r in outputs], [r['target'] for r in outputs]), 'behavior_coverage': sum(bool(r['behavior']) for r in outputs) / len(outputs), 'truncation_rate': sum(any(r['input_truncated'].values()) for r in outputs) / len(outputs), 'rows': outputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export, indent=2) + '\n')
    print(json.dumps({k:v for k,v in export.items() if k not in {'rows', 'runtime_signature'}}, indent=2))


if __name__ == '__main__':
    main()
