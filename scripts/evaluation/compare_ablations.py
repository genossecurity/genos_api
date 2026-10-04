#!/usr/bin/env python3
"""Paired group-bootstrap accuracy deltas for identical held-out commands."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scientific_validation import read_rows, normalize_command, command_of, sha256_file


def compare(left, right, iterations=2000, seed=42):
    if not left or not right:
        raise ValueError('Paired comparisons require nonempty examples')
    if iterations < 1:
        raise ValueError('iterations must be positive')
    a = {normalize_command(command_of(r)): r for r in left}
    b = {normalize_command(command_of(r)): r for r in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError('Paired comparisons require identical unique commands')
    groups = {}
    for key in a:
        x, y = a[key], b[key]
        if x['target'] != y['target'] or any(x.get(axis) != y.get(axis) for axis in ('family_id', 'source_group', 'holdout_group')):
            raise ValueError('Targets or provenance groups differ')
        # Connected holdout groups may join several families through a shared source.
        group = x.get('holdout_group') or x.get('family_id')
        if not group:
            raise ValueError('Family groups are required for cluster uncertainty')
        delta = int(y['prediction'] == y['target']) - int(x['prediction'] == x['target'])
        groups.setdefault(group, []).append(delta)
    sums = np.array([sum(v) for v in groups.values()]); counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(iterations, len(groups)))
    deltas = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {'n': len(a), 'groups': len(groups), 'accuracy_delta_right_minus_left': float(sums.sum() / counts.sum()), 'paired_group_bootstrap_95_percent_interval': np.quantile(deltas, [.025, .975]).tolist(), 'iterations': iterations, 'seed': seed, 'limitation': 'Conditional on these labels, groups, training runs, and test population; does not establish operational validity'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left',type=Path,required=True); p.add_argument('--right',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=compare(read_rows(args.left),read_rows(args.right))
    report['inputs']={str(path):sha256_file(path) for path in (args.left,args.right)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
