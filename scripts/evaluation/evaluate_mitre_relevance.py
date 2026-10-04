#!/usr/bin/env python3
"""Evaluate multilabel MITRE candidate relevance, including explicit no-technique rows."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scientific_validation import read_rows,normalize_command,command_of,sha256_file


def evaluate(annotations,predictions,minimum_score=0):
    if not 0 <= minimum_score <= 100:raise ValueError('Scores use percentages, 0 to 100')
    truth={normalize_command(command_of(r)):r for r in annotations}
    scored={normalize_command(command_of(r)):r for r in predictions}
    if len(truth)!=len(annotations) or len(scored)!=len(predictions) or truth.keys()!=scored.keys():
        raise ValueError('Expected identical unique commands in annotations and predictions')
    tp=fp=fn=none=none_with_candidates=positive=hits=0
    for key,row in truth.items():
        expected=row.get('mitre_codes')
        if not isinstance(expected,list):raise ValueError('Each row needs explicit mitre_codes; [] means none applies')
        expected=set(expected)
        candidates=scored[key].get('MITRE_codes')
        if not isinstance(candidates,list):raise ValueError('Missing MITRE candidate predictions')
        predicted={r['code'] for r in candidates[:5] if float(r['confidence'])>=minimum_score}
        tp+=len(expected&predicted);fp+=len(predicted-expected);fn+=len(expected-predicted)
        if expected:positive+=1;hits+=bool(expected&predicted)
        else:none+=1;none_with_candidates+=bool(predicted)
    return {'n':len(truth),'minimum_score':minimum_score,'micro_precision':tp/(tp+fp) if tp+fp else None,'micro_recall':tp/(tp+fn) if tp+fn else None,'positive_command_hit_rate':hits/positive if positive else None,'no_technique_count':none,'no_technique_candidate_rate':none_with_candidates/none if none else None,'true_positive_codes':tp,'false_positive_codes':fp,'missed_codes':fn,'scope':'Exact technique identifiers, top five returned candidates; any threshold must be chosen on validation only'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--minimum-score',type=float,default=0);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    predictions=json.loads(args.predictions.read_text())['rows'] if args.predictions.suffix=='.json' else read_rows(args.predictions)
    report=evaluate(read_rows(args.annotations),predictions,args.minimum_score)
    report['inputs']={str(path):sha256_file(path) for path in [args.annotations,args.predictions]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
