#!/usr/bin/env python3
"""Select multilabel behavior thresholds using validation labels only."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scientific_validation import sha256_file


def select_thresholds(scores,targets,labels):
    scores=np.asarray(scores,dtype=float);targets=np.asarray(targets,dtype=int)
    if scores.ndim!=2 or scores.shape!=targets.shape or scores.shape[1]!=len(labels) or not len(scores):
        raise ValueError('Expected matching nonempty N-by-action score and target matrices')
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any() or not np.isin(targets,[0,1]).all():
        raise ValueError('Invalid action scores or binary targets')
    thresholds={};diagnostics={}
    for column,label in enumerate(labels):
        truth=targets[:,column];pred=scores[:,column]
        if not truth.any() or truth.all():
            thresholds[label]=.5
            diagnostics[label]={'status':'insufficient_class_coverage','positive_count':int(truth.sum()),'negative_count':int((1-truth).sum())}
            continue
        order=np.argsort(-pred,kind='stable');ordered=pred[order]
        positives=np.cumsum(truth[order]);count=np.arange(1,len(pred)+1)
        ends=np.r_[ordered[:-1]!=ordered[1:],True]
        candidate_scores=ordered[ends]
        f1=2*positives[ends]/(count[ends]+truth.sum())
        best=float(f1.max());winners=np.flatnonzero(np.isclose(f1,best))
        # Stable tie break: closest threshold to the existing default.
        chosen=int(winners[np.argmin(np.abs(candidate_scores[winners]-.5))])
        thresholds[label]=float(candidate_scores[chosen])
        diagnostics[label]={'status':'validation_selected','validation_f1':best,'positive_count':int(truth.sum()),'negative_count':int((1-truth).sum())}
    return thresholds,diagnostics


def fit_export(data):
    if data.get('split')!='validation' or not data.get('independence_audit',{}).get('passed'):
        raise ValueError('Action thresholds require an audited validation export')
    if data.get('component')!='behavior' or not data.get('runtime_signature') or not data.get('training_manifest'):
        raise ValueError('Expected a behavior export with runtime and training provenance')
    if data['runtime_signature'].get('behavior_policy_sha256'):
        raise ValueError('Export scores with the default action policy before fitting a new policy')
    labels=data['action_class_names']
    thresholds,diagnostics=select_thresholds([r['action_probabilities'] for r in data['rows']],[r['action_targets'] for r in data['rows']],labels)
    return {'fit_split':'validation','method':'per_action_validation_f1','runtime_signature':data['runtime_signature'],'action_thresholds':thresholds,'diagnostics':diagnostics,'label_basis':data.get('label_basis','unverified')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    artifact=fit_export(json.loads(args.validation.read_text()))
    artifact['validation_sha256']=sha256_file(args.validation)
    if args.output.exists():raise ValueError('Choose a new policy artifact path')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(artifact,indent=2)+'\n')
    print(json.dumps({'thresholds':artifact['action_thresholds'],'diagnostics':artifact['diagnostics']},indent=2))


if __name__=='__main__':main()
