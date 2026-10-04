#!/usr/bin/env python3
"""Compare gatekeeper view policies, including the retired risk-selection baseline."""
import argparse
import json
import sys
from pathlib import Path
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scientific_validation import normalize_command,probability_metrics,sha256_file
from scripts.evaluation.compare_ablations import compare
LABELS=['Benign','Malicious','Context_Dependent']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','decoded','mean']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    exports={name:json.loads(getattr(args,name).read_text()) for name in ['raw','decoded','mean']}
    reference=None;rows={}
    for name,data in exports.items():
        if data['component']!='gatekeeper' or data['class_names']!=LABELS or data.get('calibration_applied'):
            raise ValueError('Expected uncalibrated gatekeeper exports in the declared class order')
        signature=dict(data['runtime_signature']);policy=signature.pop('view_policy')
        if policy!=name:raise ValueError('Export policy does not match argument')
        if reference is not None and signature!=reference:raise ValueError('Models or preprocessing differ between exports')
        if any(data.get(field) != exports['raw'].get(field) for field in ['split', 'training_manifest', 'dataset_manifest']):
            raise ValueError('Evaluation/training provenance differs between exports')
        reference=signature
        rows[name]={normalize_command(r['command']):r for r in data['rows']}
        if len(rows[name])!=len(data['rows']):raise ValueError('Duplicate examples')
    if not rows['raw'].keys()==rows['decoded'].keys()==rows['mean'].keys():raise ValueError('View policies must use identical examples')
    legacy={}
    for key,raw in rows['raw'].items():
        decoded=rows['decoded'][key]
        if raw['target']!=decoded['target'] or raw['target']!=rows['mean'][key]['target']:raise ValueError('Targets differ')
        risk=lambda row:row['probabilities'][1]+.55*row['probabilities'][2]
        legacy[key]=raw if risk(raw)>risk(decoded)+.03 else decoded
    rows['legacy_risk_selection']=legacy
    results={}
    for name,indexed in rows.items():
        data=list(indexed.values());targets=[r['target'] for r in data]
        predicted=[max(range(3),key=lambda i:r['probabilities'][i]) for r in data]
        benign=sum(y==0 for y in targets)
        results[name]={'metrics':probability_metrics([r['probabilities'] for r in data],targets),'macro_f1':float(f1_score(targets,predicted,labels=[0,1,2],average='macro',zero_division=0)),'benign_to_malicious_rate':sum(y==0 and pred==1 for y,pred in zip(targets,predicted))/benign if benign else None,'benign_to_context_rate':sum(y==0 and pred==2 for y,pred in zip(targets,predicted))/benign if benign else None}
        if all(r.get('family_id') for r in data):
            results[name]['paired_vs_raw']=compare(list(rows['raw'].values()),[dict(r,prediction=pred) for r,pred in zip(data,predicted)])
    report={'policies':results,'split':exports['raw']['split'],'inputs':{name:sha256_file(getattr(args,name)) for name in exports},'legacy_parameters':{'context_weight':.55,'raw_margin':.03},'interpretation':'Select policy using validation only; test/development comparisons do not authorize tuning on their labels'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
