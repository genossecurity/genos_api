#!/usr/bin/env python3
"""Validate two blind human reviews; export only agreed or explicitly adjudicated rows."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from genos.scientific_validation import read_rows,dataset_manifest
LABELS={'Benign','Malicious','Context_Dependent'}


def indexed(rows):
    result={}
    for row in rows:
        key=hashlib.sha256(row['command'].encode()).hexdigest()
        if row.get('id') != key or key in result:
            raise ValueError('Annotation ID mismatch or duplicate')
        if row.get('verdict') not in LABELS or not row.get('annotator_id') or not row.get('rationale'):
            raise ValueError('Each review needs a valid verdict, reviewer ID, and rationale')
        if not isinstance(row.get('mitre_codes'),list) or not isinstance(row.get('observable_behavior'),list):
            raise ValueError('MITRE codes and observable behavior must be lists, including [] when none applies')
        if row.get('authorization') not in {'authorized','unauthorized','unknown'}:
            raise ValueError('Authorization must be explicit, including unknown')
        result[key]=row
    return result


def reconcile(left,right,adjudications=None):
    a,b=indexed(left),indexed(right)
    if a.keys()!=b.keys():raise ValueError('Reviewers must annotate the same examples')
    adj=indexed(adjudications or [])
    if not adj.keys() <= a.keys():raise ValueError('Adjudication contains unknown examples')
    agreed,unresolved=[],[]
    agreement=0
    for key,x in a.items():
        y=b[key]
        if x['annotator_id']==y['annotator_id']:raise ValueError('Two distinct reviewers are required')
        same=x['verdict']==y['verdict'] and set(x['mitre_codes'])==set(y['mitre_codes']) and set(x['observable_behavior'])==set(y['observable_behavior']) and x['authorization']==y['authorization']
        agreement+=int(same)
        if same:resolved=x
        elif key in adj:
            resolved=adj[key]
            if resolved['annotator_id'] in {x['annotator_id'],y['annotator_id']}:raise ValueError('Adjudication requires a third reviewer')
        else:
            unresolved.append({'id':key,'command':x['command'],'reviews':[x,y]});continue
        agreed.append(dict(resolved,label=resolved['verdict'],label_basis='independent_human_review_attested',reviewer_ids=[x['annotator_id'],y['annotator_id']],adjudicated=not same))
    kappa=float(cohen_kappa_score([r['verdict'] for r in a.values()],[b[k]['verdict'] for k in a],labels=sorted(LABELS))) if len({r['verdict'] for r in list(a.values()) + list(b.values())}) > 1 else None
    # Kappa is undefined if both reviewers use only one identical class.
    if kappa is not None and not (-1 <= kappa <= 1):kappa=None
    report={'n':len(a),'full_annotation_agreement':agreement/len(a) if a else None,'verdict_cohen_kappa':kappa,'accepted':len(agreed),'unresolved':len(unresolved),'provenance_note':'Reviewer identities and independence are attestations; software cannot establish blinding or independence'}
    return agreed,unresolved,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-a',type=Path,required=True);p.add_argument('--review-b',type=Path,required=True)
    p.add_argument('--adjudications',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Choose an empty output directory')
    accepted,unresolved,report=reconcile(read_rows(args.review_a),read_rows(args.review_b),read_rows(args.adjudications) if args.adjudications else [])
    report['inputs']=dataset_manifest({'review_a':args.review_a,'review_b':args.review_b,**({'adjudications':args.adjudications} if args.adjudications else {})})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,rows in [('accepted',accepted),('unresolved',unresolved)]:
        (args.output_dir/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.output_dir/'agreement.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
