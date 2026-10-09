#!/usr/bin/env python3
"""Explicit weak-label rule baseline; never used to override runtime predictions."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from genos.scientific_validation import read_rows,command_of,sha256_file

spec=importlib.util.spec_from_file_location('labeling_rules',ROOT/'parser/build_3class_dataset.py')
rules=importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)
LABELS=['Benign','Malicious','Context_Dependent']


def predict(command):
    if rules._has_malicious_indicators(command):return 'Malicious'
    if rules._benign_is_context_dependent(command):return 'Context_Dependent'
    return 'Benign'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=read_rows(args.dataset)
    actual=[r['label'] for r in rows];predicted=[predict(command_of(r)) for r in rows]
    matrix=confusion_matrix(actual,predicted,labels=LABELS).tolist()
    benign=sum(label=='Benign' for label in actual)
    report={'baseline':'labeling_rule_family','label_names':LABELS,'n':len(rows),'accuracy':sum(a==b for a,b in zip(actual,predicted))/len(rows),'macro_f1':float(f1_score(actual,predicted,labels=LABELS,average='macro',zero_division=0)),'benign_to_malicious_rate':sum(a=='Benign' and b=='Malicious' for a,b in zip(actual,predicted))/benign if benign else None,'benign_to_context_rate':sum(a=='Benign' and b=='Context_Dependent' for a,b in zip(actual,predicted))/benign if benign else None,'confusion_matrix':matrix,'dataset_sha256':sha256_file(args.dataset),'rules_sha256':sha256_file(ROOT/'parser/build_3class_dataset.py'),'limitation':'Shares rule families with weak supervision; evaluates teacher agreement, not independently established intent'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
