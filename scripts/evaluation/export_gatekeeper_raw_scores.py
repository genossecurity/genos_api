#!/usr/bin/env python3
"""Batch-export raw-view gatekeeper scores with deployed model/preprocessing hashes.

This evaluates the verdict component under GENOS_VIEW_POLICY=raw. It does not
measure behavior coverage, decoded-view fusion, or complete-pipeline latency.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from sklearn.metrics import f1_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from genos.scientific_validation import read_rows,command_of,require_disjoint,dataset_manifest,probability_metrics,sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/derived/scientific_v2/gatekeeper')
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    paths={split:args.data_dir/f'gatekeeper_3class_{split}.csv' for split in ['train','val','test']}
    rows={split:read_rows(path) for split,path in paths.items()}
    audit=require_disjoint(rows)
    meta=json.loads((args.checkpoint_dir/'gatekeeper_meta.json').read_text())
    for split,path in paths.items():
        if meta['dataset_manifest'][split]['sha256']!=sha256_file(path):
            raise ValueError('Supplied datasets do not match training metadata')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Choose an empty export directory')
    from genos.engine import GenosEngine
    engine=GenosEngine(t1_path=str(args.checkpoint_dir/'gatekeeper.pt'),gatekeeper_meta_path=str(args.checkpoint_dir/'gatekeeper_meta.json'),view_policy='raw')
    if engine.calibration:raise ValueError('Export uncalibrated scores for fitting')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary={}
    names=engine._GATE_LABELS
    for split in ['val','test']:
        exported=[]
        for start in range(0,len(rows[split]),args.batch_size):
            batch=rows[split][start:start+args.batch_size]
            text=[command_of(row).lower().strip() for row in batch]
            encoded=engine.tokenizer(text,truncation=True,padding='max_length',max_length=engine.max_length,return_tensors='pt').to(engine.device)
            with torch.no_grad(),autocast(device_type=engine.device.type,dtype=torch.float16 if engine.device.type=='cuda' else torch.bfloat16):
                logits=engine.t1(encoded['input_ids'],encoded['attention_mask'])['verdict_logits']
            scores=torch.softmax(logits.float(),dim=1).cpu().tolist()
            for row,score in zip(batch,scores):
                exported.append({'command':command_of(row),'label':row['label'],'target':names.index(row['label']),'probabilities':score,'prediction':int(np.argmax(score)),'family_id':row.get('family_id'),'holdout_group':row.get('holdout_group')})
            if start%1000<args.batch_size:print(f'{split}: {min(start+len(batch),len(rows[split]))}/{len(rows[split])}',flush=True)
        y=[row['target'] for row in exported];pred=[row['prediction'] for row in exported]
        metrics=probability_metrics([row['probabilities'] for row in exported],y)
        metrics['macro_f1']=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0))
        metrics['confusion_matrix']=confusion_matrix(y,pred,labels=[0,1,2]).tolist()
        benign=sum(target==0 for target in y)
        metrics['benign_to_malicious_rate']=sum(target==0 and prediction==1 for target,prediction in zip(y,pred))/benign
        metrics['benign_to_context_rate']=sum(target==0 and prediction==2 for target,prediction in zip(y,pred))/benign
        export={'split':'validation' if split=='val' else 'test','component':'gatekeeper','class_names':names,'runtime_signature':engine.provenance,'calibration_applied':False,'training_manifest':dataset_manifest({'train':paths['train']}),'dataset_manifest':dataset_manifest({'evaluation':paths[split]}),'independence_audit':audit,'label_basis':'legacy_weak_supervision_unreviewed','evaluation_scope':'raw-view verdict component only; no whole-pipeline coverage or latency claim','rows':exported,'metrics':metrics}
        (args.output_dir/f'{split}.json').write_text(json.dumps(export)+'\n')
        summary[split]=metrics
    (args.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
