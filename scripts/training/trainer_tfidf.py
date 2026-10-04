#!/usr/bin/env python3
"""Train isolated RAW/structured TF-IDF ablations on identical audited splits."""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scientific_validation import read_rows, representation, require_disjoint, dataset_manifest, sha256_file, probability_metrics


def build_rf_pipeline(n_estimators=400, seed=42, character=False):
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb" if character else "word", ngram_range=(2, 5) if character else (1, 3), max_features=30000, sublinear_tf=True)),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample", max_features="sqrt", min_samples_leaf=1, random_state=seed, n_jobs=-1)),
    ])


def build_char_rf_pipeline(n_estimators=400, seed=42):
    return build_rf_pipeline(n_estimators, seed, character=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/derived/scientific_v2/mitre")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--input-format", choices=["raw", "structured"], default="structured")
    p.add_argument("--model", choices=["rf", "char_rf", "both"], default="char_rf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=400)
    args = p.parse_args()
    paths = {split: args.data_dir / f"specialist_{split}_variant_a.jsonl" for split in ("train", "val", "test")}
    rows = {split: read_rows(path) for split, path in paths.items()}
    independence = require_disjoint(rows)
    label_map = json.loads((ROOT / "config/specialist_map.json").read_text())
    unknown = {r["label"] for data in rows.values() for r in data} - label_map.keys()
    if unknown:
        raise ValueError(f"Unmapped labels must not be silently dropped: {sorted(unknown)}")
    X = {split: [representation(r, args.input_format) for r in data] for split, data in rows.items()}
    y = {split: np.array([int(label_map[r["label"]]) for r in data]) for split, data in rows.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in (["rf", "char_rf"] if args.model == "both" else [args.model]):
        output = args.output_dir / f"specialist_tfidf_{variant}.pkl"
        if output.exists():
            raise ValueError(f"Refusing to overwrite experiment: {output}")
        model = build_rf_pipeline(args.n_estimators, args.seed, character=variant == "char_rf")
        start = time.perf_counter()
        model.fit(X["train"], y["train"])
        results = {}
        for split in ("val", "test"):
            scores = model.predict_proba(X[split])
            # RF columns index classes_, not the global label map.
            predicted = model.classes_[scores.argmax(axis=1)]
            topk = model.classes_[np.argsort(scores, axis=1)[:, -5:]]
            full = np.zeros((len(scores), len(label_map)))
            full[:, model.classes_.astype(int)] = scores
            results[split] = {
                "top1_acc": float(accuracy_score(y[split], predicted)),
                "top5_acc": float(np.mean([target in rank for target, rank in zip(y[split], topk)])),
                "macro_f1": float(f1_score(y[split], predicted, labels=sorted(label_map.values()), average="macro", zero_division=0)),
                "unseen_label_rows": int(sum(target not in model.classes_ for target in y[split])),
                "calibration": probability_metrics(full, y[split]),
            }
            prediction_rows = [dict(command=r["raw_command"], family_id=r.get("family_id"), source_group=r.get("source_group"), holdout_group=r.get("holdout_group"), target=int(target), prediction=int(pred), probabilities=prob.tolist()) for r, target, pred, prob in zip(rows[split], y[split], predicted, full)]
            (args.output_dir / f"{variant}_{split}_predictions.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in prediction_rows))
        joblib.dump(model, output)
        meta = {"training_arguments": json.loads(json.dumps(vars(args), default=str)), "training_code_sha256": sha256_file(__file__), "seed": args.seed, "input_format": args.input_format, "label_map": label_map, "n_estimators": args.n_estimators, "dataset_manifest": dataset_manifest(paths), "independence_audit": independence, "evaluation_scope": "legacy_weak_labels_grouped_by_tool", "score_type": "uncalibrated_model_estimate", "checkpoint_sha256": sha256_file(output), "seconds": time.perf_counter() - start, "metrics": results}
        output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
        print(json.dumps({"variant": variant, "input_format": args.input_format, "seed": args.seed, "seconds": meta["seconds"], "metrics": {split: {k:v for k,v in m.items() if k != 'calibration'} for split,m in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
