#!/usr/bin/env python3
"""Train a calibrated linear TF-IDF gatekeeper on the grouped CodeBERT splits."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scientific_validation import command_of, dataset_manifest, normalize_command, probability_metrics, read_rows, require_disjoint, sha256_file

LABELS = ["Benign", "Malicious", "Context_Dependent"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
WORD_TOKEN_PATTERN = r"(?u)(?:--?[A-Za-z][\w-]*|/[A-Za-z](?:[\w-]*)?|\b[\w][\w./:+#-]*\b)"


def build_model(seed: int, folds: list[tuple[np.ndarray, np.ndarray]], char_features: int, word_features: int) -> Pipeline:
    vectorizers = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
            max_features=char_features, min_df=2, dtype=np.float32,
        )),
        ("word", TfidfVectorizer(
            analyzer="word", token_pattern=WORD_TOKEN_PATTERN,
            ngram_range=(1, 2), sublinear_tf=True,
            max_features=word_features, min_df=2, dtype=np.float32,
        )),
    ])
    classifier = CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000, random_state=seed),
        method="sigmoid",
        cv=folds,
        ensemble=False,
        n_jobs=1,
    )
    return Pipeline([("features", vectorizers), ("calibrated", classifier)])


def row_group(row: dict) -> str:
    for field in ("holdout_group", "family_id", "source_group", "source_family"):
        value = row.get(field)
        if value:
            return f"{field}:{value}"
    return "command:" + normalize_command(command_of(row))


def score_split(model: Pipeline, rows: list[dict]) -> tuple[dict, list[dict]]:
    texts = [command_of(row).lower().strip() for row in rows]
    targets = np.asarray([LABEL_TO_INDEX[row["label"]] for row in rows], dtype=np.int64)
    probabilities = model.predict_proba(texts)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if not np.array_equal(classes, np.arange(len(LABELS))):
        raise ValueError(f"Unexpected classifier class order: {classes.tolist()}")
    predictions = classes[np.argmax(probabilities, axis=1)]
    metrics = probability_metrics(probabilities, targets)
    metrics.update({
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "per_class_f1": {
            label: float(f1_score(targets, predictions, labels=[index], average="macro", zero_division=0))
            for index, label in enumerate(LABELS)
        },
    })
    exports = []
    for row, target, prediction, scores in zip(rows, targets, predictions, probabilities):
        exports.append({
            "command": command_of(row),
            "target": int(target),
            "label": LABELS[int(target)],
            "prediction": int(prediction),
            "probabilities": [float(value) for value in scores],
            "family_id": row.get("family_id"),
            "source_group": row.get("source_group"),
            "holdout_group": row.get("holdout_group"),
        })
    return metrics, exports


def top_linear_features(model: Pipeline, limit: int) -> dict:
    calibrated = model.named_steps["calibrated"]
    fitted_linear = calibrated.calibrated_classifiers_[0].estimator
    feature_names = model.named_steps["features"].get_feature_names_out()
    coefficients = fitted_linear.coef_
    result = {}
    for class_index, label in enumerate(LABELS):
        weights = coefficients[class_index]
        largest = np.argsort(weights)[-limit:][::-1]
        result[label] = [
            {"feature": str(feature_names[index]), "weight": float(weights[index])}
            for index in largest
        ]
    return result


def measure_latency_ms(model: Pipeline, texts: list[str], sample_size: int) -> dict:
    sample = texts[:sample_size]
    if not sample:
        return {"n": 0, "median_ms": None, "p95_ms": None}
    for text in sample[:10]:
        model.predict_proba([text])
    durations = []
    for text in sample:
        started = time.perf_counter()
        model.predict_proba([text])
        durations.append((time.perf_counter() - started) * 1000)
    return {
        "n": len(durations),
        "median_ms": float(statistics.median(durations)),
        "p95_ms": float(np.quantile(durations, 0.95)),
        "scope": "single-command TF-IDF gate only; excludes API, behavior, and MITRE inference",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/derived/scientific_v2/gatekeeper")
    parser.add_argument("--train-patch", type=Path, default=ROOT / "data/training/genos_dataset/gatekeeper_benign_core_patch_v2a.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--char-features", type=int, default=250000)
    parser.add_argument("--word-features", type=int, default=100000)
    parser.add_argument("--top-features", type=int, default=30)
    parser.add_argument("--latency-samples", type=int, default=500)
    args = parser.parse_args()
    if args.folds < 2 or min(args.char_features, args.word_features, args.top_features, args.latency_samples) < 1:
        parser.error("folds must be >= 2 and feature/sample limits must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.output_dir}")

    split_paths = {name: args.data_dir / f"gatekeeper_3class_{name}.csv" for name in ("train", "val", "test")}
    rows = {name: read_rows(path) for name, path in split_paths.items()}
    patch_rows = read_rows(args.train_patch) if args.train_patch else []
    rows["train"].extend(patch_rows)
    audit = require_disjoint(rows)
    if not audit.get("passed"):
        raise ValueError("Train/validation/test/patch group audit failed: " + json.dumps(audit))
    for split, split_rows in rows.items():
        unknown = {row.get("label") for row in split_rows} - set(LABEL_TO_INDEX)
        if unknown:
            raise ValueError(f"Unexpected labels in {split}: {sorted(unknown)}")

    train_text = [command_of(row).lower().strip() for row in rows["train"]]
    train_targets = np.asarray([LABEL_TO_INDEX[row["label"]] for row in rows["train"]], dtype=np.int64)
    groups = np.asarray([row_group(row) for row in rows["train"]])
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    cv_splits = list(splitter.split(np.zeros(len(train_targets)), train_targets, groups))
    for train_indices, calibration_indices in cv_splits:
        if set(groups[train_indices]) & set(groups[calibration_indices]):
            raise AssertionError("Calibration folds split a family/holdout group")
        if set(train_targets[train_indices]) != set(range(len(LABELS))):
            raise ValueError("A calibration training fold is missing a class")

    model = build_model(args.seed, cv_splits, args.char_features, args.word_features)
    print(f"Fitting calibrated linear gate on {len(train_text)} train rows; {len(cv_splits)} group folds.", flush=True)
    fit_started = time.perf_counter()
    model.fit(train_text, train_targets)
    fit_seconds = time.perf_counter() - fit_started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "gatekeeper_tfidf.joblib"
    joblib.dump(model, model_path)
    split_results = {}
    for split in ("val", "test"):
        metrics, exports = score_split(model, rows[split])
        prediction_path = args.output_dir / f"{split}_predictions.jsonl"
        write_jsonl(prediction_path, exports)
        split_results[split] = metrics
        print(f"{split}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)

    latency = measure_latency_ms(model, [command_of(row).lower().strip() for row in rows["test"]], args.latency_samples)
    metadata = {
        "backend": "tfidf_char_word_linear_svc_sigmoid_calibration",
        "class_names": LABELS,
        "training_arguments": json.loads(json.dumps(vars(args), default=str)),
        "normalization": "lowercase_strip",
        "feature_design": {
            "char": {"analyzer": "char_wb", "ngram_range": [2, 5], "sublinear_tf": True, "max_features": args.char_features},
            "word": {"analyzer": "word", "ngram_range": [1, 2], "token_pattern": WORD_TOKEN_PATTERN, "sublinear_tf": True, "max_features": args.word_features},
        },
        "classifier": "LinearSVC(C=1.0, class_weight=balanced) wrapped in CalibratedClassifierCV(method=sigmoid, ensemble=False)",
        "calibration": "StratifiedGroupKFold out-of-fold calibration on training groups only",
        "training_examples": len(train_text),
        "patch_examples": len(patch_rows),
        "training_group_count": len(set(groups)),
        "cv_fold_count": len(cv_splits),
        "dataset_manifest": dataset_manifest(split_paths),
        "patch_sha256": sha256_file(args.train_patch) if args.train_patch else None,
        "independence_audit": audit,
        "evaluation_scope": "same grouped split and benign training patch as CodeBERT; legacy weak labels",
        "fit_seconds": fit_seconds,
        "single_command_gate_latency": latency,
        "metrics": split_results,
        "top_linear_features": top_linear_features(model, args.top_features),
        "checkpoint_sha256": sha256_file(model_path),
        "training_code_sha256": sha256_file(__file__),
        "score_type": "cross_validated_calibrated_model_estimate",
    }
    metadata_path = args.output_dir / "gatekeeper_tfidf.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary = {"output_dir": str(args.output_dir), "fit_seconds": fit_seconds, "metrics": split_results, "single_command_gate_latency": latency}
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
