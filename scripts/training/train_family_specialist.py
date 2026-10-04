#!/usr/bin/env python3
"""Train a calibrated linear multi-label command-family specialist."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scientific_validation import dataset_manifest, read_rows, require_disjoint, sha256_file

WORD_TOKEN_PATTERN = r"(?u)(?:--?[A-Za-z][\w-]*|/[A-Za-z](?:[\w-]*)?|\b[\w][\w./:+#-]*\b)"
BENIGN_LABEL = "Benign Admin"


def build_vectorizer(char_features: int, word_features: int) -> FeatureUnion:
    return FeatureUnion([
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


def targets_for(rows: list[dict], labels: list[str]) -> np.ndarray:
    indices = {label: index for index, label in enumerate(labels)}
    targets = np.zeros((len(rows), len(labels)), dtype=np.uint8)
    for row_index, row in enumerate(rows):
        row_labels = row.get("family_labels")
        if not isinstance(row_labels, list) or not row_labels:
            raise ValueError(f"Row {row_index} has no positive family labels")
        unknown = set(row_labels) - indices.keys()
        if unknown:
            raise ValueError(f"Row {row_index} has unknown family labels: {sorted(unknown)}")
        for label in row_labels:
            targets[row_index, indices[label]] = 1
    return targets


def calibration_folds(binary_targets: np.ndarray, groups: np.ndarray, requested: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    positive_groups = len(set(groups[binary_targets == 1]))
    negative_groups = len(set(groups[binary_targets == 0]))
    maximum = min(requested, positive_groups, negative_groups)
    for fold_count in range(maximum, 1, -1):
        splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros((len(binary_targets), 1)), binary_targets, groups))
        if all(set(np.unique(binary_targets[train])) == {0, 1} for train, _ in folds):
            if all(set(groups[train]).isdisjoint(set(groups[validation])) for train, validation in folds):
                return folds
    raise ValueError(
        "Cannot make group-isolated calibration folds containing both classes "
        f"(positive groups={positive_groups}, negative groups={negative_groups})"
    )


def fit_estimators(
    features,
    targets: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    folds: int,
    seed: int,
) -> tuple[list[CalibratedClassifierCV], dict[str, int]]:
    estimators = []
    fold_counts = {}
    for label_index, label in enumerate(labels):
        binary = targets[:, label_index].astype(np.int8)
        cv = calibration_folds(binary, groups, folds, seed + label_index)
        estimator = CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000, random_state=seed + label_index),
            method="sigmoid", cv=cv, ensemble=False, n_jobs=1,
        )
        print(
            f"Fitting {label_index + 1}/{len(labels)} {label}: "
            f"positives={int(binary.sum())}, groups={len(set(groups))}, folds={len(cv)}",
            flush=True,
        )
        estimator.fit(features, binary)
        estimators.append(estimator)
        fold_counts[label] = len(cv)
    return estimators, fold_counts


def predict_probabilities(estimators: list[CalibratedClassifierCV], features) -> np.ndarray:
    columns = []
    for estimator in estimators:
        classes = np.asarray(estimator.classes_, dtype=int)
        positive_index = np.flatnonzero(classes == 1)
        if len(positive_index) != 1:
            raise ValueError(f"Calibrated family estimator lacks positive class: {classes.tolist()}")
        columns.append(estimator.predict_proba(features)[:, positive_index[0]])
    return np.column_stack(columns)


def metric_summary(targets: np.ndarray, probabilities: np.ndarray, labels: list[str], threshold: float) -> dict:
    predictions = probabilities >= threshold
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=np.arange(len(labels)), average=None, zero_division=0,
    )
    rankings = np.argsort(-probabilities, axis=1)
    top1 = np.mean([bool(targets[index, ranking[0]]) for index, ranking in enumerate(rankings)])
    top2 = np.mean([bool(targets[index, ranking[:2]].any()) for index, ranking in enumerate(rankings)])
    family_metrics = {
        label: {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    benign_index = labels.index(BENIGN_LABEL)
    benign_truth = targets[:, benign_index].astype(bool)
    benign_predicted = predictions[:, benign_index]
    tp = int(np.sum(benign_truth & benign_predicted))
    fp = int(np.sum(~benign_truth & benign_predicted))
    fn = int(np.sum(benign_truth & ~benign_predicted))
    tn = int(np.sum(~benign_truth & ~benign_predicted))
    benign_precision = tp / (tp + fp) if tp + fp else 0.0
    benign_recall = tp / (tp + fn) if tp + fn else 0.0
    benign_f1 = 2 * benign_precision * benign_recall / (benign_precision + benign_recall) if benign_precision + benign_recall else 0.0

    confusions = Counter()
    for row_index, ranking in enumerate(rankings):
        predicted_top = labels[int(ranking[0])]
        true_labels = [labels[index] for index in np.flatnonzero(targets[row_index])]
        if predicted_top not in true_labels:
            for true_label in true_labels:
                confusions[(true_label, predicted_top)] += 1

    return {
        "threshold": threshold,
        "per_family": family_metrics,
        "macro_f1": float(np.mean(f1)),
        "top1_accuracy_any_true_family": float(top1),
        "top2_accuracy_any_true_family": float(top2),
        "benign_admin_vs_rest": {
            "precision": benign_precision, "recall": benign_recall, "f1": benign_f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "top1_confusions": [
            {"true_family": true_label, "predicted_family": predicted_label, "count": int(count)}
            for (true_label, predicted_label), count in confusions.most_common(30)
        ],
        "predicted_labels_per_row_mean": float(predictions.sum(axis=1).mean()),
    }


def grouped_bootstrap(targets: np.ndarray, probabilities: np.ndarray, groups: list[str], labels: list[str], threshold: float, iterations: int, seed: int) -> dict:
    rows_by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        rows_by_group[group].append(index)
    group_names = list(rows_by_group)
    rng = np.random.default_rng(seed)
    metrics = {"macro_f1": [], "top1_accuracy_any_true_family": [], "top2_accuracy_any_true_family": [], "benign_admin_f1": []}
    family_f1 = {label: [] for label in labels}
    for _ in range(iterations):
        selected_groups = rng.choice(group_names, size=len(group_names), replace=True)
        indices = np.asarray([row_index for group in selected_groups for row_index in rows_by_group[group]], dtype=np.int64)
        current = metric_summary(targets[indices], probabilities[indices], labels, threshold)
        metrics["macro_f1"].append(current["macro_f1"])
        metrics["top1_accuracy_any_true_family"].append(current["top1_accuracy_any_true_family"])
        metrics["top2_accuracy_any_true_family"].append(current["top2_accuracy_any_true_family"])
        metrics["benign_admin_f1"].append(current["benign_admin_vs_rest"]["f1"])
        for label in labels:
            family_f1[label].append(current["per_family"][label]["f1"])
    intervals = {
        metric: np.quantile(values, [0.025, 0.975]).tolist()
        for metric, values in metrics.items()
    }
    intervals["per_family_f1"] = {
        label: np.quantile(values, [0.025, 0.975]).tolist()
        for label, values in family_f1.items()
    }
    return {"template_groups": len(group_names), "iterations": iterations, "seed": seed, "95ci": intervals}


def top_linear_features(vectorizer: FeatureUnion, estimators: list[CalibratedClassifierCV], labels: list[str], limit: int) -> dict:
    names = vectorizer.get_feature_names_out()
    result = {}
    for label, estimator in zip(labels, estimators):
        linear = estimator.calibrated_classifiers_[0].estimator
        weights = linear.coef_[0]
        indices = np.argsort(weights)[-limit:][::-1]
        result[label] = [{"feature": str(names[index]), "weight": float(weights[index])} for index in indices]
    return result


def measure_latency(vectorizer: FeatureUnion, estimators: list[CalibratedClassifierCV], texts: list[str], samples: int) -> dict:
    subset = [text.lower().strip() for text in texts[:samples]]
    for text in subset[:10]:
        features = vectorizer.transform([text])
        predict_probabilities(estimators, features)
    elapsed_ms = []
    for text in subset:
        started = time.perf_counter()
        features = vectorizer.transform([text])
        predict_probabilities(estimators, features)
        elapsed_ms.append((time.perf_counter() - started) * 1000)
    return {
        "n": len(elapsed_ms),
        "median_ms": float(statistics.median(elapsed_ms)) if elapsed_ms else None,
        "p95_ms": float(np.quantile(elapsed_ms, 0.95)) if elapsed_ms else None,
        "scope": "single-command CPU family predictions; excludes behavior, MITRE, and HTTP overhead",
    }


def export_predictions(path: Path, rows: list[dict], targets: np.ndarray, probabilities: np.ndarray, labels: list[str], threshold: float) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row, target, scores in zip(rows, targets, probabilities):
            handle.write(json.dumps({
                "command": row["command"],
                "family_targets": [labels[index] for index in np.flatnonzero(target)],
                "family_probabilities": {label: float(score) for label, score in zip(labels, scores)},
                "predicted_families": [labels[index] for index in np.flatnonzero(scores >= threshold)],
                "template_id": row["template_id"],
            }, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/derived/family_specialist_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--char-features", type=int, default=250000)
    parser.add_argument("--word-features", type=int, default=100000)
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--latency-samples", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.folds < 2 or not 0 < args.threshold < 1 or args.bootstrap_iterations < 1:
        parser.error("folds and bootstrap iterations must be positive; threshold must be in (0, 1)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.output_dir}")

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    labels = list(manifest["family_labels"])
    if len(labels) != 11 or BENIGN_LABEL not in labels:
        raise ValueError("Family dataset must contain the declared 11 labels, including Benign Admin")
    paths = {split: args.data_dir / f"family_{split}.jsonl" for split in ("train", "val", "test")}
    rows = {split: read_rows(path) for split, path in paths.items()}
    split_audit = require_disjoint({
        split: [dict(row, holdout_group=row["template_id"]) for row in split_rows]
        for split, split_rows in rows.items()
    })
    if not split_audit.get("passed"):
        raise ValueError("Family dataset split leakage: " + json.dumps(split_audit))

    texts = {split: [row["command"].lower().strip() for row in data] for split, data in rows.items()}
    targets = {split: targets_for(data, labels) for split, data in rows.items()}
    groups = np.asarray([row["template_id"] for row in rows["train"]])
    vectorizer = build_vectorizer(args.char_features, args.word_features)
    print(f"Fitting shared char+word TF-IDF vocabulary on {len(texts['train'])} training commands.", flush=True)
    fit_started = time.perf_counter()
    x_train = vectorizer.fit_transform(texts["train"])
    estimators, fold_counts = fit_estimators(x_train, targets["train"], groups, labels, args.folds, args.seed)
    fit_seconds = time.perf_counter() - fit_started

    features = {"train": x_train}
    for split in ("val", "test"):
        features[split] = vectorizer.transform(texts[split])
    results = {}
    probabilities = {}
    for split in ("val", "test"):
        probabilities[split] = predict_probabilities(estimators, features[split])
        results[split] = metric_summary(targets[split], probabilities[split], labels, args.threshold)
        if split == "test":
            results[split]["grouped_bootstrap"] = grouped_bootstrap(
                targets[split], probabilities[split],
                [row["template_id"] for row in rows[split]], labels,
                args.threshold, args.bootstrap_iterations, args.seed,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "family_specialist_tfidf.joblib"
    model_bundle = {
        "vectorizer": vectorizer,
        "estimators": estimators,
        "family_labels": labels,
        "decision_threshold": args.threshold,
        "normalization": "lowercase_strip",
    }
    joblib.dump(model_bundle, model_path)
    export_predictions(args.output_dir / "val_predictions.jsonl", rows["val"], targets["val"], probabilities["val"], labels, args.threshold)
    export_predictions(args.output_dir / "test_predictions.jsonl", rows["test"], targets["test"], probabilities["test"], labels, args.threshold)

    latency = measure_latency(vectorizer, estimators, texts["test"], args.latency_samples)
    metadata = {
        "model_type": "multi_label_family_specialist_tfidf_char_word_linear_svc",
        "family_labels": labels,
        "technique_ids_in_model_or_rows": False,
        "training_arguments": json.loads(json.dumps(vars(args), default=str)),
        "feature_design": {
            "char": {"analyzer": "char_wb", "ngram_range": [2, 5], "sublinear_tf": True},
            "word": {"token_pattern": WORD_TOKEN_PATTERN, "ngram_range": [1, 2], "sublinear_tf": True},
        },
        "calibration": "per-family sigmoid CalibratedClassifierCV with template-group-isolated folds on train only",
        "folds_per_family": fold_counts,
        "decision_threshold": args.threshold,
        "training_examples": len(rows["train"]),
        "training_template_groups": len(set(groups)),
        "dataset_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "split_manifest": manifest,
        "split_independence_audit": split_audit,
        "fit_seconds": fit_seconds,
        "single_command_latency": latency,
        "metrics": results,
        "top_linear_features": top_linear_features(vectorizer, estimators, labels, args.top_features),
        "checkpoint_sha256": sha256_file(model_path),
        "training_code_sha256": sha256_file(__file__),
        "label_basis": "weak source-derived tactic families and gatekeeper benign labels; not human-verified",
        "evaluation_scope": "template-grouped local research splits; production generalization not measured",
    }
    (args.output_dir / "family_specialist_tfidf.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "fit_seconds": fit_seconds,
        "validation": {
            "macro_f1": results["val"]["macro_f1"],
            "top1": results["val"]["top1_accuracy_any_true_family"],
            "top2": results["val"]["top2_accuracy_any_true_family"],
        },
        "test": {
            "macro_f1": results["test"]["macro_f1"],
            "top1": results["test"]["top1_accuracy_any_true_family"],
            "top2": results["test"]["top2_accuracy_any_true_family"],
            "benign_admin_vs_rest": results["test"]["benign_admin_vs_rest"],
            "grouped_bootstrap": results["test"]["grouped_bootstrap"],
        },
        "latency": latency,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
