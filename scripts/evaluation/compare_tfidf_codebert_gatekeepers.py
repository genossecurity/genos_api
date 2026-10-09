#!/usr/bin/env python3
"""Paired raw/decoded-view comparison of TF-IDF and CodeBERT gatekeepers."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import autocast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from genos.scientific_validation import command_of, dataset_manifest, probability_metrics, read_rows, require_disjoint, sha256_file
from scripts.training.train_tfidf_gatekeeper import LABELS


def decode_for_runtime(engine, command: str) -> tuple[str, bool]:
    current = command.strip()
    if not engine.is_obfuscated(current):
        return current, False
    was_obfuscated = True
    previous_entropy = engine.calculate_entropy(current)
    for _ in range(engine.max_deobfuscation_layers):
        if not engine.is_obfuscated(current):
            break
        decoded = engine.deobfuscate_layer(current)
        if decoded == current:
            break
        current = decoded
        entropy = engine.calculate_entropy(current)
        if abs(previous_entropy - entropy) < 0.01:
            break
        previous_entropy = entropy
    return current, was_obfuscated


def codebert_predict(engine, texts: list[str], batch_size: int) -> np.ndarray:
    outputs = []
    device_type = "cuda" if "cuda" in engine.device.type else "cpu"
    dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    for start in range(0, len(texts), batch_size):
        batch = [text.lower().strip() for text in texts[start:start + batch_size]]
        encoded = engine.tokenizer(
            batch, truncation=True, padding="max_length", max_length=engine.max_length,
            return_tensors="pt",
        ).to(engine.device)
        with torch.no_grad(), autocast(device_type=device_type, dtype=dtype):
            logits = engine.t1(encoded["input_ids"], encoded["attention_mask"])["verdict_logits"]
            probabilities = torch.softmax(logits.float(), dim=1)
        outputs.extend(probabilities.cpu().numpy())
        print(f"CodeBERT export: {min(start + len(batch), len(texts))}/{len(texts)}", flush=True)
    return np.asarray(outputs, dtype=np.float64)


def measure_gate_latency(engine, tfidf, texts: list[str], sample_size: int = 300) -> dict:
    sample = [text.lower().strip() for text in texts[:sample_size]]
    device_type = "cuda" if "cuda" in engine.device.type else "cpu"
    dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

    def codebert_once(text: str) -> None:
        encoded = engine.tokenizer(
            text, truncation=True, padding="max_length", max_length=engine.max_length,
            return_tensors="pt",
        ).to(engine.device)
        with torch.no_grad(), autocast(device_type=device_type, dtype=dtype):
            engine.t1(encoded["input_ids"], encoded["attention_mask"])["verdict_logits"]
        if device_type == "cuda":
            torch.cuda.synchronize(engine.device)

    def tfidf_once(text: str) -> None:
        tfidf.predict_proba([text])

    for text in sample[:10]:
        codebert_once(text)
        tfidf_once(text)
    codebert_times = []
    tfidf_times = []
    for text in sample:
        started = time.perf_counter()
        codebert_once(text)
        codebert_times.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        tfidf_once(text)
        tfidf_times.append((time.perf_counter() - started) * 1000)

    def summarize(values: list[float]) -> dict:
        return {"n": len(values), "median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, 0.95))}

    return {
        "scope": "Single-command gate only; includes tokenization for CodeBERT and vectorization for TF-IDF; excludes behavior, MITRE, and HTTP overhead.",
        "codebert_device": engine.device.type,
        "codebert": summarize(codebert_times),
        "tfidf_cpu": summarize(tfidf_times),
    }


def metric_bundle(probabilities: np.ndarray, targets: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    result = probability_metrics(probabilities, targets)
    result["accuracy"] = float(accuracy_score(targets, predictions))
    result["macro_f1"] = float(f1_score(targets, predictions, labels=[0, 1, 2], average="macro", zero_division=0))
    result["per_class_f1"] = {
        label: float(f1_score(targets, predictions, labels=[index], average="macro", zero_division=0))
        for index, label in enumerate(LABELS)
    }
    return result


def paired_group_bootstrap(
    codebert_probabilities: np.ndarray,
    tfidf_probabilities: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    iterations: int,
    seed: int,
) -> dict:
    codebert_predictions = codebert_probabilities.argmax(axis=1)
    tfidf_predictions = tfidf_probabilities.argmax(axis=1)
    group_rows: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_rows.setdefault(group, []).append(index)
    group_names = list(group_rows)
    rng = np.random.default_rng(seed)
    accuracy_deltas = []
    macro_f1_deltas = []
    for sampled in rng.integers(0, len(group_names), size=(iterations, len(group_names))):
        indices = np.asarray([row for selected in sampled for row in group_rows[group_names[selected]]], dtype=np.int64)
        truth = targets[indices]
        cb_pred = codebert_predictions[indices]
        tf_pred = tfidf_predictions[indices]
        accuracy_deltas.append(float(accuracy_score(truth, tf_pred) - accuracy_score(truth, cb_pred)))
        macro_f1_deltas.append(float(
            f1_score(truth, tf_pred, labels=[0, 1, 2], average="macro", zero_division=0)
            - f1_score(truth, cb_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        ))
    return {
        "groups": len(group_names),
        "iterations": iterations,
        "seed": seed,
        "tfidf_minus_codebert_accuracy": float(accuracy_score(targets, tfidf_predictions) - accuracy_score(targets, codebert_predictions)),
        "accuracy_delta_group_bootstrap_95ci": np.quantile(accuracy_deltas, [0.025, 0.975]).tolist(),
        "tfidf_minus_codebert_macro_f1": float(
            f1_score(targets, tfidf_predictions, labels=[0, 1, 2], average="macro", zero_division=0)
            - f1_score(targets, codebert_predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "macro_f1_delta_group_bootstrap_95ci": np.quantile(macro_f1_deltas, [0.025, 0.975]).tolist(),
    }


def evaluate_slice(
    name: str,
    mask: np.ndarray,
    codebert: np.ndarray,
    tfidf: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    iterations: int,
    seed: int,
) -> dict:
    if not np.any(mask):
        return {"n": 0, "note": "No examples in this slice"}
    indices = np.flatnonzero(mask)
    return {
        "n": len(indices),
        "codebert": metric_bundle(codebert[indices], targets[indices]),
        "tfidf": metric_bundle(tfidf[indices], targets[indices]),
        "paired_difference": paired_group_bootstrap(
            codebert[indices], tfidf[indices], targets[indices],
            [groups[index] for index in indices], iterations, seed,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/derived/scientific_v2/gatekeeper")
    parser.add_argument("--train-patch", type=Path, default=ROOT / "data/training/genos_dataset/gatekeeper_benign_core_patch_v2a.jsonl")
    parser.add_argument("--codebert-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tfidf-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite comparison report: {args.output}")
    if args.batch_size < 1 or args.bootstrap_iterations < 1:
        parser.error("Batch size and bootstrap iterations must be positive")

    paths = {split: args.data_dir / f"gatekeeper_3class_{split}.csv" for split in ("train", "val", "test")}
    rows = {split: read_rows(path) for split, path in paths.items()}
    if args.train_patch:
        rows["train"].extend(read_rows(args.train_patch))
    audit = require_disjoint(rows)
    if not audit.get("passed"):
        raise ValueError("Data disjointness check failed: " + json.dumps(audit))

    checkpoint_dir = args.codebert_checkpoint_dir.resolve()
    codebert_metadata = json.loads((checkpoint_dir / "gatekeeper_meta.json").read_text(encoding="utf-8"))
    for split, path in paths.items():
        expected = codebert_metadata["dataset_manifest"][split]["sha256"]
        if expected != sha256_file(path):
            raise ValueError(f"{split} data hash differs from CodeBERT training metadata")
    if args.train_patch and codebert_metadata.get("train_patch_jsonl"):
        if sha256_file(args.train_patch) != sha256_file(codebert_metadata["train_patch_jsonl"]):
            raise ValueError("Training patch differs from CodeBERT run")

    tfidf_path = args.tfidf_model.resolve()
    tfidf = joblib.load(tfidf_path)
    tfidf_metadata = json.loads(tfidf_path.with_suffix(".json").read_text(encoding="utf-8"))
    if tfidf_metadata.get("checkpoint_sha256") != sha256_file(tfidf_path):
        raise ValueError("TF-IDF checkpoint hash differs from its metadata")
    if list(tfidf_metadata.get("class_names", [])) != LABELS:
        raise ValueError("TF-IDF class order differs from the CodeBERT class order")

    from genos.engine import GenosEngine
    engine = GenosEngine(
        t1_path=str(checkpoint_dir / "gatekeeper.pt"),
        gatekeeper_meta_path=str(checkpoint_dir / "gatekeeper_meta.json"),
        gatekeeper_backend="codebert",
        view_policy="mean",
    )
    commands = [command_of(row) for row in rows["test"]]
    decoded = []
    obfuscated = []
    for command in commands:
        value, flagged = decode_for_runtime(engine, command)
        decoded.append(value)
        obfuscated.append(flagged)
    print(f"Decoded-view preparation complete; obfuscated={sum(obfuscated)}/{len(commands)}", flush=True)

    raw_codebert = codebert_predict(engine, commands, args.batch_size)
    decoded_codebert = codebert_predict(engine, decoded, args.batch_size)
    raw_tfidf = tfidf.predict_proba([command.lower().strip() for command in commands])
    decoded_tfidf = tfidf.predict_proba([command.lower().strip() for command in decoded])
    if not np.array_equal(np.asarray(tfidf.classes_, dtype=int), np.arange(len(LABELS))):
        raise ValueError("TF-IDF estimator classes are not in canonical runtime order")

    changed = np.asarray([raw.strip().lower() != view.strip().lower() for raw, view in zip(commands, decoded)])
    obfuscated = np.asarray(obfuscated, dtype=bool)
    mean_codebert = np.where(changed[:, None], (raw_codebert + decoded_codebert) / 2, decoded_codebert)
    mean_tfidf = np.where(changed[:, None], (raw_tfidf + decoded_tfidf) / 2, decoded_tfidf)
    targets = np.asarray([LABELS.index(row["label"]) for row in rows["test"]], dtype=np.int64)
    groups = [str(row.get("holdout_group") or row.get("family_id") or row.get("source_group") or "unknown") for row in rows["test"]]

    masks = {
        "all_raw_view": np.ones(len(commands), dtype=bool),
        "all_runtime_mean_view": np.ones(len(commands), dtype=bool),
        "obfuscated_raw_view": obfuscated,
        "obfuscated_runtime_mean_view": obfuscated,
        "not_obfuscated_raw_view": ~obfuscated,
    }
    models = {
        "all_raw_view": (raw_codebert, raw_tfidf),
        "all_runtime_mean_view": (mean_codebert, mean_tfidf),
        "obfuscated_raw_view": (raw_codebert, raw_tfidf),
        "obfuscated_runtime_mean_view": (mean_codebert, mean_tfidf),
        "not_obfuscated_raw_view": (raw_codebert, raw_tfidf),
    }
    comparisons = {}
    for index, (name, mask) in enumerate(masks.items()):
        comparisons[name] = evaluate_slice(
            name, mask, *models[name], targets, groups,
            args.bootstrap_iterations, args.seed + index,
        )

    codebert_raw_metrics = comparisons["all_raw_view"]["codebert"]
    recorded_test = codebert_metadata["test_metrics"]
    recorded_delta = abs(codebert_raw_metrics["macro_f1"] - recorded_test["macro_f1"])
    if recorded_delta > 0.01:
        raise ValueError(f"Direct CodeBERT test export differs from trainer metadata by {recorded_delta:.4f} macro-F1")

    report = {
        "scope": "Same frozen grouped test split, same 250-row benign training patch, raw lowercased command representation; weak labels.",
        "view_policies": {
            "raw_view": "Score the lowercased original command only; comparable to trainer test metrics.",
            "runtime_mean_view": "Score decoded text and average with raw scores only when runtime deobfuscation changes the command.",
        },
        "obfuscation_definition": "Genos runtime is_obfuscated detector; subset has no independent obfuscation annotations.",
        "codebert_checkpoint": str((checkpoint_dir / "gatekeeper.pt").relative_to(ROOT)),
        "codebert_seed": codebert_metadata.get("seed"),
        "tfidf_checkpoint": str(tfidf_path.relative_to(ROOT)) if tfidf_path.is_relative_to(ROOT) else str(tfidf_path),
        "codebert_inference_device": engine.device.type,
        "dataset_manifest": dataset_manifest({"test": paths["test"]}),
        "training_patch_sha256": sha256_file(args.train_patch) if args.train_patch else None,
        "independence_audit": audit,
        "codebert_trainer_test_metrics": recorded_test,
        "codebert_export_macro_f1_abs_delta_from_trainer": recorded_delta,
        "obfuscated_count": int(obfuscated.sum()),
        "deobfuscated_changed_count": int(changed.sum()),
        "single_command_gate_latency": measure_gate_latency(engine, tfidf, commands),
        "comparisons": comparisons,
        "limitations": [
            "Labels are legacy weak supervision, not independent human ground truth.",
            "Obfuscation subset is detector-defined and the raw/mean comparisons are component-only gatekeeper scores.",
            "Holdout groups are inferred/provided provenance groups; undetected template/source links may remain.",
            "TF-IDF gate latency excludes the separate CodeBERT behavior encoder and MITRE specialist in the full API.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    prediction_path = args.output.with_name(args.output.stem + "_predictions.jsonl")
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, flagged, changed_view, cb_raw, tf_raw, cb_mean, tf_mean in zip(
            rows["test"], obfuscated, changed, raw_codebert, raw_tfidf, mean_codebert, mean_tfidf,
        ):
            handle.write(json.dumps({
                "command": command_of(row),
                "label": row["label"],
                "target": LABELS.index(row["label"]),
                "family_id": row.get("family_id"),
                "holdout_group": row.get("holdout_group"),
                "is_obfuscated": bool(flagged),
                "decoded_changed": bool(changed_view),
                "codebert_raw_probabilities": cb_raw.tolist(),
                "tfidf_raw_probabilities": tf_raw.tolist(),
                "codebert_runtime_mean_probabilities": cb_mean.tolist(),
                "tfidf_runtime_mean_probabilities": tf_mean.tolist(),
            }, ensure_ascii=True) + "\n")
    print(json.dumps({
        "report": str(args.output),
        "obfuscated_count": int(obfuscated.sum()),
        "raw_macro_f1": {
            "codebert": comparisons["all_raw_view"]["codebert"]["macro_f1"],
            "tfidf": comparisons["all_raw_view"]["tfidf"]["macro_f1"],
        },
        "obfuscated_raw_macro_f1": {
            "codebert": comparisons["obfuscated_raw_view"].get("codebert", {}).get("macro_f1"),
            "tfidf": comparisons["obfuscated_raw_view"].get("tfidf", {}).get("macro_f1"),
        },
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
