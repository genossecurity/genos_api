#!/usr/bin/env python3
"""Sample validation-only family-label candidates for human review."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from genos.scientific_validation import read_rows, sha256_file


FIELDS = [
    "review_id", "split", "command", "template_id", "weak_family_labels",
    "tfidf_predicted_families", "tfidf_family_scores", "codebert_predicted_families",
    "codebert_family_scores", "review_reason", "corrected_family_labels", "review_notes",
]


def load_predictions(path: Path) -> dict[str, dict]:
    rows = read_rows(path)
    return {row["command"]: row for row in rows}


def predicted(row: dict, threshold: float) -> set[str]:
    scores = row["family_probabilities"]
    return {family for family, probability in scores.items() if float(probability) >= threshold}


def choose_reason(weak: set[str], tfidf: set[str], codebert: set[str]) -> str | None:
    weak_attack = weak - {"Benign Admin"}
    tfidf_attack = tfidf - {"Benign Admin"}
    codebert_attack = codebert - {"Benign Admin"}
    if weak_attack and not tfidf_attack and not codebert_attack:
        return "both_models_miss_weak_attack_family"
    if "Benign Admin" in weak and (tfidf_attack or codebert_attack):
        return "benign_admin_vs_attack_boundary"
    if tfidf != codebert:
        return "model_disagreement"
    if len(weak) > 1:
        return "multi_label_target_review"
    return None


def build_queue(rows, tfidf_rows, codebert_rows, count: int, seed: int, threshold: float) -> list[dict]:
    by_reason: dict[str, list[dict]] = defaultdict(list)
    seen_commands = set()
    for row in rows:
        command = row["command"]
        if command in seen_commands:
            continue
        seen_commands.add(command)
        weak = set(row["family_labels"])
        tfidf = tfidf_rows.get(command)
        codebert = codebert_rows.get(command)
        if tfidf is None or codebert is None:
            continue
        tfidf_labels = predicted(tfidf, threshold)
        codebert_labels = predicted(codebert, threshold)
        reason = choose_reason(weak, tfidf_labels, codebert_labels)
        if not reason:
            continue
        by_reason[reason].append({
            "command": command,
            "template_id": row["template_id"],
            "weak_family_labels": sorted(weak),
            "tfidf_predicted_families": sorted(tfidf_labels),
            "tfidf_family_scores": tfidf["family_probabilities"],
            "codebert_predicted_families": sorted(codebert_labels),
            "codebert_family_scores": codebert["family_probabilities"],
            "review_reason": reason,
        })

    reason_order = [
        "both_models_miss_weak_attack_family",
        "benign_admin_vs_attack_boundary",
        "multi_label_target_review",
        "model_disagreement",
    ]
    randomizer = random.Random(seed)
    for candidates in by_reason.values():
        randomizer.shuffle(candidates)
    quotas = {reason: count // len(reason_order) for reason in reason_order}
    for reason in reason_order[:count % len(reason_order)]:
        quotas[reason] += 1

    selected = []
    used_templates = set()
    for reason in reason_order:
        for row in by_reason[reason]:
            if len([item for item in selected if item["review_reason"] == reason]) >= quotas[reason]:
                break
            if row["template_id"] in used_templates:
                continue
            selected.append(row)
            used_templates.add(row["template_id"])

    if len(selected) < count:
        remaining = [row for reason in reason_order for row in by_reason[reason]
                     if row["template_id"] not in used_templates]
        randomizer.shuffle(remaining)
        selected.extend(remaining[:count - len(selected)])
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} unique-template review candidates; requested {count}")

    result = []
    for index, row in enumerate(selected[:count], start=1):
        result.append({
            "review_id": f"family-val-{index:04d}",
            "split": "validation",
            **row,
            "weak_family_labels": json.dumps(row["weak_family_labels"], ensure_ascii=True),
            "tfidf_predicted_families": json.dumps(row["tfidf_predicted_families"], ensure_ascii=True),
            "tfidf_family_scores": json.dumps(row["tfidf_family_scores"], sort_keys=True, ensure_ascii=True),
            "codebert_predicted_families": json.dumps(row["codebert_predicted_families"], ensure_ascii=True),
            "codebert_family_scores": json.dumps(row["codebert_family_scores"], sort_keys=True, ensure_ascii=True),
            "corrected_family_labels": "",
            "review_notes": "",
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/derived/family_specialist_v1")
    parser.add_argument("--tfidf-predictions", type=Path, default=ROOT / "models/experiments/family_specialist_20261004_seed42/val_predictions.jsonl")
    parser.add_argument("--codebert-predictions", type=Path, default=ROOT / "models/experiments/family_codebert_20261004_seed42/val_predictions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/derived/family_specialist_v1/relabel_validation_review.csv")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite review labels: {args.output}")
    if args.count < 1 or not 0 < args.threshold < 1:
        parser.error("count must be positive and threshold must be in (0, 1)")

    dataset_rows = read_rows(args.data_dir / "family_val.jsonl")
    tfidf = load_predictions(args.tfidf_predictions)
    codebert = load_predictions(args.codebert_predictions)
    if {row["command"] for row in dataset_rows} != tfidf.keys() or tfidf.keys() != codebert.keys():
        raise ValueError("Validation commands differ across targets and model exports")
    selected = build_queue(dataset_rows, tfidf, codebert, args.count, args.seed, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({
        "output": str(args.output), "rows": len(selected),
        "review_reason_counts": dict(Counter(row["review_reason"] for row in selected)),
        "unique_templates": len({row["template_id"] for row in selected}),
        "inputs": {"dataset": sha256_file(args.data_dir / "family_val.jsonl"),
                   "tfidf_predictions": sha256_file(args.tfidf_predictions),
                   "codebert_predictions": sha256_file(args.codebert_predictions)},
        "note": "corrected_family_labels and review_notes are blank; no model prediction was converted into a correction",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
