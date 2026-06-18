#!/usr/bin/env python3
"""Evaluate the active Tier 1 checkpoint on a real benign provenance holdout."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs" / "real_benign_holdout"
LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE_DIR))

from scripts.benchmark.tier1_case_sets import build_catastrophic_benign_cases
from scripts.benchmark.tier1_stress_test import compute_metrics, init_engine, runtime_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-jsonl",
        default=str(BASE_DIR / "data" / "training" / "genos_dataset" / "gatekeeper_real_benign_v1_holdout.jsonl"),
        help="Path to the real benign provenance holdout JSONL.",
    )
    parser.add_argument(
        "--prefix",
        default="gatekeeper_real_benign_v1",
        help="Output filename prefix under logs/real_benign_holdout.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            command = str(row.get("command", "")).strip()
            if not command:
                raise ValueError(f"Missing command in {path} line {line_no}")
            rows.append(row)
    return rows


def build_error_audit(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["true_label"] == row["predicted"]:
            continue
        grouped[f"{row['true_label']}->{row['predicted']}"] .append(row)
    return dict(grouped)


def main() -> None:
    args = parse_args()
    holdout_path = Path(args.holdout_jsonl)
    rows = load_jsonl(holdout_path)
    engine = init_engine()
    catastrophic_commands = {row["command"] for row in build_catastrophic_benign_cases()}

    y_true: list[str] = []
    y_pred: list[str] = []
    audit_rows: list[dict[str, object]] = []
    routing_counts: Counter[str] = Counter()
    public_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    error_source_type_counts: Counter[str] = Counter()
    catastrophic_failures: list[dict[str, object]] = []

    for row in tqdm(rows, total=len(rows), desc="Real benign holdout", unit="case", dynamic_ncols=True, mininterval=0.1, smoothing=0.0):
        runtime = runtime_predict(engine, str(row["command"]))
        predicted = runtime["internal_label"]
        source_type = str(row.get("source_type", "unknown"))
        y_true.append("Benign")
        y_pred.append(predicted)
        routing_counts[predicted] += 1
        public_counts[runtime["public_label"]] += 1
        source_type_counts[source_type] += 1
        audit_row = {
            "command": row["command"],
            "true_label": "Benign",
            "predicted": predicted,
            "public_label": runtime["public_label"],
            "confidence": runtime["label_confidence"],
            "runtime_probs": runtime["label_probabilities"],
            "source_type": source_type,
            "label_basis": row.get("label_basis", ""),
            "provenance_source": row.get("provenance_source", ""),
            "source_uri": row.get("source_uri", ""),
            "holdout_group": row.get("holdout_group", ""),
            "source_name": row.get("source_name", ""),
        }
        audit_rows.append(audit_row)
        if predicted != "Benign":
            error_source_type_counts[source_type] += 1
        if str(row["command"]) in catastrophic_commands and predicted == "Malicious":
            catastrophic_failures.append(audit_row)

    metrics = compute_metrics(y_true, y_pred)
    benign_to_malicious = sum(1 for label in y_pred if label == "Malicious")
    non_benign = sum(1 for label in y_pred if label != "Benign")

    summary = {
        "holdout_path": str(holdout_path),
        "active_model_path": str(BASE_DIR / "models" / "gatekeeper.pt"),
        "active_meta_path": str(BASE_DIR / "config" / "gatekeeper_meta.json"),
        "n": len(rows),
        "metrics": metrics,
        "benign_to_malicious": benign_to_malicious,
        "non_benign_count": non_benign,
        "non_benign_rate": non_benign / max(1, len(rows)),
        "routing_counts": dict(routing_counts),
        "public_label_counts": dict(public_counts),
        "source_type_counts": dict(source_type_counts),
        "error_source_type_counts": dict(error_source_type_counts),
        "catastrophic_overlap": {
            "cases_in_holdout": sum(1 for row in rows if str(row["command"]) in catastrophic_commands),
            "malicious_count": len(catastrophic_failures),
            "failures": catastrophic_failures[:50],
        },
        "error_audit_path": str(LOG_DIR / f"{args.prefix}_errors.json"),
    }

    summary_path = LOG_DIR / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (LOG_DIR / f"{args.prefix}_errors.json").write_text(json.dumps(build_error_audit(audit_rows), indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[+] Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()