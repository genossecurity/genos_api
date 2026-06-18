#!/usr/bin/env python3
"""Focused Tier 1 sanity benchmark for benign-first correction work.

Runs four 100-command buckets against the active runtime checkpoint:
  - trivial_benign
  - routine_admin
  - context_needed_dual_use
  - direct_abuse

Primary acceptance gate:
  - trivial benign recall > 95%
  - trivial benign -> malicious = 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs" / "tier1_sanity"
LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE_DIR))

from scripts.benchmark.tier1_case_sets import build_catastrophic_benign_cases, build_tier1_sanity_buckets
from scripts.benchmark.tier1_stress_test import compute_metrics, init_engine, runtime_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="c_current", help="Output filename prefix under logs/tier1_sanity")
    return parser.parse_args()


def evaluate_bucket(engine, rows: list[dict[str, str]], desc: str) -> tuple[dict, list[dict[str, object]]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    audit_rows: list[dict[str, object]] = []
    for row in tqdm(rows, total=len(rows), desc=desc, unit="case", dynamic_ncols=True, mininterval=0.1, smoothing=0.0):
        runtime = runtime_predict(engine, row["command"])
        predicted = runtime["internal_label"]
        y_true.append(row["label"])
        y_pred.append(predicted)
        audit_rows.append(
            {
                "bucket": row["bucket"],
                "command": row["command"],
                "true_label": row["label"],
                "predicted": predicted,
                "public_label": runtime["public_label"],
                "confidence": runtime["label_confidence"],
                "runtime_probs": runtime["label_probabilities"],
            }
        )
    metrics = compute_metrics(y_true, y_pred)
    routing_counts = Counter(y_pred)
    return {
        "n": len(rows),
        "metrics": metrics,
        "routing_counts": dict(routing_counts),
    }, audit_rows


def build_error_audit(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["true_label"] == row["predicted"]:
            continue
        error_type = f"{row['true_label']}->{row['predicted']}"
        grouped.setdefault(error_type, []).append(row)
    return grouped


def focused_audit(engine) -> dict[str, dict[str, object]]:
    commands = [
        "pwd",
        "cal",
        "lsmem",
        "docker ps",
        "kubectl get pods",
        "stat /etc/nginx",
        "find /var/log -maxdepth 2 -type f",
    ]
    results: dict[str, dict[str, object]] = {}
    for command in tqdm(commands, total=len(commands), desc="Focused audit", unit="case", dynamic_ncols=True, mininterval=0.1, smoothing=0.0):
        runtime = runtime_predict(engine, command)
        results[command] = {
            "internal_label": runtime["internal_label"],
            "public_label": runtime["public_label"],
            "confidence": runtime["label_confidence"],
            "runtime_probs": runtime["label_probabilities"],
        }
    return results


def catastrophic_benign_gate(engine) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    cases = build_catastrophic_benign_cases()
    for row in tqdm(cases, total=len(cases), desc="Catastrophic benign gate", unit="case", dynamic_ncols=True, mininterval=0.1, smoothing=0.0):
        runtime = runtime_predict(engine, row["command"])
        predicted = runtime["internal_label"]
        if predicted == "Malicious":
            failures.append(
                {
                    "command": row["command"],
                    "predicted": predicted,
                    "public_label": runtime["public_label"],
                    "confidence": runtime["label_confidence"],
                    "runtime_probs": runtime["label_probabilities"],
                }
            )
    return {
        "n": len(cases),
        "target": 0,
        "malicious_count": len(failures),
        "passed": len(failures) == 0,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    print("[1/4] Loading engine and sanity buckets...", flush=True)
    engine = init_engine()
    buckets = build_tier1_sanity_buckets()

    summary = {
        "active_model_path": str(BASE_DIR / "models" / "gatekeeper.pt"),
        "active_meta_path": str(BASE_DIR / "config" / "gatekeeper_meta.json"),
        "bucket_sizes": {name: len(rows) for name, rows in buckets.items()},
        "buckets": {},
    }
    all_audit_rows: dict[str, list[dict[str, object]]] = {}

    for index, bucket_name in enumerate(("trivial_benign", "routine_admin", "context_needed_dual_use", "direct_abuse"), start=2):
        print(f"[{index}/4] Evaluating {bucket_name}...", flush=True)
        bucket_summary, audit_rows = evaluate_bucket(engine, buckets[bucket_name], desc=bucket_name)
        summary["buckets"][bucket_name] = bucket_summary
        all_audit_rows[bucket_name] = audit_rows

    trivial_rows = all_audit_rows["trivial_benign"]
    benign_to_malicious = sum(1 for row in trivial_rows if row["predicted"] == "Malicious")
    trivial_recall = summary["buckets"]["trivial_benign"]["metrics"]["per_class"]["Benign"]["recall"]
    catastrophic_gate = catastrophic_benign_gate(engine)
    summary["acceptance_gate"] = {
        "catastrophic_benign": catastrophic_gate,
        "trivial_benign_recall": trivial_recall,
        "trivial_benign_recall_target": 0.95,
        "trivial_benign_to_malicious": benign_to_malicious,
        "trivial_benign_to_malicious_target": 0,
        "passed": catastrophic_gate["passed"] and trivial_recall > 0.95 and benign_to_malicious == 0,
    }
    summary["focused_audit"] = focused_audit(engine)

    summary_path = LOG_DIR / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for bucket_name, audit_rows in all_audit_rows.items():
        (LOG_DIR / f"{args.prefix}_{bucket_name}_errors.json").write_text(
            json.dumps(build_error_audit(audit_rows), indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2))
    print(f"[+] Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()