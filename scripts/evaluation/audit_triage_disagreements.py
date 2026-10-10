#!/usr/bin/env python3
"""Export gate, rule, and family disagreements for command-line triage review."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_row(engine, row: dict) -> dict:
    command = row["command"]
    result = engine.scan(command, run_specialist=False, collect_iocs=False, use_baseline=False)
    family = engine.specialist.predict_family_specialist(result.get("deobfuscated_cmd") or command)
    selected = [item["family"] for item in family.get("predicted_families", [])]
    benign_family = "Benign Admin" in selected
    attack_family = any(name != "Benign Admin" for name in selected)
    gate_nonbenign = result["label"] != "Benign"
    expected = row.get("verdict") or row.get("expected_verdict")
    if expected not in {None, "Benign", "Malicious", "Context_Dependent", "Suspicious"}:
        raise ValueError(f"Unsupported review verdict: {expected!r}")
    expected_nonbenign = None if expected is None else expected != "Benign"
    reviewer_ids = row.get("reviewer_ids") or []
    provisional = bool(expected is not None and (
        "single-reviewer authorized pass" in str(row.get("rationale", "")).lower()
        or any(str(reviewer).startswith("copilot_authorized_") for reviewer in reviewer_ids)
    ))
    gate_family_conflict = gate_nonbenign and benign_family and not attack_family
    label_disagreement = expected_nonbenign is not None and gate_nonbenign != expected_nonbenign
    return {
        "command": command,
        "probe_group": row.get("probe_group"),
        "review_verdict": expected,
        "label_basis": row.get("label_basis"),
        "review_provenance": ("single_reviewer_provisional" if provisional else
                              "reported_review_unverified" if expected is not None else None),
        "source_context": row.get("source_context"),
        "gate_label": result["label"],
        "model_top_label": result["gatekeeper"]["model_top_label"],
        "gate_probabilities_pct": result["class_probabilities"],
        "routing_policy": result["routing_policy"],
        "triggered_features": result["triggered_features"],
        "selected_families": selected,
        "family_scores_pct": {item["family"]: item["probability"]
                              for item in family.get("all_family_scores", [])},
        "gate_family_conflict": gate_family_conflict,
        "label_disagreement": label_disagreement,
        "review_priority": ("provisional_label_disagreement" if label_disagreement and provisional else
                            "label_disagreement" if label_disagreement else
                            "gate_family_conflict" if gate_family_conflict else
                            "feature_override" if result["routing_policy"] == "feature_override" else None),
    }


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(Counter)
    for row in rows:
        group = row.get("probe_group") or "unassigned"
        groups[group]["total"] += 1
        groups[group]["gate_nonbenign"] += row["gate_label"] != "Benign"
        groups[group]["gate_family_conflicts"] += row["gate_family_conflict"]
        groups[group]["label_disagreements"] += row["label_disagreement"]
        groups[group]["feature_overrides"] += row["routing_policy"] == "feature_override"
    return {
        "total": len(rows),
        "labeled": sum(row["review_verdict"] is not None for row in rows),
        "single_reviewer_provisional": sum(row["review_provenance"] == "single_reviewer_provisional" for row in rows),
        "review_verdicts": dict(Counter(row["review_verdict"] for row in rows if row["review_verdict"])),
        "gate_labels": dict(Counter(row["gate_label"] for row in rows)),
        "gate_family_conflicts": sum(row["gate_family_conflict"] for row in rows),
        "label_disagreements": sum(row["label_disagreement"] for row in rows),
        "groups": {name: dict(counts) for name, counts in groups.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blind-review-queue", type=Path,
                        help="Optional model-blind, targeted error-analysis queue; never use as an untouched test set")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; existing audit results are preserved")
    if args.blind_review_queue and args.blind_review_queue.exists():
        parser.error("Choose a new blind review queue path")
    from genos.engine import GenosEngine

    engine = GenosEngine()
    rows = [audit_row(engine, row) for source in args.input for row in read_rows(source)]
    report = {
        "scope": "Disagreement discovery; predictions and unverified reviews are not ground truth",
        "inputs": [str(path) for path in args.input],
        "checkpoint_hashes": {"gatekeeper": engine.provenance["gatekeeper_sha256"],
                              "family_specialist": engine.provenance["family_specialist_sha256"]},
        "implementation_hashes": engine.provenance["implementation_sha256"],
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    queue_count = 0
    if args.blind_review_queue:
        seen = set()
        queue = []
        for row in rows:
            if not (row["gate_family_conflict"] or row["label_disagreement"]):
                continue
            normalized = row["command"].strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            queue.append({"id": hashlib.sha256(normalized.encode()).hexdigest(),
                          "command": row["command"], "source_context": row["source_context"]})
        args.blind_review_queue.parent.mkdir(parents=True, exist_ok=True)
        with args.blind_review_queue.open("w", encoding="utf-8") as handle:
            for row in queue:
                handle.write(json.dumps(row) + "\n")
        queue_count = len(queue)
    print(json.dumps({"output": str(args.output), "blind_review_queue":
                      str(args.blind_review_queue) if args.blind_review_queue else None,
                      "queue_count": queue_count, "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
