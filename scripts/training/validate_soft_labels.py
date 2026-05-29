#!/usr/bin/env python3
"""Validate Gatekeeper soft-label JSONL files and summarize annotation coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from trainer1 import normalize_soft_target, soft_target_to_auxiliary_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Soft-label JSONL files to validate")
    return parser.parse_args()


def validate_file(path: Path) -> dict:
    rows = 0
    source_types = Counter()
    label_bases = Counter()
    dominant_labels = Counter()
    non_benign_values = []
    ordinal_risk_values = []
    examples_by_basis = defaultdict(list)

    with path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            command = str(payload.get("command", "")).strip()
            if not command:
                raise ValueError(f"Missing command in {path} row {row_idx}")
            label_basis = str(payload.get("label_basis", "")).strip()
            source_type = str(payload.get("source_type", "")).strip()
            if not label_basis:
                raise ValueError(f"Missing label_basis in {path} row {row_idx}")
            if not source_type:
                raise ValueError(f"Missing source_type in {path} row {row_idx}")

            soft_target = normalize_soft_target(payload.get("soft_target", {}), row_idx=row_idx, source_path=str(path))
            non_benign, _, ordinal_risk = soft_target_to_auxiliary_targets(soft_target)

            rows += 1
            source_types[source_type] += 1
            label_bases[label_basis] += 1
            dominant_labels[["Routine_Operational", "Direct_Abuse", "Needs_Context"][max(range(3), key=lambda i: soft_target[i])]] += 1
            non_benign_values.append(non_benign)
            ordinal_risk_values.append(ordinal_risk)
            if len(examples_by_basis[label_basis]) < 3:
                examples_by_basis[label_basis].append(command)

    return {
        "path": str(path),
        "rows": rows,
        "source_types": dict(source_types),
        "label_bases": dict(label_bases),
        "dominant_evidence": dict(dominant_labels),
        "non_benign_mean": (sum(non_benign_values) / rows) if rows else 0.0,
        "ordinal_risk_mean": (sum(ordinal_risk_values) / rows) if rows else 0.0,
        "examples_by_basis": dict(examples_by_basis),
    }


def main() -> None:
    args = parse_args()
    summaries = [validate_file(Path(path)) for path in args.paths]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()