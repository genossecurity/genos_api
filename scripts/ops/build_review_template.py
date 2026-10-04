#!/usr/bin/env python3
"""Create a spreadsheet-friendly review template for blind annotation queues.

This turns a queue JSONL file into CSV rows that an annotator can fill in using
Excel, Sheets, or other spreadsheet tooling. The output preserves the command ID,
keeps the command visible, and exposes the required fields from the scientific
validation rubric.

Example:
  python3 scripts/ops/build_review_template.py \
    --queue data/derived/scientific_v2/gatekeeper/annotation_queue.jsonl \
    --output data/derived/scientific_v2/gatekeeper/annotation_queue_review_template.csv \
    --annotator-id reviewer_a
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "id",
    "command",
    "family_id",
    "source_context",
    "observable_behavior",
    "authorization",
    "verdict",
    "mitre_codes",
    "rationale",
    "annotator_id",
]


def read_queue(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_value(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True, help="Input annotation queue .jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    parser.add_argument("--annotator-id", default="", help="Optional reviewer ID to prefill for each row")
    args = parser.parse_args()

    if not args.queue.exists():
        candidates = [
            args.queue,
            args.queue.with_name("annotation_queue.jsonl"),
            args.queue.with_name("annotation_validation_queue.jsonl"),
            args.queue.with_name("annotation_test_queue.jsonl"),
            args.queue.with_name("annotation_validation_queue.csv"),
        ]
        for candidate in candidates:
            if candidate.exists():
                args.queue = candidate
                break
        else:
            raise FileNotFoundError(f"Queue file not found: {args.queue}")

    rows = list(read_queue(args.queue))
    if not rows:
        raise ValueError(f"Queue is empty: {args.queue}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            record = {field: safe_value(row.get(field)) for field in FIELDS}
            if args.annotator_id:
                record["annotator_id"] = args.annotator_id
            writer.writerow(record)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
