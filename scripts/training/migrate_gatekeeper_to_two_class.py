#!/usr/bin/env python3
"""Collapse the gatekeeper's direct-abuse class into Context_Dependent."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collapse(label: str) -> str:
    label = str(label).strip()
    if label == "Benign":
        return label
    if label in {"Malicious", "Suspicious", "Context_Dependent"}:
        return "Context_Dependent"
    raise ValueError(f"Unsupported gatekeeper label: {label!r}")


def migrate_csv(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0]) if rows else []
    for row in rows:
        row["label"] = collapse(row["label"])
        if "original_label" in row and row["original_label"] in {"Benign", "Malicious", "Suspicious", "Context_Dependent"}:
            row["original_label"] = collapse(row["original_label"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def migrate_jsonl(path: Path) -> None:
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["label"] = collapse(row["label"])
        row["label_schema"] = "benign_context_v1"
        output.append(json.dumps(row, ensure_ascii=True))
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/derived/scientific_v2/gatekeeper"))
    parser.add_argument("--patch", type=Path, default=Path("data/training/genos_dataset/gatekeeper_benign_core_patch_v2a.jsonl"))
    args = parser.parse_args()
    for split in ("train", "val", "test"):
        migrate_csv(args.data_dir / f"gatekeeper_3class_{split}.csv")
    migrate_jsonl(args.patch)
    print(json.dumps({"schema": "benign_context_v1", "classes": ["Benign", "Context_Dependent"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
