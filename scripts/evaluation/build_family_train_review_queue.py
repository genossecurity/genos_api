#!/usr/bin/env python3
"""Create a deterministic, label-balanced training review queue without model scores."""
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
from scientific_validation import read_rows, sha256_file

FIELDS = ["review_id", "split", "command", "template_id", "weak_family_labels",
          "source_kinds", "label_bases", "corrected_family_labels", "review_notes"]


def build_queue(rows: list[dict], labels: list[str], count: int, seed: int) -> list[dict]:
    randomizer = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["template_id"]].append(row)
    candidates = [randomizer.choice(group) for group in groups.values()]
    randomizer.shuffle(candidates)
    by_label = {label: [row for row in candidates if label in row["family_labels"]] for label in labels}
    selected = []
    used = set()
    # Equal attention to rare families. Benign samples fill spare slots below.
    quota = max(1, count // len(labels))
    for label in labels:
        for row in by_label[label]:
            if sum(label in item["family_labels"] for item in selected) >= quota:
                break
            if row["template_id"] in used:
                continue
            selected.append(row)
            used.add(row["template_id"])
    for row in candidates:
        if len(selected) >= count:
            break
        if row["template_id"] not in used:
            selected.append(row)
            used.add(row["template_id"])
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} distinct training templates; requested {count}")
    return selected[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/derived/family_specialist_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/derived/family_specialist_v1/relabel_train_review.csv")
    parser.add_argument("--count", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite review labels: {args.output}")
    if args.count < 1:
        parser.error("count must be positive")
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    source = args.data_dir / "family_train.jsonl"
    selected = build_queue(read_rows(source), manifest["family_labels"], args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            writer.writerow({
                "review_id": f"family-train-{index:04d}", "split": "train",
                "command": row["command"], "template_id": row["template_id"],
                "weak_family_labels": json.dumps(row["family_labels"]),
                "source_kinds": json.dumps(row.get("source_kinds", [])),
                "label_bases": json.dumps(row.get("label_bases", [])),
                "corrected_family_labels": "", "review_notes": "",
            })
    print(json.dumps({
        "output": str(args.output), "rows": len(selected),
        "family_counts": dict(Counter(label for row in selected for label in row["family_labels"])),
        "unique_templates": len({row["template_id"] for row in selected}),
        "train_sha256": sha256_file(source),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
