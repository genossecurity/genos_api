#!/usr/bin/env python3
"""Apply explicit family-label corrections to train/validation, preserving test."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scientific_validation import read_rows, require_disjoint, sha256_file


def read_corrections(paths: list[Path], labels: set[str]) -> dict[tuple[str, str], dict]:
    corrections = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for line, row in enumerate(csv.DictReader(handle), start=2):
                raw = (row.get("corrected_family_labels") or "").strip()
                if not raw:
                    continue
                split = (row.get("split") or "").strip().lower()
                split = "val" if split == "validation" else split
                if split not in {"train", "val"}:
                    raise ValueError(f"{path}:{line}: corrections are allowed only for train and validation")
                try:
                    selected = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line}: corrected labels must be a JSON array") from error
                if not isinstance(selected, list) or not selected or any(not isinstance(item, str) for item in selected):
                    raise ValueError(f"{path}:{line}: corrected labels must be a nonempty JSON string array")
                if len(selected) != len(set(selected)) or set(selected) - labels:
                    raise ValueError(f"{path}:{line}: duplicate or unknown corrected family")
                if "Benign Admin" in selected and len(selected) > 1:
                    raise ValueError(f"{path}:{line}: Benign Admin cannot coexist with attack families")
                command = row.get("command") or ""
                template_id = row.get("template_id") or ""
                if not command or not template_id:
                    raise ValueError(f"{path}:{line}: command and template_id are required")
                key = (split, command)
                if key in corrections:
                    raise ValueError(f"{path}:{line}: duplicate correction for {key}")
                corrections[key] = {
                    "labels": selected,
                    "template_id": template_id,
                    "weak_family_labels": row.get("weak_family_labels") or "",
                    "review_id": row.get("review_id") or "",
                    "review_notes": row.get("review_notes") or "",
                    "file": str(path),
                    "line": line,
                }
    if not corrections:
        raise ValueError("No completed corrections were supplied; refusing an unchanged retrain")
    return corrections


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/derived/family_specialist_v1")
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.output_dir}")

    source_manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    labels = list(manifest["family_labels"])
    if len(labels) != 11 or len(set(labels)) != 11:
        raise ValueError("Expected exactly 11 distinct family labels")
    corrections = read_corrections(args.reviews, set(labels))
    source_paths = {split: args.data_dir / f"family_{split}.jsonl" for split in ("train", "val", "test")}
    rows = {split: read_rows(path) for split, path in source_paths.items()}
    by_key = {(split, row["command"]): row for split in ("train", "val") for row in rows[split]}
    if len(by_key) != len(rows["train"]) + len(rows["val"]):
        raise ValueError("Duplicate command within a development split")
    unknown = set(corrections) - set(by_key)
    if unknown:
        raise ValueError(f"Corrections not found in train/validation: {sorted(unknown)[:5]}")

    changed = Counter()
    audit = []
    for key, correction in corrections.items():
        row = by_key[key]
        if row["template_id"] != correction["template_id"]:
            raise ValueError(f"Template mismatch for {key}")
        if correction["weak_family_labels"]:
            try:
                original_labels = json.loads(correction["weak_family_labels"])
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid weak_family_labels for {key}") from error
            if not isinstance(original_labels, list) or set(original_labels) != set(row["family_labels"]):
                raise ValueError(f"Original family labels changed since review for {key}")
        before = row["family_labels"]
        after = sorted(correction["labels"], key=labels.index)
        if before == after:
            continue
        row["family_labels"] = after
        row["label_bases"] = sorted(set(row.get("label_bases", [])) | {"explicit_family_correction_unverified"})
        changed[key[0]] += 1
        audit.append({
            "split": key[0], "command": key[1], "template_id": row["template_id"],
            "before": before, "after": after, "review_id": correction["review_id"],
            "review_notes": correction["review_notes"], "review_file": correction["file"],
            "review_line": correction["line"],
        })
    if not audit:
        raise ValueError("All supplied labels match their source rows; refusing an unchanged retrain")
    independence = require_disjoint({
        split: [dict(row, holdout_group=row["template_id"]) for row in split_rows]
        for split, split_rows in rows.items()
    })
    if not independence["passed"]:
        raise ValueError(f"Split independence failed: {independence}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, source in source_paths.items():
        target = args.output_dir / source.name
        if split == "test":
            shutil.copyfile(source, target)
        else:
            write_jsonl(target, rows[split])
    with (args.output_dir / "correction_audit.jsonl").open("w", encoding="utf-8") as handle:
        for record in audit:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    manifest.update({
        "parent_manifest_sha256": sha256_file(source_manifest_path),
        "review_files": [{"path": str(path), "sha256": sha256_file(path)} for path in args.reviews],
        "label_basis": "Original weak labels plus explicit command-text corrections on development rows; correction provenance is in the audit; test labels remain weak and frozen",
        "corrections_by_split": dict(changed),
        "correction_audit_sha256": sha256_file(args.output_dir / "correction_audit.jsonl"),
        "independence_audit": independence,
        "label_counts_by_split": {
            split: dict(Counter(label for row in split_rows for label in row["family_labels"]))
            for split, split_rows in rows.items()
        },
        "multi_label_rows_by_split": {
            split: sum(len(row["family_labels"]) > 1 for row in split_rows)
            for split, split_rows in rows.items()
        },
        "outputs": {
            split: {"path": str(args.output_dir / source.name), "sha256": sha256_file(args.output_dir / source.name), "rows": len(rows[split])}
            for split, source in source_paths.items()
        },
    })
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "corrections_by_split": dict(changed),
                      "test_sha256_unchanged": sha256_file(source_paths["test"]) == sha256_file(args.output_dir / "family_test.jsonl")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
