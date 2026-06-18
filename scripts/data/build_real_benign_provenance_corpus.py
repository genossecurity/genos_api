#!/usr/bin/env python3
"""Build a real benign operational provenance corpus.

This script intentionally does not synthesize variants or benchmark-directed
repairs. It ingests observed benign command sources, preserves provenance
metadata per row, deduplicates against the current Tier 1 train/val/test sets,
and creates a held-out real-benign benchmark split that is never used for
training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.data.data_scraper import norm_key, normalize_command


DEFAULT_DEDUP_CSVS = [
    BASE_DIR / "data" / "training" / "genos_dataset" / "gatekeeper_3class_train.csv",
    BASE_DIR / "data" / "training" / "genos_dataset" / "gatekeeper_3class_val.csv",
    BASE_DIR / "data" / "training" / "genos_dataset" / "gatekeeper_3class_test.csv",
]
DEFAULT_DEDUP_JSONL_GLOB = "gatekeeper_benign_core_patch*.jsonl"
HISTORY_PREFIX_RE = re.compile(r"^\s*:\s*\d+:\d+;")
NUMBERED_HISTORY_RE = re.compile(r"^\s*\d+\*?\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(BASE_DIR / "config" / "benign_provenance_sources.template.json"),
        help="JSON manifest describing real benign provenance sources.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "data" / "training" / "genos_dataset"),
        help="Directory for output train/holdout corpora and manifest.",
    )
    parser.add_argument(
        "--output-stem",
        default="real_benign_operational_provenance_v1",
        help="Output filename stem.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of accepted rows reserved for the never-train-on real benign benchmark.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "sources" not in manifest or not isinstance(manifest["sources"], list):
        raise ValueError(f"Manifest {path} must contain a top-level 'sources' list")
    return manifest


def resolve_source_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    manifest_relative = (base_dir / candidate).resolve()
    if manifest_relative.exists():
        return manifest_relative
    repo_relative = (BASE_DIR / candidate).resolve()
    return repo_relative


def load_existing_command_keys(output_dir: Path) -> set[str]:
    existing: set[str] = set()
    for csv_path in DEFAULT_DEDUP_CSVS:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                command = normalize_command(str(row.get("command", "")))
                if command:
                    existing.add(norm_key(command))
    for jsonl_path in sorted(output_dir.glob(DEFAULT_DEDUP_JSONL_GLOB)):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                command = normalize_command(str(row.get("command", "")))
                if command:
                    existing.add(norm_key(command))
    return existing


def parse_history_line(line: str) -> str:
    stripped = line.rstrip("\n")
    stripped = HISTORY_PREFIX_RE.sub("", stripped)
    stripped = NUMBERED_HISTORY_RE.sub("", stripped)
    return stripped.strip()


def iter_text_commands(path: Path, *, history_mode: bool) -> list[str]:
    commands: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = parse_history_line(line) if history_mode else line.strip()
            if not raw or raw.startswith("#"):
                continue
            commands.append(raw)
    return commands


def build_row(
    command: str,
    *,
    label: str,
    source_type: str,
    label_basis: str,
    provenance_source: str,
    source_uri: str,
    holdout_group: str,
    source_name: str,
) -> dict[str, str]:
    normalized = normalize_command(command)
    if not normalized:
        return {}
    return {
        "command": normalized,
        "label": label,
        "source_type": source_type,
        "label_basis": label_basis,
        "provenance_source": provenance_source,
        "source_uri": source_uri,
        "holdout_group": holdout_group,
        "source_name": source_name,
    }


def load_source_rows(base_dir: Path, source_spec: dict[str, object]) -> list[dict[str, str]]:
    path_value = str(source_spec.get("path", "")).strip()
    if not path_value:
        raise ValueError("Each source entry must define 'path'")
    source_path = resolve_source_path(base_dir, path_value)
    data_format = str(source_spec.get("format", "jsonl")).strip().lower()
    source_type = str(source_spec.get("source_type", "")).strip()
    if not source_type:
        raise ValueError(f"Source {source_path} is missing required field 'source_type'")
    label_basis = str(source_spec.get("label_basis", "routine_operational_provenance")).strip()
    default_label = str(source_spec.get("default_label", "Benign")).strip() or "Benign"
    source_uri = str(source_spec.get("source_uri", "")).strip()
    source_name = str(source_spec.get("source_name", source_path.stem)).strip() or source_path.stem
    default_holdout_group = str(source_spec.get("holdout_group", source_name)).strip() or source_name
    provenance_source = str(source_spec.get("provenance_source", source_name)).strip() or source_name

    rows: list[dict[str, str]] = []
    if data_format == "jsonl":
        with source_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                entry = json.loads(payload)
                row = build_row(
                    str(entry.get("command", "")),
                    label=str(entry.get("label", default_label)).strip() or default_label,
                    source_type=str(entry.get("source_type", source_type)).strip() or source_type,
                    label_basis=str(entry.get("label_basis", label_basis)).strip() or label_basis,
                    provenance_source=str(entry.get("provenance_source", provenance_source)).strip() or provenance_source,
                    source_uri=str(entry.get("source_uri", source_uri)).strip(),
                    holdout_group=str(entry.get("holdout_group", default_holdout_group)).strip() or default_holdout_group,
                    source_name=str(entry.get("source_name", source_name)).strip() or source_name,
                )
                if row:
                    rows.append(row)
                elif str(entry.get("command", "")).strip():
                    raise ValueError(f"Unable to normalize command in {source_path} line {line_no}")
        return rows

    if data_format not in {"history", "text"}:
        raise ValueError(f"Unsupported source format: {data_format}")

    commands = iter_text_commands(source_path, history_mode=data_format == "history")
    for command in commands:
        row = build_row(
            command,
            label=default_label,
            source_type=source_type,
            label_basis=label_basis,
            provenance_source=provenance_source,
            source_uri=source_uri,
            holdout_group=default_holdout_group,
            source_name=source_name,
        )
        if row:
            rows.append(row)
    return rows


def split_rows(rows: list[dict[str, str]], holdout_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("holdout_group") or norm_key(row["command"])].append(row)

    grouped_rows = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(grouped_rows)

    total_rows = len(rows)
    holdout_target = max(1, int(round(total_rows * holdout_ratio))) if total_rows else 0
    holdout: list[dict[str, str]] = []
    train: list[dict[str, str]] = []
    holdout_group_count = 0
    for group_name, group_rows in grouped_rows:
        if len(holdout) < holdout_target:
            holdout.extend(group_rows)
            holdout_group_count += 1
        else:
            train.extend(group_rows)
    if not train and holdout:
        train, holdout = holdout[:-1], holdout[-1:]
    return train, holdout, {
        "groups_total": len(grouped_rows),
        "groups_in_holdout": holdout_group_count,
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    existing_keys = load_existing_command_keys(output_dir)
    accepted_rows: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    source_stats: list[dict[str, object]] = []
    deduped_against_existing = 0
    deduped_internal = 0

    for source_spec in manifest["sources"]:
        rows = load_source_rows(manifest_path.parent, source_spec)
        accepted_before = len(accepted_rows)
        source_seen = 0
        source_existing_dupes = 0
        source_internal_dupes = 0
        for row in rows:
            key = norm_key(row["command"])
            if key in existing_keys:
                deduped_against_existing += 1
                source_existing_dupes += 1
                continue
            if key in seen_keys:
                deduped_internal += 1
                source_internal_dupes += 1
                continue
            seen_keys.add(key)
            accepted_rows.append(row)
            source_seen += 1
        source_stats.append(
            {
                "source_name": str(source_spec.get("source_name", Path(str(source_spec.get("path", ""))).stem)),
                "path": str(source_spec.get("path", "")),
                "format": str(source_spec.get("format", "jsonl")),
                "rows_loaded": len(rows),
                "rows_accepted": len(accepted_rows) - accepted_before,
                "deduped_against_existing": source_existing_dupes,
                "deduped_internal": source_internal_dupes,
            }
        )

    train_rows, holdout_rows, split_stats = split_rows(accepted_rows, args.holdout_ratio, args.seed)

    train_path = output_dir / f"{args.output_stem}_train.jsonl"
    holdout_path = output_dir / f"{args.output_stem}_holdout.jsonl"
    build_manifest_path = output_dir / f"{args.output_stem}_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(holdout_path, holdout_rows)

    build_manifest = {
        "seed": args.seed,
        "source_manifest_path": str(manifest_path),
        "output_stem": args.output_stem,
        "holdout_ratio": args.holdout_ratio,
        "rows_total": len(accepted_rows),
        "rows_train": len(train_rows),
        "rows_holdout": len(holdout_rows),
        "deduped_against_existing": deduped_against_existing,
        "deduped_internal": deduped_internal,
        "source_type_counts": dict(Counter(row["source_type"] for row in accepted_rows)),
        "label_basis_counts": dict(Counter(row["label_basis"] for row in accepted_rows)),
        "holdout_group_counts": dict(Counter(row["holdout_group"] for row in holdout_rows)),
        "split_stats": split_stats,
        "source_stats": source_stats,
    }
    build_manifest_path.write_text(json.dumps(build_manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "train_path": str(train_path),
                "holdout_path": str(holdout_path),
                "build_manifest_path": str(build_manifest_path),
                **build_manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()