#!/usr/bin/env python3
"""Audit existing data, quarantine contradictions, and prepare future grouped runs.

The new splits cannot validate a checkpoint trained on the old splits. Generated
family groups are conservative tool groups, not independently verified templates.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from genos.scientific_validation import command_of, dataset_manifest, normalize_command, read_rows, require_disjoint, split_audit

def inferred_family(command):
    # A deliberately coarse split axis; never claim this proves template isolation.
    tokens = re.findall(r"[^\s\"']+", normalize_command(command))
    if not tokens:
        raise ValueError("Empty command")
    first = tokens[0].replace("\\", "/").rsplit("/", 1)[-1]
    return "tool:" + first


def prepare(splits, excluded, seed=42):
    by_command = defaultdict(list)
    for split, rows in splits.items():
        for row in rows:
            copied = dict(row, previous_split=split)
            by_command[normalize_command(command_of(row))].append(copied)
    retained, quarantined = [], []
    for command, rows in sorted(by_command.items()):
        labels = {str(r.get("label", r.get("stage_label"))) for r in rows}
        actions = {tuple(sorted(r.get("action_tags", []))) for r in rows}
        reason = "conflicting_labels" if len(labels) > 1 or len(actions) > 1 else "benchmark_exposure" if command in excluded else None
        if reason:
            quarantined.extend(dict(row, quarantine_reason=reason) for row in rows)
            continue
        row = rows[0]
        provided_family = bool(row.get("family_id"))
        row["family_id"] = row.get("family_id") or inferred_family(command_of(row))
        row["family_basis"] = "provided" if provided_family else "inferred_first_executable"
        # Preserve all grouping constraints from duplicates, not just the first row.
        row["_group_keys"] = sorted({(field, str(r[field])) for r in rows for field in ("family_id", "source_group", "holdout_group") if r.get(field)})
        row["_group_keys"].append(("family_id", row["family_id"]))
        row["label_basis"] = row.get("label_basis") or "legacy_weak_supervision_unreviewed"
        retained.append(row)

    parent = list(range(len(retained)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(retained):
        for key in row["_group_keys"]:
            if key in owners:
                parent[root(i)] = root(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(retained):
        groups[root(i)].append(row)
    grouped = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)
    result = {name: [] for name in ("train", "val", "test")}
    targets = {"train": len(retained) * .8, "val": len(retained) * .1, "test": len(retained) * .1}
    for rows in grouped:
        name = max(result, key=lambda key: targets[key] - len(result[key]))
        group_id = hashlib.sha256(json.dumps(sorted({tuple(k) for r in rows for k in r['_group_keys']})).encode()).hexdigest()[:20]
        for row in rows:
            row.pop("_group_keys")
            row["holdout_group"] = group_id
        result[name].extend(rows)
    require_disjoint(result)
    return result, quarantined


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["gatekeeper", "mitre", "behavior"], default="gatekeeper")
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--benchmark-jsonl", type=Path, help="Development cases to exclude from all new splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--annotation-count", type=int, default=300)
    args = p.parse_args()
    defaults = {"gatekeeper": "genos_dataset", "mitre": "genos_residual_expanded", "behavior": "genos_behavior"}
    input_dir = args.input_dir or ROOT / "data/training" / defaults[args.kind]
    pattern = {"gatekeeper": "gatekeeper_3class_{}.csv", "mitre": "specialist_{}_variant_a.jsonl", "behavior": "behavior_{}.jsonl"}[args.kind]
    paths = {name: input_dir / pattern.format(name) for name in ("train", "val", "test")}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty; frozen artifacts must not be overwritten")
    splits = {name: read_rows(path) for name, path in paths.items()}
    benchmarks = defaultdict(list)
    if args.benchmark_jsonl:
        for row in read_rows(args.benchmark_jsonl):
            benchmarks[row.get("benchmark", "development")].append(row)
    excluded = {normalize_command(command_of(r)) for rows in benchmarks.values() for r in rows}
    report = {"benchmark_exclusions": dataset_manifest({"benchmarks": args.benchmark_jsonl}) if args.benchmark_jsonl else {}, "kind": args.kind, "seed": args.seed, "inputs": dataset_manifest(paths), "before": split_audit(splits)}
    if args.kind == "gatekeeper":
        patch_path = ROOT / "data/training/genos_dataset/gatekeeper_benign_core_patch_v2a.jsonl"
        patch = read_rows(patch_path)
        patch_commands = {normalize_command(command_of(r)) for r in patch}
        excluded |= patch_commands
        report["benchmark_patch"] = dataset_manifest({"patch": patch_path})
        report["benchmark_patch_overlaps"] = {name: {"n": len(rows), "overlap": sum(normalize_command(command_of(r)) in patch_commands for r in rows)} for name, rows in benchmarks.items()}
    clean, quarantine = prepare(splits, excluded, args.seed)
    report["after"] = require_disjoint(clean)
    report["quarantine"] = dict(Counter(r["quarantine_reason"] for r in quarantine))
    report["label_counts"] = {name: dict(Counter(r.get("label", r.get("stage_label")) for r in rows)) for name, rows in clean.items()}
    train_labels = set(report["label_counts"]["train"])
    report["unseen_eval_labels"] = {name: sorted(set(report["label_counts"][name]) - train_labels) for name in ("val", "test")}
    report["limitations"] = ["Requires fresh training; old checkpoints have seen rows reassigned to evaluation", "Legacy labels remain weak supervision; independent annotation pending", "Inferred tool groups do not prove template or source independence; review family/source provenance"]
    outputs = {}
    for name, rows in clean.items():
        path = args.output_dir / pattern.format(name)
        outputs[name] = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.kind == "gatekeeper":
            fields = sorted({k for row in rows for k in row})
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            write_jsonl(path, rows)
    write_jsonl(args.output_dir / "quarantine.jsonl", quarantine)
    # Reviewers receive neither weak labels nor model predictions.
    pool = list(clean["test"])
    random.Random(args.seed).shuffle(pool)
    queue = [{"id": hashlib.sha256(command_of(r).encode()).hexdigest(), "command": command_of(r), "family_id": r["family_id"], "source_context": None, "observable_behavior": None, "authorization": "unknown", "verdict": None, "mitre_codes": None, "annotator_id": None, "rationale": None} for r in pool[:args.annotation_count]]
    write_jsonl(args.output_dir / "annotation_queue.jsonl", queue)
    validation_pool = list(clean["val"])
    random.Random(args.seed).shuffle(validation_pool)
    validation_queue = [{"id": hashlib.sha256(command_of(r).encode()).hexdigest(), "command": command_of(r), "family_id": r["family_id"], "source_context": None, "observable_behavior": None, "authorization": "unknown", "verdict": None, "mitre_codes": None, "annotator_id": None, "rationale": None} for r in validation_pool[:args.annotation_count]]
    write_jsonl(args.output_dir / "annotation_validation_queue.jsonl", validation_queue)
    report["outputs"] = dataset_manifest(outputs)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("kind", "before", "after", "quarantine", "unseen_eval_labels", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
