#!/usr/bin/env python3
"""Build a joint multi-label tactic-family dataset from prepared weak labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from genos.engine import GenosEngine
from genos.scientific_validation import command_of, normalize_command, read_rows, require_disjoint, sha256_file

FAMILY_LABELS = [
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Command-and-Control / Payload Retrieval",
    "Exfiltration",
    "Impact",
    "Benign Admin",
]
PHASE_TO_FAMILY = {
    "execution": "Execution",
    "persistence": "Persistence",
    "privilege-escalation": "Privilege Escalation",
    "defense-evasion": "Defense Evasion",
    "credential-access": "Credential Access",
    "discovery": "Discovery",
    "lateral-movement": "Lateral Movement",
    "command-and-control": "Command-and-Control / Payload Retrieval",
    "exfiltration": "Exfiltration",
    "impact": "Impact",
}
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def load_technique_families(path: Path) -> tuple[dict[str, set[str]], Counter]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, set[str]] = defaultdict(set)
    for item in bundle.get("objects", []):
        if item.get("type") != "attack-pattern" or item.get("x_mitre_deprecated") or item.get("x_mitre_revoked"):
            continue
        phases = {
            phase.get("phase_name")
            for phase in item.get("kill_chain_phases", [])
            if phase.get("kill_chain_name") == "mitre-attack"
        }
        families = {PHASE_TO_FAMILY[phase] for phase in phases if phase in PHASE_TO_FAMILY}
        for reference in item.get("external_references", []):
            if reference.get("source_name") == "mitre-attack" and reference.get("external_id"):
                mapping[str(reference["external_id"])].update(families)
    return dict(mapping), Counter()


def residual_template(command: str, parser_engine: GenosEngine) -> str:
    structured, _ = parser_engine._build_behavior_input(command)
    marker = "\nRESIDUAL: "
    if marker not in structured:
        return normalize_command(command)
    residual = structured.split(marker, 1)[1].split("\nFEATURES:", 1)[0]
    residual = normalize_command(residual)
    return residual or normalize_command(command)


def collect_rows(data_root: Path, cti_path: Path, benign_patch: Path | None) -> tuple[list[dict], dict]:
    technique_families, _ = load_technique_families(cti_path)
    source_paths: dict[str, Path] = {}
    source_rows: list[dict] = []
    skipped = Counter()
    omitted_phases = Counter()

    for split in ("train", "val", "test"):
        attack_path = data_root / "mitre" / f"specialist_{split}_variant_a.jsonl"
        benign_path = data_root / "gatekeeper" / f"gatekeeper_3class_{split}.csv"
        source_paths[f"attack_{split}"] = attack_path
        source_paths[f"gatekeeper_{split}"] = benign_path
        for row in read_rows(attack_path):
            technique = str(row.get("label", ""))
            families = technique_families.get(technique)
            if families is None:
                skipped["technique_missing_from_local_attack_catalog"] += 1
                continue
            if not families:
                skipped["only_excluded_attack_tactics"] += 1
                continue
            source_rows.append({
                "command": command_of(row),
                "families": set(families),
                "source_kind": "weak_attack_technique_label",
                "label_basis": row.get("label_basis", "legacy_weak_supervision_unreviewed"),
            })
        for row in read_rows(benign_path):
            if row.get("label") == "Benign":
                source_rows.append({
                    "command": command_of(row),
                    "families": {"Benign Admin"},
                    "source_kind": "weak_gatekeeper_benign_label",
                    "label_basis": row.get("label_basis", "legacy_weak_supervision_unreviewed"),
                })

    if benign_patch is not None:
        source_paths["benign_training_patch"] = benign_patch
        for row in read_rows(benign_patch):
            if row.get("label") != "Benign":
                raise ValueError("Benign patch contains non-Benign labels")
            source_rows.append({
                "command": command_of(row),
                "families": {"Benign Admin"},
                "source_kind": "curated_benign_training_patch",
                "label_basis": row.get("label_basis", "unspecified"),
            })

    by_command: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        normalized = normalize_command(row["command"])
        if not normalized:
            skipped["empty_command"] += 1
            continue
        by_command[normalized].append(row)

    merged = []
    conflict_ids = []
    for normalized, duplicates in by_command.items():
        origins = {"benign" if "Benign Admin" in row["families"] else "attack" for row in duplicates}
        if len(origins) > 1:
            conflict_ids.append(hashlib.sha256(normalized.encode()).hexdigest())
            skipped["same_command_benign_attack_conflict"] += 1
            continue
        merged.append({
            "command": duplicates[0]["command"],
            "normalized_command": normalized,
            "family_labels": sorted(set().union(*(row["families"] for row in duplicates)), key=FAMILY_LABELS.index),
            "source_kinds": sorted({row["source_kind"] for row in duplicates}),
            "label_bases": sorted({str(row["label_basis"]) for row in duplicates}),
            "source_rows_merged": len(duplicates),
        })

    parser_engine = GenosEngine.__new__(GenosEngine)
    for row in merged:
        residual = residual_template(row["command"], parser_engine)
        row["template_id"] = hashlib.sha256(residual.lower().encode()).hexdigest()

    source_manifest = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in source_paths.items()}
    audit_info = {
        "source_manifest": source_manifest,
        "source_rows": len(source_rows),
        "unique_commands_before_conflicts": len(by_command),
        "merged_commands": len(merged),
        "conflict_command_hashes": sorted(conflict_ids),
        "skipped_counts": dict(skipped),
        "attack_rows_excluded_by_tactic": int(skipped["only_excluded_attack_tactics"]),
        "attack_tactic_source_sha256": sha256_file(cti_path),
    }
    return merged, audit_info


def split_by_template(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    templates: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        templates[row["template_id"]].append(row)

    label_totals = Counter(label for row in rows for label in row["family_labels"])
    total_rows = len(rows)
    target_rows = {split: total_rows * ratio for split, ratio in SPLIT_RATIOS.items()}
    target_labels = {
        split: {label: count * SPLIT_RATIOS[split] for label, count in label_totals.items()}
        for split in SPLIT_RATIOS
    }
    current_rows = Counter()
    current_labels = {split: Counter() for split in SPLIT_RATIOS}

    randomizer = random.Random(seed)
    template_ids = list(templates)
    randomizer.shuffle(template_ids)
    template_ids.sort(key=lambda key: len(templates[key]), reverse=True)
    output = {split: [] for split in SPLIT_RATIOS}

    for template_id in template_ids:
        template_rows = templates[template_id]
        template_labels = Counter(label for row in template_rows for label in row["family_labels"])
        costs = {}
        for split in SPLIT_RATIOS:
            size_target = max(target_rows[split], 1.0)
            size_before = current_rows[split] - target_rows[split]
            size_after = current_rows[split] + len(template_rows) - target_rows[split]
            cost = (size_after * size_after - size_before * size_before) / size_target
            for label, count in template_labels.items():
                label_target = max(target_labels[split].get(label, 0.0), 1.0)
                before = current_labels[split][label] - target_labels[split].get(label, 0.0)
                after = current_labels[split][label] + count - target_labels[split].get(label, 0.0)
                cost += 2.0 * (after * after - before * before) / label_target
            costs[split] = cost
        selected = min(costs, key=costs.get)
        output[selected].extend(template_rows)
        current_rows[selected] += len(template_rows)
        current_labels[selected].update(template_labels)

    require_disjoint({
        split: [dict(row, holdout_group=row["template_id"]) for row in split_rows]
        for split, split_rows in output.items()
    })
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/derived/scientific_v2")
    parser.add_argument("--attack-catalog", type=Path, default=ROOT / "data/training/genos_cache/mitre_cti.json")
    parser.add_argument("--benign-patch", type=Path, default=ROOT / "data/training/genos_dataset/gatekeeper_benign_core_patch_v2a.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {args.output_dir}")

    rows, audit = collect_rows(args.data_root, args.attack_catalog, args.benign_patch)
    split_rows = split_by_template(rows, args.seed)
    for split in split_rows:
        split_rows[split] = [
            {key: value for key, value in row.items() if key != "normalized_command"}
            for row in split_rows[split]
        ]
    split_audit = require_disjoint({
        split: [dict(row, holdout_group=row["template_id"]) for row in values]
        for split, values in split_rows.items()
    })
    label_counts = {
        split: dict(Counter(label for row in values for label in row["family_labels"]))
        for split, values in split_rows.items()
    }
    multi_label_counts = {
        split: sum(len(row["family_labels"]) > 1 for row in values)
        for split, values in split_rows.items()
    }
    outputs = {}
    for split, values in split_rows.items():
        path = args.output_dir / f"family_{split}.jsonl"
        write_jsonl(path, values)
        outputs[split] = {"path": str(path), "sha256": sha256_file(path), "rows": len(values)}

    manifest = {
        "task": "multi_label_attack_tactic_families",
        "family_labels": FAMILY_LABELS,
        "labels_are_multi_hot": True,
        "split_seed": args.seed,
        "split_strategy": "80/10/10 greedy balance over parser residual-template groups; normalized commands/templates held together",
        "input_audit": audit,
        "independence_audit": split_audit,
        "label_counts_by_split": label_counts,
        "multi_label_rows_by_split": multi_label_counts,
        "outputs": outputs,
        "label_basis": "weak legacy technique mappings plus weak Gatekeeper Benign labels; not human-verified",
        "excluded_tactic_phases": sorted(set({
            phase.get("phase_name")
            for item in json.loads(args.attack_catalog.read_text(encoding="utf-8")).get("objects", [])
            if item.get("type") == "attack-pattern"
            for phase in item.get("kill_chain_phases", [])
            if phase.get("kill_chain_name") == "mitre-attack" and phase.get("phase_name") not in PHASE_TO_FAMILY
        })),
        "technique_ids_in_model_rows": False,
        "note": "Technique IDs are used only during source conversion and are omitted from emitted family rows and model features.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "family_labels": FAMILY_LABELS,
        "rows": {split: output["rows"] for split, output in outputs.items()},
        "label_counts_by_split": label_counts,
        "multi_label_rows_by_split": multi_label_counts,
        "skipped_counts": audit["skipped_counts"],
        "independence_passed": split_audit["passed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
