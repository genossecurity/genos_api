#!/usr/bin/env python3
"""Process behavior_augment.csv through the residual pipeline, split 80/10/10,
and merge with the existing genos_residual_expanded data.

Writes merged splits to data/training/genos_residual_merged/.
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "parser"))

from build_residual_dataset import process_row  # noqa: E402

AUGMENT_CSV   = BASE_DIR / "data/training/genos_dataset/behavior_augment.csv"
EXISTING_DIR  = BASE_DIR / "data/training/genos_residual_expanded"
OUTPUT_DIR    = BASE_DIR / "data/training/genos_residual_merged"
SEED = 42


def load_augment_csv(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            line = ",".join(row)
            idx = line.rfind(",")
            cmd, label = line[:idx].strip(), line[idx + 1:].strip()
            if cmd and label:
                rows.append((cmd, label))
    return rows


def residual_rows(raw_rows):
    out = []
    for i, (cmd, label) in enumerate(raw_rows, 1):
        print(f"  processing {i}/{len(raw_rows)}: {cmd[:60]}", end="\r", flush=True)
        rec = process_row(cmd, label)
        out.append({
            "input_text": rec["input_a"],
            "label": label,
            "rule_strength": rec["rule_strength"],
            "raw_command": cmd,
            "residual": rec["residual"],
            "features": rec["features"],
            "fired_rules": rec["fired_rules"],
        })
    print()
    return out


def load_existing(split: str) -> list:
    path = EXISTING_DIR / f"specialist_{split}_variant_a.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_split(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)

    print("[*] Loading and processing augmentation CSV...")
    raw = load_augment_csv(AUGMENT_CSV)
    print(f"    {len(raw)} commands loaded")

    # Deduplicate against existing data (by raw_command)
    existing_cmds = set()
    for split in ("train", "val", "test"):
        for row in load_existing(split):
            existing_cmds.add(row.get("raw_command", "").strip().lower())

    raw = [(cmd, label) for cmd, label in raw
           if cmd.strip().lower() not in existing_cmds]
    print(f"    {len(raw)} after dedup against existing data")

    # Label distribution of new rows
    label_counts = Counter(label for _, label in raw)
    print("\n[*] New command distribution by technique:")
    for tech, n in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {tech}: {n}")

    print("\n[*] Running residual pipeline...")
    new_rows = residual_rows(raw)

    # Shuffle and split 80/10/10
    random.shuffle(new_rows)
    n = len(new_rows)
    n_val  = max(1, int(n * 0.10))
    n_test = max(1, int(n * 0.10))
    n_train = n - n_val - n_test

    new_train = new_rows[:n_train]
    new_val   = new_rows[n_train:n_train + n_val]
    new_test  = new_rows[n_train + n_val:]

    print(f"\n[*] New split sizes: train={len(new_train)} val={len(new_val)} test={len(new_test)}")

    # Merge with existing
    print("[*] Merging with existing residual_expanded data...")
    for split, new in [("train", new_train), ("val", new_val), ("test", new_test)]:
        existing = load_existing(split)
        merged = existing + new
        random.shuffle(merged)
        out_path = OUTPUT_DIR / f"specialist_{split}_variant_a.jsonl"
        write_split(merged, out_path)
        print(f"    {split}: {len(existing)} existing + {len(new)} new = {len(merged)} total")

    # Final balance check per stage (using build_behavior_dataset mapping)
    print("\n[*] Checking stage distribution in merged train split...")
    from engine import GenosEngine

    TACTIC_TO_STAGE = {
        "Reconnaissance": "Discovery / Recon", "Discovery": "Discovery / Recon",
        "Execution": "Execution", "Persistence": "Persistence",
        "Privilege Escalation": "Privilege Escalation", "Defense Evasion": "Defense Evasion",
        "Credential Access": "Credential Access", "Collection": "Collection / Staging",
        "Command and Control": "C2 / Remote Access", "Lateral Movement": "Lateral Movement",
        "Exfiltration": "Exfiltration", "Impact": "Impact",
        "Initial Access": "Context Required",
    }
    TECHNIQUE_TO_STAGE = {"T1105": "Payload Retrieval"}

    stage_counts = Counter()
    with open(OUTPUT_DIR / "specialist_train_variant_a.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tech = r.get("label", "")
            tactic = GenosEngine._TECHNIQUE_TO_TACTIC.get(tech, "")
            stage = TECHNIQUE_TO_STAGE.get(tech) or TACTIC_TO_STAGE.get(tactic, "Unknown")
            stage_counts[stage] += 1

    print(f"\n{'Stage':<30} {'Count':>6}  {'Bar'}")
    print("-" * 60)
    total = sum(stage_counts.values())
    for stage, n in sorted(stage_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(n / total * 40)
        print(f"  {stage:<28} {n:>6}  {bar}")
    print(f"\n  Total train rows: {total}")


if __name__ == "__main__":
    main()
