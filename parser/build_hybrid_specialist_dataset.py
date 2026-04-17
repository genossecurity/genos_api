"""
build_hybrid_specialist_dataset.py

Reads the existing specialist CSVs and runs the full hybrid pipeline on each
command, producing JSONL files ready for trainer2_hybrid.py.

Pipeline per row:
  raw_command → deobfuscator → parser → semantic_features → rule_engine
             → residual_text → candidate_mask → JSONL row

Output fields per line:
    raw_command            str
    label                  str   (MITRE technique ID)
    hybrid_text            str   (semantic tags + command text)
    deobfuscated_command   str|null
    semantic_tags          list[str]
    candidate_mitre_ids    list[str]
    candidate_mask         list[int]   (0/1 × num_classes)
    prior_vector           list[float] (0-1  × num_classes)
    num_candidates         int
    true_label_in_candidates  bool
    used_fallback          bool
    evidence               list[str]

Usage:
    cd parser/
    python3 build_hybrid_specialist_dataset.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure parser/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from parser import parse_command
from semantic_features import build_semantic_features
from rule_engine import build_rule_result
from residual_text import build_hybrid_text, build_semantic_tags
from candidate_mask import build_candidate_mask, compute_coverage_stats

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "training" / "genos_dataset"
OUTPUT_DIR = DATA_DIR  # write alongside the CSVs

SPECIALIST_MAP_PATH = BASE_DIR / "config" / "specialist_map.json"

SPLITS = [
    ("specialist_train.csv", "hybrid_specialist_train.jsonl"),
    ("specialist_val.csv",   "hybrid_specialist_val.jsonl"),
    ("specialist_test.csv",  "hybrid_specialist_test.jsonl"),
]


def load_specialist_map() -> dict:
    with open(SPECIALIST_MAP_PATH, "r") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list:
    """Read CSV with header 'command,mitre_id'. Return list of (command, label)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        assert "command" in header and "mitre_id" in header, f"Unexpected header: {header}"
        for line in f:
            line = line.strip()
            if not line:
                continue
            # CSV may have commas inside quoted fields
            if line.startswith('"'):
                # Find the last comma that separates command from mitre_id
                last_comma = line.rfind(",")
                command = line[:last_comma].strip('"')
                label = line[last_comma + 1:].strip()
            else:
                last_comma = line.rfind(",")
                command = line[:last_comma]
                label = line[last_comma + 1:].strip()
            if command and label:
                rows.append((command, label))
    return rows


def process_split(
    csv_name: str,
    jsonl_name: str,
    specialist_map: dict,
    split_label: str,
) -> dict:
    """Process one split. Returns coverage stats dict."""
    csv_path = DATA_DIR / csv_name
    out_path = OUTPUT_DIR / jsonl_name

    print(f"\n[*] Processing {split_label}: {csv_path.name} → {jsonl_name}")
    rows = read_csv_rows(csv_path)
    print(f"    {len(rows)} rows loaded")

    mask_results = []
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, (raw_cmd, label) in enumerate(rows):
            try:
                parsed = parse_command(raw_cmd)
            except Exception:
                # Some training rows are fragments; parser should handle them
                # but if it truly fails, produce a minimal parse
                parsed = {"raw_command": raw_cmd}

            sem = build_semantic_features(parsed)
            rule = build_rule_result(parsed, sem)
            hybrid_text = build_hybrid_text(parsed, sem)
            tags = build_semantic_tags(sem)

            mask_info = build_candidate_mask(
                rule, specialist_map,
                true_label=label,
                expand_neighbors=True,
            )

            record = {
                "raw_command":               raw_cmd,
                "label":                     label,
                "hybrid_text":               hybrid_text,
                "deobfuscated_command":       parsed.get("deobfuscated_command"),
                "semantic_tags":             tags,
                "candidate_mitre_ids":       mask_info["candidate_mitre_ids"],
                "candidate_mask":            mask_info["candidate_mask"],
                "prior_vector":              mask_info["prior_vector"],
                "num_candidates":            mask_info["num_candidates"],
                "true_label_in_candidates":  mask_info["true_label_in_candidates"],
                "used_fallback":             mask_info["used_fallback"],
                "evidence":                  rule.get("evidence", []),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            mask_results.append(mask_info)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"    [{i + 1}/{len(rows)}] {elapsed:.1f}s")

    elapsed = time.time() - t0
    stats = compute_coverage_stats(mask_results)
    print(f"    Done in {elapsed:.1f}s. Wrote {out_path}")
    print(f"    Coverage: {stats['organic_coverage_pct']:.1f}% organic "
          f"({stats['organic_true_label_in_candidates']}/{stats['total']})")
    print(f"    Fallback: {stats['fallback_pct']:.1f}% "
          f"({stats['fallback_count']}/{stats['total']})")
    print(f"    Candidates: avg={stats['avg_candidate_size']:.1f}, "
          f"min={stats['min_candidate_size']}, "
          f"max={stats['max_candidate_size']}, "
          f"median={stats['median_candidate_size']}")
    return stats


def main():
    specialist_map = load_specialist_map()
    print(f"[*] Specialist map: {len(specialist_map)} classes")

    all_stats = {}
    for csv_name, jsonl_name in SPLITS:
        split_label = csv_name.replace(".csv", "")
        stats = process_split(csv_name, jsonl_name, specialist_map, split_label)
        all_stats[split_label] = stats

    # Summary
    print("\n" + "=" * 70)
    print("HYBRID DATASET BUILD SUMMARY")
    print("=" * 70)
    for split, stats in all_stats.items():
        print(f"\n  {split}:")
        print(f"    samples:          {stats['total']}")
        print(f"    organic coverage: {stats['organic_coverage_pct']:.1f}%")
        print(f"    fallback rate:    {stats['fallback_pct']:.1f}%")
        print(f"    avg candidates:   {stats['avg_candidate_size']:.1f}")

    # Data quality warnings
    print("\n" + "-" * 70)
    print("DATA QUALITY NOTES:")
    print("-" * 70)
    for split, stats in all_stats.items():
        cov = stats["organic_coverage_pct"]
        if cov < 95:
            fb = stats["fallback_pct"]
            effective = cov + (100 - cov) * (fb / 100) if fb > 0 else cov
            print(f"  WARNING: {split} organic coverage = {cov:.1f}%")
            print(f"           With fallback-as-coverage: ~{effective:.1f}%")
            if cov < 80:
                print(f"           ⚠ Rule engine may need more families or broader mappings")
        else:
            print(f"  OK: {split} organic coverage = {cov:.1f}%")

    # Data quality note
    print("\n  NOTE: If organic coverage is below 95%, consider adding more")
    print("        rule-engine families or broadening MITRE neighbour mappings.")

    # Write summary
    summary_path = OUTPUT_DIR / "hybrid_build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n[+] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
