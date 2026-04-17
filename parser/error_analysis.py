"""
error_analysis.py — Per-rule error analysis for soft-prior hybrid system.

Reports:
  - All breaks grouped by fired rule(s), bucket, predicted vs true label
  - Per-rule fix/broke/ratio
  - Top offending rules by break count
  - Most improved and most degraded commands

Usage:
    cd parser/
    python3 error_analysis.py [--split specialist_val] [--alpha-strong 2.0] [--alpha-weak 1.5]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_hybrid_priors import load_model_and_map, load_split, evaluate


def per_rule_analysis(buckets):
    """Compute per-rule fix/broke stats and return sorted tables."""
    all_records = buckets["strong"] + buckets["weak"] + buckets["none"]

    # Per-rule counters
    rule_stats = defaultdict(lambda: {
        "fires": 0, "fixes": 0, "breaks": 0,
        "fix_examples": [], "break_examples": [],
    })

    for r in all_records:
        rules = r.get("fired_rules", [])
        if not rules:
            continue
        is_fix = not r["baseline_correct"] and r["hybrid_correct"]
        is_break = r["baseline_correct"] and not r["hybrid_correct"]

        for rule_name in rules:
            rule_stats[rule_name]["fires"] += 1
            if is_fix:
                rule_stats[rule_name]["fixes"] += 1
                rule_stats[rule_name]["fix_examples"].append(r)
            if is_break:
                rule_stats[rule_name]["breaks"] += 1
                rule_stats[rule_name]["break_examples"].append(r)

    return dict(rule_stats)


def print_break_analysis(buckets):
    """Print all breaks grouped by rule, bucket, predicted vs true."""
    all_records = buckets["strong"] + buckets["weak"] + buckets["none"]
    breaks = [r for r in all_records if r["baseline_correct"] and not r["hybrid_correct"]]

    print("\n" + "=" * 80)
    print("  BREAK ANALYSIS — All cases where hybrid broke correct baseline")
    print("=" * 80)
    print(f"\n  Total breaks: {len(breaks)}")

    # Group by fired rules (as tuple)
    by_rules = defaultdict(list)
    for r in breaks:
        key = tuple(sorted(r["fired_rules"]))
        by_rules[key].append(r)

    for rule_set, records in sorted(by_rules.items(), key=lambda x: -len(x[1])):
        print(f"\n  Rules: {list(rule_set)} ({len(records)} breaks)")
        for r in records:
            print(f"    [{r['strength']}] {r['cmd'][:90]}")
            print(f"      true={r['label']}  baseline={r['baseline_pred']}  hybrid={r['hybrid_pred']}")


def print_per_rule_table(rule_stats):
    """Print per-rule fix/broke ratio table sorted by impact."""
    print("\n" + "=" * 80)
    print("  PER-RULE FIX/BROKE TABLE")
    print("=" * 80)
    print(f"\n  {'Rule':<40s} {'Fires':>6s} {'Fixes':>6s} {'Breaks':>7s} {'Ratio':>8s} {'Net':>5s}")
    print("  " + "-" * 72)

    # Sort by breaks descending, then by fires descending
    sorted_rules = sorted(
        rule_stats.items(),
        key=lambda x: (-x[1]["breaks"], -x[1]["fires"])
    )

    for rule_name, s in sorted_rules:
        if s["fixes"] == 0 and s["breaks"] == 0:
            ratio_str = "n/a"
        elif s["breaks"] == 0:
            ratio_str = f"{s['fixes']}:0"
        else:
            ratio_str = f"{s['fixes']/s['breaks']:.1f}:1"
        net = s["fixes"] - s["breaks"]
        net_str = f"+{net}" if net >= 0 else str(net)
        print(f"  {rule_name:<40s} {s['fires']:>6d} {s['fixes']:>6d} {s['breaks']:>7d} {ratio_str:>8s} {net_str:>5s}")


def print_top_offenders(rule_stats, top_n=5):
    """Print top N worst offending rules by break count with examples."""
    print("\n" + "=" * 80)
    print(f"  TOP {top_n} WORST OFFENDING RULES (by break count)")
    print("=" * 80)

    sorted_rules = sorted(
        rule_stats.items(),
        key=lambda x: (-x[1]["breaks"], x[1]["fixes"])
    )

    for i, (rule_name, s) in enumerate(sorted_rules[:top_n]):
        if s["breaks"] == 0:
            break
        print(f"\n  #{i+1}: {rule_name}")
        print(f"      fires={s['fires']}  fixes={s['fixes']}  breaks={s['breaks']}")

        # Pattern: what true labels get broken
        true_labels = defaultdict(int)
        hybrid_preds = defaultdict(int)
        for r in s["break_examples"]:
            true_labels[r["label"]] += 1
            hybrid_preds[r["hybrid_pred"]] += 1

        print(f"      True labels of broken cases: {dict(true_labels)}")
        print(f"      Wrong hybrid predictions:    {dict(hybrid_preds)}")

        # Show examples
        for r in s["break_examples"][:3]:
            print(f"        cmd: {r['cmd'][:90]}")
            print(f"        true={r['label']} baseline={r['baseline_pred']} hybrid={r['hybrid_pred']}")


def print_most_improved(buckets, top_n=10):
    """Show most improved and most degraded commands."""
    all_records = buckets["strong"] + buckets["weak"] + buckets["none"]
    fixed = [r for r in all_records if not r["baseline_correct"] and r["hybrid_correct"]]
    broke = [r for r in all_records if r["baseline_correct"] and not r["hybrid_correct"]]

    print("\n" + "=" * 80)
    print(f"  MOST IMPROVED COMMANDS ({len(fixed)} total, showing {min(top_n, len(fixed))})")
    print("=" * 80)
    for r in fixed[:top_n]:
        print(f"  [{r['strength']}] {r['cmd'][:90]}")
        print(f"    true={r['label']}  baseline={r['baseline_pred']}  hybrid={r['hybrid_pred']}")
        print(f"    rules={r['fired_rules']}")

    print("\n" + "=" * 80)
    print(f"  MOST DEGRADED COMMANDS ({len(broke)} total, showing all)")
    print("=" * 80)
    for r in broke:
        print(f"  [{r['strength']}] {r['cmd'][:90]}")
        print(f"    true={r['label']}  baseline={r['baseline_pred']}  hybrid={r['hybrid_pred']}")
        print(f"    rules={r['fired_rules']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="specialist_val")
    ap.add_argument("--alpha-strong", type=float, default=2.0)
    ap.add_argument("--alpha-weak", type=float, default=1.5)
    ap.add_argument("--alpha-none", type=float, default=0.0)
    args = ap.parse_args()

    alpha_overrides = {
        "strong": args.alpha_strong,
        "weak": args.alpha_weak,
        "none": args.alpha_none,
    }

    print("Loading model...", file=sys.stderr)
    model, tokenizer, specialist_map, idx_to_mitre, device, model_num_classes = load_model_and_map()

    print(f"Loading {args.split}...", file=sys.stderr)
    rows = load_split(args.split)

    print(f"Evaluating {len(rows)} commands...", file=sys.stderr)
    buckets = evaluate(rows, model, tokenizer, specialist_map, idx_to_mitre,
                       device, alpha_overrides, model_num_classes)

    rule_stats = per_rule_analysis(buckets)

    print_break_analysis(buckets)
    print_per_rule_table(rule_stats)
    print_top_offenders(rule_stats, top_n=5)
    print_most_improved(buckets, top_n=10)


if __name__ == "__main__":
    main()
