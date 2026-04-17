"""
eval_hybrid_priors.py — Evaluate soft-prior fusion on the specialist model.

Compares baseline (raw logits) vs hybrid (logits + priors) accuracy.
Reports per-bucket impact, regressions, and example cases.

Usage:
    cd parser/
    python3 eval_hybrid_priors.py [--split val] [--alpha-strong 2.0] [--alpha-weak 1.0]
"""

import argparse
import csv
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parser import parse_command
from semantic_features import build_semantic_features
from rule_engine import build_rule_result
from candidate_mask import build_prior_vector, fuse_logits_with_priors
from engine import Tier2_Specialist


def load_model_and_map():
    base = os.path.join(os.path.dirname(__file__), "..")
    map_path = os.path.join(base, "config", "specialist_map.json")
    model_path = os.path.join(base, "models", "specialist.pt")

    with open(map_path) as f:
        specialist_map = json.load(f)  # {mitre_id: int_index}

    # Determine actual model output size from checkpoint
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model_num_classes = state["classifier.4.bias"].shape[0]

    # Filter specialist_map to only indices the model covers
    active_map = {k: int(v) for k, v in specialist_map.items() if int(v) < model_num_classes}
    idx_to_mitre = {v: k for k, v in active_map.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    model = Tier2_Specialist(num_classes=model_num_classes).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, tokenizer, active_map, idx_to_mitre, device, model_num_classes


def load_split(split_name):
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "training", "genos_dataset"
    )
    fname = os.path.join(data_dir, f"{split_name}.csv")
    rows = []
    with open(fname) as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            line = ",".join(r)
            idx = line.rfind(",")
            rows.append((line[:idx], line[idx + 1 :]))
    return rows


def evaluate(rows, model, tokenizer, specialist_map, idx_to_mitre, device,
             alpha_overrides, model_num_classes):
    max_len = 256

    buckets = {"strong": [], "weak": [], "none": []}
    total = len(rows)
    skipped = 0

    for i, (cmd, label) in enumerate(rows):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{total}...", file=sys.stderr)

        # Skip labels the model doesn't know about
        if label not in specialist_map:
            skipped += 1
            continue

        # Rule engine
        parsed = parse_command(cmd)
        sem = build_semantic_features(parsed)
        rule = build_rule_result(parsed, sem)
        pv_info = build_prior_vector(rule, specialist_map,
                                     alpha_overrides=alpha_overrides)

        # Tokenize
        enc = tokenizer(cmd, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Inference
        with torch.no_grad():
            logits = model(input_ids, attention_mask)  # (1, model_num_classes)

        # Baseline prediction
        baseline_pred_idx = torch.argmax(logits, dim=1).item()
        baseline_pred = idx_to_mitre.get(baseline_pred_idx, f"UNK_{baseline_pred_idx}")
        baseline_top3 = torch.topk(logits, 3, dim=1).indices[0].tolist()
        baseline_top3_ids = [idx_to_mitre.get(i, f"UNK_{i}") for i in baseline_top3]

        # Hybrid prediction — build prior vector sized to model output
        pv_full = [0.0] * model_num_classes
        for j, v in enumerate(pv_info["prior_vector"]):
            if j < model_num_classes:
                pv_full[j] = v
        pv_tensor = torch.tensor(pv_full, dtype=logits.dtype, device=device).unsqueeze(0)
        fused = logits + pv_tensor
        hybrid_pred_idx = torch.argmax(fused, dim=1).item()
        hybrid_pred = idx_to_mitre.get(hybrid_pred_idx, f"UNK_{hybrid_pred_idx}")
        hybrid_top3 = torch.topk(fused, 3, dim=1).indices[0].tolist()
        hybrid_top3_ids = [idx_to_mitre.get(i, f"UNK_{i}") for i in hybrid_top3]

        record = {
            "cmd": cmd[:120],
            "label": label,
            "strength": pv_info["rule_strength"],
            "alpha": pv_info["alpha"],
            "baseline_pred": baseline_pred,
            "hybrid_pred": hybrid_pred,
            "baseline_top3": baseline_top3_ids,
            "hybrid_top3": hybrid_top3_ids,
            "baseline_correct": baseline_pred == label,
            "hybrid_correct": hybrid_pred == label,
            "baseline_top3_correct": label in baseline_top3_ids,
            "hybrid_top3_correct": label in hybrid_top3_ids,
            "fired_rules": pv_info["fired_rules"],
        }
        buckets[pv_info["rule_strength"]].append(record)

    if skipped:
        print(f"  Skipped {skipped} commands (label not in model's class set)", file=sys.stderr)

    return buckets


def report(buckets):
    all_records = buckets["strong"] + buckets["weak"] + buckets["none"]
    total = len(all_records)

    def stats(records):
        n = len(records)
        if n == 0:
            return {"n": 0}
        bl_top1 = sum(1 for r in records if r["baseline_correct"])
        hy_top1 = sum(1 for r in records if r["hybrid_correct"])
        bl_top3 = sum(1 for r in records if r["baseline_top3_correct"])
        hy_top3 = sum(1 for r in records if r["hybrid_top3_correct"])
        fixed = sum(1 for r in records
                    if not r["baseline_correct"] and r["hybrid_correct"])
        broke = sum(1 for r in records
                    if r["baseline_correct"] and not r["hybrid_correct"])
        return {
            "n": n,
            "bl_top1": bl_top1, "bl_top1_pct": bl_top1 / n * 100,
            "hy_top1": hy_top1, "hy_top1_pct": hy_top1 / n * 100,
            "bl_top3": bl_top3, "bl_top3_pct": bl_top3 / n * 100,
            "hy_top3": hy_top3, "hy_top3_pct": hy_top3 / n * 100,
            "fixed": fixed, "broke": broke,
        }

    print("\n" + "=" * 72)
    print("  SOFT PRIOR FUSION — BASELINE vs HYBRID")
    print("=" * 72)

    for bucket_name in ["strong", "weak", "none", "ALL"]:
        records = buckets.get(bucket_name, all_records) if bucket_name != "ALL" else all_records
        s = stats(records)
        if s["n"] == 0:
            continue
        print(f"\n  {bucket_name.upper():6s} bucket ({s['n']} commands, {s['n']/total*100:.1f}%)")
        print(f"    Baseline  top-1: {s['bl_top1_pct']:5.1f}%  top-3: {s['bl_top3_pct']:5.1f}%")
        print(f"    Hybrid    top-1: {s['hy_top1_pct']:5.1f}%  top-3: {s['hy_top3_pct']:5.1f}%")
        print(f"    Delta     top-1: {s['hy_top1_pct'] - s['bl_top1_pct']:+5.1f}pp")
        print(f"    Fixed (baseline wrong → hybrid right): {s['fixed']}")
        print(f"    Broke (baseline right → hybrid wrong): {s['broke']}")

    # Show example fixes and regressions
    fixed = [r for r in all_records
             if not r["baseline_correct"] and r["hybrid_correct"]]
    broke = [r for r in all_records
             if r["baseline_correct"] and not r["hybrid_correct"]]

    if fixed:
        print(f"\n  EXAMPLES: Hybrid FIXED ({len(fixed)} total, showing up to 5)")
        for r in fixed[:5]:
            print(f"    [{r['strength']}] {r['cmd'][:80]}")
            print(f"      label={r['label']}  baseline={r['baseline_pred']}  hybrid={r['hybrid_pred']}")
            print(f"      rules={r['fired_rules']}")

    if broke:
        print(f"\n  EXAMPLES: Hybrid BROKE ({len(broke)} total, showing up to 5)")
        for r in broke[:5]:
            print(f"    [{r['strength']}] {r['cmd'][:80]}")
            print(f"      label={r['label']}  baseline={r['baseline_pred']}  hybrid={r['hybrid_pred']}")
            print(f"      rules={r['fired_rules']}")

    print("\n" + "=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="specialist_val",
                    help="Dataset split (specialist_train/specialist_val/specialist_test)")
    ap.add_argument("--alpha-strong", type=float, default=2.0)
    ap.add_argument("--alpha-weak", type=float, default=1.0)
    ap.add_argument("--alpha-none", type=float, default=0.0)
    args = ap.parse_args()

    alpha_overrides = {
        "strong": args.alpha_strong,
        "weak": args.alpha_weak,
        "none": args.alpha_none,
    }

    print(f"Loading model...", file=sys.stderr)
    model, tokenizer, specialist_map, idx_to_mitre, device, model_num_classes = load_model_and_map()

    print(f"Loading {args.split}...", file=sys.stderr)
    rows = load_split(args.split)

    print(f"Evaluating {len(rows)} commands (alpha: strong={args.alpha_strong}, "
          f"weak={args.alpha_weak}, none={args.alpha_none})...", file=sys.stderr)
    print(f"Model classes: {model_num_classes}, Map classes: {len(specialist_map)}", file=sys.stderr)
    buckets = evaluate(rows, model, tokenizer, specialist_map, idx_to_mitre,
                       device, alpha_overrides, model_num_classes)

    report(buckets)


if __name__ == "__main__":
    main()
