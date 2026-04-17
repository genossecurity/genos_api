"""
eval_residual_priors.py — Evaluate soft-prior fusion on top of Variant A model.

Loads specialist_residual_a.pt and runs two eval modes on val + test:
  1. Baseline: Variant A model, pure logits, no priors
  2. Hybrid:   Variant A model + soft-prior fusion (logits + alpha * prior_vector)

Input text for each command is rebuilt using the exact Variant A format:
  RAW: {cmd}
  RESIDUAL: {residual}
  FEATURES: {tags}     <- omitted if no tags fire

This matches the format used during training and is the correct inference path.

Usage:
    cd /home/snake/genos_api/genos_api
    python3 parser/eval_residual_priors.py [--alpha-strong 2.0] [--alpha-weak 1.5]
    python3 parser/eval_residual_priors.py --sweep          # auto-tune alphas
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parser import parse_command
from semantic_features import build_semantic_features
from rule_engine import build_rule_result
from candidate_mask import build_prior_vector
from build_residual_dataset import build_residual, build_feature_tags
from engine import Tier2_Specialist


# ─────────────────────────────────────────────────────────────────────────────
# Model + tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_variant_a(device):
    base = os.path.join(os.path.dirname(__file__), "..")
    map_path  = os.path.join(base, "config", "specialist_map.json")
    ckpt_path = os.path.join(base, "models", "specialist_residual_a.pt")

    with open(map_path) as f:
        label_map = json.load(f)          # {mitre_id: int_index}
    num_classes = len(label_map)
    idx_to_mitre = {int(v): k for k, v in label_map.items()}

    # Verify checkpoint covers same class count
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_classes = state["classifier.4.bias"].shape[0]
    if ckpt_classes != num_classes:
        print(f"[!] WARNING: checkpoint has {ckpt_classes} classes, "
              f"map has {num_classes}. Using checkpoint size.", file=sys.stderr)
        num_classes = ckpt_classes
        idx_to_mitre = {v: k for k, v in label_map.items() if int(v) < num_classes}

    model = Tier2_Specialist(num_classes=num_classes).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[!] Missing keys: {missing[:5]}", file=sys.stderr)
    model.eval()
    print(f"[+] Loaded specialist_residual_a.pt  ({num_classes} classes)", file=sys.stderr)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base", use_fast=True
    )

    return model, tokenizer, label_map, idx_to_mitre, num_classes


# ─────────────────────────────────────────────────────────────────────────────
# Input text builder (Variant A format — must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────

def build_variant_a_text(cmd: str) -> tuple:
    """
    Run parser + semantic + rule pipeline and produce:
      - variant_a_text: str  (the input text for the model)
      - rule_result: dict    (for prior vector construction)

    Returns (variant_a_text, rule_result).
    """
    parsed  = parse_command(cmd)
    sem     = build_semantic_features(parsed)
    rules   = build_rule_result(parsed, sem)

    residual     = build_residual(parsed, sem, rules)
    feature_tags = build_feature_tags(sem, rules)

    parts = [f"RAW: {cmd}", f"RESIDUAL: {residual}"]
    if feature_tags:
        parts.append(f"FEATURES: {' '.join(feature_tags)}")
    return "\n".join(parts), rules


# ─────────────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_split(split_name: str):
    """Load genos_dataset CSV → list of (command, label)."""
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
            rows.append((line[:idx], line[idx + 1:]))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    rows,
    model,
    tokenizer,
    label_map,
    idx_to_mitre,
    num_classes,
    device,
    alpha_strong,
    alpha_weak,
    max_len=256,
):
    """
    Evaluate every command against Variant A baseline and Variant A + priors.
    Returns dict of buckets {strong, weak, none}, each a list of records.
    """
    alpha_overrides = {
        "strong": alpha_strong,
        "weak": alpha_weak,
        "none": 0.0,
    }

    buckets = {"strong": [], "weak": [], "none": []}
    skipped = 0
    total = len(rows)

    for i, (cmd, label) in enumerate(rows):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total}...", file=sys.stderr)

        if label not in label_map:
            skipped += 1
            continue

        # ── Build Variant A input text + get rule result
        input_text, rules = build_variant_a_text(cmd)

        # ── Tokenise
        enc = tokenizer(
            input_text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        # ── Inference
        with torch.no_grad():
            logits = model(ids, mask)           # (1, num_classes)

        # ── Baseline predictions
        bl_top1_idx = torch.argmax(logits, dim=1).item()
        bl_top3_idx = torch.topk(logits, min(3, num_classes), dim=1).indices[0].tolist()
        bl_top1  = idx_to_mitre.get(bl_top1_idx, f"UNK_{bl_top1_idx}")
        bl_top3  = [idx_to_mitre.get(j, f"UNK_{j}") for j in bl_top3_idx]

        # ── Prior vector
        pv_info = build_prior_vector(rules, label_map, alpha_overrides=alpha_overrides)
        pv_full = [0.0] * num_classes
        for j, v in enumerate(pv_info["prior_vector"]):
            if j < num_classes:
                pv_full[j] = v
        pv_tensor = torch.tensor(pv_full, dtype=logits.dtype, device=device).unsqueeze(0)

        # ── Hybrid predictions
        fused = logits + pv_tensor
        hy_top1_idx = torch.argmax(fused, dim=1).item()
        hy_top3_idx = torch.topk(fused, min(3, num_classes), dim=1).indices[0].tolist()
        hy_top1  = idx_to_mitre.get(hy_top1_idx, f"UNK_{hy_top1_idx}")
        hy_top3  = [idx_to_mitre.get(j, f"UNK_{j}") for j in hy_top3_idx]

        record = {
            "cmd":              cmd[:120],
            "label":            label,
            "strength":         pv_info["rule_strength"],
            "alpha":            pv_info["alpha"],
            "fired_rules":      pv_info["fired_rules"],
            "bl_top1":          bl_top1,
            "hy_top1":          hy_top1,
            "bl_top3":          bl_top3,
            "hy_top3":          hy_top3,
            "bl_correct":       bl_top1 == label,
            "hy_correct":       hy_top1 == label,
            "bl_top3_correct":  label in bl_top3,
            "hy_top3_correct":  label in hy_top3,
        }
        buckets[pv_info["rule_strength"]].append(record)

    if skipped:
        print(f"  Skipped {skipped} rows (unknown label)", file=sys.stderr)

    return buckets


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _stats(records):
    n = len(records)
    if n == 0:
        return None
    bl1 = sum(r["bl_correct"]      for r in records)
    hy1 = sum(r["hy_correct"]      for r in records)
    bl3 = sum(r["bl_top3_correct"] for r in records)
    hy3 = sum(r["hy_top3_correct"] for r in records)
    fixed = sum(1 for r in records if not r["bl_correct"] and r["hy_correct"])
    broke = sum(1 for r in records if r["bl_correct"]     and not r["hy_correct"])
    return {
        "n": n,
        "bl_acc": bl1 / n, "hy_acc": hy1 / n,
        "bl_top3": bl3 / n, "hy_top3": hy3 / n,
        "d_acc": (hy1 - bl1) / n,
        "d_top3": (hy3 - bl3) / n,
        "fixed": fixed, "broke": broke,
        "ratio": f"{fixed}:{broke}",
    }


def _compute_macro_f1(records, key_correct):
    """Compute macro F1 from per-record hits."""
    # Collect unique labels
    labels = list({r["label"] for r in records})
    label_set = set(labels)

    mf = 0.0
    wf = 0.0
    total = len(records)

    for cls in labels:
        tp = sum(1 for r in records if r["label"] == cls and r[key_correct])
        fp = sum(1 for r in records if r["label"] != cls and r[f"{'bl' if 'bl' in key_correct else 'hy'}_top1"] == cls)
        fn = sum(1 for r in records if r["label"] == cls and not r[key_correct])
        sup = sum(1 for r in records if r["label"] == cls)
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        mf  += f1
        wf  += f1 * sup

    nc = max(1, len(labels))
    return mf / nc, wf / total


def _macro_f1_for_bucket(records, use_hybrid=False):
    key_top1     = "hy_top1" if use_hybrid else "bl_top1"
    key_correct  = "hy_correct" if use_hybrid else "bl_correct"
    labels = list({r["label"] for r in records})
    mf = 0.0
    wf = 0.0
    total = len(records)
    for cls in labels:
        tp  = sum(1 for r in records if r["label"] == cls and r[key_correct])
        fp  = sum(1 for r in records if r["label"] != cls and r[key_top1] == cls)
        fn  = sum(1 for r in records if r["label"] == cls and not r[key_correct])
        sup = sum(1 for r in records if r["label"] == cls)
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        mf  += f1
        wf  += f1 * sup
    nc = max(1, len(labels))
    return mf / nc, wf / total


def report(split_name: str, buckets: dict, alpha_strong: float, alpha_weak: float):
    all_rec = buckets["strong"] + buckets["weak"] + buckets["none"]

    W  = 76
    sep = "=" * W

    print(f"\n{sep}")
    print(f"  VARIANT A BASELINE vs VARIANT A + SOFT PRIORS — {split_name.upper()}")
    print(f"  (alpha: strong={alpha_strong}, weak={alpha_weak}, none=0.0)")
    print(sep)

    # ── Summary table
    bl_mf, bl_wf = _macro_f1_for_bucket(all_rec, use_hybrid=False)
    hy_mf, hy_wf = _macro_f1_for_bucket(all_rec, use_hybrid=True)
    s = _stats(all_rec)

    print(f"\n  {'Metric':<22s} {'Variant A':>12s} {'+ Priors':>12s} {'Delta':>10s}")
    print(f"  {'-'*56}")
    print(f"  {'Accuracy':<22s} {s['bl_acc']*100:>11.2f}% {s['hy_acc']*100:>11.2f}% {s['d_acc']*100:>+9.2f}pp")
    print(f"  {'Top-3':<22s} {s['bl_top3']*100:>11.2f}% {s['hy_top3']*100:>11.2f}% {s['d_top3']*100:>+9.2f}pp")
    print(f"  {'Macro F1':<22s} {bl_mf:>12.4f} {hy_mf:>12.4f} {hy_mf-bl_mf:>+10.4f}")
    print(f"  {'Weighted F1':<22s} {bl_wf:>12.4f} {hy_wf:>12.4f} {hy_wf-bl_wf:>+10.4f}")
    print(f"  {'Fixed / Broke':<22s} {'':>12s} {s['fixed']:>5d} fixed  {s['broke']:>5d} broke  ({s['ratio']})")

    # ── Per-bucket table
    print(f"\n  {'Bucket':<10s} {'N':>5s} {'BL_Acc':>8s} {'HY_Acc':>8s} {'D_Acc':>8s} {'BL_Top3':>8s} {'HY_Top3':>8s} {'Fixed':>6s} {'Broke':>6s}")
    print(f"  {'-'*72}")
    for bname in ["strong", "weak", "none", "ALL"]:
        recs = all_rec if bname == "ALL" else buckets.get(bname, [])
        s2 = _stats(recs)
        if s2 is None:
            continue
        print(
            f"  {bname.upper():<10s} {s2['n']:>5d} "
            f"{s2['bl_acc']*100:>7.1f}% {s2['hy_acc']*100:>7.1f}% "
            f"{s2['d_acc']*100:>+7.1f}pp "
            f"{s2['bl_top3']*100:>7.1f}% {s2['hy_top3']*100:>7.1f}% "
            f"{s2['fixed']:>6d} {s2['broke']:>6d}"
        )

    # ── Fix / broke details
    fixed_cases = [r for r in all_rec if not r["bl_correct"] and r["hy_correct"]]
    broke_cases = [r for r in all_rec if r["bl_correct"] and not r["hy_correct"]]

    print(f"\n  FIXED CASES ({len(fixed_cases)} total, showing up to 8)")
    print(f"  {'-'*72}")
    for r in fixed_cases[:8]:
        print(f"    [{r['strength']}] {r['cmd'][:82]}")
        print(f"      label={r['label']}  baseline={r['bl_top1']}  hybrid={r['hy_top1']}")
        print(f"      rules={r['fired_rules']}")

    print(f"\n  BROKE CASES ({len(broke_cases)} total, showing all)")
    print(f"  {'-'*72}")
    for r in broke_cases:
        print(f"    [{r['strength']}] {r['cmd'][:82]}")
        print(f"      label={r['label']}  baseline={r['bl_top1']}  hybrid={r['hy_top1']}")
        print(f"      rules={r['fired_rules']}")

    # ── Fix:broke per rule
    rule_stats = defaultdict(lambda: {"fires": 0, "fixes": 0, "breaks": 0})
    for r in all_rec:
        for rn in r["fired_rules"]:
            rule_stats[rn]["fires"] += 1
            if not r["bl_correct"] and r["hy_correct"]:
                rule_stats[rn]["fixes"] += 1
            if r["bl_correct"] and not r["hy_correct"]:
                rule_stats[rn]["breaks"] += 1

    if rule_stats:
        print(f"\n  PER-RULE IMPACT TABLE")
        print(f"  {'-'*72}")
        print(f"  {'Rule':<42s} {'Fires':>6s} {'Fixes':>6s} {'Breaks':>7s} {'Net':>5s} {'Ratio':>8s}")
        for rn, s3 in sorted(rule_stats.items(), key=lambda x: -abs(x[1]["fixes"] - x[1]["breaks"])):
            net = s3["fixes"] - s3["breaks"]
            if s3["breaks"] == 0:
                ratio = "inf" if s3["fixes"] > 0 else "0.00"
            else:
                ratio = f"{s3['fixes'] / s3['breaks']:.2f}"
            print(f"  {rn:<42s} {s3['fires']:>6d} {s3['fixes']:>6d} {s3['breaks']:>7d} {net:>+5d} {ratio:>8s}")

    print(f"\n{sep}\n")

    return {
        "bl_acc": s["bl_acc"], "hy_acc": s["hy_acc"],
        "bl_top3": s["bl_top3"], "hy_top3": s["hy_top3"],
        "bl_mf": bl_mf, "hy_mf": hy_mf,
        "bl_wf": bl_wf, "hy_wf": hy_wf,
        "fixed": s["fixed"], "broke": s["broke"],
    }


def recommend_config(val_results: dict, test_results: dict,
                     alpha_strong: float, alpha_weak: float):
    """Print a production config recommendation."""
    W = 76
    sep = "=" * W
    print(f"\n{sep}")
    print(f"  PRODUCTION CONFIG RECOMMENDATION")
    print(sep)

    val_net  = val_results["fixed"]  - val_results["broke"]
    test_net = test_results["fixed"] - test_results["broke"]
    val_f1_delta  = val_results["hy_mf"]  - val_results["bl_mf"]
    test_f1_delta = test_results["hy_mf"] - test_results["bl_mf"]

    if val_f1_delta > 0 and test_f1_delta > 0:
        verdict = "ENABLE soft-prior fusion"
        note    = "Both val and test macro F1 improve with priors."
    elif val_f1_delta > 0 or test_f1_delta > 0:
        verdict = "ENABLE soft-prior fusion (marginal gain)"
        note    = "One split improves; the other is roughly neutral."
    else:
        verdict = "SKIP soft-prior fusion (no benefit on Variant A)"
        note    = "Priors do not improve Variant A; disable or reduce alphas."

    print(f"\n  Verdict: {verdict}")
    print(f"  Reason:  {note}")
    print(f"\n  Val   macro F1 delta: {val_f1_delta:+.4f}   fix:broke={val_results['fixed']}:{val_results['broke']}")
    print(f"  Test  macro F1 delta: {test_f1_delta:+.4f}   fix:broke={test_results['fixed']}:{test_results['broke']}")
    print(f"\n  Recommended candidate_mask.py alphas:")
    print(f"    strong: {alpha_strong}")
    print(f"    weak:   {alpha_weak}")
    print(f"    none:   0.0")
    print(f"\n  engine.py / app.py: use specialist_residual_a.pt for Tier2_Specialist")
    print(f"  Input text format at inference: Variant A (RAW + RESIDUAL + FEATURES)")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Alpha sweep (optional)
# ─────────────────────────────────────────────────────────────────────────────

def alpha_sweep(rows, model, tokenizer, label_map, idx_to_mitre, num_classes, device):
    """
    Quickly sweep alpha_strong and alpha_weak on val set to find best combo.
    Optimises for macro F1 gain over baseline.
    """
    import itertools
    candidates_strong = [1.0, 1.5, 2.0, 2.5, 3.0]
    candidates_weak   = [0.5, 1.0, 1.5, 2.0]

    print("\n[*] Alpha sweep on val set (optimising macro F1 delta)...", file=sys.stderr)
    best = None

    for as_, aw in itertools.product(candidates_strong, candidates_weak):
        buckets = evaluate(rows, model, tokenizer, label_map, idx_to_mitre,
                           num_classes, device, as_, aw)
        all_rec = buckets["strong"] + buckets["weak"] + buckets["none"]
        bl_mf, _ = _macro_f1_for_bucket(all_rec, use_hybrid=False)
        hy_mf, _ = _macro_f1_for_bucket(all_rec, use_hybrid=True)
        delta = hy_mf - bl_mf
        fixed = sum(1 for r in all_rec if not r["bl_correct"] and r["hy_correct"])
        broke = sum(1 for r in all_rec if r["bl_correct"] and not r["hy_correct"])
        print(f"  as={as_:.1f}  aw={aw:.1f}  -> delta_f1={delta:+.4f}  fix:broke={fixed}:{broke}", file=sys.stderr)
        if best is None or delta > best["delta"]:
            best = {"alpha_strong": as_, "alpha_weak": aw, "delta": delta}

    print(f"\n[+] Best: alpha_strong={best['alpha_strong']}, "
          f"alpha_weak={best['alpha_weak']}  (delta_f1={best['delta']:+.4f})", file=sys.stderr)
    return best["alpha_strong"], best["alpha_weak"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Eval Variant A + soft priors")
    ap.add_argument("--alpha-strong", type=float, default=2.0)
    ap.add_argument("--alpha-weak",   type=float, default=1.5)
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep alpha values on val set before final eval")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}", file=sys.stderr)

    model, tokenizer, label_map, idx_to_mitre, num_classes = load_variant_a(device)

    val_rows  = load_split("specialist_val")
    test_rows = load_split("specialist_test")
    print(f"[*] Val: {len(val_rows)} rows  |  Test: {len(test_rows)} rows", file=sys.stderr)

    alpha_strong = args.alpha_strong
    alpha_weak   = args.alpha_weak

    if args.sweep:
        alpha_strong, alpha_weak = alpha_sweep(
            val_rows, model, tokenizer, label_map, idx_to_mitre, num_classes, device
        )

    print(f"\n[*] Evaluating val  (alpha: strong={alpha_strong}, weak={alpha_weak})...", file=sys.stderr)
    val_buckets  = evaluate(val_rows,  model, tokenizer, label_map, idx_to_mitre,
                            num_classes, device, alpha_strong, alpha_weak)

    print(f"[*] Evaluating test (alpha: strong={alpha_strong}, weak={alpha_weak})...", file=sys.stderr)
    test_buckets = evaluate(test_rows, model, tokenizer, label_map, idx_to_mitre,
                            num_classes, device, alpha_strong, alpha_weak)

    val_summary  = report("val",  val_buckets,  alpha_strong, alpha_weak)
    test_summary = report("test", test_buckets, alpha_strong, alpha_weak)

    recommend_config(val_summary, test_summary, alpha_strong, alpha_weak)


if __name__ == "__main__":
    main()
