#!/usr/bin/env python3
"""Benchmark the heuristic_bootstrap behavior path against the labeled test set.

Evaluates stage exact-match accuracy and action-tag F1 on the 829-sample
behavior_test.jsonl, which was generated from the rule engine itself and
therefore measures internal consistency rather than ground-truth correctness.
A separate --curated flag runs the smaller (11-sample) hand-verified fixture.

Usage:
    python scripts/benchmark/heuristic_behavior_benchmark.py
    python scripts/benchmark/heuristic_behavior_benchmark.py --curated
    python scripts/benchmark/heuristic_behavior_benchmark.py --out logs/heuristic_behavior_results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from engine import GenosEngine

TEST_FIXTURE   = BASE_DIR / "data/training/genos_behavior/behavior_test.jsonl"
CURATED_FIXTURE = BASE_DIR / "data/training/genos_behavior/behavior_eval_curated.jsonl"


# ── schema normalisation ───────────────────────────────────────────────────────

def load_test_rows(path: Path) -> list[dict]:
    """Load behavior_test.jsonl  →  unified {command, expected_stage, expected_actions}."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "command":          r.get("raw_command") or r.get("command", ""),
                "expected_stage":   r.get("stage_label")  or r.get("expected_stage", ""),
                "expected_actions": r.get("action_tags")   or r.get("expected_actions") or [],
                "rule_strength":    r.get("rule_strength", ""),
                "source_mitre":     r.get("source_mitre", ""),
            })
    return rows


def load_curated_rows(path: Path) -> list[dict]:
    """Load behavior_eval_curated.jsonl  →  unified schema."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "command":          r.get("command", ""),
                "expected_stage":   r.get("expected_stage", ""),
                "expected_actions": r.get("expected_actions") or [],
                "rule_strength":    "",
                "source_mitre":     "",
            })
    return rows


# ── metrics ───────────────────────────────────────────────────────────────────

def f1(expected: set, predicted: set) -> float:
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    overlap = len(expected & predicted)
    prec = overlap / len(predicted)
    rec  = overlap / len(expected)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


# ── main ──────────────────────────────────────────────────────────────────────

def run(rows: list[dict], engine: GenosEngine, verbose: bool = False) -> dict:
    stage_correct  = 0
    action_f1_sum  = 0.0
    action_f1_heuristic_sum = 0.0
    heuristic_count = 0

    failures = []
    per_stage: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    per_strength: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0, "f1_sum": 0.0})

    COL_W = 68
    print(f"\n{'#':>4}  {'Stage?':>6}  {'ActF1':>5}  {'Model':>18}  Command")
    print("─" * 100)

    for i, row in enumerate(rows, 1):
        result  = engine.scan(row["command"])
        beh     = result.get("behavior") or {}
        pred_stage   = beh.get("stage", "")
        pred_actions = set(beh.get("action_tags") or [])
        exp_actions  = set(row["expected_actions"])
        model_type   = beh.get("model_type", "unknown")

        stage_ok  = pred_stage == row["expected_stage"]
        act_f1    = f1(exp_actions, pred_actions)

        stage_correct += int(stage_ok)
        action_f1_sum += act_f1

        if model_type == "heuristic_bootstrap":
            heuristic_count += 1
            action_f1_heuristic_sum += act_f1

        per_stage[row["expected_stage"]]["total"]   += 1
        per_stage[row["expected_stage"]]["correct"] += int(stage_ok)

        rs = row.get("rule_strength") or "none"
        per_strength[rs]["total"]   += 1
        per_strength[rs]["correct"] += int(stage_ok)
        per_strength[rs]["f1_sum"]  += act_f1

        cmd_preview = row["command"][:COL_W].replace("\n", " ")
        mark = "✓" if stage_ok else "✗"
        if verbose or not stage_ok or act_f1 < 0.5:
            print(f"{i:>4}  {mark:>6}  {act_f1:>5.2f}  {model_type:>18}  {cmd_preview}")
            if not stage_ok:
                print(f"       exp_stage={row['expected_stage']!r}  pred_stage={pred_stage!r}")
            if act_f1 < 1.0 and (exp_actions or pred_actions):
                print(f"       exp_actions={sorted(exp_actions)}  pred_actions={sorted(pred_actions)}")

        if not stage_ok or act_f1 < 1.0:
            failures.append({
                "command":          row["command"],
                "expected_stage":   row["expected_stage"],
                "predicted_stage":  pred_stage,
                "expected_actions": sorted(exp_actions),
                "predicted_actions": sorted(pred_actions),
                "action_f1":        round(act_f1, 3),
                "rule_strength":    row.get("rule_strength", ""),
                "source_mitre":     row.get("source_mitre", ""),
                "model_type":       model_type,
            })

    n = len(rows)
    stage_acc   = stage_correct / max(n, 1)
    mean_act_f1 = action_f1_sum / max(n, 1)

    print("\n" + "═" * 100)
    print(f"OVERALL  n={n}  stage_accuracy={stage_acc:.1%}  mean_action_f1={mean_act_f1:.3f}")

    if heuristic_count:
        heur_f1 = action_f1_heuristic_sum / heuristic_count
        print(f"heuristic_bootstrap only  n={heuristic_count}  mean_action_f1={heur_f1:.3f}")

    print("\nPer-stage breakdown:")
    for stage, counts in sorted(per_stage.items(), key=lambda x: -x[1]["total"]):
        acc = counts["correct"] / counts["total"]
        print(f"  {stage:<30}  {counts['correct']:>3}/{counts['total']:<3}  {acc:.1%}")

    print("\nPer-rule-strength breakdown:")
    for strength in ["strong", "medium", "weak", "none"]:
        d = per_strength.get(strength)
        if not d or not d["total"]:
            continue
        acc  = d["correct"] / d["total"]
        mf1  = d["f1_sum"] / d["total"]
        print(f"  {strength:<8}  {d['correct']:>3}/{d['total']:<4}  stage_acc={acc:.1%}  mean_action_f1={mf1:.3f}")

    return {
        "n":                n,
        "stage_accuracy":   round(stage_acc, 4),
        "mean_action_f1":   round(mean_act_f1, 4),
        "heuristic_n":      heuristic_count,
        "per_stage":        {k: {"accuracy": round(v["correct"]/v["total"], 4), **v} for k, v in per_stage.items()},
        "per_strength":     {k: {"stage_accuracy": round(v["correct"]/v["total"], 4),
                                  "mean_action_f1": round(v["f1_sum"]/v["total"], 4),
                                  **{kk: vv for kk, vv in v.items() if kk != "f1_sum"}}
                             for k, v in per_strength.items() if v["total"]},
        "failures":         failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", action="store_true",
                        help="Use the 11-sample hand-verified fixture instead of the 829-sample test set")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print every row, not just failures")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write full JSON results to this path")
    args = parser.parse_args()

    fixture = CURATED_FIXTURE if args.curated else TEST_FIXTURE
    rows    = load_curated_rows(fixture) if args.curated else load_test_rows(fixture)

    print(f"[*] Loading engine…")
    engine = GenosEngine()

    print(f"[*] Running benchmark on {len(rows)} rows from {fixture.name}…")
    results = run(rows, engine, verbose=args.verbose)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[*] Results written to {args.out}")


if __name__ == "__main__":
    main()
