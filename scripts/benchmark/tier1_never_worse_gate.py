#!/usr/bin/env python3
"""Compare a Tier 1 candidate against a baseline and fail obvious regressions.

This is an evaluation gate only. It does not modify runtime behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
STRESS_DIR = BASE_DIR / "logs" / "tier1_stress"
SANITY_DIR = BASE_DIR / "logs" / "tier1_sanity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-prefix", required=True, help="Summary prefix for the candidate run.")
    parser.add_argument("--baseline-prefix", default="benign_patch_v2a", help="Summary prefix for the baseline run.")
    parser.add_argument("--fp-tolerance", type=float, default=0.005, help="Allowed expanded-benign FP regression over baseline.")
    parser.add_argument("--hard-negative-f1-tolerance", type=float, default=0.005, help="Allowed hard-negative macro-F1 drop from baseline.")
    parser.add_argument("--threshold-hard-negative-f1-tolerance", type=float, default=0.005, help="Allowed thresholded hard-negative macro-F1 drop from baseline.")
    parser.add_argument("--robustness-regression-tolerance", type=int, default=0, help="Allowed additional robustness mismatches over baseline.")
    parser.add_argument("--routine-admin-recall-floor", type=float, default=0.70, help="Minimum routine-admin recall from sanity benchmark.")
    parser.add_argument("--held-out-macro-f1-floor", type=float, default=0.93, help="Minimum held-out macro F1 from stress benchmark.")
    parser.add_argument("--output-json", default="", help="Optional path to save the gate report JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_candidate_catastrophic_count(candidate_prefix: str, candidate_sanity: dict) -> int:
    acceptance_gate = candidate_sanity.get("acceptance_gate", {})
    catastrophic_gate = acceptance_gate.get("catastrophic_benign")
    if catastrophic_gate is None:
        raise ValueError(
            "Candidate sanity summary is missing acceptance_gate.catastrophic_benign. "
            f"Rerun scripts/benchmark/tier1_sanity_benchmark.py --prefix {candidate_prefix} with the updated benchmark first."
        )
    return int(catastrophic_gate["malicious_count"])


def criterion(name: str, actual, expected, passed: bool, details: str = "") -> dict[str, object]:
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "passed": passed,
        "details": details,
    }


def main() -> None:
    args = parse_args()
    candidate_stress = load_json(STRESS_DIR / f"{args.candidate_prefix}_summary.json")
    candidate_sanity = load_json(SANITY_DIR / f"{args.candidate_prefix}_summary.json")
    baseline_stress = load_json(STRESS_DIR / f"{args.baseline_prefix}_summary.json")
    baseline_sanity = load_json(SANITY_DIR / f"{args.baseline_prefix}_summary.json")

    candidate_catastrophic = get_candidate_catastrophic_count(args.candidate_prefix, candidate_sanity)
    candidate_parity = candidate_stress["parity"]["mismatch_count"]
    candidate_fp = candidate_stress["expanded_benign"]["fp_rate"]
    baseline_fp = baseline_stress["expanded_benign"]["fp_rate"]
    candidate_hard_f1 = candidate_stress["hard_negative"]["metrics"]["macro_f1"]
    baseline_hard_f1 = baseline_stress["hard_negative"]["metrics"]["macro_f1"]
    candidate_threshold_hard_f1 = candidate_stress["threshold_policy"]["hard_negative_metrics"]["macro_f1"]
    baseline_threshold_hard_f1 = baseline_stress["threshold_policy"]["hard_negative_metrics"]["macro_f1"]
    candidate_held_out = candidate_stress["held_out_test"]["metrics"]["macro_f1"]
    candidate_routine_admin = candidate_sanity["buckets"]["routine_admin"]["metrics"]["per_class"]["Benign"]["recall"]
    candidate_robustness = candidate_stress["robustness"]["mismatch_count"]
    baseline_robustness = baseline_stress["robustness"]["mismatch_count"]
    baseline_trivial_malicious = baseline_sanity["acceptance_gate"]["trivial_benign_to_malicious"]
    candidate_trivial_malicious = candidate_sanity["acceptance_gate"]["trivial_benign_to_malicious"]

    checks = [
        criterion(
            "catastrophic_benign_zero_malicious",
            candidate_catastrophic,
            0,
            candidate_catastrophic == 0,
            "Obvious routine commands may not route to Malicious.",
        ),
        criterion(
            "runtime_parity_zero",
            candidate_parity,
            0,
            candidate_parity == 0,
            "Direct model labels and runtime internal labels must match.",
        ),
        criterion(
            "expanded_benign_fp_never_worse",
            candidate_fp,
            f"<= {baseline_fp + args.fp_tolerance:.6f}",
            candidate_fp <= baseline_fp + args.fp_tolerance,
            f"Baseline={baseline_fp:.6f}, tolerance={args.fp_tolerance:.6f}",
        ),
        criterion(
            "hard_negative_macro_f1_never_worse",
            candidate_hard_f1,
            f">= {baseline_hard_f1 - args.hard_negative_f1_tolerance:.6f}",
            candidate_hard_f1 >= baseline_hard_f1 - args.hard_negative_f1_tolerance,
            f"Baseline={baseline_hard_f1:.6f}, tolerance={args.hard_negative_f1_tolerance:.6f}",
        ),
        criterion(
            "threshold_hard_negative_macro_f1_never_worse",
            candidate_threshold_hard_f1,
            f">= {baseline_threshold_hard_f1 - args.threshold_hard_negative_f1_tolerance:.6f}",
            candidate_threshold_hard_f1 >= baseline_threshold_hard_f1 - args.threshold_hard_negative_f1_tolerance,
            f"Baseline={baseline_threshold_hard_f1:.6f}, tolerance={args.threshold_hard_negative_f1_tolerance:.6f}",
        ),
        criterion(
            "held_out_macro_f1_floor",
            candidate_held_out,
            f">= {args.held_out_macro_f1_floor:.6f}",
            candidate_held_out >= args.held_out_macro_f1_floor,
            "Candidate must preserve held-out quality.",
        ),
        criterion(
            "routine_admin_recall_floor",
            candidate_routine_admin,
            f">= {args.routine_admin_recall_floor:.6f}",
            candidate_routine_admin >= args.routine_admin_recall_floor,
            "Candidate must remain usable for routine operations.",
        ),
        criterion(
            "robustness_never_worse",
            candidate_robustness,
            f"<= {baseline_robustness + args.robustness_regression_tolerance}",
            candidate_robustness <= baseline_robustness + args.robustness_regression_tolerance,
            f"Baseline={baseline_robustness}, tolerance={args.robustness_regression_tolerance}",
        ),
        criterion(
            "trivial_benign_malicious_never_worse",
            candidate_trivial_malicious,
            f"<= {baseline_trivial_malicious}",
            candidate_trivial_malicious <= baseline_trivial_malicious,
            f"Baseline trivial benign->Malicious count={baseline_trivial_malicious}",
        ),
    ]
    report = {
        "candidate_prefix": args.candidate_prefix,
        "baseline_prefix": args.baseline_prefix,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()