#!/usr/bin/env python3
"""Curated behavior benchmark for the runtime behavior encoder.

Evaluates stage exact-match and action-tag overlap on a small hand-checked set.
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from engine import GenosEngine


DEFAULT_FIXTURE = BASE_DIR / "data" / "training" / "genos_behavior" / "behavior_eval_curated.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    return parser.parse_args()


def load_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def f1_score(expected: set[str], predicted: set[str]) -> float:
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    overlap = len(expected & predicted)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    args = parse_args()
    rows = load_rows(args.fixture)
    engine = GenosEngine()

    stage_correct = 0
    action_f1_total = 0.0

    print(f"[*] Evaluating {len(rows)} curated behavior samples...")
    print(f"{'#':>3}  {'Stage':>5}  {'ActF1':>5}  Command")
    print("-" * 88)

    for index, row in enumerate(rows, start=1):
        result = engine.scan(row["command"])
        behavior = result.get("behavior") or {}
        predicted_stage = behavior.get("stage")
        predicted_actions = set(behavior.get("action_tags") or [])
        expected_actions = set(row.get("expected_actions") or [])

        stage_ok = predicted_stage == row["expected_stage"]
        stage_correct += int(stage_ok)
        action_f1 = f1_score(expected_actions, predicted_actions)
        action_f1_total += action_f1

        print(
            f"{index:>3}  {'yes' if stage_ok else 'no':>5}  {action_f1:>5.2f}  {row['command'][:64]}"
        )
        if not stage_ok or action_f1 < 1.0:
            print(
                json.dumps(
                    {
                        "expected_stage": row["expected_stage"],
                        "predicted_stage": predicted_stage,
                        "expected_actions": sorted(expected_actions),
                        "predicted_actions": sorted(predicted_actions),
                    },
                    indent=2,
                )
            )

    stage_accuracy = stage_correct / max(1, len(rows))
    mean_action_f1 = action_f1_total / max(1, len(rows))
    print("\nSUMMARY")
    print(json.dumps({"n": len(rows), "stage_accuracy": stage_accuracy, "mean_action_f1": mean_action_f1}, indent=2))


if __name__ == "__main__":
    main()