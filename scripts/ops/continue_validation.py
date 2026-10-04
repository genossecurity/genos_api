#!/usr/bin/env python3
"""Continue the scientific-validation pipeline after provisional review labels exist.

This script encapsulates the next operational steps after a single-authorized
review pass has been created and audited.

It does not manufacture scientific truth. It simply stages the next required
pipeline steps under the project’s documented protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEWED_DIR = ROOT / "data" / "derived" / "reviewed_v1"
STATE_PATH = ROOT / "artifacts" / "scientific_validation" / "annotation_status.json"


def read_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def print_header() -> None:
    print("Scientific validation continuation")
    print("=" * 36)
    print("This stage assumes the provisional review pass has already been generated and audited.")


def print_next_steps() -> None:
    print("1) Freeze the reviewed validation labels and keep the untouched final test closed.")
    print("2) Review source/template provenance and grouped holdout integrity.")
    print("3) Retrain the gatekeeper on clean scientific splits.")
    print("4) Retrain behavior with the fixed protocol and multiple seeds.")
    print("5) Fit thresholds on reviewed validation data only.")
    print("6) Fit calibration on validation data only.")
    print("7) Run one final evaluation on the untouched test set.")
    print("8) Run the regression and runtime smoke checks.")


def maybe_run(cmd: list[str], dry_run: bool) -> int:
    cmd_text = " ".join(str(part) for part in cmd)
    print(f"$ {cmd_text}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the next commands without executing them")
    parser.add_argument("--run", action="store_true", help="Execute the next validation commands")
    parser.add_argument("--reviewed-dir", type=Path, default=REVIEWED_DIR, help="Reviewed label directory to use")
    args = parser.parse_args()

    print_header()
    print_next_steps()

    if not args.run and not args.dry_run:
        args.dry_run = True

    commands = [
        [
            "venv/bin/python",
            "scripts/training/trainer1.py",
            "--data-dir",
            "data/derived/scientific_v2/gatekeeper",
            "--stage-loss-weight",
            "0",
            "--seed",
            "42",
            "--output-dir",
            "models/experiments/my_study/gatekeeper_42",
        ],
        [
            "venv/bin/python",
            "scripts/training/train_behavior_encoder.py",
            "--data-dir",
            "data/derived/scientific_v2/behavior",
            "--input-format",
            "raw",
            "--seed",
            "42",
            "--epochs",
            "4",
            "--output",
            "models/experiments/my_study/behavior_raw_42/behavior_encoder.pt",
        ],
        [
            "venv/bin/python",
            "scripts/evaluation/calibrate_scores.py",
            "--validation",
            "data/derived/validation_predictions.json",
            "--test",
            "data/derived/test_predictions.json",
            "--output",
            "data/derived/gatekeeper_calibration.json",
        ],
        [
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "venv/bin/python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "scripts/evaluation",
            "-p",
            "test_*.py",
            "-v",
        ],
        [
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "venv/bin/python",
            "scripts/evaluation/smoke_runtime.py",
            "--api",
            "--output",
            "/tmp/runtime_smoke.json",
        ],
    ]

    print("\nNext commands to run:")
    for cmd in commands:
        rc = maybe_run(cmd, args.dry_run)
        if rc != 0 and args.run:
            print(f"Command failed with exit code {rc}: {' '.join(str(x) for x in cmd)}")
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
