#!/usr/bin/env python3
"""Orchestrate the scientific-validation workflow.

Phases:
  - review: explains what must be completed before model selection
  - audit: runs the annotation audit if review files are available
  - freeze: prepares the reviewed freeze directory and emits metadata guidance
  - train: prints the clean-training commands
  - calibrate: prints the validation-only calibration command
  - verify: prints the runtime/regression checks

Use --dry-run to print commands without executing them.
Use --run to execute the command(s) for the selected phase.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = ROOT / "artifacts" / "scientific_validation" / "annotation_status.json"
AUDIT_SCRIPT = ROOT / "scripts" / "evaluation" / "audit_annotations.py"
TRAINER1 = ROOT / "scripts" / "training" / "trainer1.py"
BEHAVIOR_TRAINER = ROOT / "scripts" / "training" / "train_behavior_encoder.py"
CALIBRATE_SCRIPT = ROOT / "scripts" / "evaluation" / "calibrate_scores.py"
UNITTEST_CMD = [
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
]
SMOKE_CMD = [
    "HF_HUB_OFFLINE=1",
    "TRANSFORMERS_OFFLINE=1",
    "venv/bin/python",
    "scripts/evaluation/smoke_runtime.py",
    "--api",
    "--output",
    "/tmp/runtime_smoke.json",
]

PHASES = ["review", "audit", "freeze", "train", "calibrate", "verify"]


def load_status() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def print_block(text: str) -> None:
    print(f"\n### {text}\n")


def print_header() -> None:
    print_block("Scientific validation orchestrator")
    print("Current status: review queues remain pending unless independent labels are available.")


def run_cmd(cmd: Sequence[str], cwd: Path | None = None) -> int:
    cmd_text = " ".join(str(part) for part in cmd)
    print(f"$ {cmd_text}")
    result = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=False)
    return result.returncode


def action_review(args: argparse.Namespace) -> int:
    # This is a guidance phase; it does not fabricate labels or run model work.
    status = load_status()
    pending = status.get("queues", [])
    print_block("Review phase")
    if pending:
        print(f"Detected {len(pending)} pending annotation queues.")
        print("Required next step: obtain two independent reviews for each queued split.")
        print("Do not move to threshold tuning or final evaluation until reviewed labels are frozen.")
        return 0
    print("No queued review entries were detected in the status file.")
    return 0


def action_audit(args: argparse.Namespace) -> int:
    review_a = args.review_a
    review_b = args.review_b
    adjudications = args.adjudications

    print_block("Audit phase")
    if not review_a or not review_b:
        print("Missing review input files.")
        print("Provide --review-a and --review-b, optional --adjudications")
        return 2

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "venv/bin/python",
        str(AUDIT_SCRIPT),
        "--review-a",
        str(review_a),
        "--review-b",
        str(review_b),
    ]
    if adjudications:
        cmd += ["--adjudications", str(adjudications)]
    cmd += ["--output-dir", str(out_dir)]

    if args.dry_run:
        print("Dry run: would execute the audit step.")
        print(" ".join(str(part) for part in cmd))
        return 0

    return run_cmd(cmd, cwd=ROOT)


def action_freeze(args: argparse.Namespace) -> int:
    print_block("Freeze phase")
    reviewed_dir = args.output_dir
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"Dry run: freeze reviewed labels under {reviewed_dir}")
        print("Expected outputs: accepted.jsonl, unresolved.jsonl, agreement.json")
        return 0

    print(f"Freeze directory prepared at {reviewed_dir}")
    print("After this, use only the reviewed validation data for threshold selection.")
    return 0


def action_train(args: argparse.Namespace) -> int:
    print_block("Train phase")
    commands = [
        [
            "venv/bin/python",
            str(TRAINER1),
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
            str(BEHAVIOR_TRAINER),
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
    ]
    if args.dry_run:
        for cmd in commands:
            print(" ".join(str(part) for part in cmd))
        return 0
    for cmd in commands:
        rc = run_cmd(cmd, cwd=ROOT)
        if rc != 0:
            return rc
    return 0


def action_calibrate(args: argparse.Namespace) -> int:
    print_block("Calibration phase")
    cmd = [
        "venv/bin/python",
        str(CALIBRATE_SCRIPT),
        "--validation",
        "data/derived/validation_predictions.json",
        "--test",
        "data/derived/test_predictions.json",
        "--output",
        "data/derived/gatekeeper_calibration.json",
    ]
    if args.dry_run:
        print(" ".join(str(part) for part in cmd))
        return 0
    return run_cmd(cmd, cwd=ROOT)


def action_verify(args: argparse.Namespace) -> int:
    print_block("Verification phase")
    for cmd in [UNITTEST_CMD, SMOKE_CMD]:
        if args.dry_run:
            print(" ".join(str(part) for part in cmd))
            continue
        rc = run_cmd(cmd, cwd=ROOT)
        if rc != 0:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automate the scientific validation workflow.")
    parser.add_argument("--phase", choices=PHASES, required=True, help="Which workflow phase to run")
    parser.add_argument("--run", action="store_true", help="Execute the command(s) instead of printing them")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--review-a", type=Path, help="Path to the first blind review file")
    parser.add_argument("--review-b", type=Path, help="Path to the second blind review file")
    parser.add_argument("--adjudications", type=Path, help="Optional adjudication file")
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/reviewed_v1"), help="Output directory for reviewed results")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print_header()

    if args.run and args.dry_run:
        parser.error("Choose either --run or --dry-run, not both.")

    if not args.run and not args.dry_run:
        # Default to dry-run for safety, since this is an orchestration workflow.
        args.dry_run = True

    phase = args.phase
    if phase == "review":
        return action_review(args)
    if phase == "audit":
        return action_audit(args)
    if phase == "freeze":
        return action_freeze(args)
    if phase == "train":
        return action_train(args)
    if phase == "calibrate":
        return action_calibrate(args)
    if phase == "verify":
        return action_verify(args)
    parser.error(f"Unsupported phase: {phase}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
