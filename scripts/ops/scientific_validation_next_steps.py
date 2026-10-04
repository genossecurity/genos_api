#!/usr/bin/env python3
"""Summarize the remaining scientific-validation work before deployment.

This script reads the validation report and annotation status and prints a
prioritized checklist of the next actions required before making any accuracy
or deployment claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "scientific_validation"
ANNOTATION_STATUS = ARTIFACTS / "annotation_status.json"
REPORT = ARTIFACTS / "REPORT.md"
PROTOCOL = ARTIFACTS / "full_training_protocol.json"
RUNTIME_SMOKE = ARTIFACTS / "runtime_smoke.json"
DATA_ROOT = ROOT / "data" / "derived"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - simple guard
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def parse_annotation_status() -> List[Dict[str, Any]]:
    data = load_json(ANNOTATION_STATUS)
    return data.get("queues", [])


def summary_text() -> str:
    lines: List[str] = []
    lines.append("Scientific Validation Next Steps")
    lines.append("=" * 34)

    annotation_queues = parse_annotation_status()
    pending = [q for q in annotation_queues if q.get("status", "").startswith("awaiting")]
    completed = [q for q in annotation_queues if q.get("status", "") == "review_complete"]

    lines.append(f"Annotation queues: {len(annotation_queues)} total | {len(pending)} pending | {len(completed)} complete")
    if pending:
        lines.append("Immediate blocker: independent reviews are still required before deployment claims.")
    else:
        lines.append("No pending queue entries were detected; review workflow can proceed to adjudication and freeze.")

    lines.append("")
    lines.append("Priority order:")
    lines.append("  1. Obtain two independent reviews for every queued validation/test command.")
    lines.append("  2. Adjudicate disagreements with a third independent reviewer.")
    lines.append("  3. Freeze reviewed labels and use only reviewed validation data for threshold fitting.")
    lines.append("  4. Review source/template provenance and grouped holdout integrity.")
    lines.append("  5. Retrain gatekeeper and behavior models on clean scientific splits.")
    lines.append("  6. Fit calibration on validation data only, then evaluate once on the untouched final test.")
    lines.append("  7. Promote only after runtime smoke and infrastructure checks pass.")

    lines.append("")
    lines.append("Required artifacts / checks:")
    for path, label in [
        (REPORT, "Scientific validation report"),
        (PROTOCOL, "Fixed training protocol"),
        (RUNTIME_SMOKE, "Runtime smoke export"),
        (ROOT / "docs" / "scientific_validation.md", "Validation workflow"),
    ]:
        exists = "yes" if path.exists() else "no"
        lines.append(f"  - {label}: {exists}")

    lines.append("")
    lines.append("Representative commands:")
    lines.append("  - venv/bin/python scripts/evaluation/audit_annotations.py --review-a review_a.jsonl --review-b review_b.jsonl --adjudications adjudications.jsonl --output-dir data/derived/reviewed_v1")
    lines.append("  - venv/bin/python scripts/training/trainer1.py --data-dir data/derived/scientific_v2/gatekeeper --stage-loss-weight 0 --seed 42 --output-dir models/experiments/my_study/gatekeeper_42")
    lines.append("  - venv/bin/python scripts/training/train_behavior_encoder.py --data-dir data/derived/scientific_v2/behavior --input-format raw --seed 42 --epochs 4 --output models/experiments/my_study/behavior_raw_42/behavior_encoder.pt")
    lines.append("  - venv/bin/python scripts/evaluation/calibrate_scores.py --validation data/derived/validation_predictions.json --test data/derived/test_predictions.json --output data/derived/gatekeeper_calibration.json")
    lines.append("  - HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python -m unittest discover -s scripts/evaluation -p 'test_*.py' -v")

    return "\n".join(lines)


def summary_json() -> Dict[str, Any]:
    annotation_queues = parse_annotation_status()
    pending = [q for q in annotation_queues if q.get("status", "").startswith("awaiting")]
    review_dir = DATA_ROOT / "reviewed_v1"
    artifacts = {
        "report": REPORT.exists(),
        "protocol": PROTOCOL.exists(),
        "runtime_smoke": RUNTIME_SMOKE.exists(),
        "reviewed_data_dir": review_dir.exists(),
        "doc": (ROOT / "docs" / "scientific_validation.md").exists(),
    }

    return {
        "status": "blocked" if pending else "ready_for_adjudication",
        "pending_annotation_queues": len(pending),
        "total_annotation_queues": len(annotation_queues),
        "artifacts": artifacts,
        "next_actions": [
            "Obtain two independent reviews for every queued validation/test example.",
            "Adjudicate disagreements with a third independent reviewer.",
            "Freeze reviewed labels before any threshold selection.",
            "Review source/template provenance and grouped holdout integrity.",
            "Retrain gatekeeper and behavior models on clean scientific splits.",
            "Fit calibration on validation data only, then evaluate once on the untouched final test.",
            "Run runtime smoke and infrastructure checks before promotion.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of the human summary.")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(summary_json(), indent=2))
    else:
        print(summary_text())


if __name__ == "__main__":
    main()
