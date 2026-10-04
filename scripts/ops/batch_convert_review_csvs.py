#!/usr/bin/env python3
"""Batch-convert all generated review CSV templates to JSONL.

This script is intended for the human-review workflow after reviewers have
filled in the generated CSV sheets. It converts every template in the scientific
validation directories into a JSONL review file in the same folder.

Usage:
  python3 scripts/ops/batch_convert_review_csvs.py --base-dir data/derived/scientific_v2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "scripts" / "ops" / "convert_review_csv_to_jsonl.py"


def iter_templates(base_dir: Path):
    for csv_path in sorted(base_dir.glob("**/annotation_*_review_template.csv")):
        if "_review_template.csv" not in csv_path.name:
            continue
        yield csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("data/derived/scientific_v2"), help="Base scientific validation directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    templates = list(iter_templates(args.base_dir))
    if not templates:
        raise FileNotFoundError(f"No review templates found under {args.base_dir}")

    for csv_path in templates:
        out_path = csv_path.with_suffix(".jsonl")
        cmd = [
            sys.executable,
            str(CONVERTER),
            "--csv",
            str(csv_path),
            "--output",
            str(out_path),
        ]
        print("$ " + " ".join(str(part) for part in cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, cwd=str(ROOT), check=False)
            if result.returncode != 0:
                return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
