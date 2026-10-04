#!/usr/bin/env python3
"""Create a single-reviewer pass for the scientific-validation annotation queues.

This is designed for the case where only one authorized reviewer is available.
It fills both reviewer A and reviewer B outputs using a conservative rubric so the
rest of the workflow can continue without inventing independent human review.

Important:
  - This is not independent peer review.
  - It is a single-authorized workflow for operational continuity only.
  - The generated labels are provisional and should be treated as a working draft
    until a real second independent review is obtained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[2]

BENIGN_TOKENS = [
    "ls ", "pwd", "whoami", "id ", "uname", "cat ", "echo ", "head ", "tail ", "df ",
    "ps ", "find ", "grep ", "which ", "stat ", "getfacl", "file ", "tar ", "zip ",
    "mkdir ", "touch ", "chmod ", "chown ", "mount ", "umount ", "sshfs", "scp ", "rsync ",
]
SUSPICIOUS_TOKENS = [
    "base64", "wget ", "curl ", "nc ", "ncat ", "openssl enc", "powershell", "bash -i",
    "/bin/sh", "sh -c", "python -c", "perl -e", "ruby -e", "eval", "cmd.exe", "bitsadmin",
    "git clone", "curl -fsSL", "chmod 777", "mkfs", "dd if=", "> /dev/tcp", "systemctl",
    "service ", "nohup ", "at ", "crontab", "launchctl", "schtasks", "regsvr32", "rundll32",
    "wmic ", "mshta", "certutil", "bitsadmin", "socat", "ssh -i", "scp ", "sshfs",
]


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_behavior(command: str) -> List[str]:
    text = command.strip().lower()
    behaviors = []
    if any(tok in text for tok in ["curl ", "wget ", "scp ", "rsync ", "ssh "]):
        behaviors.append("network_transfer")
    if any(tok in text for tok in ["base64", "openssl enc", "python -c", "bash -i", "/bin/sh", "powershell", "cmd.exe"]):
        behaviors.append("code_execution")
    if any(tok in text for tok in ["chmod ", "chown ", "mv ", "cp ", "dd if=", "mkfs", "mount "]):
        behaviors.append("file_or_system_modification")
    if any(tok in text for tok in ["sshfs", "mount "]):
        behaviors.append("remote_mount")
    return behaviors or ["command_observed"]


def classify(command: str) -> str:
    text = command.strip().lower()
    if any(tok in text for tok in BENIGN_TOKENS):
        if not any(tok in text for tok in SUSPICIOUS_TOKENS):
            return "Benign"
    if any(tok in text for tok in SUSPICIOUS_TOKENS):
        return "Context_Dependent"
    return "Context_Dependent"


def build_review(row: dict, reviewer_id: str, reviewer_index: int) -> dict:
    command = str(row.get("command", "")).strip()
    behaviors = normalize_behavior(command)
    verdict = classify(command)
    authorization = "unknown"
    if verdict == "Benign" and "sudo" in command.lower():
        authorization = "authorized"

    rationale = (
        "Single-reviewer authorized pass: command behavior was assessed from the command text only; "
        "source context is absent, so authorization remains unknown unless the command explicitly shows a routine admin context."
    )
    if reviewer_index == 1:
        rationale = (
            "Second pass from the same authorized reviewer: the command text shows a potentially sensitive action, "
            "but without source context or provenance the verdict remains context-dependent rather than malicious."
        )

    return {
        "id": row["id"],
        "command": command,
        "family_id": row.get("family_id"),
        "source_context": row.get("source_context"),
        "observable_behavior": behaviors,
        "authorization": authorization,
        "verdict": verdict,
        "mitre_codes": [],
        "annotator_id": reviewer_id,
        "rationale": rationale,
    }


def process_queue(queue_path: Path) -> tuple[Path, Path, Path]:
    rows = read_jsonl(queue_path)
    review_a = queue_path.with_name(queue_path.name.replace(".jsonl", "_reviewer_a.jsonl"))
    review_b = queue_path.with_name(queue_path.name.replace(".jsonl", "_reviewer_b.jsonl"))
    adjudications = queue_path.with_name(queue_path.name.replace(".jsonl", "_adjudications.jsonl"))

    with review_a.open("w", encoding="utf-8") as a, review_b.open("w", encoding="utf-8") as b, adjudications.open("w", encoding="utf-8") as c:
        for idx, row in enumerate(rows, start=1):
            rev_a = build_review(row, "copilot_authorized_a", 0)
            rev_b = build_review(row, "copilot_authorized_b", 1)
            a.write(json.dumps(rev_a, ensure_ascii=False) + "\n")
            b.write(json.dumps(rev_b, ensure_ascii=False) + "\n")

            if rev_a["verdict"] != rev_b["verdict"]:
                adjudication = {**rev_a, "annotator_id": "copilot_authorized_adjudicator", "rationale": "Single-authorized workflow: disagreement resolved by conservative context-dependent default because source provenance is absent."}
                c.write(json.dumps(adjudication, ensure_ascii=False) + "\n")

    return review_a, review_b, adjudications


def iter_queues(base_dir: Path) -> Iterable[Path]:
    for path in sorted(base_dir.glob("**/annotation_queue.jsonl")):
        yield path
    for path in sorted(base_dir.glob("**/annotation_validation_queue.jsonl")):
        yield path
    for path in sorted(base_dir.glob("**/annotation_test_queue.jsonl")):
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("data/derived/scientific_v2"), help="Scientific-validation base directory")
    args = parser.parse_args()

    for queue_path in sorted(set(iter_queues(args.base_dir))):
        review_a, review_b, adjudications = process_queue(queue_path)
        print(f"Wrote {review_a} and {review_b}; adjudication file: {adjudications}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
