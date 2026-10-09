#!/usr/bin/env python3
"""Audit grouped data, train model experiments, then benchmark completed runs.

Defaults to dry-run. Existing frozen datasets are audited and never rewritten;
use --phase prepare to build fresh datasets under the selected run directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREPARE_SCRIPT = ROOT / "scripts/data/prepare_scientific_splits.py"
GATEKEEPER_TRAINER = ROOT / "scripts/training/trainer1.py"
MITRE_TRAINER = ROOT / "scripts/training/trainer_tfidf.py"
BEHAVIOR_TRAINER = ROOT / "scripts/training/train_behavior_encoder.py"
BENCHMARK_SCRIPT = ROOT / "scripts/evaluation/benchmark_training_runs.py"
TASK_FILES = {
    "gatekeeper": {
        "train": "gatekeeper_3class_train.csv",
        "val": "gatekeeper_3class_val.csv",
        "test": "gatekeeper_3class_test.csv",
    },
    "mitre": {
        "train": "specialist_train_variant_a.jsonl",
        "val": "specialist_val_variant_a.jsonl",
        "test": "specialist_test_variant_a.jsonl",
    },
    "behavior": {
        "train": "behavior_train.jsonl",
        "val": "behavior_val.jsonl",
        "test": "behavior_test.jsonl",
    },
}
SOURCE_DIRS = {
    "gatekeeper": "genos_dataset",
    "mitre": "genos_residual_expanded",
    "behavior": "genos_behavior",
}
PHASES = ("audit", "prepare", "train", "benchmark", "all")


def parse_csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated value")
    return values


def audit_data(data_root: Path, gatekeeper_train_patch: Path | None = None) -> dict:
    from genos.scientific_validation import read_rows, require_disjoint, sha256_file

    report = {"data_root": str(data_root.resolve()), "tasks": {}}
    for task, filenames in TASK_FILES.items():
        task_dir = data_root / task
        manifest_path = task_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing frozen data manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("after", {}).get("passed"):
            raise ValueError(f"Manifest split audit failed for {task}: {manifest_path}")

        paths = {split: task_dir / name for split, name in filenames.items()}
        base_rows = {split: read_rows(path) for split, path in paths.items()}
        if any(not split_rows for split_rows in base_rows.values()):
            raise ValueError(f"Empty train/validation/test split for {task}")
        for split, path in paths.items():
            expected = manifest.get("outputs", {}).get(split, {}).get("sha256")
            if not expected or sha256_file(path) != expected:
                raise ValueError(f"Frozen output hash mismatch for {task}/{split}: {path}")
        rows = {split: list(split_rows) for split, split_rows in base_rows.items()}
        training_patch = None
        if task == "gatekeeper" and gatekeeper_train_patch is not None:
            patch_path = gatekeeper_train_patch.resolve()
            patch_rows = read_rows(patch_path)
            if not patch_rows:
                raise ValueError(f"Gatekeeper training patch is empty: {patch_path}")
            rows["train"].extend(patch_rows)
            training_patch = {
                "path": str(patch_path),
                "sha256": sha256_file(patch_path),
                "examples": len(patch_rows),
                "label_counts": dict(Counter(str(row.get("label", "<missing>")) for row in patch_rows)),
            }
        independence = require_disjoint(rows)
        if not independence.get("passed"):
            raise ValueError(f"Current command/group overlap found for {task}: {independence}")

        target_key = "stage_label" if task == "behavior" else "label"
        label_counts = {
            split: dict(Counter(str(row.get(target_key, "<missing>")) for row in split_rows))
            for split, split_rows in rows.items()
        }
        label_basis = sorted({
            str(row.get("label_basis", "legacy_unverified"))
            for split_rows in rows.values() for row in split_rows
        })
        report["tasks"][task] = {
            "counts": {split: len(split_rows) for split, split_rows in base_rows.items()},
            "effective_training_rows": len(rows["train"]),
            "label_counts": label_counts,
            "label_basis": label_basis,
            "quarantine": manifest.get("quarantine", {}),
            "independence_audit": independence,
            "manifest": str(manifest_path.resolve()),
        }
        if training_patch:
            report["tasks"][task]["training_patch"] = training_patch

    status_path = ROOT / "artifacts/scientific_validation/annotation_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        report["annotation_reviews"] = {
            "user_direction": status.get("user_direction"),
            "pending_queues": sum(
                queue.get("status") == "awaiting_two_independent_reviews"
                for queue in status.get("queues", [])
            ),
            "note": status.get("note"),
        }
    report["label_warning"] = (
        "Structural split checks do not correct weak labels. Review label_basis and quarantine; "
        "do not treat same-reviewer passes as independent ground truth."
    )
    return report


def persist_json(path: Path, value: dict) -> None:
    serialized = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def print_audit_summary(report: dict) -> None:
    summary = {
        "data_root": report["data_root"],
        "tasks": {
            task: {
                "counts": data["counts"],
                "effective_training_rows": data["effective_training_rows"],
                "label_basis": data["label_basis"],
                "quarantine": data["quarantine"],
                "disjoint": data["independence_audit"]["passed"],
                "training_patch": data.get("training_patch"),
            }
            for task, data in report["tasks"].items()
        },
        "annotation_reviews": report.get("annotation_reviews"),
        "label_warning": report["label_warning"],
    }
    print(json.dumps(summary, indent=2))


def prepare_commands(args: argparse.Namespace) -> tuple[Path, list[list[str]]]:
    prepared_root = args.run_dir.resolve() / "prepared_data"
    benchmark_path = args.benchmark_jsonl or ROOT / "data/derived/scientific_v1/development_benchmarks.jsonl"
    commands = []
    for task in TASK_FILES:
        output_dir = prepared_root / task
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite prepared split directory: {output_dir}")
        command = [
            sys.executable, str(PREPARE_SCRIPT), "--kind", task,
            "--output-dir", str(output_dir), "--seed", str(args.split_seed),
        ]
        if benchmark_path and benchmark_path.exists():
            command.extend(["--benchmark-jsonl", str(benchmark_path)])
        commands.append(command)
    return prepared_root, commands


def training_commands(args: argparse.Namespace, data_root: Path) -> list[tuple[list[str], Path, Path]]:
    commands = []
    seeds = [int(seed) for seed in args.seeds]
    representations = args.representations
    run_dir = args.run_dir.resolve()
    python = sys.executable

    if "gatekeeper" in args.components:
        for seed in seeds:
            output_dir = run_dir / "gatekeeper" / f"seed_{seed}"
            checkpoint = output_dir / "gatekeeper.pt"
            metadata = output_dir / "gatekeeper_meta.json"
            command = [
                python, str(GATEKEEPER_TRAINER), "--data-dir", str(data_root / "gatekeeper"),
                "--output-dir", str(output_dir), "--seed", str(seed),
                "--epochs", str(args.gatekeeper_epochs),
            ]
            if args.gatekeeper_train_patch_jsonl is not None:
                command.extend(["--train-patch-jsonl", str(args.gatekeeper_train_patch_jsonl.resolve())])
            commands.append((command, checkpoint, metadata))

    if "mitre" in args.components:
        mitre_models = ["rf", "char_rf"] if args.mitre_model == "both" else [args.mitre_model]
        for representation in representations:
            for seed in seeds:
                output_dir = run_dir / "mitre" / representation / f"seed_{seed}"
                for model in mitre_models:
                    checkpoint = output_dir / f"specialist_tfidf_{model}.pkl"
                    metadata = checkpoint.with_suffix(".json")
                    command = [
                        python, str(MITRE_TRAINER), "--data-dir", str(data_root / "mitre"),
                        "--output-dir", str(output_dir), "--input-format", representation,
                        "--model", model, "--seed", str(seed),
                        "--n-estimators", str(args.mitre_estimators),
                    ]
                    commands.append((command, checkpoint, metadata))

    if "behavior" in args.components:
        for representation in representations:
            for seed in seeds:
                checkpoint = run_dir / "behavior" / representation / f"seed_{seed}" / "behavior_encoder.pt"
                metadata = checkpoint.with_suffix(".json")
                command = [
                    python, str(BEHAVIOR_TRAINER), "--data-dir", str(data_root / "behavior"),
                    "--input-format", representation, "--seed", str(seed),
                    "--epochs", str(args.behavior_epochs), "--output", str(checkpoint),
                ]
                if args.amp:
                    command.append("--amp")
                commands.append((command, checkpoint, metadata))
    return commands


def execute(command: Sequence[str], cwd: Path, log_path: Path | None = None) -> int:
    print("$ " + shlex.join(str(part) for part in command), flush=True)
    if log_path is None:
        return subprocess.run(list(command), cwd=cwd, check=False).returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + shlex.join(str(part) for part in command) + "\n")
        log.flush()
        process = subprocess.Popen(
            list(command), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()


def run_prepare(args: argparse.Namespace, dry_run: bool) -> Path:
    prepared_root, commands = prepare_commands(args)
    if dry_run:
        for command in commands:
            print("$ " + shlex.join(command))
        return prepared_root
    for command in commands:
        status = execute(command, ROOT)
        if status:
            raise subprocess.CalledProcessError(status, command)
    report = audit_data(prepared_root, args.gatekeeper_train_patch_jsonl)
    persist_json(args.run_dir.resolve() / "data_audit.json", report)
    return prepared_root


def run_training(args: argparse.Namespace, data_root: Path, dry_run: bool) -> tuple[int, bool]:
    commands = training_commands(args, data_root)
    log_path = args.run_dir.resolve() / "training.log"
    completed = 0
    deadline = time.monotonic() + args.max_runtime_hours * 3600
    for job_number, (command, checkpoint, metadata) in enumerate(commands, start=1):
        if checkpoint.exists() and metadata.exists():
            completed += 1
            print(f"[train {job_number}/{len(commands)}] already complete: {checkpoint}", flush=True)
            continue
        if checkpoint.exists() or metadata.exists():
            completed_retry = None
            retry_number = 1
            while True:
                retry_dir = checkpoint.parent / f"retry_{retry_number}"
                retry_checkpoint = retry_dir / checkpoint.name
                retry_metadata = retry_dir / metadata.name
                if retry_checkpoint.exists() and retry_metadata.exists():
                    completed_retry = (retry_checkpoint, retry_metadata)
                    break
                if not retry_dir.exists():
                    break
                retry_number += 1
            if completed_retry:
                completed += 1
                print(
                    f"[train {job_number}/{len(commands)}] completed retry found: {completed_retry[0]}",
                    flush=True,
                )
                continue
            command = list(command)
            if "--output-dir" in command:
                output_flag = "--output-dir"
            elif "--output" in command:
                output_flag = "--output"
            else:
                raise RuntimeError(f"Cannot safely retry partial experiment command: {command}")
            output_index = command.index(output_flag) + 1
            command[output_index] = str(retry_dir)
            checkpoint, metadata = retry_checkpoint, retry_metadata
            print(
                f"[train {job_number}/{len(commands)}] preserving partial output; "
                f"restarting this model under {retry_dir}",
                flush=True,
            )
        if dry_run:
            print(f"[train {job_number}/{len(commands)}] $ " + shlex.join(command))
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"[train] Time budget of {args.max_runtime_hours:g} hours reached; "
                f"stopping before job {job_number}/{len(commands)}.",
                flush=True,
            )
            return completed, True
        print(
            f"\n[train {job_number}/{len(commands)}] starting {checkpoint} "
            f"(time budget remaining: {remaining / 3600:.2f}h)",
            flush=True,
        )
        status = execute(command, ROOT, log_path)
        if status:
            raise subprocess.CalledProcessError(status, command)
        if not checkpoint.is_file() or not metadata.is_file():
            raise RuntimeError(f"Trainer exited without expected artifacts: {checkpoint}, {metadata}")
        completed += 1
        print(f"[train {job_number}/{len(commands)}] completed: {checkpoint}", flush=True)
    return completed, False


def run_benchmark(args: argparse.Namespace, dry_run: bool) -> None:
    output = args.run_dir.resolve() / "benchmark.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark evidence: {output}")
    command = [sys.executable, str(BENCHMARK_SCRIPT), "--run-dir", str(args.run_dir.resolve())]
    if dry_run:
        print("$ " + shlex.join(command))
        return
    status = execute(command, ROOT)
    if status:
        raise subprocess.CalledProcessError(status, command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Unique output directory for this experiment")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/derived/scientific_v2")
    parser.add_argument("--prepare-data", action="store_true", help="Build fresh grouped splits before training")
    parser.add_argument(
        "--gatekeeper-train-patch-jsonl",
        type=Path,
        default=ROOT / "data/training/genos_dataset/gatekeeper_benign_core_patch_v2a.jsonl",
        help="Leakage-checked, training-only gatekeeper examples; defaults to the frozen benign core patch",
    )
    parser.add_argument("--no-gatekeeper-train-patch", action="store_true", help="Disable the default gatekeeper training patch")
    parser.add_argument("--benchmark-jsonl", type=Path, help="Additional development cases excluded when preparing splits")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--components", type=parse_csv_values, default=["gatekeeper", "mitre", "behavior"])
    parser.add_argument("--seeds", type=parse_csv_values, default=["42", "43", "44"])
    parser.add_argument("--representations", type=parse_csv_values, default=["raw", "structured"])
    parser.add_argument("--mitre-model", choices=["rf", "char_rf", "both"], default="char_rf")
    parser.add_argument("--mitre-estimators", type=int, default=400)
    parser.add_argument("--gatekeeper-epochs", type=int, default=5)
    parser.add_argument("--behavior-epochs", type=int, default=4)
    parser.add_argument(
        "--max-runtime-hours", type=float, default=6.0,
        help="Stop starting model jobs after this training wall-clock budget; finish the current model first",
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Enable CUDA AMP for behavior-model training (default; ignored on CPU)",
    )
    parser.add_argument("--run", action="store_true", help="Execute; otherwise print a dry-run command plan")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if set(args.components) - set(TASK_FILES):
        parser.error(f"Unknown components: {sorted(set(args.components) - set(TASK_FILES))}")
    if set(args.representations) - {"raw", "structured"}:
        parser.error("--representations accepts raw and/or structured")
    if args.mitre_estimators < 1 or args.gatekeeper_epochs < 1 or args.behavior_epochs < 1:
        parser.error("Epoch and estimator counts must be positive")
    if args.max_runtime_hours <= 0:
        parser.error("--max-runtime-hours must be positive")
    if args.no_gatekeeper_train_patch:
        args.gatekeeper_train_patch_jsonl = None
    dry_run = not args.run
    if args.phase == "prepare":
        run_prepare(args, dry_run)
        if not dry_run:
            print(f"Prepared and audited grouped datasets under {args.run_dir.resolve() / 'prepared_data'}")
        return 0
    if args.phase == "benchmark":
        run_benchmark(args, dry_run)
        return 0

    data_root = args.run_dir.resolve() / "prepared_data" if args.prepare_data else args.data_root.resolve()
    if args.phase in {"audit", "train", "all"}:
        if args.prepare_data:
            data_root = run_prepare(args, dry_run)
            if dry_run and args.phase == "audit":
                print(f"Prepared data would be written under {data_root}")
                return 0
        else:
            report = audit_data(data_root, args.gatekeeper_train_patch_jsonl)
            print_audit_summary(report)
            if not dry_run:
                persist_json(args.run_dir.resolve() / "data_audit.json", report)
        if args.phase == "audit":
            return 0

    if args.phase in {"train", "all"}:
        completed, stopped_for_budget = run_training(args, data_root, dry_run)
        if completed:
            run_benchmark(args, dry_run)
        elif dry_run:
            run_benchmark(args, dry_run)
        else:
            print("[train] No completed checkpoints to benchmark.", flush=True)
        if stopped_for_budget:
            print(f"[train] Partial run retained under {args.run_dir.resolve()}; rerun the same command to continue.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
