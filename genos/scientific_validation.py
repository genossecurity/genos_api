"""Shared provenance, representation, and evaluation checks (no model downloads)."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

PREPROCESSING_VERSION = "command-v2-mean-views"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_command(command):
    # Match the gatekeeper's actual input normalization; preserve internal spaces.
    return str(command).lower().strip()


def read_rows(path):
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        if path.suffix == ".csv":
            return list(csv.DictReader(handle))
        return [json.loads(line) for line in handle if line.strip()]


def command_of(row):
    command = row.get("command", row.get("raw_command"))
    if command is None or not str(command).strip():
        raise ValueError("Every row must have a nonempty command or raw_command")
    return str(command)


def representation(row, input_format):
    if input_format == "raw":
        return command_of(row)
    if input_format == "structured":
        return row["input_text"]
    raise ValueError(f"Unknown input format: {input_format}")


def split_audit(splits):
    commands, groups, conflicts = {}, {}, {}
    for name, rows in splits.items():
        commands[name] = defaultdict(set)
        groups[name] = set()
        for row in rows:
            key = normalize_command(command_of(row))
            label = row.get("label", row.get("stage_label"))
            if label is None:
                label = json.dumps(row.get("soft_target"), sort_keys=True)
            commands[name][key].add(str(label))
            # All supplied provenance group axes must be disjoint.
            for field in ("family_id", "source_group", "holdout_group"):
                if row.get(field):
                    groups[name].add((field, str(row[field])))
        conflicts[name] = sum(len(labels) > 1 for labels in commands[name].values())
    pairs = {}
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = commands[left].keys() & commands[right].keys()
            pairs[f"{left}:{right}"] = {
                "commands": len(overlap),
                "conflicting_commands": sum(commands[left][key] != commands[right][key] for key in overlap),
                "provenance_groups": len(groups[left] & groups[right]),
            }
    return {
        "normalization": "lowercase_strip",
        "counts": {name: len(rows) for name, rows in splits.items()},
        "within_split_label_conflicts": conflicts,
        "overlap": pairs,
        "passed": not any(conflicts.values()) and not any(v["commands"] or v["provenance_groups"] for v in pairs.values()),
    }


def require_disjoint(splits):
    report = split_audit(splits)
    if not report["passed"]:
        raise ValueError("Dataset independence check failed: " + json.dumps(report, sort_keys=True))
    return report


def dataset_manifest(paths):
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def probability_metrics(probabilities, targets):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(targets, dtype=int)
    if p.ndim != 2 or len(p) != len(y) or not len(y):
        raise ValueError("Expected nonempty N-by-C probabilities and N targets")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Scores must be finite normalized probability estimates")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("Target outside class map")
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = pred == y
    ece, bins = 0.0, []
    for i in range(10):
        mask = (conf >= i / 10) & ((conf < (i + 1) / 10) if i < 9 else (conf <= 1))
        n = int(mask.sum())
        if n:
            acc, confidence = float(correct[mask].mean()), float(conf[mask].mean())
            ece += n / len(y) * abs(acc - confidence)
            bins.append({"lower": i / 10, "upper": (i + 1) / 10, "n": n, "accuracy": acc, "confidence": confidence})
    return {
        "n": len(y), "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "brier": float(((p - np.eye(p.shape[1])[y]) ** 2).sum(axis=1).mean()),
        "ece_10_equal_width": float(ece), "reliability_bins": bins,
    }


def temperature_scale(probabilities, temperature):
    import numpy as np
    if not 0 < float(temperature) < float("inf"):
        raise ValueError("Temperature must be positive and finite")
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Expected normalized probability estimates")
    logits = np.log(np.clip(p, 1e-12, 1)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)
