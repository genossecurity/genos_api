#!/usr/bin/env python3
"""Summarize trainer test benchmarks without selecting on test metrics."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scientific_validation import read_rows


def _candidate(path: Path, metadata: dict) -> dict | None:
    args = metadata.get("training_arguments") or {}
    if not args:
        return None
    if "label_names" in metadata and "val_metrics" in metadata and "test_metrics" in metadata:
        component = "gatekeeper"
        representation = "raw"
        model = "codebert"
        validation = metadata["val_metrics"]
        test = metadata["test_metrics"]
        primary_metric = "macro_f1"
    elif "input_format" in args and "metrics" in metadata:
        component = "mitre"
        representation = args["input_format"]
        model = args.get("model", path.stem.replace("specialist_tfidf_", ""))
        validation = metadata["metrics"].get("val", {})
        test = metadata["metrics"].get("test", {})
        primary_metric = "macro_f1"
    elif "stage_map" in metadata and "test_metrics" in metadata:
        component = "behavior"
        representation = args.get("input_format", metadata.get("input_format", "unknown"))
        model = "codebert_multitask"
        validation = metadata.get("val_metrics", {})
        test = metadata["test_metrics"]
        primary_metric = "stage_macro_f1"
    else:
        return None

    if not validation or not test:
        raise ValueError(f"Missing validation/test benchmark metrics in {path}")
    seed = args.get("seed", metadata.get("seed"))
    return {
        "component": component,
        "representation": representation,
        "model": model,
        "seed": seed,
        "validation": validation,
        "test": test,
        "selection_metric": primary_metric,
        "selection_score": validation.get(primary_metric),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "metadata_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "evaluation_scope": metadata.get("evaluation_scope", "unspecified"),
        "independence_audit_passed": bool((metadata.get("independence_audit") or {}).get("passed")),
        "prediction_dir": path.parent,
    }


def collect_candidates(run_dir: Path) -> list[dict]:
    candidates = []
    for path in sorted(run_dir.rglob("*.json")):
        if path.name == "benchmark.json":
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate = _candidate(path, metadata)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _public_candidate(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if key != "prediction_dir"}


def summarize(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("No completed trainer metadata found; train models before benchmarking")

    groups = defaultdict(list)
    for candidate in candidates:
        key = (candidate["component"], candidate["representation"], candidate["model"])
        groups[key].append(candidate)

    summaries = []
    best = {}
    for (component, representation, model), runs in sorted(groups.items()):
        metric = runs[0]["selection_metric"]
        scores = [run["selection_score"] for run in runs if run["selection_score"] is not None]
        if not scores:
            raise ValueError(f"Validation metric {metric} missing for {component}/{representation}/{model}")
        summaries.append({
            "component": component,
            "representation": representation,
            "model": model,
            "runs": len(runs),
            "seed_values": sorted(run["seed"] for run in runs if run["seed"] is not None),
            "validation_metric": metric,
            "validation_mean": statistics.fmean(scores),
            "validation_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "test_metrics_by_seed": [
                {"seed": run["seed"], **run["test"]} for run in sorted(runs, key=lambda item: item["seed"])
            ],
            "evaluation_scopes": sorted({run["evaluation_scope"] for run in runs}),
            "all_split_audits_passed": all(run["independence_audit_passed"] for run in runs),
        })
        chosen = max(runs, key=lambda run: float("-inf") if run["selection_score"] is None else run["selection_score"])
        prior = best.get(component)
        if prior is None or chosen["selection_score"] > prior["selection_score"]:
            best[component] = chosen

    paired = _paired_mitre_comparisons(candidates)
    return {
        "scope": "Grouped prepared test split; metrics inherit the dataset label basis and do not establish independent ground truth.",
        "selection_policy": "Best checkpoint/configuration is identified using validation metric only; test metrics are descriptive and never select a model.",
        "candidate_count": len(candidates),
        "summaries": summaries,
        "best_by_validation": {component: _public_candidate(candidate) for component, candidate in sorted(best.items())},
        "paired_mitre_representation_comparisons": paired,
    }


def _paired_mitre_comparisons(candidates: list[dict]) -> list[dict]:
    from scripts.evaluation.compare_ablations import compare

    indexed = {}
    for candidate in candidates:
        if candidate["component"] != "mitre":
            continue
        key = (candidate["model"], candidate["seed"], candidate["representation"])
        indexed[key] = candidate

    results = []
    for model, seed, representation in sorted(indexed):
        if representation != "raw":
            continue
        raw = indexed[(model, seed, "raw")]
        structured = indexed.get((model, seed, "structured"))
        if structured is None:
            continue
        left = raw["prediction_dir"] / f"{model}_test_predictions.jsonl"
        right = structured["prediction_dir"] / f"{model}_test_predictions.jsonl"
        if not left.exists() or not right.exists():
            continue
        result = compare(read_rows(left), read_rows(right))
        results.append({"model": model, "seed": seed, "comparison": "structured_minus_raw", **result})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to RUN_DIR/benchmark.json")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "benchmark.json").resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite benchmark evidence: {output}")
    report = summarize(collect_candidates(run_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "best_by_validation"}, indent=2))
    print(f"Saved benchmark report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
