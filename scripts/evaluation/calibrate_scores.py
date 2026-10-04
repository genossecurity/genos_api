#!/usr/bin/env python3
"""Fit post-fusion temperature on validation exports; optionally assess test exports."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scientific_validation import probability_metrics, temperature_scale, sha256_file, require_disjoint


def fit_temperature(probabilities, targets):
    # Search only validation NLL. Include T=1 to avoid a worse numerical optimum.
    def loss(log_temperature):
        return probability_metrics(temperature_scale(probabilities, np.exp(log_temperature)), targets)["nll"]
    result = minimize_scalar(loss, bounds=(-4, 4), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fit failed")
    return float(np.exp(result.x)) if result.fun < loss(0) else 1.0


def fit_export(data):
    if data.get("split") != "validation" or not data.get("independence_audit", {}).get("passed"):
        raise ValueError("Calibration requires an audited validation export")
    if data.get("calibration_applied"):
        raise ValueError("Export uncalibrated scores before fitting")
    if not data.get("runtime_signature") or not data.get("training_manifest"):
        raise ValueError("Calibration requires runtime and training provenance")
    rows = data["rows"]
    probabilities = [r["probabilities"] for r in rows]
    targets = [r["target"] for r in rows]
    temperature = fit_temperature(probabilities, targets)
    return temperature, {"before": probability_metrics(probabilities, targets), "after": probability_metrics(temperature_scale(probabilities, temperature), targets)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--test", type=Path)
    p.add_argument("--existing", type=Path, help="Merge another calibrated component from the identical runtime")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.validation.read_text())
    temperature, metrics = fit_export(data)
    component = data["component"]
    if component not in {"gatekeeper", "mitre", "behavior"}:
        raise ValueError("Unsupported calibration component")
    artifact = {"fit_split": "validation", "method": "post_fusion_temperature", "runtime_signature": data["runtime_signature"], "temperatures": {component: temperature}, "validation_sha256": sha256_file(args.validation), "class_names": data["class_names"], "label_basis": data.get("label_basis", "unverified"), "validation_metrics": metrics}
    if args.test:
        test = json.loads(args.test.read_text())
        if test.get("split") != "test" or not test.get("independence_audit", {}).get("passed") or test.get("calibration_applied"):
            raise ValueError("Expected an audited, uncalibrated test export")
        if any(test.get(k) != data.get(k) for k in ("runtime_signature", "component", "class_names", "training_manifest")):
            raise ValueError("Validation and test provenance differ")
        require_disjoint({"validation": data["rows"], "test": test["rows"]})
        scores, targets = [r["probabilities"] for r in test["rows"]], [r["target"] for r in test["rows"]]
        artifact["test_metrics"] = {"before": probability_metrics(scores, targets), "after": probability_metrics(temperature_scale(scores, temperature), targets)}
        artifact["test_sha256"] = sha256_file(args.test)
    if args.existing:
        existing = json.loads(args.existing.read_text())
        if existing.get("runtime_signature") != artifact["runtime_signature"] or existing.get("fit_split") != "validation":
            raise ValueError("Cannot merge calibration from another runtime or fit split")
        if component in existing.get("temperatures", {}):
            raise ValueError("Component already fitted in existing artifact")
        artifact["temperatures"] = {**existing["temperatures"], **artifact["temperatures"]}
        artifact["previous_components"] = existing
    if args.output.exists():
        raise ValueError("Calibration artifacts are immutable; choose a new output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + '\n')
    print(json.dumps({"temperature": temperature, "validation": metrics}, indent=2))


if __name__ == '__main__':
    main()
