#!/usr/bin/env python3
"""Measure local end-to-end scan latency; this is a runtime, not accuracy, test."""

import argparse
import base64
import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def summarize(samples):
    ordered = sorted(samples)
    return {"n": len(ordered), "median_ms": round(statistics.median(ordered), 3),
            "p95_ms": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3),
            "min_ms": round(ordered[0], 3), "max_ms": round(ordered[-1], 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.output.exists():
        parser.error("Use positive repeats, nonnegative warmups, and a new output path")

    import torch
    from genos.engine import GenosEngine

    engine = GenosEngine()
    command = "curl https://example.org/tool.sh -o /tmp/tool.sh"
    suspicious_command = "cat /etc/shadow"
    encoded = base64.b64encode(suspicious_command.encode()).decode()
    workloads = [
        ("benign_tier1", "whoami", False, False),
        ("url_tier1", command, False, False),
        ("url_with_iocs", command, False, True),
        ("suspicious_tier1", suspicious_command, False, False),
        ("suspicious_with_iocs", suspicious_command, False, True),
        ("suspicious_full", suspicious_command, True, True),
        ("encoded_tier1", encoded, False, False),
        ("encoded_full", encoded, True, True),
    ]
    results = {}
    for name, cmd, specialist, iocs in workloads:
        for _ in range(args.warmups):
            engine.scan(cmd, run_specialist=specialist, collect_iocs=iocs, use_baseline=False)
        samples = []
        specialist_runs = 0
        for _ in range(args.repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            response = engine.scan(cmd, run_specialist=specialist, collect_iocs=iocs, use_baseline=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - started) / 1e6)
            specialist_runs += bool(response.get("should_run_specialist"))
        results[name] = {**summarize(samples), "specialist_runs": specialist_runs,
                         "label": response["label"], "input_truncated": response.get("input_truncated")}
        if name in {"suspicious_full", "encoded_full"} and specialist_runs != args.repeats:
            raise AssertionError(f"{name} did not exercise specialist inference on every sample")

    report = {
        "scope": "In-process end-to-end GenosEngine.scan latency; excludes HTTP and accuracy validation",
        "device": engine.device.type, "gatekeeper_backend": engine.gatekeeper_backend,
        "specialist_mode": engine.specialist_mode, "view_policy": engine.view_policy,
        "max_tokens": engine.max_length, "repeats": args.repeats, "warmups": args.warmups,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "workloads": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
