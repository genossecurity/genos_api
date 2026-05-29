#!/usr/bin/env python3
"""Build the Gatekeeper soft-label pilot dataset from fixed annotation policies.

The pilot is intentionally constructed from three evidence groups:
  - Routine_Operational (targets benign FP clusters from stress testing)
  - Needs_Context (dual-use and sensitive inspection commands)
  - Direct_Abuse (clear command-only abuse procedures)

Outputs deterministic train/val/test JSONL files plus a manifest JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.benchmark.tier1_case_sets import build_expanded_benign_commands, build_hard_negative_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(BASE_DIR / "data" / "training" / "genos_dataset"))
    return parser.parse_args()


def _soft_target(routine: float, needs_context: float, direct_abuse: float) -> dict[str, float]:
    total = routine + needs_context + direct_abuse
    if total <= 0:
        raise ValueError("Soft target must sum to a positive value")
    return {
        "Routine_Operational": round(routine / total, 6),
        "Needs_Context": round(needs_context / total, 6),
        "Direct_Abuse": round(direct_abuse / total, 6),
    }


ROUTINE_POLICIES = {
    "linux_admin": (_soft_target(0.85, 0.15, 0.00), "routine_linux_admin_inspection", "expanded_benign_stress"),
    "docker": (_soft_target(0.85, 0.15, 0.00), "routine_container_inspection", "expanded_benign_stress"),
    "kubernetes": (_soft_target(0.75, 0.25, 0.00), "routine_kubernetes_operations_with_context_relevance", "expanded_benign_stress"),
    "logs": (_soft_target(0.75, 0.25, 0.00), "routine_log_review_with_context_relevance", "expanded_benign_stress"),
    "networking_benign": (_soft_target(0.75, 0.25, 0.00), "routine_network_inspection", "expanded_benign_stress"),
    "filesystem_matrix": (_soft_target(0.85, 0.15, 0.00), "routine_filesystem_inventory", "expanded_benign_stress"),
    "linux_service_matrix": (_soft_target(0.85, 0.15, 0.00), "routine_service_inventory", "expanded_benign_stress"),
    "config_review": (_soft_target(0.75, 0.25, 0.00), "routine_config_review_with_context_relevance", "expanded_benign_stress"),
    "package_managers": (_soft_target(0.85, 0.15, 0.00), "routine_package_inventory", "expanded_benign_stress"),
}

NEEDS_CONTEXT_POLICIES = {
    "context_recon": (_soft_target(0.15, 0.80, 0.05), "dual_use_recon_without_telemetry", "hard_negative_context"),
    "context_sensitive_access": (_soft_target(0.15, 0.75, 0.10), "sensitive_access_without_direct_abuse", "hard_negative_context"),
    "cloud_cli": (_soft_target(0.20, 0.75, 0.05), "cloud_inventory_needs_context", "expanded_benign_contextual"),
    "database_admin": (_soft_target(0.20, 0.75, 0.05), "database_inventory_needs_context", "expanded_benign_contextual"),
    "existing_context_non_synthetic": (_soft_target(0.15, 0.80, 0.05), "existing_context_label_non_synthetic", "existing_3class_train_non_synthetic"),
}

DIRECT_ABUSE_POLICIES = {
    "malicious_execution": (_soft_target(0.00, 0.05, 0.95), "direct_remote_execution_abuse", "hard_negative_abuse"),
    "malicious_persistence": (_soft_target(0.00, 0.05, 0.95), "direct_persistence_abuse", "hard_negative_abuse"),
    "malicious_impact": (_soft_target(0.00, 0.05, 0.95), "direct_impact_abuse", "hard_negative_abuse"),
    "context_sensitive_access_malicious": (_soft_target(0.00, 0.05, 0.95), "credential_material_access_with_direct_abuse", "hard_negative_abuse"),
    "existing_malicious_non_synthetic": (_soft_target(0.00, 0.05, 0.95), "existing_malicious_label_non_synthetic", "existing_3class_train_non_synthetic"),
}


def load_training_rows() -> list[dict[str, str]]:
    csv_path = BASE_DIR / "data" / "training" / "genos_dataset" / "gatekeeper_3class_train.csv"
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_command(command: str) -> str:
    return " ".join(command.strip().split())


def _is_non_synthetic(row: dict[str, str]) -> bool:
    return row.get("original_label") != "Synthetic"


def _looks_like_real_command(command: str) -> bool:
    command = command.strip()
    return bool(command) and len(command) >= 8 and (" " in command or "/" in command or "\\" in command)


def _malicious_family(command: str) -> str | None:
    lower = command.lower()
    if any(token in lower for token in ["/dev/tcp/", "nc -e", "socket.socket", "fsockopen", "tcpsocket.open", "socat exec", "/bin/sh -i"]):
        return "reverse_shell_family"
    if (("curl" in lower or "wget" in lower or "invoke-webrequest" in lower or "bitsadmin" in lower or "certutil" in lower)
            and ("| bash" in lower or "| sh" in lower or " && /tmp/" in lower or "payload.exe" in lower or "downloadstring" in lower)):
        return "remote_payload_execution_family"
    if any(token in lower for token in ["authorized_keys", "sudoers", " crontab ", "/etc/cron", "currentversion\\run", "schtasks ", "startup", "rc.local", "launchagents", "systemctl enable malicious", "useradd -o -u 0"]):
        return "persistence_family"
    if any(token in lower for token in ["rm -rf /", "dd if=/dev/zero", "mkfs.", "delete shadows", "delete catalog", "cipher /w:", "wevtutil cl", "disable auditd", "disable realtimemonitoring", "ufw disable", "setenforce 0", "shred -u", "truncate -s 0"]):
        return "impact_family"
    if (("curl" in lower or "wget" in lower) and any(token in lower for token in ["@/etc/", "@/root/.ssh", "@/var/log/"])):
        return "sensitive_exfil_family"
    return None


def build_rows(seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = random.Random(seed)
    hard_negative = build_hard_negative_cases()
    expanded_benign = build_expanded_benign_commands()
    training_rows = load_training_rows()

    by_hn_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hard_negative:
        by_hn_bucket[row["bucket"]].append(row)

    by_benign_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in expanded_benign:
        by_benign_bucket[row["bucket"]].append(row)

    selected: list[dict[str, object]] = []
    seen_commands: set[str] = set()
    source_counts: Counter[str] = Counter()
    dominant_counts: Counter[str] = Counter()

    def add_row(command: str, soft_target: dict[str, float], label_basis: str, source_type: str) -> bool:
        normalized = _normalize_command(command)
        if normalized in seen_commands:
            return False
        seen_commands.add(normalized)
        selected.append(
            {
                "command": normalized,
                "soft_target": soft_target,
                "label_basis": label_basis,
                "source_type": source_type,
            }
        )
        dominant = max(soft_target, key=soft_target.get)
        source_counts[source_type] += 1
        dominant_counts[dominant] += 1
        return True

    routine_plan = [
        ("linux_admin", 70),
        ("docker", 45),
        ("kubernetes", 40),
        ("logs", 25),
        ("networking_benign", 25),
        ("filesystem_matrix", 35),
        ("linux_service_matrix", 20),
        ("config_review", 20),
        ("package_managers", 20),
    ]
    for bucket, count in routine_plan:
        soft_target, label_basis, source_type = ROUTINE_POLICIES[bucket]
        for row in by_benign_bucket[bucket][:count]:
            add_row(row["command"], soft_target, label_basis, source_type)

    needs_context_plan = [
        ("context_recon", 40),
        ("context_sensitive_access", 32),
        ("cloud_cli", 22),
        ("database_admin", 18),
    ]
    for bucket, count in needs_context_plan:
        soft_target, label_basis, source_type = NEEDS_CONTEXT_POLICIES[bucket]
        source_rows = by_hn_bucket[bucket] if bucket in by_hn_bucket else by_benign_bucket[bucket]
        added = 0
        for row in source_rows:
            if bucket == "context_sensitive_access" and row["label"] != "Context_Dependent":
                continue
            if add_row(row["command"], soft_target, label_basis, source_type):
                added += 1
            if added == count:
                break

    direct_abuse_buckets = [
        ("malicious_execution", lambda row: row["label"] == "Malicious"),
        ("malicious_persistence", lambda row: row["label"] == "Malicious"),
        ("malicious_impact", lambda row: row["label"] == "Malicious"),
        ("context_sensitive_access", lambda row: row["label"] == "Malicious"),
    ]
    for bucket, predicate in direct_abuse_buckets:
        policy_key = bucket if bucket != "context_sensitive_access" else "context_sensitive_access_malicious"
        soft_target, label_basis, source_type = DIRECT_ABUSE_POLICIES[policy_key]
        for row in by_hn_bucket[bucket]:
            if predicate(row):
                add_row(row["command"], soft_target, label_basis, source_type)

    # Top off needs-context from existing non-synthetic context rows.
    soft_target, label_basis, source_type = NEEDS_CONTEXT_POLICIES["existing_context_non_synthetic"]
    context_fallback = [
        row for row in training_rows
        if row["label"] == "Context_Dependent" and _is_non_synthetic(row) and _looks_like_real_command(row["command"])
    ]
    rng.shuffle(context_fallback)
    while dominant_counts["Needs_Context"] < 200:
        if not context_fallback:
            raise RuntimeError("Ran out of non-synthetic context rows while building the pilot")
        row = context_fallback.pop()
        add_row(row["command"], soft_target, label_basis, source_type)

    # Top off direct-abuse from existing non-synthetic malicious rows with clear direct-abuse families.
    soft_target, base_label_basis, source_type = DIRECT_ABUSE_POLICIES["existing_malicious_non_synthetic"]
    malicious_fallback = []
    for row in training_rows:
        if row["label"] != "Malicious" or not _is_non_synthetic(row) or not _looks_like_real_command(row["command"]):
            continue
        family = _malicious_family(row["command"])
        if family is None:
            continue
        malicious_fallback.append((row, family))
    rng.shuffle(malicious_fallback)
    while dominant_counts["Direct_Abuse"] < 200:
        if not malicious_fallback:
            raise RuntimeError("Ran out of non-synthetic direct-abuse rows while building the pilot")
        row, family = malicious_fallback.pop()
        add_row(row["command"], soft_target, f"{base_label_basis}:{family}", source_type)

    if len(selected) != 700:
        raise RuntimeError(f"Expected 700 pilot rows, built {len(selected)}")

    manifest = {
        "seed": seed,
        "rows": len(selected),
        "dominant_evidence": dict(dominant_counts),
        "source_type_counts": dict(source_counts),
        "bucket_counts": {
            "routine_selected": {bucket: count for bucket, count in routine_plan},
            "needs_context_fixed": {bucket: count for bucket, count in needs_context_plan},
            "direct_abuse_curated": {
                "malicious_execution": len(by_hn_bucket["malicious_execution"]),
                "malicious_persistence": len(by_hn_bucket["malicious_persistence"]),
                "malicious_impact": sum(1 for row in by_hn_bucket["malicious_impact"] if row["label"] == "Malicious"),
                "context_sensitive_access_malicious": sum(1 for row in by_hn_bucket["context_sensitive_access"] if row["label"] == "Malicious"),
            },
        },
    }
    return selected, manifest


def stratified_split(rows: list[dict[str, object]], seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        dominant = max(row["soft_target"], key=row["soft_target"].get)
        groups[dominant].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for _, group_rows in groups.items():
        rng.shuffle(group_rows)
        n = len(group_rows)
        train_cut = int(round(n * 0.8))
        val_cut = int(round(n * 0.9))
        train_rows.extend(group_rows[:train_cut])
        val_rows.extend(group_rows[train_cut:val_cut])
        test_rows.extend(group_rows[val_cut:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize_split(rows: list[dict[str, object]]) -> dict[str, object]:
    source_types = Counter(row["source_type"] for row in rows)
    dominant = Counter(max(row["soft_target"], key=row["soft_target"].get) for row in rows)
    return {
        "rows": len(rows),
        "source_types": dict(source_types),
        "dominant_evidence": dict(dominant),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, manifest = build_rows(args.seed)
    train_rows, val_rows, test_rows = stratified_split(rows, args.seed)

    train_path = output_dir / "gatekeeper_soft_label_pilot_train.jsonl"
    val_path = output_dir / "gatekeeper_soft_label_pilot_val.jsonl"
    test_path = output_dir / "gatekeeper_soft_label_pilot_test.jsonl"
    manifest_path = output_dir / "gatekeeper_soft_label_pilot_manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)

    manifest.update(
        {
            "train": summarize_split(train_rows),
            "val": summarize_split(val_rows),
            "test": summarize_split(test_rows),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()