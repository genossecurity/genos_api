import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from engine import GenosEngine

DATA_DIR = BASE_DIR / "data" / "training" / "genos_residual_expanded"
OUTPUT_DIR = BASE_DIR / "data" / "training" / "genos_behavior"
CONFIG_DIR = BASE_DIR / "config"

TACTIC_TO_STAGE = {
    "Reconnaissance": "Discovery / Recon",
    "Discovery": "Discovery / Recon",
    "Execution": "Execution",
    "Persistence": "Persistence",
    "Privilege Escalation": "Privilege Escalation",
    "Defense Evasion": "Defense Evasion",
    "Credential Access": "Credential Access",
    "Collection": "Collection / Staging",
    "Command and Control": "C2 / Remote Access",
    "Lateral Movement": "Lateral Movement",
    "Exfiltration": "Exfiltration",
    "Impact": "Impact",
    "Initial Access": "Context Required",
}

TECHNIQUE_TO_STAGE = {
    # Keep a distinct download/retrieval stage only for explicit ingress transfer techniques.
    "T1105": "Payload Retrieval",
}

RULE_TO_STAGE = {
    "archive_collected_data": "Collection / Staging",
    "credential_access": "Credential Access",
    "credential_dumping": "Credential Access",
    "exfiltration": "Exfiltration",
    "ingress_tool_transfer": "Payload Retrieval",
}

FEATURE_TO_ACTION = {
    "ENCODED_PAYLOAD": "use_encoded_payload",
    "INLINE_EXEC": "execute_inline_code",
    "PIPE": "pipe_data",
    "ARCHIVE_CREATE": "archive_data",
    "ARCHIVE_EXTRACT": "extract_archive",
}

RULE_TO_ACTION = {
    "obfuscated_files_or_information": "use_obfuscation",
    "command_and_scripting_interpreter": "execute_interpreter",
    "signed_binary_proxy_execution": "use_signed_proxy_binary",
    "remote_services": "remote_execution",
    "ingress_tool_transfer": "download_remote_resource",
    "archive_collected_data": "archive_data",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def normalize_rule_name(value: str) -> str:
    value = value.replace("_rule_", "")
    return value.strip("_")


def derive_stage(mitre_id: str, features: list[str], fired_rules: list[str]) -> str:
    if mitre_id in TECHNIQUE_TO_STAGE:
        return TECHNIQUE_TO_STAGE[mitre_id]

    for feature in features:
        if not feature.startswith("RULE:"):
            continue
        stage = RULE_TO_STAGE.get(feature.split(":", 1)[1].lower())
        if stage:
            return stage

    for rule in fired_rules:
        stage = RULE_TO_STAGE.get(normalize_rule_name(rule))
        if stage:
            return stage

    tactic = GenosEngine._TECHNIQUE_TO_TACTIC.get(mitre_id)
    return TACTIC_TO_STAGE.get(tactic, "Context Required")


def derive_action_tags(features: list[str], fired_rules: list[str], input_text: str) -> list[str]:
    tags = set()

    for feature in features:
        if feature in FEATURE_TO_ACTION:
            tags.add(FEATURE_TO_ACTION[feature])
        if feature.startswith("RULE:"):
            mapped = RULE_TO_ACTION.get(feature.split(":", 1)[1].lower())
            if mapped:
                tags.add(mapped)

    for rule in fired_rules:
        mapped = RULE_TO_ACTION.get(normalize_rule_name(rule))
        if mapped:
            tags.add(mapped)

    return sorted(tags)


def convert_file(input_path: Path, output_path: Path, stage_counts: Counter, action_counts: Counter):
    rows = []
    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            features = list(row.get("features") or [])
            fired_rules = list(row.get("fired_rules") or [])
            tactic = GenosEngine._TECHNIQUE_TO_TACTIC.get(row["label"])
            stage = derive_stage(row["label"], features, fired_rules)
            action_tags = derive_action_tags(features, fired_rules, row["input_text"])

            output_row = {
                "input_text": row["input_text"],
                "raw_command": row.get("raw_command", ""),
                "stage_label": stage,
                "action_tags": action_tags,
                "source_mitre": row["label"],
                "source_tactic": tactic,
                "source_features": features,
                "source_rules": fired_rules,
                "rule_strength": row.get("rule_strength", "none"),
            }
            rows.append(output_row)
            stage_counts[stage] += 1
            for tag in action_tags:
                action_counts[tag] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage_counts = Counter()
    action_counts = Counter()
    split_names = ["train", "val", "test"]

    for split in split_names:
        input_path = args.input_dir / f"specialist_{split}_variant_a.jsonl"
        output_path = args.output_dir / f"behavior_{split}.jsonl"
        convert_file(input_path, output_path, stage_counts, action_counts)
        print(f"[+] Wrote {output_path}")

    stage_map = {label: index for index, label in enumerate(sorted(stage_counts))}
    action_map = {label: index for index, label in enumerate(sorted(action_counts))}

    (CONFIG_DIR / "behavior_stage_map.json").write_text(json.dumps(stage_map, indent=2), encoding="utf-8")
    (CONFIG_DIR / "behavior_action_map.json").write_text(json.dumps(action_map, indent=2), encoding="utf-8")
    summary = {
        "stage_counts": dict(stage_counts),
        "action_tag_counts": dict(action_counts),
        "n_stages": len(stage_map),
        "n_action_tags": len(action_map),
    }
    (args.output_dir / "behavior_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[+] Wrote {(CONFIG_DIR / 'behavior_stage_map.json')}")
    print(f"[+] Wrote {(CONFIG_DIR / 'behavior_action_map.json')}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
