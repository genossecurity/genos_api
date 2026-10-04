#!/usr/bin/env python3
"""Convert a filled CSV review sheet back into JSONL for the annotation audit tool.

This is the normal handoff between spreadsheet-based annotation and the
scientific-validation audit tool.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REQUIRED_FIELDS = {
    "id",
    "command",
    "family_id",
    "source_context",
    "observable_behavior",
    "authorization",
    "verdict",
    "mitre_codes",
    "rationale",
    "annotator_id",
}


def parse_list(value: str):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split(";") if part.strip()]


def clean_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "null", "None"}:
        return None
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV review sheet")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL review file")
    args = parser.parse_args()

    with args.csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required fields: {sorted(missing)}")

        rows = list(reader)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            obj = {
                "id": clean_value(row.get("id")),
                "command": clean_value(row.get("command")),
                "family_id": clean_value(row.get("family_id")),
                "source_context": clean_value(row.get("source_context")),
                "observable_behavior": parse_list(row.get("observable_behavior")),
                "authorization": clean_value(row.get("authorization")),
                "verdict": clean_value(row.get("verdict")),
                "mitre_codes": parse_list(row.get("mitre_codes")),
                "rationale": clean_value(row.get("rationale")),
                "annotator_id": clean_value(row.get("annotator_id")),
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Converted {len(rows)} rows from {args.csv} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
