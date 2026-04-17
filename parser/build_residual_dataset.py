"""
build_residual_dataset.py — Transform specialist CSVs into pipeline-aligned
residual training data.

The specialist model currently learns syntax + structure + intent.
This script strips what the pipeline (parser + semantic features + rule engine)
already determines, leaving only the *residual* — the ambiguity the model must
learn to resolve.

Outputs JSONL files for three dataset variants:
  A) RAW + RESIDUAL + FEATURES  (recommended first)
  B) RESIDUAL + FEATURES        (semi-abstracted)
  C) RESIDUAL only              (fully abstracted)

Usage:
    cd parser/
    python3 build_residual_dataset.py [--out-dir ../data/training/genos_residual]
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parser import parse_command
from semantic_features import build_semantic_features
from rule_engine import build_rule_result
from residual_text import build_semantic_tags

# ─────────────────────────────────────────────────────────────────────────────
# Action vocabulary — maps executables/subcommands to abstract action tokens
# ─────────────────────────────────────────────────────────────────────────────

_ACTION_MAP: Dict[str, str] = {
    # Transfer / download
    "curl":     "TRANSFER",
    "wget":     "TRANSFER",
    "certutil": "TRANSFER",
    "bitsadmin": "TRANSFER",
    "invoke-webrequest": "TRANSFER",
    "invoke-restmethod":  "TRANSFER",
    "start-bitstransfer": "TRANSFER",
    "scp": "TRANSFER",
    "sftp": "TRANSFER",

    # Registry
    "reg":     "REGISTRY",
    "reg.exe": "REGISTRY",
    "set-itemproperty":  "REGISTRY",
    "new-itemproperty":  "REGISTRY",
    "remove-itemproperty": "REGISTRY",
    "get-itemproperty":  "REGISTRY",

    # Scheduled tasks
    "schtasks":    "SCHED_TASK",
    "schtasks.exe": "SCHED_TASK",
    "at":          "SCHED_TASK",
    "at.exe":      "SCHED_TASK",
    "crontab":     "SCHED_TASK",

    # Service control
    "sc":        "SERVICE",
    "sc.exe":    "SERVICE",
    "systemctl": "SERVICE",
    "service":   "SERVICE",
    "new-service": "SERVICE",
    "set-service": "SERVICE",

    # Process / execution
    "powershell":     "POWERSHELL",
    "powershell.exe": "POWERSHELL",
    "pwsh":           "POWERSHELL",
    "pwsh.exe":       "POWERSHELL",
    "cmd":     "CMD_SHELL",
    "cmd.exe": "CMD_SHELL",
    "bash":    "SHELL",
    "sh":      "SHELL",
    "zsh":     "SHELL",

    # WMI
    "wmic":    "WMI",
    "wmic.exe": "WMI",

    # Net commands
    "net":     "NET_CMD",
    "net.exe": "NET_CMD",
    "net1":    "NET_CMD",
    "net1.exe": "NET_CMD",
    "dsquery": "NET_CMD",

    # Discovery
    "whoami":   "IDENTITY_QUERY",
    "id":       "IDENTITY_QUERY",
    "ipconfig": "NET_QUERY",
    "ifconfig": "NET_QUERY",
    "netstat":  "NET_QUERY",
    "arp":      "NET_QUERY",
    "nslookup": "NET_QUERY",
    "systeminfo": "SYS_QUERY",
    "hostname":   "SYS_QUERY",
    "uname":      "SYS_QUERY",
    "tasklist":   "PROC_QUERY",
    "ps":         "PROC_QUERY",
    "get-process": "PROC_QUERY",
    "wmic":       "WMI",
    "dir":        "FILE_ENUM",
    "ls":         "FILE_ENUM",
    "find":       "FILE_ENUM",
    "get-childitem": "FILE_ENUM",

    # Credential access
    "mimikatz": "CRED_TOOL",
    "secretsdump": "CRED_TOOL",
    "procdump": "PROC_DUMP",

    # Impact
    "vssadmin":  "SHADOW_MGMT",
    "wbadmin":   "SHADOW_MGMT",
    "bcdedit":   "BOOT_CONFIG",

    # Permission
    "chmod":  "PERMISSION",
    "chown":  "PERMISSION",
    "icacls": "PERMISSION",
    "cacls":  "PERMISSION",
    "takeown": "PERMISSION",
    "attrib": "ATTRIB",

    # File operations
    "copy":  "FILE_COPY",
    "xcopy": "FILE_COPY",
    "robocopy": "FILE_COPY",
    "cp":    "FILE_COPY",
    "move":  "FILE_MOVE",
    "mv":    "FILE_MOVE",
    "del":   "FILE_DELETE",
    "rm":    "FILE_DELETE",
    "erase": "FILE_DELETE",
    "type":  "FILE_READ",
    "cat":   "FILE_READ",
    "more":  "FILE_READ",

    # Archive
    "tar":     "ARCHIVE",
    "zip":     "ARCHIVE",
    "unzip":   "ARCHIVE",
    "gzip":    "ARCHIVE",
    "7z":      "ARCHIVE",
    "compress-archive":   "ARCHIVE",
    "expand-archive":     "ARCHIVE",

    # Evasion
    "rundll32":    "PROXY_EXEC",
    "rundll32.exe": "PROXY_EXEC",
    "regsvr32":    "PROXY_EXEC",
    "regsvr32.exe": "PROXY_EXEC",
    "mshta":       "PROXY_EXEC",
    "mshta.exe":   "PROXY_EXEC",
    "cscript":     "SCRIPT_HOST",
    "wscript":     "SCRIPT_HOST",
    "msiexec":     "INSTALLER_EXEC",
    "msiexec.exe": "INSTALLER_EXEC",
}

_SUBCOMMAND_MAP: Dict[Tuple[str, str], str] = {
    # reg subcommands
    ("reg", "add"):    "WRITE",
    ("reg", "delete"):  "DELETE",
    ("reg", "query"):   "QUERY",
    ("reg", "export"):  "EXPORT",
    ("reg", "import"):  "IMPORT",
    ("reg", "save"):    "EXPORT",
    ("reg", "load"):    "IMPORT",
    ("reg", "copy"):    "COPY",
    # sc subcommands
    ("sc", "create"):  "CREATE",
    ("sc", "config"):  "MODIFY",
    ("sc", "delete"):  "DELETE",
    ("sc", "query"):   "QUERY",
    ("sc", "start"):   "START",
    ("sc", "stop"):    "STOP",
    # net subcommands
    ("net", "user"):    "USER_MGMT",
    ("net", "group"):   "GROUP_QUERY",
    ("net", "localgroup"): "GROUP_QUERY",
    ("net", "share"):   "SHARE_QUERY",
    ("net", "view"):    "NET_ENUM",
    ("net", "use"):     "NET_CONNECT",
    ("net", "stop"):    "SVC_STOP",
    ("net", "start"):   "SVC_START",
    # schtasks subcommands
    ("schtasks", "create"):  "CREATE",
    ("schtasks", "delete"):  "DELETE",
    ("schtasks", "query"):   "QUERY",
    ("schtasks", "change"):  "MODIFY",
    ("schtasks", "/create"): "CREATE",
    ("schtasks", "/delete"): "DELETE",
    ("schtasks", "/query"):  "QUERY",
    ("schtasks", "/change"): "MODIFY",
}

# Registry path classification
_AUTORUN_PREFIXES = [
    r"currentversion\run",
    r"currentversion\runonce",
    r"currentversion\runonceex",
    r"currentversion\runservices",
    r"currentversion\policies\explorer\run",
    r"wow6432node\microsoft\windows\currentversion\run",
    r"session manager\bootexecute",
    r"currentversion\winlogon",
    r"environment\userinitmprlogonscript",
]

_SECURITY_REG_HINTS = [
    "defender", "firewall", "antivirus", "disableantispyware",
    "disablerealtimemonitoring", "tamperprotection", "securityhealth",
    "safeboot", "real-time protection", "spynet",
]

_CRED_REG_HINTS = [
    "sam", "security", "system", "ntds", "lsa", "credential",
    "wdigest", "tspkg", "livessp",
]


def _classify_registry_path(reg_path: str) -> str:
    """Classify a registry path into a semantic bucket."""
    rp = reg_path.lower().replace("/", "\\")
    if any(prefix in rp for prefix in _AUTORUN_PREFIXES):
        return "AUTORUN"
    if any(hint in rp for hint in _SECURITY_REG_HINTS):
        return "SECURITY"
    if any(hint in rp for hint in _CRED_REG_HINTS):
        return "CREDENTIAL"
    if "services\\" in rp:
        return "SERVICE_REG"
    if "classes\\" in rp and ("clsid" in rp or "progid" in rp or "shell" in rp):
        return "COM_HIJACK"
    if "policies\\" in rp:
        return "POLICY"
    return "GENERAL"


# ─────────────────────────────────────────────────────────────────────────────
# Residual builder
# ─────────────────────────────────────────────────────────────────────────────

# Patterns to strip from command text
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_URL_RE = re.compile(r"https?://\S+|ftp://\S+", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"\b(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|io|edu|gov|mil|info|biz|co|uk|de|ru|cn|xyz)\b",
    re.IGNORECASE,
)
_UNC_RE = re.compile(r"\\\\[^\s\\]+(?:\\[^\s\\]+)*")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]{4,}\b")
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_NUMERIC_RE = re.compile(r"\b\d{3,}\b")  # 3+ digit numbers
_GUID_RE = re.compile(
    r"\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}?"
)
_WIN_PATH_RE = re.compile(
    r"[A-Za-z]:\\(?:[^\s\\\"]+\\)*[^\s\\\"]*", re.IGNORECASE
)
_UNIX_PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}")
_FLAG_RE = re.compile(r"(?:^|\s)(?:--?[a-zA-Z][\w-]*|/[a-zA-Z][\w]*)")
_REG_PATH_RE = re.compile(
    r"(?:HKLM|HKCU|HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER|HKEY_CLASSES_ROOT|"
    r"HKCR|HKU|HKEY_USERS)(?:\\[^\s\"]+)+",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"\"[^\"]{20,}\"")  # long quoted strings


def build_residual(
    parsed: dict,
    features: dict,
    rules: dict,
) -> str:
    """
    Build a residual representation that strips deterministic syntax and keeps
    only intent-disambiguating signal.

    Removes: exact paths, URLs, IPs, flags, registry paths, long numeric/hex/
             base64 tokens, GUIDs, long quoted strings.
    Keeps:   action intent tokens, behavioral context, relationship tokens.
    """
    tokens = []

    exe = parsed.get("executable", "").lower()
    sub = (parsed.get("subcommand") or "").lower()

    # 1. Action token from executable
    action = _ACTION_MAP.get(exe)
    if action:
        tokens.append(action)

    # 2. Sub-action token
    sub_action = _SUBCOMMAND_MAP.get((exe, sub))
    if sub_action:
        tokens.append(sub_action)
    elif sub and not sub_action:
        # Keep the subcommand as-is if it's meaningful
        if sub not in {"exe", ""}:
            tokens.append(sub.upper())

    # 3. Registry path classification
    for rp in parsed.get("registry_paths", []):
        reg_type = _classify_registry_path(rp)
        tokens.append(f"REG_{reg_type}")

    # 4. Target type tokens (abstract, not exact values)
    if parsed.get("remote_targets") or parsed.get("urls"):
        tokens.append("REMOTE_TARGET")
    if parsed.get("ips"):
        tokens.append("IP_TARGET")
    if parsed.get("domains"):
        tokens.append("DOMAIN_TARGET")

    # 5. File extension tokens (keep the extension type, not the path)
    _EXT_MAP = {
        ".exe": "EXE_FILE", ".dll": "DLL_FILE", ".bat": "BAT_FILE",
        ".ps1": "PS1_FILE", ".vbs": "VBS_FILE", ".js": "JS_FILE",
        ".hta": "HTA_FILE", ".msi": "MSI_FILE", ".sys": "SYS_FILE",
        ".scr": "SCR_FILE", ".pif": "PIF_FILE", ".cmd": "CMD_FILE",
        ".lnk": "LNK_FILE", ".inf": "INF_FILE",
        ".py": "PY_FILE", ".sh": "SH_FILE", ".pl": "PL_FILE",
        ".zip": "ZIP_FILE", ".rar": "RAR_FILE", ".7z": "7Z_FILE",
        ".tar": "TAR_FILE", ".gz": "GZ_FILE",
        ".doc": "DOC_FILE", ".docx": "DOC_FILE", ".xls": "XLS_FILE",
        ".pdf": "PDF_FILE",
        ".sam": "CRED_FILE", ".dit": "CRED_FILE",
    }
    seen_ext = set()
    for fp in parsed.get("file_paths", []):
        fp_lower = fp.lower()
        for ext, tok in _EXT_MAP.items():
            if fp_lower.endswith(ext) and tok not in seen_ext:
                tokens.append(tok)
                seen_ext.add(tok)
                break

    # 6. Pipe / redirect / chain context
    if parsed.get("has_pipe"):
        tokens.append("PIPED")
    if parsed.get("has_redirect"):
        tokens.append("REDIRECT")
    if parsed.get("has_chain"):
        tokens.append("CHAINED")

    # 7. LOLBin indicator
    if parsed.get("lolbin_matches"):
        tokens.append("LOLBIN")

    # 8. Encoded / obfuscation markers
    if parsed.get("encoded_markers"):
        tokens.append("ENCODED")
    if parsed.get("obfuscation_markers"):
        tokens.append("OBFUSCATED")
    if parsed.get("deobfuscated_command"):
        tokens.append("DEOBFUSCATED")

    # 9. Inline execution
    if parsed.get("inline_code"):
        tokens.append("INLINE_EXEC")

    # 10. Interpreter markers
    for marker in parsed.get("interpreter_markers", []):
        tokens.append(f"INTERP_{marker.upper()}")

    # 11. Stripped command body — remove deterministic noise
    cmd_text = parsed.get("deobfuscated_command") or parsed.get("raw_command", "")

    # Strip in order: URLs, IPs, UNC, reg paths, win paths, unix paths,
    # GUIDs, hex, base64, long quoted, numeric
    stripped = cmd_text
    stripped = _URL_RE.sub(" <URL> ", stripped)
    stripped = _UNC_RE.sub(" <UNC> ", stripped)
    stripped = _REG_PATH_RE.sub(" <REGPATH> ", stripped)
    stripped = _WIN_PATH_RE.sub(" <PATH> ", stripped)
    stripped = _UNIX_PATH_RE.sub(" <PATH> ", stripped)
    stripped = _IP_RE.sub(" <IP> ", stripped)
    stripped = _DOMAIN_RE.sub(" <DOMAIN> ", stripped)
    stripped = _GUID_RE.sub(" <GUID> ", stripped)
    stripped = _HEX_RE.sub(" <HEX> ", stripped)
    stripped = _BASE64_RE.sub(" <B64> ", stripped)
    stripped = _QUOTED_RE.sub(" <QUOTED> ", stripped)
    stripped = _NUMERIC_RE.sub(" <NUM> ", stripped)

    # Collapse whitespace
    stripped = re.sub(r"\s+", " ", stripped).strip()

    # Remove flag tokens (but keep the stripped command structure)
    # We keep flags in the stripped text because they carry some intent signal
    # (e.g., /f = force, /create = action), but we've already extracted the
    # key action from _SUBCOMMAND_MAP above.

    # Build final residual: ACTION_TOKENS | STRIPPED_CMD
    residual_prefix = " ".join(tokens)
    if residual_prefix and stripped:
        return residual_prefix + " | " + stripped
    elif residual_prefix:
        return residual_prefix
    elif stripped:
        return stripped
    else:
        return "UNKNOWN_COMMAND"


# ─────────────────────────────────────────────────────────────────────────────
# Feature tag builder (uses semantic_features + rule metadata)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_tags(features: dict, rules: dict) -> List[str]:
    """
    Build structured feature tags from semantic features + rule metadata.
    Returns space-separated tag list for the FEATURES: field.
    """
    tags = []

    # Semantic feature tags (from residual_text.py canonical list)
    tags.extend(build_semantic_tags(features))

    # Rule-derived tags: which rule classes were nominated
    for cls in rules.get("candidate_classes", []):
        # Convert "defense_evasion:modify_registry" → "RULE:modify_registry"
        short = cls.split(":")[-1] if ":" in cls else cls
        tags.append(f"RULE:{short}")

    return tags


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> List[Tuple[str, str]]:
    """Load specialist CSV → list of (command, label)."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for r in reader:
            line = ",".join(r)
            idx = line.rfind(",")
            rows.append((line[:idx], line[idx + 1:]))
    return rows


def process_row(
    raw_cmd: str,
    label: str,
) -> dict:
    """
    Run full pipeline on a single row and produce the JSONL record.
    """
    parsed = parse_command(raw_cmd)
    features = build_semantic_features(parsed)
    rules = build_rule_result(parsed, features)

    residual = build_residual(parsed, features, rules)
    feature_tags = build_feature_tags(features, rules)

    # Dataset variant A: RAW + RESIDUAL + FEATURES
    parts_a = []
    parts_a.append(f"RAW: {raw_cmd}")
    parts_a.append(f"RESIDUAL: {residual}")
    if feature_tags:
        parts_a.append(f"FEATURES: {' '.join(feature_tags)}")
    input_a = "\n".join(parts_a)

    # Dataset variant B: RESIDUAL + FEATURES
    parts_b = [f"RESIDUAL: {residual}"]
    if feature_tags:
        parts_b.append(f"FEATURES: {' '.join(feature_tags)}")
    input_b = "\n".join(parts_b)

    # Dataset variant C: RESIDUAL only
    input_c = residual

    return {
        "label": label,
        "raw_command": raw_cmd,
        "residual": residual,
        "features": feature_tags,
        "rule_strength": rules.get("rule_strength", "none"),
        "fired_rules": rules.get("fired_rules", []),
        "input_a": input_a,
        "input_b": input_b,
        "input_c": input_c,
    }


def build_split(
    split_name: str,
    data_dir: str,
    out_dir: str,
) -> dict:
    """Process one split and write JSONL files. Returns stats dict."""
    csv_path = os.path.join(data_dir, f"{split_name}.csv")
    rows = load_csv(csv_path)

    records = []
    for i, (cmd, label) in enumerate(rows):
        if (i + 1) % 1000 == 0:
            print(f"  {split_name}: {i+1}/{len(rows)}...", file=sys.stderr)
        rec = process_row(cmd, label)
        records.append(rec)

    # Write 3 variant files
    for variant in ["a", "b", "c"]:
        out_path = os.path.join(out_dir, f"{split_name}_variant_{variant}.jsonl")
        with open(out_path, "w") as f:
            for rec in records:
                row = {
                    "input_text": rec[f"input_{variant}"],
                    "label": rec["label"],
                    "rule_strength": rec["rule_strength"],
                    "raw_command": rec["raw_command"],
                    "residual": rec["residual"],
                    "features": rec["features"],
                    "fired_rules": rec["fired_rules"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Compute stats
    stats = compute_stats(records, rows)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(records: List[dict], raw_rows: List[Tuple[str, str]]) -> dict:
    """Compute diagnostics for a processed split."""
    n = len(records)

    # Token length comparison (space-split approximation)
    raw_lengths = [len(cmd.split()) for cmd, _ in raw_rows]
    residual_lengths = [len(rec["residual"].split()) for rec in records]

    avg_raw = sum(raw_lengths) / n if n else 0
    avg_res = sum(residual_lengths) / n if n else 0
    reduction = (1 - avg_res / avg_raw) * 100 if avg_raw > 0 else 0

    # Vocabulary size
    raw_vocab = set()
    res_vocab = set()
    for cmd, _ in raw_rows:
        raw_vocab.update(cmd.lower().split())
    for rec in records:
        res_vocab.update(rec["residual"].lower().split())

    # Short residuals (<5 tokens)
    short_count = sum(1 for rec in records if len(rec["residual"].split()) < 5)

    # Distribution by rule_strength
    strength_dist = Counter(rec["rule_strength"] for rec in records)

    # Feature tag frequency
    tag_freq = Counter()
    for rec in records:
        for tag in rec["features"]:
            tag_freq[tag] += 1

    # Labels preserved check
    labels_match = all(
        rec["label"] == raw_rows[i][1]
        for i, rec in enumerate(records)
    )

    # Empty residuals
    empty_count = sum(1 for rec in records if not rec["residual"].strip())

    # Fired rules distribution
    rule_freq = Counter()
    for rec in records:
        for rule in rec["fired_rules"]:
            rule_freq[rule] += 1

    return {
        "n": n,
        "avg_raw_tokens": round(avg_raw, 1),
        "avg_residual_tokens": round(avg_res, 1),
        "token_reduction_pct": round(reduction, 1),
        "raw_vocab_size": len(raw_vocab),
        "residual_vocab_size": len(res_vocab),
        "short_residual_pct": round(short_count / n * 100, 1) if n else 0,
        "short_residual_count": short_count,
        "empty_residual_count": empty_count,
        "strength_distribution": dict(strength_dist),
        "labels_preserved": labels_match,
        "top_feature_tags": tag_freq.most_common(20),
        "top_fired_rules": rule_freq.most_common(15),
    }


def print_stats(split_name: str, stats: dict):
    """Pretty-print diagnostics for one split."""
    print(f"\n{'='*72}")
    print(f"  {split_name.upper()} — {stats['n']} rows")
    print(f"{'='*72}")
    print(f"  Token lengths:")
    print(f"    Raw average:      {stats['avg_raw_tokens']} tokens")
    print(f"    Residual average: {stats['avg_residual_tokens']} tokens")
    print(f"    Reduction:        {stats['token_reduction_pct']}%")
    print(f"  Vocabulary:")
    print(f"    Raw vocab:        {stats['raw_vocab_size']}")
    print(f"    Residual vocab:   {stats['residual_vocab_size']}")
    print(f"  Residual quality:")
    print(f"    Short (<5 tok):   {stats['short_residual_count']} ({stats['short_residual_pct']}%)")
    print(f"    Empty:            {stats['empty_residual_count']}")
    print(f"  Labels preserved:   {'YES' if stats['labels_preserved'] else 'NO ⚠️'}")
    print(f"  Strength distribution:")
    for k, v in sorted(stats["strength_distribution"].items()):
        print(f"    {k:8s}: {v:5d} ({v/stats['n']*100:.1f}%)")
    print(f"  Top feature tags:")
    for tag, count in stats["top_feature_tags"][:10]:
        print(f"    {tag:<35s} {count:5d}")
    print(f"  Top fired rules:")
    for rule, count in stats["top_fired_rules"][:10]:
        print(f"    {rule:<40s} {count:5d}")


def print_examples(split_name: str, data_dir: str, n_examples: int = 10):
    """Show example transformations."""
    csv_path = os.path.join(data_dir, f"{split_name}.csv")
    rows = load_csv(csv_path)

    print(f"\n{'='*72}")
    print(f"  EXAMPLE TRANSFORMATIONS ({split_name}, showing {n_examples})")
    print(f"{'='*72}")

    # Pick diverse examples: sample from different strength buckets
    examples = []
    for cmd, label in rows[:200]:
        rec = process_row(cmd, label)
        examples.append(rec)

    # Select diverse: some strong, some weak, some none
    by_strength = defaultdict(list)
    for ex in examples:
        by_strength[ex["rule_strength"]].append(ex)

    selected = []
    for strength in ["strong", "weak", "none"]:
        bucket = by_strength[strength]
        selected.extend(bucket[:min(4, len(bucket))])
    selected = selected[:n_examples]

    for i, rec in enumerate(selected):
        print(f"\n  [{i+1}] {rec['rule_strength'].upper()}")
        print(f"  RAW:      {rec['raw_command'][:90]}")
        print(f"  RESIDUAL: {rec['residual'][:90]}")
        print(f"  FEATURES: {' '.join(rec['features'][:8])}")
        print(f"  LABEL:    {rec['label']}")
        print(f"  RULES:    {rec['fired_rules']}")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(out_dir: str, splits: List[str]) -> bool:
    """Validate output JSONL files. Returns True if all checks pass."""
    print(f"\n{'='*72}")
    print("  SANITY CHECKS")
    print(f"{'='*72}")

    all_ok = True

    for split_name in splits:
        for variant in ["a", "b", "c"]:
            path = os.path.join(out_dir, f"{split_name}_variant_{variant}.jsonl")
            if not os.path.exists(path):
                print(f"  FAIL: Missing {path}")
                all_ok = False
                continue

            with open(path) as f:
                lines = f.readlines()

            n = len(lines)
            # Check parseable
            bad_json = 0
            empty_input = 0
            empty_label = 0
            labels = set()
            for line in lines:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad_json += 1
                    continue
                if not rec.get("input_text", "").strip():
                    empty_input += 1
                if not rec.get("label", "").strip():
                    empty_label += 1
                labels.add(rec.get("label", ""))

            checks = [
                ("rows", n > 0, f"{n} rows"),
                ("valid JSON", bad_json == 0, f"{bad_json} bad"),
                ("no empty inputs", empty_input == 0, f"{empty_input} empty"),
                ("no empty labels", empty_label == 0, f"{empty_label} empty"),
                ("multi-class", len(labels) > 1, f"{len(labels)} classes"),
            ]

            failed = [c for c in checks if not c[1]]
            status = "PASS" if not failed else "FAIL"
            print(f"  {status}: {split_name}_variant_{variant}.jsonl "
                  f"({n} rows, {len(labels)} classes)")
            if failed:
                for name, _, detail in failed:
                    print(f"    ⚠️  {name}: {detail}")
                all_ok = False

    # Cross-split leakage check
    for variant in ["a"]:
        train_path = os.path.join(out_dir, f"specialist_train_variant_{variant}.jsonl")
        val_path = os.path.join(out_dir, f"specialist_val_variant_{variant}.jsonl")
        test_path = os.path.join(out_dir, f"specialist_test_variant_{variant}.jsonl")

        if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            train_cmds = set()
            with open(train_path) as f:
                for line in f:
                    rec = json.loads(line)
                    train_cmds.add(rec["raw_command"])

            for check_name, check_path in [("val", val_path), ("test", test_path)]:
                overlap = 0
                with open(check_path) as f:
                    for line in f:
                        rec = json.loads(line)
                        if rec["raw_command"] in train_cmds:
                            overlap += 1
                if overlap > 0:
                    print(f"  ⚠️  LEAKAGE: {overlap} {check_name} commands found in train")
                    all_ok = False
                else:
                    print(f"  PASS: No leakage between train and {check_name}")

    # Determinism check: process same command twice
    test_cmd = "reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v test /d evil.exe"
    r1 = process_row(test_cmd, "T1547")
    r2 = process_row(test_cmd, "T1547")
    if r1["residual"] == r2["residual"] and r1["features"] == r2["features"]:
        print("  PASS: Deterministic transformation")
    else:
        print("  FAIL: Non-deterministic transformation!")
        all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build pipeline-aligned residual training datasets"
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "data", "training", "genos_residual"
        ),
    )
    ap.add_argument("--examples", type=int, default=10,
                    help="Number of example transformations to show")
    args = ap.parse_args()

    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "training", "genos_dataset"
    )
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    splits = ["specialist_train", "specialist_val", "specialist_test"]
    all_stats = {}

    # Process each split
    for split_name in splits:
        print(f"\nProcessing {split_name}...", file=sys.stderr)
        stats = build_split(split_name, data_dir, out_dir)
        all_stats[split_name] = stats
        print_stats(split_name, stats)

    # Show examples from val
    print_examples("specialist_val", data_dir, n_examples=args.examples)

    # Sanity checks
    ok = run_sanity_checks(out_dir, splits)

    # Summary recommendation
    print(f"\n{'='*72}")
    print("  RECOMMENDATION")
    print(f"{'='*72}")
    print("  Start training with: variant_a (RAW + RESIDUAL + FEATURES)")
    print("  Reason: preserves full signal while adding structured context")
    print("  After baseline: try variant_b to measure raw vs residual lift")
    print(f"  Output directory: {out_dir}")
    print(f"{'='*72}")

    if not ok:
        print("\n  ⚠️  Some sanity checks FAILED — review before training")
        sys.exit(1)
    else:
        print("\n  All sanity checks PASSED")


if __name__ == "__main__":
    main()
