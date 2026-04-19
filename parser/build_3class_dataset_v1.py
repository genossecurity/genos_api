#!/usr/bin/env python3
"""Relabel the gatekeeper (benign) and specialist (malicious) CSVs into a
unified 3-class dataset:  Benign | Malicious | Context_Dependent

Usage:
    python parser/build_3class_dataset.py [--dry-run]

Outputs (into data/training/genos_dataset/):
    gatekeeper_3class_train.csv
    gatekeeper_3class_val.csv
    gatekeeper_3class_test.csv

Format:  command,label,original_label,mitre_id
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "training", "genos_dataset")

# ═══════════════════════════════════════════════════════════════════════════════
# RULE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── MITRE techniques that are inherently context-dependent ──────────────────
# Discovery tactic
DISCOVERY_TECHNIQUES = {
    "T1007",  # System Service Discovery
    "T1010",  # Application Window Discovery
    "T1012",  # Query Registry
    "T1016",  # System Network Configuration Discovery
    "T1018",  # Remote System Discovery
    "T1033",  # System Owner/User Discovery
    "T1040",  # Network Sniffing
    "T1046",  # Network Service Discovery (nmap, etc.)
    "T1049",  # System Network Connections Discovery
    "T1057",  # Process Discovery
    "T1069",  # Permission Groups Discovery
    "T1082",  # System Information Discovery
    "T1083",  # File and Directory Discovery
    "T1087",  # Account Discovery
    "T1120",  # Peripheral Device Discovery
    "T1135",  # Network Share Discovery
    "T1201",  # Password Policy Discovery
    "T1217",  # Browser Information Discovery
    "T1518",  # Software Discovery
    "T1526",  # Cloud Service Discovery
    "T1580",  # Cloud Infrastructure Discovery
    "T1613",  # Container and Resource Discovery
    "T1614",  # System Location Discovery
    "T1615",  # Group Policy Discovery
    "T1619",  # Cloud Storage Object Discovery
    "T1622",  # Debugger Evasion
    "T1654",  # Log Enumeration
}

# Collection techniques that are really just "reading data"
COLLECTION_TECHNIQUES = {
    "T1005",  # Data from Local System (file reads)
}

CTX_TECHNIQUES = DISCOVERY_TECHNIQUES | COLLECTION_TECHNIQUES

# ── Malicious-indicator patterns: commands matching these STAY malicious ─────
# even if their MITRE technique is in CTX_TECHNIQUES
_MALICIOUS_INDICATORS = re.compile(
    r"|".join([
        # Reverse shell patterns
        r"/dev/tcp/",
        r"mkfifo\s.*/tmp/",
        r"\bexec\s+\d+<>/dev/",
        # Download-and-execute chains
        r"curl\s.*\|\s*(?:ba)?sh",
        r"wget\s.*\|\s*(?:ba)?sh",
        r"curl\s.*-o\s.*/tmp/.*&&.*chmod",
        r"wget\s.*-O\s.*/tmp/.*&&.*chmod",
        # Encoded payload execution
        r"base64\s+-d.*\|\s*(?:ba)?sh",
        r"eval\s.*\$\(",
        r"-enc\s+[A-Za-z0-9+/=]{20,}",
        # Known attacker domains/paths (placeholder patterns)
        r"evil\.com",
        r"malware\.",
        r"attacker\.",
        r"c2\.",
        # Credential dumping
        r"/etc/shadow",
        r"mimikatz",
        r"sekurlsa",
        r"hashdump",
        # Persistence
        r"crontab.*\|.*(?:ba)?sh",
        r"/etc/sudoers",
    ]),
    re.IGNORECASE,
)


def _has_malicious_indicators(cmd: str) -> bool:
    """Return True if the command contains strong attacker-intent signals."""
    return bool(_MALICIOUS_INDICATORS.search(cmd))


# ── Benign → Context_Dependent patterns ─────────────────────────────────────

# Sensitive system file paths (reading / statting these is ambiguous)
_SENSITIVE_FILES = re.compile(
    r"(?:/etc/passwd|/etc/group|/etc/hostname|/etc/os-release"
    r"|/etc/resolv\.conf|/etc/ssh/|/etc/crontab|/etc/fstab"
    r"|/etc/hosts(?:$|[^n]))",  # /etc/hosts but not /etc/hostname again
    re.IGNORECASE,
)

# Security-relevant log files
_SECURITY_LOGS = re.compile(
    r"/var/log/(?:auth|secure|audit|btmp|wtmp|faillog|lastlog)",
    re.IGNORECASE,
)

# Network enumeration commands (standalone or with flags)
_NET_ENUM = re.compile(
    r"^\s*(?:netstat|ifconfig|ip\s+(?:addr|route|neigh|link)\b)",
    re.IGNORECASE,
)

# Standalone system enumeration commands
_SYS_ENUM_STANDALONE = re.compile(
    r"^\s*(?:hostname|uname)\s*(?:$|-)",
    re.IGNORECASE,
)

# SUID / capability discovery
_SUID_DISCOVERY = re.compile(
    r"find\s+/\s.*-perm\s+-[24]000",
    re.IGNORECASE,
)

# Process / user enumeration
_USER_ENUM = re.compile(
    r"^\s*(?:who|finger|lastlog|getent\s+(?:passwd|group))\b",
    re.IGNORECASE,
)


def _benign_is_context_dependent(cmd: str) -> bool:
    """Return True if a currently-benign command should be context_dependent."""
    return bool(
        _SENSITIVE_FILES.search(cmd)
        or _SECURITY_LOGS.search(cmd)
        or _NET_ENUM.search(cmd)
        or _SYS_ENUM_STANDALONE.search(cmd)
        or _SUID_DISCOVERY.search(cmd)
        or _USER_ENUM.search(cmd)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def relabel_split(split: str, dry_run: bool = False):
    """Process one split (train / val / test).

    Returns (rows, stats) where rows is a list of
    (command, new_label, original_label, mitre_id) tuples.
    """
    gatekeeper_path = os.path.join(BASE, f"gatekeeper_{split}.csv")
    specialist_path = os.path.join(BASE, f"specialist_{split}.csv")

    stats = Counter()
    rows = []

    # ── Process benign commands ──────────────────────────────────────────
    with open(gatekeeper_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            cmd = row[0]
            if _benign_is_context_dependent(cmd):
                new_label = "Context_Dependent"
                stats["benign→ctx"] += 1
            else:
                new_label = "Benign"
                stats["benign→benign"] += 1
            rows.append((cmd, new_label, "Benign", "Benign"))

    # ── Process malicious commands ───────────────────────────────────────
    with open(specialist_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            cmd, mitre_id = row[0], row[1]

            if mitre_id in CTX_TECHNIQUES and not _has_malicious_indicators(cmd):
                new_label = "Context_Dependent"
                stats["malicious→ctx"] += 1
            else:
                new_label = "Malicious"
                stats["malicious→malicious"] += 1
            rows.append((cmd, new_label, "Malicious", mitre_id))

    # ── Write output ─────────────────────────────────────────────────────
    if not dry_run:
        out_path = os.path.join(BASE, f"gatekeeper_3class_{split}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["command", "label", "original_label", "mitre_id"])
            writer.writerows(rows)

    return rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    print("=" * 72)
    print("  GATEKEEPER 3-CLASS RELABELING")
    print("=" * 72)

    grand_stats = Counter()
    for split in ("train", "val", "test"):
        rows, stats = relabel_split(split, dry_run=args.dry_run)
        grand_stats += stats

        total = len(rows)
        benign = sum(1 for r in rows if r[1] == "Benign")
        malicious = sum(1 for r in rows if r[1] == "Malicious")
        ctx = sum(1 for r in rows if r[1] == "Context_Dependent")

        print(f"\n── {split.upper()} ({'dry run' if args.dry_run else 'written'}) ──")
        print(f"  Total:             {total:>6}")
        print(f"  Benign:            {benign:>6}  ({100*benign/total:.1f}%)")
        print(f"  Malicious:         {malicious:>6}  ({100*malicious/total:.1f}%)")
        print(f"  Context_Dependent: {ctx:>6}  ({100*ctx/total:.1f}%)")
        print(f"  ── Transitions ──")
        print(f"    benign→benign:       {stats['benign→benign']:>5}")
        print(f"    benign→ctx:          {stats['benign→ctx']:>5}")
        print(f"    malicious→malicious: {stats['malicious→malicious']:>5}")
        print(f"    malicious→ctx:       {stats['malicious→ctx']:>5}")

    # ── Grand totals ─────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  GRAND TOTALS")
    print(f"{'=' * 72}")
    for k in ("benign→benign", "benign→ctx", "malicious→malicious", "malicious→ctx"):
        print(f"  {k:30s} {grand_stats[k]:>6}")
    total_ctx = grand_stats["benign→ctx"] + grand_stats["malicious→ctx"]
    print(f"  {'total context_dependent':30s} {total_ctx:>6}")

    # ── Show example relabeled commands ──────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  SAMPLE RELABELED COMMANDS (from train)")
    print(f"{'=' * 72}")

    rows, _ = relabel_split("train", dry_run=True)

    benign_to_ctx = [(r[0], r[3]) for r in rows if r[1] == "Context_Dependent" and r[2] == "Benign"]
    mal_to_ctx = [(r[0], r[3]) for r in rows if r[1] == "Context_Dependent" and r[2] == "Malicious"]

    print(f"\n  Benign → Context_Dependent (showing up to 15):")
    for cmd, _ in benign_to_ctx[:15]:
        print(f"    {cmd[:90]}")

    print(f"\n  Malicious → Context_Dependent (showing up to 15):")
    for cmd, tid in mal_to_ctx[:15]:
        print(f"    [{tid}] {cmd[:80]}")

    # Show malicious commands that STAYED malicious despite being in a ctx technique
    stayed = [(r[0], r[3]) for r in rows
              if r[1] == "Malicious" and r[2] == "Malicious" and r[3] in CTX_TECHNIQUES]
    if stayed:
        print(f"\n  Kept as Malicious (ctx technique but malicious indicators) ({len(stayed)} total):")
        for cmd, tid in stayed[:10]:
            print(f"    [{tid}] {cmd[:80]}")

    if args.dry_run:
        print(f"\n  ** DRY RUN — no files written **")
    else:
        print(f"\n  Files written to: {os.path.abspath(BASE)}/gatekeeper_3class_*.csv")


if __name__ == "__main__":
    main()
