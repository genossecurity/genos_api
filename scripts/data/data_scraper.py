#!/usr/bin/env python3
"""
GENOS dataset builder.

Scrapes benign and malicious shell command datasets from verified public sources,
normalizes them, dedupes them, splits 80/10/10, and writes CSVs compatible with
the existing gatekeeper/specialist pipeline.

Sources
-------
Benign (Gatekeeper):
  - NL2Bash (TellinaTool/nl2bash)          ~10K   raw .cm file
  - tldr-pages (tldr-pages/tldr)            ~25K   tarball + markdown parser
  - SLP nl2bash copy (dtrizna/slp)          ~10K   raw .cm file (normalized)
  - (optional) MPSD benign PowerShell corpus (das-lab/mpsd) - flagged off by default

Malicious (Specialist):
  - Atomic Red Team (redcanaryco/atomic-red-team)  ~5-8K  tarball + YAML parser
  - LOLBAS (LOLBAS-Project/LOLBAS)                 ~2K    tarball + YAML parser
  - GTFOBins (GTFOBins/GTFOBins.github.io)         ~2K    tarball + markdown parser
  - MITRE ATT&CK STIX (mitre/cti)                  ~5K    JSON, procedure examples

Usage
-----
  # first run (downloads everything):
  python build_dataset.py

  # re-run (uses cache, only re-downloads sources that are missing):
  python build_dataset.py

  # force re-download everything:
  python build_dataset.py --refresh

  # skip certain sources:
  python build_dataset.py --skip atomic_red_team,gtfobins

  # change output dir:
  python build_dataset.py --out ./my_dataset

Output
------
  <out>/gatekeeper_train.csv   80%
  <out>/gatekeeper_val.csv     10%
  <out>/gatekeeper_test.csv    10%
  <out>/specialist_train.csv   80%
  <out>/specialist_val.csv     10%
  <out>/specialist_test.csv    10%
  <out>/provenance.json        where each row came from (for auditing)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import urllib.request
import urllib.error

try:
    import yaml  # PyYAML; required for Atomic Red Team / LOLBAS
except ImportError:
    yaml = None


SEED = 42

# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

@dataclass
class SourceSpec:
    key: str
    kind: str                 # 'benign' | 'malicious'
    url: str
    fetch_type: str           # 'raw' | 'tarball'
    parser: str               # name of parser function
    note: str = ""


SOURCES: list[SourceSpec] = [
    # ---- BENIGN ----
    SourceSpec(
        key="nl2bash",
        kind="benign",
        url="https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm",
        fetch_type="raw",
        parser="parse_nl2bash",
        note="~10K bash commands, 102 utilities, MIT license",
    ),
    # Note: dtrizna/slp's nl2bash.cm is a near-duplicate of TellinaTool/nl2bash with IPs/domains
    # masked. Using both adds only ~60 unique rows after dedup, so we keep only the original.
    SourceSpec(
        key="tldr_pages",
        kind="benign",
        url="https://codeload.github.com/tldr-pages/tldr/tar.gz/refs/heads/main",
        fetch_type="tarball",
        parser="parse_tldr_pages",
        note="~4K binaries with ~5-7 real invocations each, CC-BY-4.0",
    ),

    # ---- MALICIOUS ----
    SourceSpec(
        key="atomic_red_team",
        kind="malicious",
        url="https://codeload.github.com/redcanaryco/atomic-red-team/tar.gz/refs/heads/master",
        fetch_type="tarball",
        parser="parse_atomic_red_team",
        note="MITRE-mapped atomic tests across ~290 techniques, MIT license",
    ),
    SourceSpec(
        key="lolbas",
        kind="malicious",
        url="https://codeload.github.com/LOLBAS-Project/LOLBAS/tar.gz/refs/heads/master",
        fetch_type="tarball",
        parser="parse_lolbas",
        note="Windows living-off-the-land binaries; attack_mitre_attack_ids in YAML",
    ),
    SourceSpec(
        key="gtfobins",
        kind="malicious",
        url="https://codeload.github.com/GTFOBins/GTFOBins.github.io/tar.gz/refs/heads/master",
        fetch_type="tarball",
        parser="parse_gtfobins",
        note="Unix living-off-the-land binaries; maps functions -> technique heuristically",
    ),
    SourceSpec(
        key="mitre_cti",
        kind="malicious",
        url="https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
        fetch_type="raw",
        parser="parse_mitre_cti",
        note="MITRE ATT&CK STIX; procedure examples from known threat actor reports",
    ),
]


# ---------------------------------------------------------------------------
# Download helpers w/ caching
# ---------------------------------------------------------------------------

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, refresh: bool = False) -> Path:
    """Download url to dest, skipping if already cached. Streams to avoid RAM blowup."""
    if dest.exists() and not refresh:
        return dest
    _safe_mkdir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "genos-dataset-builder/1.0 (+github.com)"},
    )
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 256)
    tmp.replace(dest)
    return dest


def _human(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024 ** 2: return f"{n/1024:.1f} KB"
    if n < 1024 ** 3: return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.1f} GB"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+)\}\}")          # tldr {{placeholder}}
_ART_PLACEHOLDER_RE = re.compile(r"#\{([^{}]+)\}")         # Atomic Red Team #{input}
_WHITESPACE_RE = re.compile(r"\s+")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\.){1,}[a-z]{2,}\b", re.I)


def _replace_tldr_placeholder(m: re.Match) -> str:
    """`{{path/to/file}}` -> `path/to/file` (last slash-part is what a user would actually type)."""
    inner = m.group(1).strip()
    # take the last path segment if it looks like a path
    if "/" in inner and " " not in inner:
        return inner
    return inner


def _replace_art_placeholder(m: re.Match) -> str:
    """`#{input_arg}` -> sensible placeholder (keep variable-y look so model doesn't memorize)."""
    name = m.group(1).strip().split(":")[0]
    return f"${name}"


def normalize_command(
    cmd: str,
    collapse_ws: bool = True,
    strip_placeholders: bool = True,
    mask_ips_domains: bool = False,
) -> str:
    """
    Normalize a command string. Returns "" if the command is unusable.
    We intentionally do NOT lowercase here — lowercasing happens downstream
    in the engine's tokenization pipeline.
    """
    if cmd is None:
        return ""
    s = cmd.strip()
    if not s:
        return ""

    if strip_placeholders:
        s = _PLACEHOLDER_RE.sub(_replace_tldr_placeholder, s)
        s = _ART_PLACEHOLDER_RE.sub(_replace_art_placeholder, s)

    # strip leading '$' or '#' prompt artifacts ("$ ls -la" -> "ls -la")
    s = re.sub(r"^\s*[\$#>]\s+", "", s)

    # strip leading 'PS> ' or 'PS C:\> ' PowerShell prompts
    s = re.sub(r"^\s*PS[^>]*>\s+", "", s)

    if mask_ips_domains:
        s = _IP_RE.sub("1.1.1.1", s)
        # be careful: don't mask "file.py" etc. Require at least one dot-separated TLD-ish.
        # (keep this off by default; enable if mixing with dtrizna/slp nl2bash)

    if collapse_ws:
        s = _WHITESPACE_RE.sub(" ", s).strip()

    # reject garbage
    if len(s) < 2:
        return ""
    if len(s) > 2048:
        return ""
    # reject anything that's clearly not a command (pure punctuation, pure numbers, etc.)
    if not re.search(r"[A-Za-z]", s):
        return ""
    # reject markdown table rows, bullet artifacts
    if s.startswith(("|", "- ", "* ", "> ")):
        return ""
    return s


def norm_key(cmd: str) -> str:
    """Normalized key for deduplication (case-insensitive, whitespace-collapsed)."""
    return _WHITESPACE_RE.sub(" ", cmd.lower().strip())


# ---------------------------------------------------------------------------
# Source parsers
# ---------------------------------------------------------------------------
# Each parser returns list[dict] with fields: command, mitre_id, source

# ---- BENIGN ----

def parse_nl2bash(raw_bytes: bytes) -> list[dict]:
    """One command per line."""
    rows = []
    text = raw_bytes.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        c = normalize_command(line, strip_placeholders=False)
        if c:
            rows.append({"command": c, "mitre_id": "Benign", "source": "nl2bash"})
    return rows


def parse_slp_nl2bash(raw_bytes: bytes) -> list[dict]:
    """Same format as NL2Bash. Source is already normalized (IPs/domains)."""
    rows = []
    text = raw_bytes.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        c = normalize_command(line, strip_placeholders=False)
        if c:
            rows.append({"command": c, "mitre_id": "Benign", "source": "slp_nl2bash"})
    return rows


def parse_tldr_pages(tarball_path: Path) -> list[dict]:
    """
    tldr-pages stores each command as a markdown file under pages/<platform>/<name>.md.
    Each example looks like:
        - Description text:
          `actual command --flags {{placeholder}}`
    We want only the content inside backticks.
    """
    rows = []
    # pattern: a line that is ONLY a backticked command (tldr's convention)
    cmd_line_re = re.compile(r"^`([^`]+)`\s*$")
    with tarfile.open(tarball_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            # path looks like tldr-main/pages/linux/ls.md or pages.xx/linux/ls.md
            parts = Path(member.name).parts
            if len(parts) < 3: continue
            if "pages" not in parts[1]:  # only pages/, not pages.xx/ translations
                continue
            if parts[1] != "pages":  # ignore non-English translations
                continue
            if not member.name.endswith(".md"):
                continue
            f = tf.extractfile(member)
            if f is None: continue
            text = f.read().decode("utf-8", errors="ignore")
            for line in text.splitlines():
                m = cmd_line_re.match(line.strip())
                if not m:
                    continue
                c = normalize_command(m.group(1))
                if c:
                    rows.append({"command": c, "mitre_id": "Benign", "source": "tldr_pages"})
    return rows


# ---- MALICIOUS ----

_ATOMIC_TECHNIQUE_RE = re.compile(r"atomic-red-team-[^/]+/atomics/(T\d+(?:\.\d+)?)/\1\.yaml$")
_VAR_ASSIGN_RE = re.compile(r"^\s*(\$[A-Za-z_]\w*|[A-Z][A-Z0-9_]*|export\s+[A-Za-z_]\w*)\s*=")
_SHELL_ESCAPE_ARTIFACT_RE = re.compile(r"^[!:\\]")  # things like "!/bin/sh", ":(){}" fragments

def _split_multiline_command(block: str) -> list[str]:
    """
    Atomic Red Team / GTFOBins / MITRE CTI 'command' blocks can be multiline scripts.
    We split them into individual candidate commands because the specialist classifies
    single commands. We also drop noise lines that aren't standalone commands:

    - comments (# ...)
    - variable assignments ($x = ..., FOO=bar, export FOO=...) — these aren't commands
      the model should learn on their own; they're context for the next line
    - shell escape artifacts (lines starting with !, :, \\) that come from YAML multiline
      continuation or here-doc fragments
    - pure operator lines (just `|`, `&&`, `||`, `;`)
    """
    out = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if _VAR_ASSIGN_RE.match(line):
            continue
        if _SHELL_ESCAPE_ARTIFACT_RE.match(line):
            continue
        # reject pure operator/punctuation lines
        if re.match(r"^[|&;<>{}()\[\]\s]+$", line):
            continue
        # reject lines that are just a closing brace/paren from a heredoc
        if line in {"EOF", "EOT", "END", "}", ")", "]", "'", '"'}:
            continue
        out.append(line)
    return out


def parse_atomic_red_team(tarball_path: Path) -> list[dict]:
    if yaml is None:
        raise RuntimeError("PyYAML required for Atomic Red Team parsing. pip install pyyaml")
    rows = []
    with tarfile.open(tarball_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".yaml"):
                continue
            m = _ATOMIC_TECHNIQUE_RE.search(member.name)
            if not m:
                continue
            technique_id = m.group(1).split(".")[0]  # T1234.001 -> T1234
            f = tf.extractfile(member)
            if f is None: continue
            try:
                data = yaml.safe_load(f.read())
            except yaml.YAMLError:
                continue
            if not isinstance(data, dict):
                continue
            atomics = data.get("atomic_tests") or []
            for atomic in atomics:
                executor = atomic.get("executor") or {}
                if not isinstance(executor, dict):
                    continue
                cmd_block = executor.get("command")
                if not cmd_block or not isinstance(cmd_block, str):
                    continue
                for raw_line in _split_multiline_command(cmd_block):
                    c = normalize_command(raw_line)
                    if c:
                        rows.append({
                            "command": c,
                            "mitre_id": technique_id,
                            "source": "atomic_red_team",
                        })
    return rows


def parse_lolbas(tarball_path: Path) -> list[dict]:
    """
    LOLBAS yml files live under yml/OSBinaries, yml/OSScripts, yml/OSLibraries, yml/OtherMSBinaries.
    Structure:
        Commands:
          - Command: "..."
            Description: ...
            MitreID: T1105
    """
    if yaml is None:
        raise RuntimeError("PyYAML required for LOLBAS parsing")
    rows = []
    with tarfile.open(tarball_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile(): continue
            if not member.name.endswith(".yml"): continue
            if "/yml/" not in member.name: continue
            f = tf.extractfile(member)
            if f is None: continue
            try:
                data = yaml.safe_load(f.read())
            except yaml.YAMLError:
                continue
            if not isinstance(data, dict):
                continue
            commands = data.get("Commands") or []
            for entry in commands:
                if not isinstance(entry, dict): continue
                cmd = entry.get("Command")
                mitre = entry.get("MitreID")
                if not cmd or not mitre: continue
                # normalize mitre id: LOLBAS sometimes uses T1218.011 - keep only parent
                mitre_parent = str(mitre).split(".")[0].strip().upper()
                if not re.match(r"^T\d{4}$", mitre_parent):
                    continue
                c = normalize_command(cmd)
                if c:
                    rows.append({
                        "command": c,
                        "mitre_id": mitre_parent,
                        "source": "lolbas",
                    })
    return rows


# GTFOBins functions -> a very coarse MITRE mapping. Intentionally conservative;
# anything not in this map is dropped rather than mislabeled.
_GTFOBINS_FUNC_TO_MITRE = {
    "shell":               "T1059",   # command and scripting interpreter
    "reverse-shell":       "T1059",
    "bind-shell":          "T1059",
    "non-interactive-bind-shell": "T1059",
    "non-interactive-reverse-shell": "T1059",
    "command":             "T1059",
    "upload":              "T1105",   # ingress tool transfer
    "download":            "T1105",
    "file-upload":         "T1105",   # legacy aliases (pre-2024 naming)
    "file-download":       "T1105",
    "file-write":          "T1222",   # file/directory permissions modification (proxy)
    "file-read":           "T1005",   # data from local system
    "library-load":        "T1574",   # hijack execution flow
    "suid":                "T1548",   # abuse elevation control mechanism
    "sudo":                "T1548",
    "capabilities":        "T1548",
    "limited-suid":        "T1548",
    "privilege-escalation": "T1548",
    "inherit":             "T1548",   # inherits elevated context
}


def parse_gtfobins(tarball_path: Path) -> list[dict]:
    """
    GTFOBins stores each binary under _gtfobins/<name>.md with YAML frontmatter:
        ---
        functions:
          shell:
            - code: |
                binary --flags
        ---
    """
    # GTFOBins files use .md extension but the whole file is a YAML document
    # delimited by `---` and `...` (YAML doc start/end markers, NOT Jekyll frontmatter).
    # safe_load handles these markers natively.
    if yaml is None:
        raise RuntimeError("PyYAML required for GTFOBins parsing")
    rows = []
    with tarfile.open(tarball_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile(): continue
            if "/_gtfobins/" not in member.name: continue
            # GTFOBins files may have no extension (bare names like 'curl', '7z')
            # or .md extension depending on repo version — accept both.
            f = tf.extractfile(member)
            if f is None: continue
            text = f.read().decode("utf-8", errors="ignore")
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError:
                continue
            if not isinstance(data, dict):
                continue
            functions = data.get("functions") or {}
            if not isinstance(functions, dict):
                continue
            for fname, entries in functions.items():
                mitre = _GTFOBINS_FUNC_TO_MITRE.get(fname)
                if not mitre:
                    continue
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict): continue
                    code = entry.get("code")
                    if not code or not isinstance(code, str): continue
                    for raw_line in _split_multiline_command(code):
                        c = normalize_command(raw_line)
                        if c:
                            rows.append({
                                "command": c,
                                "mitre_id": mitre,
                                "source": "gtfobins",
                            })
    return rows


def parse_mitre_cti(raw_bytes: bytes) -> list[dict]:
    """
    Pull procedure examples from MITRE ATT&CK STIX. Procedure examples live on
    'relationship' objects of type 'uses' where target is an attack-pattern.
    The 'description' field often embeds code/command snippets inside backticks or <code> tags.
    """
    rows = []
    try:
        data = json.loads(raw_bytes)
    except json.JSONDecodeError:
        return rows
    objs = data.get("objects") or []
    # First pass: map STIX id -> MITRE external id for attack-patterns
    ap_id_to_mitre: dict[str, str] = {}
    for o in objs:
        if o.get("type") != "attack-pattern":
            continue
        for ext in o.get("external_references") or []:
            if ext.get("source_name") == "mitre-attack":
                ext_id = ext.get("external_id", "")
                if re.match(r"^T\d{4}(?:\.\d+)?$", ext_id):
                    ap_id_to_mitre[o["id"]] = ext_id.split(".")[0]
                    break

    # Second pass: pull descriptions from 'uses' relationships and from attack-pattern bodies
    code_re = re.compile(r"<code>(.+?)</code>", re.DOTALL | re.IGNORECASE)
    backtick_re = re.compile(r"`([^`\n]{3,500})`")

    def _extract_commands_from_description(text: str) -> list[str]:
        if not text: return []
        found: list[str] = []
        found.extend(code_re.findall(text))
        found.extend(backtick_re.findall(text))
        return found

    for o in objs:
        t = o.get("type")
        desc = None
        mitre = None
        if t == "relationship" and o.get("relationship_type") == "uses":
            target = o.get("target_ref", "")
            mitre = ap_id_to_mitre.get(target)
            desc = o.get("description")
        elif t == "attack-pattern":
            mitre = ap_id_to_mitre.get(o.get("id", ""))
            desc = o.get("description") or o.get("x_mitre_detection")
        if not desc or not mitre:
            continue
        for snippet in _extract_commands_from_description(desc):
            # procedure descriptions often have multiline shell
            for raw_line in _split_multiline_command(snippet):
                c = normalize_command(raw_line)
                # reject obvious prose (code blocks sometimes contain sentences)
                if c and not re.search(r"[.?!]\s+[A-Z]", c) and len(c.split()) <= 60:
                    rows.append({
                        "command": c,
                        "mitre_id": mitre,
                        "source": "mitre_cti",
                    })
    return rows


PARSERS: dict[str, Callable] = {
    "parse_nl2bash": parse_nl2bash,
    "parse_slp_nl2bash": parse_slp_nl2bash,
    "parse_tldr_pages": parse_tldr_pages,
    "parse_atomic_red_team": parse_atomic_red_team,
    "parse_lolbas": parse_lolbas,
    "parse_gtfobins": parse_gtfobins,
    "parse_mitre_cti": parse_mitre_cti,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _print_header(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)


def _run_source(spec: SourceSpec, cache_dir: Path, refresh: bool) -> tuple[list[dict], dict]:
    """Download + parse a single source. Returns rows and status dict."""
    status = {"key": spec.key, "kind": spec.kind, "ok": False, "rows": 0, "error": ""}
    # cache path
    ext = ".json" if spec.url.endswith(".json") else (
        ".tar.gz" if spec.fetch_type == "tarball" else ".raw"
    )
    dest = cache_dir / f"{spec.key}{ext}"

    t0 = time.time()
    print(f"[*] {spec.key:22s}  fetching {spec.url}")
    try:
        _download(spec.url, dest, refresh=refresh)
    except urllib.error.HTTPError as e:
        status["error"] = f"HTTP {e.code}"
        print(f"    ! FAILED: {status['error']}")
        return [], status
    except urllib.error.URLError as e:
        status["error"] = f"URL {e.reason}"
        print(f"    ! FAILED: {status['error']}")
        return [], status
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
        print(f"    ! FAILED: {status['error']}")
        return [], status

    size = dest.stat().st_size
    elapsed = time.time() - t0
    print(f"    downloaded {_human(size)} in {elapsed:.1f}s")

    # parse
    parser = PARSERS[spec.parser]
    try:
        if spec.fetch_type == "raw":
            with open(dest, "rb") as f:
                raw = f.read()
            rows = parser(raw)
        else:
            rows = parser(dest)
    except Exception as e:
        status["error"] = f"parse failed: {type(e).__name__}: {e}"
        print(f"    ! PARSE FAILED: {status['error']}")
        return [], status

    status["rows"] = len(rows)
    if len(rows) == 0:
        status["ok"] = False
        status["error"] = "parser returned 0 rows (check parser logic or upstream format change)"
        print(f"    ! WARN: parser returned 0 rows — likely a parser bug or upstream format change")
    else:
        status["ok"] = True
        print(f"    parsed {len(rows):,} raw rows")
    return rows, status


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    """Dedup by normalized key. Keeps first occurrence, which preserves source attribution."""
    seen: set[str] = set()
    out = []
    dups = 0
    for r in rows:
        k = norm_key(r["command"])
        if k in seen:
            dups += 1
            continue
        seen.add(k)
        out.append(r)
    return out, dups


def stratified_split(
    rows: list[dict],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    stratify_by: str = "mitre_id",
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified 80/10/10 split. Guarantees no row overlap (splits are a partition).
    For classes with <3 samples, puts all in train and warns.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[str(r[stratify_by])].append(r)

    train, val, test = [], [], []
    too_small = []
    for key, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        if n < 3:
            too_small.append((key, n))
            train.extend(items)
            continue
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 1:
            # rebalance: steal from train
            n_test = 1
            n_train = n - n_val - n_test
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    if too_small:
        print(f"    [!] {len(too_small)} classes had <3 samples and went entirely to train")

    # shuffle final splits so consumers don't see class clustering
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def assert_no_leakage(train: list[dict], val: list[dict], test: list[dict], label: str) -> None:
    tr = {norm_key(r["command"]) for r in train}
    vl = {norm_key(r["command"]) for r in val}
    ts = {norm_key(r["command"]) for r in test}
    tr_vl = len(tr & vl)
    tr_ts = len(tr & ts)
    vl_ts = len(vl & ts)
    if tr_vl or tr_ts or vl_ts:
        raise AssertionError(
            f"[{label}] leakage detected: train∩val={tr_vl} train∩test={tr_ts} val∩test={vl_ts}"
        )
    print(f"    [{label}] leakage check: OK (0 overlap across splits)")


def write_csv(rows: list[dict], path: Path, include_source: bool = False) -> None:
    _safe_mkdir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if include_source:
            w = csv.DictWriter(f, fieldnames=["command", "mitre_id", "source"])
        else:
            w = csv.DictWriter(f, fieldnames=["command", "mitre_id"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize(name: str, rows: list[dict]) -> dict:
    counts = Counter(r["mitre_id"] for r in rows)
    sources = Counter(r.get("source", "?") for r in rows)
    first_tok = Counter(r["command"].split()[0].lower() for r in rows if r["command"])
    lengths = [len(r["command"]) for r in rows]
    summary = {
        "name": name,
        "total_rows": len(rows),
        "unique_commands": len({norm_key(r["command"]) for r in rows}),
        "num_classes": len(counts),
        "unique_first_tokens": len(first_tok),
        "sources": dict(sources),
        "class_count_min": min(counts.values()) if counts else 0,
        "class_count_max": max(counts.values()) if counts else 0,
        "class_count_median": sorted(counts.values())[len(counts)//2] if counts else 0,
        "len_min": min(lengths) if lengths else 0,
        "len_max": max(lengths) if lengths else 0,
        "len_mean": (sum(lengths)/len(lengths)) if lengths else 0,
        "short_cmds_under_20_chars": sum(1 for L in lengths if L < 20),
    }
    return summary


def _print_summary(s: dict) -> None:
    print(f"\n  {s['name']}")
    print(f"    rows            : {s['total_rows']:,}")
    print(f"    unique commands : {s['unique_commands']:,}")
    print(f"    classes         : {s['num_classes']}")
    print(f"    first tokens    : {s['unique_first_tokens']}")
    print(f"    class balance   : min={s['class_count_min']}  median={s['class_count_median']}  max={s['class_count_max']}")
    print(f"    cmd length      : min={s['len_min']}  mean={s['len_mean']:.1f}  max={s['len_max']}")
    print(f"    short (<20ch)   : {s['short_cmds_under_20_chars']:,}")
    print(f"    sources         : {s['sources']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./genos_dataset", help="Output directory for CSVs")
    ap.add_argument("--cache", default="./genos_cache", help="Download cache directory")
    ap.add_argument("--refresh", action="store_true", help="Re-download even if cached")
    ap.add_argument("--skip", default="", help="Comma-separated source keys to skip")
    ap.add_argument("--max-per-class", type=int, default=500,
                    help="Cap rows per MITRE class after dedup (reduces ART over-representation)")
    ap.add_argument("--min-per-class", type=int, default=10,
                    help="Drop MITRE classes with fewer than this many samples (default 10)")
    ap.add_argument("--keep-source-column", action="store_true",
                    help="Include 'source' column in output CSVs")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    cache_dir = Path(args.cache).resolve()
    _safe_mkdir(out_dir)
    _safe_mkdir(cache_dir)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    if yaml is None:
        print("[!] PyYAML not installed. `pip install pyyaml` to enable YAML-based sources.")
        print("    Will skip atomic_red_team, lolbas, gtfobins.")
        skip |= {"atomic_red_team", "lolbas", "gtfobins"}

    # -------- Phase 1: download + parse --------
    _print_header("PHASE 1/4  DOWNLOAD + PARSE")
    active = [s for s in SOURCES if s.key not in skip]
    print(f"Sources to scrape: {len(active)} / {len(SOURCES)} (skipping: {sorted(skip) or 'none'})")
    statuses = []
    all_benign, all_malicious = [], []

    for i, spec in enumerate(active, 1):
        remaining = len(active) - i
        print(f"\n[{i}/{len(active)}] {spec.key} ({spec.kind})  — {remaining} source(s) remaining")
        print(f"    {spec.note}")
        rows, status = _run_source(spec, cache_dir, refresh=args.refresh)
        statuses.append(status)
        if spec.kind == "benign":
            all_benign.extend(rows)
        else:
            all_malicious.extend(rows)

    # Status table
    _print_header("DOWNLOAD SUMMARY")
    print(f"{'source':<22} {'kind':<10} {'status':<8} {'rows':>10}  note/error")
    print("-" * 78)
    for s in statuses:
        st = "OK" if s["ok"] else "FAIL"
        err = "" if s["ok"] else s["error"]
        print(f"{s['key']:<22} {s['kind']:<10} {st:<8} {s['rows']:>10,}  {err}")
    print(f"\nTotal benign rows    : {len(all_benign):,}")
    print(f"Total malicious rows : {len(all_malicious):,}")

    if not all_benign or not all_malicious:
        print("[!] Empty benign or malicious set. Aborting.")
        return 2

    # -------- Phase 2: normalize + dedupe --------
    _print_header("PHASE 2/4  DEDUPE + NORMALIZE")
    benign_rows, benign_dups = deduplicate(all_benign)
    print(f"  Benign     : {len(all_benign):,} -> {len(benign_rows):,} unique  ({benign_dups:,} dupes removed)")
    mal_rows, mal_dups = deduplicate(all_malicious)
    print(f"  Malicious  : {len(all_malicious):,} -> {len(mal_rows):,} unique  ({mal_dups:,} dupes removed)")

    # Cross-contamination check: the same command labeled both benign AND malicious
    # (e.g. tldr documenting `whoami` while ART labels `whoami` as T1087).
    # Default policy: REMOVE from the malicious set, because in the gatekeeper context
    # the bare command is benign. If present in a chained sequence, different story.
    benign_keys = {norm_key(r["command"]) for r in benign_rows}
    before_cross = len(mal_rows)
    mal_rows = [r for r in mal_rows if norm_key(r["command"]) not in benign_keys]
    removed_cross = before_cross - len(mal_rows)
    print(f"  Cross-contam: removed {removed_cross:,} malicious rows that also exist in benign set")
    print(f"                (bare commands like `whoami`/`pwd` stay benign-only)")

    # Cap per-class (reduces Atomic Red Team dominance)
    cap = args.max_per_class
    if cap > 0:
        by_class: dict[str, list[dict]] = defaultdict(list)
        for r in mal_rows:
            by_class[r["mitre_id"]].append(r)
        rng = random.Random(SEED)
        capped = []
        capped_count = 0
        for cls, items in by_class.items():
            if len(items) > cap:
                rng.shuffle(items)
                capped_count += len(items) - cap
                items = items[:cap]
            capped.extend(items)
        rng.shuffle(capped)
        mal_rows = capped
        print(f"  Class cap  : {cap}/class -> removed {capped_count:,} over-sampled rows")

    # Drop malicious classes with too few samples to stratify-split (<3)
    # Drop malicious classes with too few samples.
    # Below --min-per-class, classes get no real learning signal AND mess with stratify split.
    # Stratify also needs at least 3 samples per class to put one in each of train/val/test.
    cls_counts = Counter(r["mitre_id"] for r in mal_rows)
    min_n = max(3, args.min_per_class)
    tiny_classes = [c for c, n in cls_counts.items() if n < min_n]
    if tiny_classes:
        total_tiny_rows = sum(cls_counts[c] for c in tiny_classes)
        print(f"  Dropping {len(tiny_classes)} classes with <{min_n} samples "
              f"({total_tiny_rows} rows total)")
        print(f"    examples: {sorted(tiny_classes)[:10]}{'...' if len(tiny_classes)>10 else ''}")
        mal_rows = [r for r in mal_rows if r["mitre_id"] not in tiny_classes]
        print(f"  Remaining malicious rows: {len(mal_rows):,} across "
              f"{len(set(r['mitre_id'] for r in mal_rows))} classes")

    # -------- Phase 3: split --------
    _print_header("PHASE 3/4  STRATIFIED SPLIT 80/10/10")
    print("  Gatekeeper (benign) — stratified by first-token")
    # for the gatekeeper, we pseudo-stratify by first-token to guarantee each split
    # sees the full distribution of command families.
    for r in benign_rows:
        r["_strat"] = r["command"].split()[0].lower() if r["command"] else "_"
    gk_train, gk_val, gk_test = stratified_split(benign_rows, stratify_by="_strat")
    for r in benign_rows:
        r.pop("_strat", None)
    assert_no_leakage(gk_train, gk_val, gk_test, "gatekeeper")

    print("\n  Specialist (malicious) — stratified by mitre_id")
    sp_train, sp_val, sp_test = stratified_split(mal_rows, stratify_by="mitre_id")
    assert_no_leakage(sp_train, sp_val, sp_test, "specialist")

    # -------- Phase 4: write --------
    _print_header("PHASE 4/4  WRITE OUTPUT")
    files = [
        ("gatekeeper_train.csv", gk_train),
        ("gatekeeper_val.csv",   gk_val),
        ("gatekeeper_test.csv",  gk_test),
        ("specialist_train.csv", sp_train),
        ("specialist_val.csv",   sp_val),
        ("specialist_test.csv",  sp_test),
    ]
    for name, rows in files:
        path = out_dir / name
        write_csv(rows, path, include_source=args.keep_source_column)
        print(f"  wrote  {path}  ({len(rows):,} rows)")

    # provenance file
    prov = {
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "sources": [s.__dict__ for s in active],
        "statuses": statuses,
        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "max_per_class": args.max_per_class,
        "summaries": {
            "gatekeeper_train": summarize("gatekeeper_train", gk_train),
            "gatekeeper_val":   summarize("gatekeeper_val", gk_val),
            "gatekeeper_test":  summarize("gatekeeper_test", gk_test),
            "specialist_train": summarize("specialist_train", sp_train),
            "specialist_val":   summarize("specialist_val", sp_val),
            "specialist_test":  summarize("specialist_test", sp_test),
        },
    }
    prov_path = out_dir / "provenance.json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2, default=str)
    print(f"  wrote  {prov_path}")

    # final report
    _print_header("FINAL REPORT")
    for name, rows in files:
        _print_summary(summarize(name, rows))

    _print_header("DONE")
    print(f"Output directory: {out_dir}")
    print(f"Cache directory : {cache_dir}  (safe to delete)")
    print("\nNext steps:")
    print(f"  1. Inspect the provenance.json for per-source contributions.")
    print(f"  2. Compare gatekeeper_train first-tokens vs your FP list.")
    print(f"  3. Copy CSVs into your data/training/ directory and retrain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())