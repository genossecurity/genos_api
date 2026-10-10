"""
evidence.py — Indicator of Compromise (IOC) and Evidence Extraction Engine for Genos.

Performs regex parsing, flag analysis, and indicator synthesis:
- Network indicators (URLs, domains, IPs, ports)
- Target filesystem and registry paths
- LOLBin (Living Off the Land) proxy binary execution
- High-signal command-line flags
- Interpreters and execution styles
- Analyst-facing evidence summary generation
"""

import os
import re
import sys
import ipaddress
from urllib.parse import urlsplit
from typing import Any, Dict, List, Optional, Set

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARSER_DIR = os.path.join(BASE_DIR, "parser")
if _PARSER_DIR not in sys.path:
    sys.path.insert(0, _PARSER_DIR)

try:
    from parser import parse_command as _parse_command
    from semantic_features import build_semantic_features as _build_semantic_features
    from rule_engine import build_rule_result as _build_rule_result
    _RESIDUAL_PIPELINE_AVAILABLE = True
except ImportError:
    _RESIDUAL_PIPELINE_AVAILABLE = False


# ── High-Signal Flag Constants ────────────────────────────────────────────────

HIGH_SIGNAL_FLAGS = frozenset({
    "-enc", "-encodedcommand", "-e", "-c", "/c", "-nop", "-noni",
    "-noninteractive", "-windowstyle", "-w", "-exec", "-executionpolicy",
    "-ep", "-bypass", "-command", "/create", "/sc", "/tr", "/tn",
    "/f", "/d", "/t", "/v", "/s", "/add", "/delete",
    "-split", "-f", "-o", "--output", "-urlcache",
    "-decode", "-encode", "-decodefile", "-p", "--post-file",
    "-b64", "--allow-overwrite", "--on-download-complete",
    "-perm", "+4000", "+2000", "-rf", "--flush",
    "/set", "/get", "/query", "/export", "/import",
})


# ── Semantic Feature Labels to Surface ────────────────────────────────────────

SURFACE_SEM_FEATURES = frozenset({
    "downloads_remote_resource",
    "writes_executable_like_file",
    "modifies_registry_autorun",
    "creates_scheduled_task",
    "creates_or_modifies_service",
    "archive_create",
    "archive_extract",
    "deletes_shadow_copies",
    "remote_execution_or_session",
    "transfers_file_to_remote",
    "runs_interpreter",
    "executes_inline_code",
    "enumerates_identity",
    "enumerates_network_config",
    "reads_credential_store",
    "uses_encoded_payload",
    "uses_obfuscation",
    "uses_signed_proxy_binary",
})


# ── Interpreter Mappings ──────────────────────────────────────────────────────

INTERPRETER_NAMES = {
    "bash": "bash", "sh": "sh", "zsh": "zsh", "fish": "fish",
    "python": "python", "python3": "python", "py": "python",
    "perl": "perl", "ruby": "ruby", "php": "php",
    "node": "node", "node.exe": "node",
    "powershell": "powershell", "powershell.exe": "powershell",
    "pwsh": "powershell", "cmd": "cmd", "cmd.exe": "cmd",
    "wscript": "wscript", "cscript": "cscript",
    "mshta": "mshta", "mshta.exe": "mshta",
}

LOLBIN_CANDIDATES = frozenset({
    "certutil", "mshta", "rundll32", "regsvr32", "wmic", "bitsadmin",
    "powershell", "powershell.exe", "cmd", "cmd.exe", "wscript", "cscript",
    "bash", "sh", "curl", "wget",
})


# ── Evidence Synthesis Helpers ────────────────────────────────────────────────

def generate_evidence_summary(exe: str, sem: dict, fired_rules: list) -> str:
    """Generate a compact analyst-facing evidence sentence describing detected indicators."""
    parts = []
    if exe:
        parts.append(exe)
    if sem.get("uses_encoded_payload") or sem.get("uses_obfuscation"):
        parts.append("encoded/obfuscated execution")
    if sem.get("downloads_remote_resource"):
        parts.append("remote resource download")
    if sem.get("executes_inline_code"):
        parts.append("inline code execution")
    if sem.get("modifies_registry_autorun"):
        parts.append("registry autorun persistence")
    if sem.get("creates_scheduled_task"):
        parts.append("scheduled task creation")
    if sem.get("deletes_shadow_copies"):
        parts.append("shadow copy deletion")
    if sem.get("remote_execution_or_session"):
        parts.append("remote session/execution")
    if sem.get("enumerates_identity"):
        parts.append("identity enumeration")
    if sem.get("reads_credential_store"):
        parts.append("credential store access")
    if sem.get("uses_signed_proxy_binary"):
        parts.append("signed binary proxy execution")

    if not parts:
        if fired_rules:
            return f"Triggered rules: {', '.join(fired_rules[:3])}"
        return "No prominent high-signal indicators detected"

    return f"{parts[0]} displaying " + ", ".join(parts[1:]) if len(parts) > 1 else f"{parts[0]} execution"


def build_evidence(
    parsed: dict,
    sem: dict,
    rule_result: dict,
    was_obfuscated: bool = False,
    deobfuscated_cmd: Optional[str] = None,
) -> dict:
    """Build curated evidence dictionary from parsed artifacts, semantics, and rules."""
    exe = (parsed.get("executable") or "").lower()
    flags = parsed.get("flags") or []

    # Execution identity
    platform = parsed.get("platform") or "unknown"
    interpreter = (
        INTERPRETER_NAMES.get(exe)
        or (parsed.get("interpreter_markers") or [None])[0]
        or None
    )

    # High-signal flags
    high_signal_flags = sorted({
        f.lower() for f in flags
        if f.lower() in HIGH_SIGNAL_FLAGS
    })

    # Structural behavior
    has_pipe = bool(parsed.get("has_pipe"))
    has_redirect = bool(parsed.get("has_redirect"))
    has_chain = bool(parsed.get("has_chain"))
    inline_code = bool(parsed.get("inline_code")) or bool(sem.get("executes_inline_code"))

    # Obfuscation
    uses_encoded_payload = bool(sem.get("uses_encoded_payload"))
    uses_obfuscation_flag = bool(sem.get("uses_obfuscation")) or was_obfuscated
    obfuscation_markers = list(parsed.get("encoded_markers") or []) + list(parsed.get("obfuscation_markers") or [])
    deob_cmd = deobfuscated_cmd or parsed.get("deobfuscated_command") or None

    # LOLBins
    lolbin_matches = list(parsed.get("lolbin_matches") or [])
    if exe and exe in LOLBIN_CANDIDATES:
        if exe not in lolbin_matches:
            lolbin_matches.insert(0, exe)
    uses_signed_proxy = bool(sem.get("uses_signed_proxy_binary")) or bool(lolbin_matches)

    # Surface semantic features
    semantic_features = [
        k for k in SURFACE_SEM_FEATURES
        if sem.get(k)
    ]

    # Rule metadata
    rule_strength = rule_result.get("rule_strength", "none")
    raw_rules = rule_result.get("fired_rules") or []
    fired_rules = [r.replace("_rule_", "").replace("_", " ") for r in raw_rules]

    # Evidence summary sentence
    evidence_summary = generate_evidence_summary(exe, sem, fired_rules)

    # Primary artifact classification
    primary_artifact_type = None
    if parsed.get("registry_paths"):
        primary_artifact_type = "registry"
    elif sem.get("creates_scheduled_task"):
        primary_artifact_type = "task"
    elif sem.get("creates_or_modifies_service"):
        primary_artifact_type = "service"
    elif sem.get("archive_create") or sem.get("archive_extract"):
        primary_artifact_type = "archive"
    elif (parsed.get("urls") or parsed.get("remote_targets") or sem.get("downloads_remote_resource")):
        primary_artifact_type = "network"
    elif sem.get("runs_interpreter") or sem.get("executes_inline_code"):
        primary_artifact_type = "script"
    elif parsed.get("file_paths"):
        primary_artifact_type = "file"

    # Execution style
    execution_style = None
    if sem.get("downloads_remote_resource") and (sem.get("executes_inline_code") or has_pipe):
        execution_style = "download-and-execute"
    elif sem.get("creates_scheduled_task"):
        execution_style = "scheduled"
    elif sem.get("remote_execution_or_session"):
        execution_style = "remote-session"
    elif sem.get("executes_inline_code") or inline_code:
        execution_style = "inline"
    elif sem.get("downloads_remote_resource"):
        execution_style = "download-and-execute"

    return {
        "platform": platform,
        "executable": parsed.get("executable") or None,
        "subcommand": parsed.get("subcommand") or None,
        "interpreter": interpreter,
        "high_signal_flags": high_signal_flags,
        "file_paths": list(parsed.get("file_paths") or []),
        "registry_paths": list(parsed.get("registry_paths") or []),
        "local_targets": list(parsed.get("local_targets") or []),
        "remote_targets": list(parsed.get("remote_targets") or []),
        "urls": list(parsed.get("urls") or []),
        "domains": list(parsed.get("domains") or []),
        "ips": list(parsed.get("ips") or []),
        "ports": list(parsed.get("ports") or []),
        "has_pipe": has_pipe,
        "has_redirect": has_redirect,
        "has_chain": has_chain,
        "inline_code": inline_code,
        "uses_encoded_payload": uses_encoded_payload,
        "uses_obfuscation": uses_obfuscation_flag,
        "obfuscation_markers": obfuscation_markers,
        "deobfuscated_command": deob_cmd,
        "lolbin_matches": lolbin_matches,
        "uses_signed_proxy_binary": uses_signed_proxy,
        "semantic_features": semantic_features,
        "rule_strength": rule_strength,
        "fired_rules": fired_rules,
        "evidence_summary": evidence_summary,
        "primary_artifact_type": primary_artifact_type,
        "execution_style": execution_style,
    }


_SEGMENT_SPLIT_RE = re.compile(r"\|\||&&|;|\||&(?!>)")
_PREFIX_WRAPPERS = frozenset({"sudo", "env", "nohup", "time", "exec", "command", "nice", "xargs"})


def extract_binary_inventory(*commands: Optional[str]) -> List[dict]:
    """Inventory every binary invoked across pipe/chain segments of the given command views."""
    inventory: Dict[str, dict] = {}
    for command in commands:
        if not command:
            continue
        for segment in _SEGMENT_SPLIT_RE.split(command):
            tokens = segment.strip().split()
            while tokens and (re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=\S*", tokens[0]) or tokens[0].lower() in _PREFIX_WRAPPERS):
                tokens.pop(0)
            if not tokens:
                continue
            name = os.path.basename(tokens[0].strip("\"'(){}")).lower()
            if not name or not re.fullmatch(r"[\w.+-]+", name):
                continue
            entry = inventory.setdefault(name, {"binary": name, "flags": [], "roles": []})
            for tok in tokens[1:]:
                if tok.startswith("-") or (tok.startswith("/") and len(tok) <= 3):
                    if tok not in entry["flags"]:
                        entry["flags"].append(tok)
            if name in LOLBIN_CANDIDATES and "lolbin" not in entry["roles"]:
                entry["roles"].append("lolbin")
            if name in INTERPRETER_NAMES and "interpreter" not in entry["roles"]:
                entry["roles"].append("interpreter")
    return list(inventory.values())


_HASH_RE = re.compile(r"(?<![A-Fa-f0-9])(?:[A-Fa-f0-9]{64}|[A-Fa-f0-9]{40}|[A-Fa-f0-9]{32})(?![A-Fa-f0-9])")
_DEFANGED_URL_RE = re.compile(r"\bhxxps?://[^\s\"'<>|]+", re.I)
_IPV6_CANDIDATE_RE = re.compile(r"(?<![\w:])\[?[0-9A-Fa-f:]{3,}\]?(?![\w:])")
_URL_RE = re.compile(r"\b(?:https?|ftp)://[^\s\"'<>|]+", re.I)


def _extra_indicators(text: str) -> dict:
    ipv6 = []
    for match in _IPV6_CANDIDATE_RE.finditer(text):
        candidate = match.group().strip("[]")
        if ":" not in candidate:
            continue
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.version == 6 and str(address) not in ipv6:
            ipv6.append(str(address))
    defanged = [re.sub(r"\[\.\]", ".", value.rstrip(";,.!)"), flags=re.I).replace("hxxps", "https").replace("hxxp", "http")
                for value in _DEFANGED_URL_RE.findall(text)]
    return {"hashes": list(dict.fromkeys(match.group().lower() for match in _HASH_RE.finditer(text))),
            "ipv6": ipv6, "defanged_urls": list(dict.fromkeys(defanged))}


def _indicator_records(views: list[tuple[str, str, str | None]], parsed_by_view: dict) -> list[dict]:
    records = []
    seen = set()
    kinds = {"urls": "url", "domains": "domain", "ips": "ip", "ports": "port",
             "file_paths": "file", "registry_paths": "registry", "hashes": "hash",
             "ipv6": "ip", "defanged_urls": "defanged_url"}
    for source_view, text, decoder in views:
        parsed = parsed_by_view.get(source_view) or {}
        extra = _extra_indicators(text)
        values = {key: parsed.get(key, []) for key in ("urls", "domains", "ips", "ports", "file_paths", "registry_paths")}
        values.update(extra)
        if not parsed:
            values["urls"] = _URL_RE.findall(text)
            values["domains"] = []
            values["ips"] = []
            for url in values["urls"]:
                try:
                    host = urlsplit(url).hostname
                except ValueError:
                    continue
                if host:
                    try:
                        address = ipaddress.ip_address(host)
                        values["ips"].append(str(address))
                    except ValueError:
                        values["domains"].append(host.lower())
        for field, items in values.items():
            for value in items:
                key = (source_view, field, value)
                if key in seen:
                    continue
                seen.add(key)
                match = re.search(re.escape(value), text, re.I)
                records.append({"type": kinds[field], "value": value, "source_view": source_view,
                                "span": [match.start(), match.end()] if match else None,
                                "decoder": decoder, "confidence": "observed" if match else "derived"})
    return records


def collect_indicator_evidence(
    raw_cmd: str,
    decoded_cmd: str,
    was_obfuscated: bool = False,
    *, view_cache: dict | None = None, decoding_trace: dict | None = None,
) -> dict:
    """Extract observable IOC indicators and evidence from both raw and decoded command views."""
    cache = view_cache if view_cache is not None else {}
    raw_entry = cache.setdefault(raw_cmd, {})
    raw_parsed = raw_entry.get("parsed")
    if raw_parsed is None:
        raw_parsed = _parse_command(raw_cmd, deobfuscate_input=False)
        raw_entry["parsed"] = raw_parsed
    sem = raw_entry.get("sem")
    if sem is None:
        sem = _build_semantic_features(raw_parsed)
        raw_entry["sem"] = sem
    parsed = dict(raw_parsed)
    parsed_by_view = {"raw": raw_parsed}
    if decoded_cmd != raw_cmd:
        decoded_entry = cache.setdefault(decoded_cmd, {})
        decoded = decoded_entry.get("parsed")
        if decoded is None:
            decoded = _parse_command(decoded_cmd, deobfuscate_input=False)
            decoded_entry["parsed"] = decoded
        parsed_by_view["decoded"] = decoded
        decoded_sem = decoded_entry.get("sem")
        if decoded_sem is None:
            decoded_sem = _build_semantic_features(decoded)
            decoded_entry["sem"] = decoded_sem
        for key, value in decoded_sem.items():
            if value and not sem.get(key):
                sem[key] = value
        for key in (
            "file_paths", "registry_paths", "urls", "domains", "ips", "ports",
            "lolbin_matches", "local_targets", "remote_targets", "flags",
            "encoded_markers", "obfuscation_markers",
        ):
            parsed[key] = list(dict.fromkeys([*(parsed.get(key) or []), *(decoded.get(key) or [])]))
        for key in ("has_pipe", "has_redirect", "has_chain", "inline_code"):
            parsed[key] = bool(parsed.get(key) or decoded.get(key))
    rules = _build_rule_result(parsed, sem)
    evidence = build_evidence(
        parsed,
        sem,
        rules,
        was_obfuscated=was_obfuscated,
        deobfuscated_cmd=decoded_cmd if was_obfuscated else None,
    )
    evidence["binaries"] = extract_binary_inventory(raw_cmd, decoded_cmd if decoded_cmd != raw_cmd else None)
    views = [("raw", raw_cmd, None)]
    if decoding_trace:
        for index, step in enumerate(decoding_trace.get("steps", [])):
            if step["text"] != decoded_cmd:
                views.append((f"layer_{index + 1}", step["text"], step["transform"]))
    if decoded_cmd != raw_cmd:
        last_transform = (decoding_trace or {}).get("steps", [])
        views.append(("decoded", decoded_cmd, last_transform[-1]["transform"] if last_transform else None))
    records = _indicator_records(views, parsed_by_view)
    evidence["indicator_records"] = records
    evidence["hashes"] = list(dict.fromkeys(row["value"] for row in records if row["type"] == "hash"))
    evidence["ipv6"] = list(dict.fromkeys(row["value"] for row in records if row["type"] == "ip" and ":" in row["value"]))
    evidence["defanged_urls"] = list(dict.fromkeys(row["value"] for row in records if row["type"] == "defanged_url"))
    for field, kind in (("urls", "url"), ("domains", "domain"), ("ips", "ip")):
        evidence[field] = list(dict.fromkeys([*evidence[field], *(row["value"] for row in records if row["type"] == kind)]))
    return evidence


class EvidenceExtractor:
    """Configurable Evidence & IOC Extractor."""

    def __init__(self):
        pass

    def collect_indicator_evidence(self, raw_cmd: str, decoded_cmd: str, was_obfuscated: bool = False) -> dict:
        return collect_indicator_evidence(raw_cmd, decoded_cmd, was_obfuscated)

    def build_evidence(self, parsed: dict, sem: dict, rule_result: dict, was_obfuscated: bool = False, deobfuscated_cmd: Optional[str] = None) -> dict:
        return build_evidence(parsed, sem, rule_result, was_obfuscated, deobfuscated_cmd)

    def generate_evidence_summary(self, exe: str, sem: dict, fired_rules: list) -> str:
        return generate_evidence_summary(exe, sem, fired_rules)
