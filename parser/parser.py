import copy
import ipaddress
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from deobfuscator import deobfuscate, is_obfuscated

DEFAULT_SCHEMA: Dict[str, Any] = {
    "raw_command": "",
    "normalized_command": "",
    "platform": "unknown",
    "executable": "",
    "subcommand": None,
    "flags": [],
    "positional_args": [],
    "has_pipe": False,
    "has_redirect": False,
    "has_chain": False,
    "pipe_operators": [],
    "redirect_operators": [],
    "chain_operators": [],
    "file_paths": [],
    "registry_paths": [],
    "urls": [],
    "domains": [],
    "ips": [],
    "ports": [],
    "local_targets": [],
    "remote_targets": [],
    "interpreter_markers": [],
    "encoded_markers": [],
    "archive_indicators": [],
    "lolbin_matches": [],
    "inline_code": False,
    "obfuscation_markers": [],
    "deobfuscated_command": None,
}

PIPE_OPERATORS = {"|"}
REDIRECT_OPERATORS = {">", ">>", "<", "<<", "2>", "2>>", "1>", "1>>"}
CHAIN_OPERATORS = {"&&", "||", ";"}

INTERPRETERS = {
    "bash": "bash",
    "sh": "sh",
    "zsh": "zsh",
    "fish": "fish",
    "python": "python",
    "python3": "python",
    "py": "python",
    "perl": "perl",
    "ruby": "ruby",
    "php": "php",
    "node": "node",
    "node.exe": "node",
    "powershell": "powershell",
    "powershell.exe": "powershell",
    "pwsh": "powershell",
    "cmd": "cmd",
    "cmd.exe": "cmd",
    "wscript": "wscript",
    "cscript": "cscript",
}

WINDOWS_EXECUTABLES = {
    "powershell", "powershell.exe", "pwsh", "cmd", "cmd.exe", "reg", "reg.exe",
    "schtasks", "schtasks.exe", "wmic", "wmic.exe", "certutil", "certutil.exe",
    "mshta", "mshta.exe", "rundll32", "rundll32.exe", "vssadmin", "vssadmin.exe",
    "bitsadmin", "bitsadmin.exe", "ipconfig", "ipconfig.exe", "net", "net.exe",
    "sc", "sc.exe", "wscript", "wscript.exe", "cscript", "cscript.exe",
}

LINUX_EXECUTABLES = {
    "ls", "whoami", "pwd", "cat", "curl", "wget", "tar", "unzip", "ssh", "scp",
    "rsync", "docker", "systemctl", "find", "grep", "awk", "sed", "bash", "sh",
    "python3", "python", "zip", "gunicorn", "flask", "node", "id", "uname",
    "ifconfig", "ip", "netstat", "ss", "dig", "nslookup", "traceroute", "ping",
}

LOLBINS = {
    "windows": {
        "certutil", "mshta", "rundll32", "regsvr32", "wmic", "powershell",
        "bitsadmin", "schtasks", "vssadmin", "reg", "cmd", "wscript", "cscript",
    },
    "linux": {
        "bash", "sh", "curl", "wget", "ssh", "scp", "tar", "find", "python",
        "python3", "perl", "ruby", "php", "awk", "sed",
    },
}

ARCHIVE_EXES = {"tar", "zip", "unzip", "7z", "7za", "gzip", "gunzip", "rar", "unrar"}
ARCHIVE_EXTENSIONS = (".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar")
WINDOWS_REG_PATH_RE = re.compile(r"\b(?:HKLM|HKCU|HKCR|HKU|HKCC)\\[^\s\"']+", re.I)
URL_RE = re.compile(r"\b(?:https?|ftp)://[^\s\"'<>|]+", re.I)
DOMAIN_RE = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b", re.I)
PORT_RE = re.compile(r":(\d{1,5})(?:\b|/)")
ENCODED_MARKERS = [
    (re.compile(r"(?i)(?:^|\s)-enc(?:odedcommand)?(?:\s|$)"), "encoded_command_flag"),
    (re.compile(r"(?i)frombase64|string"), "base64_string_decode"),
    (re.compile(r"\b[A-Za-z0-9+/]{20,}={0,2}\b"), "base64_blob"),
    (re.compile(r"\\x[0-9a-fA-F]{2}"), "hex_escape"),
]
OBFUSCATION_MARKERS = [
    (re.compile(r"(?i)\[char\]"), "char_cast"),
    (re.compile(r"(?i)frombase64"), "base64_decode_api"),
    (re.compile(r"(?i)encodedcommand|-enc\b"), "encoded_command"),
    (re.compile(r"(?i)\b(?:\$\w{8,}|%[A-Za-z0-9_]+%)\b"), "variable_indirection"),
    (re.compile(r"\\x[0-9a-fA-F]{2}"), "hex_escape"),
]
INLINE_FLAG_HINTS = {"-c", "/c", "-e", "-enc", "-encodedcommand"}


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item is None:
            continue
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _load_schema(schema_path: Optional[str] = None) -> Dict[str, Any]:
    candidates: List[Path] = []
    if schema_path:
        candidates.append(Path(schema_path))
    candidates.append(Path(__file__).with_name("parser_schema.json"))
    candidates.append(Path.cwd() / "parser_schema.json")

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return copy.deepcopy(DEFAULT_SCHEMA)


def _new_result(raw_command: str, schema_path: Optional[str] = None) -> Dict[str, Any]:
    base = copy.deepcopy(_load_schema(schema_path))
    base["raw_command"] = raw_command
    base["normalized_command"] = normalize_command(raw_command)
    return base


def normalize_command(raw_command: str) -> str:
    text = str(raw_command).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def _safe_posix_split(text: str) -> List[str]:
    # If it looks like a Windows command, avoid shlex
    if "\\" in text or re.search(r"\b(reg|schtasks|wmic|cmd|powershell)\b", text, re.I):
        return re.findall(r'"[^"]*"|\S+', text)
    
    try:
        return shlex.split(text, posix=True)
    except Exception:
        return re.findall(r'"[^"]*"|\S+', text)


def _extract_operators(text: str) -> Dict[str, List[str]]:
    pipes = re.findall(r"(?<!\|)\|(?!\|)", text)
    redirects = re.findall(r"2>>|2>|1>>|1>|>>|<<|>|<", text)
    chains = re.findall(r"&&|\|\||;", text)
    return {
        "pipe_operators": pipes,
        "redirect_operators": redirects,
        "chain_operators": chains,
    }


def _is_flag(token: str) -> bool:
    if not token:
        return False
    if token in PIPE_OPERATORS | REDIRECT_OPERATORS | CHAIN_OPERATORS:
        return False
    if token.startswith("--"):
        return True
    if token.startswith("-") and len(token) > 1:
        return True
    if token.startswith("/") and len(token) > 1:
        # likely Windows flag, but avoid treating absolute unix paths as flags
        if token.startswith("//"):
            return False
        if "/" in token[1:] and not re.fullmatch(r"/[A-Za-z0-9_-]+", token):
            return False
        return True
    return False


def _looks_like_windows_path(token: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:\\", token)) or token.startswith(".\\") or token.startswith("..\\")


def _looks_like_unix_path(token: str) -> bool:
    if token in {".", "..", "~"}:
        return True
    return token.startswith("/") or token.startswith("./") or token.startswith("../") or token.startswith("~/")

def _looks_like_remote_target(token: str) -> bool:
    if "@" in token:   # ← add this
        return True
    if token.startswith("\\\\"):
        return True
    return False

def _looks_like_file_path(token: str) -> bool:
    if _looks_like_windows_path(token) or _looks_like_unix_path(token) or _looks_like_remote_target(token):
        return True
    if any(token.lower().endswith(ext) for ext in (
        ".exe", ".dll", ".hta", ".ps1", ".bat", ".cmd", ".sh", ".py",
        ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".log", ".txt",
        ".csv", ".json", ".yaml", ".yml", ".conf", ".cfg", ".bak", ".pdf",
    )):
        return True
    return False


def _extract_urls(text: str) -> List[str]:
    return _dedupe_keep_order(URL_RE.findall(text))


def _extract_ips(text: str) -> List[str]:
    ips: List[str] = []
    for token in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
        try:
            ipaddress.ip_address(token)
            ips.append(token)
        except ValueError:
            continue
    return _dedupe_keep_order(ips)


def _extract_domains(text: str, urls: List[str], ips: List[str]) -> List[str]:
    FILELIKE_EXTENSIONS = {
        ".exe", ".dll", ".bat", ".cmd", ".ps1", ".sh", ".py",
        ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar",
        ".txt", ".log", ".json", ".yaml", ".yml",
        ".cfg", ".conf", ".pdf", ".bin", ".dat", ".tmp",
        ".hta", ".vbs", ".js", ".jar", ".msi", ".iso", ".img",
    }
    candidates = DOMAIN_RE.findall(text)
    url_hosts: List[str] = []
    for url in urls:
        host = re.sub(r"^[a-z]+://", "", url, flags=re.I).split("/")[0].split("@")[-1]
        host = host.split(":")[0]
        if host:
            url_hosts.append(host)
    domains = []
    ip_set = set(ips)
    for d, from_url in [(c, False) for c in candidates] + [(h, True) for h in url_hosts]:
        dl = d.lower().strip(".")

        if not dl or dl in ip_set:
            continue

        # skip IP-like
        if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", dl):
            continue

        # skip file-like names (payload.exe, backup.tar.gz, etc.)
        if any(dl.endswith(ext) for ext in FILELIKE_EXTENSIONS):
            continue

        # skip CamelCase artifacts from deobfuscated code (e.g. Net.WebClient,
        # System.IO.File) — real hostnames never have uppercase-initial labels.
        # Only applied to free-text regex hits, not URL hosts.
        if not from_url and any(label and label[0].isupper() for label in d.split(".")):
            continue

        domains.append(dl)
    return _dedupe_keep_order(domains)


def _extract_ports(text: str, urls: List[str], remote_targets: List[str]) -> List[str]:
    ports: List[str] = []
    for source in [text] + urls + remote_targets:
        for match in PORT_RE.findall(source):
            try:
                port_int = int(match)
                if 0 < port_int <= 65535:
                    ports.append(str(port_int))
            except ValueError:
                pass
    return _dedupe_keep_order(ports)


def _extract_registry_paths(text: str) -> List[str]:
    return _dedupe_keep_order(WINDOWS_REG_PATH_RE.findall(text))


def _detect_platform(executable: str, flags: List[str], registry_paths: List[str]) -> str:
    exe = executable.lower()
    if registry_paths:
        return "windows"
    if exe in WINDOWS_EXECUTABLES:
        return "windows"
    if exe in LINUX_EXECUTABLES:
        return "linux"
    if any(f.startswith("/") for f in flags):
        return "windows"
    return "unknown"


def _extract_interpreter_markers(tokens: List[str]) -> List[str]:
    found: List[str] = []
    for tok in tokens:
        if _is_flag(tok):
            continue
        base = os.path.basename(tok).lower()
        if base in INTERPRETERS:
            found.append(INTERPRETERS[base])
    return _dedupe_keep_order(found)


def _extract_encoded_markers(text: str) -> List[str]:
    hits = []
    for pattern, label in ENCODED_MARKERS:
        if pattern.search(text):
            hits.append(label)
    return _dedupe_keep_order(hits)


def _extract_obfuscation_markers(text: str) -> List[str]:
    hits = []
    for pattern, label in OBFUSCATION_MARKERS:
        if pattern.search(text):
            hits.append(label)
    return _dedupe_keep_order(hits)


def _extract_archive_indicators(executable: str, tokens: List[str]) -> List[str]:
    indicators: List[str] = []
    exe = executable.lower()
    token_blob = " ".join(tokens).lower()

    if exe in ARCHIVE_EXES:
        indicators.append("archive_tool")
    if exe == "tar":
        if any("x" in t.lower().replace("-", "") for t in tokens if t.startswith("-")):
            indicators.append("archive_extract")
        if any("c" in t.lower().replace("-", "") for t in tokens if t.startswith("-")):
            indicators.append("archive_create")
        if any("z" in t.lower().replace("-", "") for t in tokens if t.startswith("-")):
            indicators.append("compression_gzip")
    if exe in {"zip", "7z", "7za", "rar", "gzip"}:
        indicators.append("archive_create")
    if exe in {"unzip", "unrar", "gunzip"}:
        indicators.append("archive_extract")
    if any(tok.lower().endswith(ARCHIVE_EXTENSIONS) for tok in tokens):
        indicators.append("archive_path")
    if "backup" in token_blob:
        indicators.append("backup_keyword")
    return _dedupe_keep_order(indicators)


def _extract_lolbin_matches(executable: str, platform: str) -> List[str]:
    exe = executable.lower()
    out: List[str] = []
    if platform in LOLBINS and exe in LOLBINS[platform]:
        out.append(exe)
    else:
        # fall back to checking both when platform is unknown
        for p in LOLBINS.values():
            if exe in p:
                out.append(exe)
                break
    return out


def _pick_executable(tokens: List[str]) -> str:
    for tok in tokens:
        if tok in PIPE_OPERATORS | REDIRECT_OPERATORS | CHAIN_OPERATORS:
            continue
        if tok in {">", ">>", "<", "<<", "2>", "2>>", "1>", "1>>"}:
            continue
        return os.path.basename(tok.strip("\"'"))
    return ""


def _pick_subcommand(executable: str, tokens: List[str]) -> Optional[str]:
    if not executable:
        return None
    base_tokens = [t for t in tokens if t not in PIPE_OPERATORS | REDIRECT_OPERATORS | CHAIN_OPERATORS | REDIRECT_OPERATORS]
    if not base_tokens:
        return None

    # skip executable
    remainder = base_tokens[1:]
    if not remainder:
        return None

    exe = executable.lower()
    candidate = remainder[0].strip("\"'")
    if _is_flag(candidate):
        return None

    # common CLIs with real subcommands
    if exe in {"docker", "kubectl", "git", "reg", "sc", "systemctl", "wmic", "net", "aws", "gcloud", "az"}:
        return candidate
    if exe in {"vssadmin"} and candidate in {"create", "delete", "list", "resize"}:
        return candidate
    return None


def _extract_flags_and_args(tokens: List[str], executable: str, subcommand: Optional[str]) -> Dict[str, List[str]]:
    filtered = []
    skip_first = True
    subcommand_skipped = subcommand is None
    for tok in tokens:
        if tok in PIPE_OPERATORS | REDIRECT_OPERATORS | CHAIN_OPERATORS | REDIRECT_OPERATORS:
            continue
        if skip_first:
            skip_first = False
            continue
        if not subcommand_skipped and tok.strip("\"'") == subcommand:
            subcommand_skipped = True
            continue
        filtered.append(tok.strip("\"'"))

    flags: List[str] = []
    args: List[str] = []

    for tok in filtered:
        if _is_flag(tok):
            flags.append(tok)
        else:
            args.append(tok)
    return {"flags": _dedupe_keep_order(flags), "positional_args": _dedupe_keep_order(args)}


def _classify_targets(
    file_paths: List[str],
    urls: List[str],
    remote_tokens: List[str],
    registry_paths: List[str],
) -> Dict[str, List[str]]:
    local_targets = _dedupe_keep_order(file_paths + registry_paths)
    remote_targets = _dedupe_keep_order(urls + remote_tokens)
    return {"local_targets": local_targets, "remote_targets": remote_targets}


def parse_command(raw_command: str, schema_path: Optional[str] = None) -> Dict[str, Any]:
    result = _new_result(raw_command, schema_path=schema_path)
    normalized = result["normalized_command"]

    # Deobfuscate if needed.
    # structural parsing (executable, flags, operators) uses the original command.
    # network/path extraction uses the deobfuscated form so buried IOCs are surfaced.
    extraction_target = normalized
    if is_obfuscated(normalized):
        deobfuscated = deobfuscate(normalized)
        if deobfuscated != normalized:
            result["deobfuscated_command"] = deobfuscated
            extraction_target = deobfuscated

    operator_info = _extract_operators(extraction_target)
    result["pipe_operators"] = operator_info["pipe_operators"]
    result["redirect_operators"] = operator_info["redirect_operators"]
    result["chain_operators"] = operator_info["chain_operators"]
    result["has_pipe"] = bool(result["pipe_operators"])
    result["has_redirect"] = bool(result["redirect_operators"])
    result["has_chain"] = bool(result["chain_operators"])

    tokens = _safe_posix_split(normalized)
    executable = _pick_executable(tokens)
    subcommand = _pick_subcommand(executable, tokens)
    flag_arg_info = _extract_flags_and_args(tokens, executable, subcommand)

    urls = _extract_urls(extraction_target)
    ips = _extract_ips(extraction_target)
    registry_paths = _extract_registry_paths(extraction_target)

    extraction_tokens = _safe_posix_split(extraction_target)
    file_paths: List[str] = []
    remote_tokens: List[str] = []
    for tok in extraction_tokens:
        clean = tok.strip("\"'")
        if clean in PIPE_OPERATORS | REDIRECT_OPERATORS | CHAIN_OPERATORS:
            continue
        if _is_flag(clean):
            continue
        if clean in urls:
            continue
        if _looks_like_remote_target(clean):
            remote_tokens.append(clean)
            continue
        if _looks_like_file_path(clean):
            file_paths.append(clean)

    domains = _extract_domains(extraction_target, urls, ips)
    targets = _classify_targets(_dedupe_keep_order(file_paths), urls, _dedupe_keep_order(remote_tokens), registry_paths)
    ports = _extract_ports(extraction_target, urls, targets["remote_targets"])
    interpreters = _extract_interpreter_markers(tokens)
    encoded_markers = _extract_encoded_markers(normalized)
    obfuscation_markers = _extract_obfuscation_markers(normalized)

    platform = _detect_platform(executable, flag_arg_info["flags"], registry_paths)
    archive_indicators = _extract_archive_indicators(executable, tokens)
    lolbin_matches = _extract_lolbin_matches(executable, platform)

    inline_code = False
    lowered_flags = {f.lower() for f in flag_arg_info["flags"]}
    if any(flag in lowered_flags for flag in INLINE_FLAG_HINTS):
        inline_code = True
    if any(m in interpreters for m in ("python", "bash", "powershell", "cmd")) and flag_arg_info["positional_args"]:
        if any(flag in lowered_flags for flag in {"-c", "/c", "-e", "-enc", "-encodedcommand"}):
            inline_code = True

    result.update(
        {
            "platform": platform,
            "executable": executable.lower() if executable else "",
            "subcommand": subcommand.lower() if isinstance(subcommand, str) else None,
            "flags": flag_arg_info["flags"],
            "positional_args": flag_arg_info["positional_args"],
            "file_paths": _dedupe_keep_order(file_paths),
            "registry_paths": registry_paths,
            "urls": urls,
            "domains": domains,
            "ips": ips,
            "ports": ports,
            "local_targets": targets["local_targets"],
            "remote_targets": targets["remote_targets"],
            "interpreter_markers": interpreters,
            "encoded_markers": encoded_markers,
            "archive_indicators": archive_indicators,
            "lolbin_matches": lolbin_matches,
            "inline_code": inline_code,
            "obfuscation_markers": obfuscation_markers,
        }
    )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a command into the Genos parser schema.")
    parser.add_argument("command", nargs="?", help="Command string to parse")
    parser.add_argument("--schema", dest="schema_path", default=None, help="Optional path to parser_schema.json")
    args = parser.parse_args()

    if not args.command:
        print("Usage: python parser.py '<command>'")
        raise SystemExit(1)

    parsed = parse_command(args.command, schema_path=args.schema_path)
    print(json.dumps(parsed, indent=2))
