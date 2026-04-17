import os
from typing import Any, Dict, List

WINDOWS_AUTORUN_PREFIXES = [
    r"hkcu\software\microsoft\windows\currentversion\run",
    r"hklm\software\microsoft\windows\currentversion\run",
    r"hkcu\software\microsoft\windows\currentversion\runonce",
    r"hklm\software\microsoft\windows\currentversion\runonce",
]

IDENTITY_ENUM_EXES = {"whoami", "id"}
NETWORK_ENUM_EXES = {"ipconfig", "ifconfig", "ip", "netstat", "ss", "route", "arp", "hostname", "nslookup", "dig"}
USER_GROUP_ENUM_EXES = {"net", "dsquery", "getent"}
REMOTE_EXEC_EXES = {"wmic", "psexec", "winrs", "ssh", "scp", "sftp"}
SIGNED_PROXY_BINARIES = {
    "certutil", "mshta", "rundll32", "regsvr32", "wmic",
    "bitsadmin", "powershell", "powershell.exe", "cmd", "cmd.exe",
}
ARCHIVE_EXES = {"tar", "zip", "unzip", "7z", "7za", "rar", "unrar", "gzip", "gunzip"}
DOWNLOAD_EXES = {"curl", "wget", "certutil", "bitsadmin", "powershell", "powershell.exe"}
SERVICE_CONTROL_EXES = {"sc", "sc.exe", "systemctl", "service"}
TASK_SCHEDULER_EXES = {"schtasks", "schtasks.exe", "at", "cron", "crontab", "launchctl", "systemd-run"}
CREDENTIAL_PATH_HINTS = {
    "/etc/shadow", "/etc/passwd", "sam", "ntds.dit", "lsass", "security", "policy\\secrets"
}
EXECUTABLE_EXTENSIONS = (".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".hta", ".sh", ".py", ".jar")
ARCHIVE_EXTENSIONS = (".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".bz2", ".xz")


def _lower_list(items: List[str]) -> List[str]:
    return [str(x).lower() for x in items]


def _contains_any(texts: List[str], needles: List[str]) -> bool:
    haystack = " ".join(texts)
    return any(n in haystack for n in needles)


def _has_executable_like_path(paths: List[str]) -> bool:
    for p in paths:
        pl = p.lower()
        if pl.endswith(EXECUTABLE_EXTENSIONS):
            return True
    return False


def _has_archive_like_path(paths: List[str]) -> bool:
    for p in paths:
        pl = p.lower()
        if pl.endswith(ARCHIVE_EXTENSIONS):
            return True
    return False


def _is_registry_autorun(registry_paths: List[str]) -> bool:
    reg_l = _lower_list(registry_paths)
    for p in reg_l:
        for prefix in WINDOWS_AUTORUN_PREFIXES:
            if p.startswith(prefix):
                return True
    return False


def _has_flag(flags: List[str], wanted: List[str]) -> bool:
    flags_l = _lower_list(flags)
    wanted_l = {w.lower() for w in wanted}
    return any(f in wanted_l for f in flags_l)


def _path_contains_any(paths: List[str], needles: List[str]) -> bool:
    paths_l = _lower_list(paths)
    return any(n in p for p in paths_l for n in needles)


def build_semantic_features(parsed: Dict[str, Any]) -> Dict[str, Any]:
    exe = str(parsed.get("executable", "")).lower()
    subcommand = parsed.get("subcommand")
    sub_l = str(subcommand).lower() if subcommand else None

    flags = parsed.get("flags", []) or []
    flags_l = _lower_list(flags)
    positional_args = parsed.get("positional_args", []) or []
    args_l = _lower_list(positional_args)

    file_paths = parsed.get("file_paths", []) or []
    file_paths_l = _lower_list(file_paths)
    registry_paths = parsed.get("registry_paths", []) or []
    urls = parsed.get("urls", []) or []
    remote_targets = parsed.get("remote_targets", []) or []
    local_targets = parsed.get("local_targets", []) or []
    interpreter_markers = parsed.get("interpreter_markers", []) or []
    encoded_markers = parsed.get("encoded_markers", []) or []
    archive_indicators = parsed.get("archive_indicators", []) or []
    lolbin_matches = parsed.get("lolbin_matches", []) or []
    obfuscation_markers = parsed.get("obfuscation_markers", []) or []
    deobfuscated_command = parsed.get("deobfuscated_command")
    normalized_command = str(parsed.get("normalized_command", "")).lower()

    has_remote_target = bool(remote_targets or urls)
    has_local_target = bool(local_targets or file_paths or registry_paths)
    writes_local_file = bool(file_paths)
    writes_executable_like_file = _has_executable_like_path(file_paths)
    archive_create = "archive_create" in archive_indicators
    archive_extract = "archive_extract" in archive_indicators

    downloads_remote_resource = (
        exe in DOWNLOAD_EXES and has_remote_target
    ) or (
        has_remote_target and writes_local_file and exe in {"curl", "wget", "certutil", "bitsadmin"}
    )

    transfers_file_to_remote = (
        exe in {"scp", "rsync", "sftp"} and has_remote_target
    ) or (
        has_remote_target and any("@" in str(t) for t in remote_targets) and exe in {"scp", "ssh", "rsync"}
    )

    executes_inline_code = bool(parsed.get("inline_code")) or (
        bool(interpreter_markers) and _has_flag(flags, ["-c", "/c", "-e", "-enc", "-encodedcommand"])
    )

    uses_encoded_payload = bool(encoded_markers)
    uses_obfuscation = bool(obfuscation_markers) or bool(parsed.get("was_obfuscated")) or bool(parsed.get("was_deobfuscated"))

    creates_scheduled_task = (
        exe in TASK_SCHEDULER_EXES and (
            sub_l in {"add", "create"} or
            _has_flag(flags, ["/create"]) or
            "create" in args_l
        )
    )

    modifies_registry_autorun = (
        exe in {"reg", "reg.exe"} and
        sub_l in {"add", "delete", "import"} and
        _is_registry_autorun(registry_paths)
    )

    enumerates_identity = (
        exe in IDENTITY_ENUM_EXES or
        (exe == "net" and sub_l == "user" and not has_remote_target)
    )

    enumerates_network_config = (
        exe in NETWORK_ENUM_EXES or
        (exe == "net" and sub_l in {"view", "use", "share"})
    )

    enumerates_users_or_groups = (
        (exe == "net" and sub_l in {"user", "group", "localgroup"}) or
        exe == "dsquery" or
        (exe == "getent" and any(x in args_l for x in {"passwd", "group"}))
    )

    uses_signed_proxy_binary = exe in SIGNED_PROXY_BINARIES or bool(lolbin_matches)

    remote_execution_or_session = (
        exe in REMOTE_EXEC_EXES and has_remote_target
    ) or (
        exe == "wmic" and sub_l == "process"
    )

    service_control = (
        exe in SERVICE_CONTROL_EXES or
        (exe == "sc.exe") or
        (exe == "sc")
    )

    creates_or_modifies_service = (
        exe in {"sc", "sc.exe", "systemctl"} and (
            sub_l in {"create", "config", "enable", "link"} or
            _has_flag(flags, ["create", "config"])
        )
    )

    reads_credential_store = (
        _path_contains_any(file_paths + local_targets, list(CREDENTIAL_PATH_HINTS)) or
        exe in {"mimikatz", "procdump", "procdump.exe"}
    )

    deletes_shadow_copies = (
        exe in {"vssadmin", "wmic"} and (
            sub_l == "delete" or
            "delete" in args_l
        ) and (
            "shadows" in args_l or "shadowcopy" in normalized_command
        )
    )

    runs_interpreter = bool(interpreter_markers)

    compression_or_archiving = bool(archive_indicators) or exe in ARCHIVE_EXES or _has_archive_like_path(file_paths)

    features = {
        "downloads_remote_resource": downloads_remote_resource,
        "transfers_file_to_remote": transfers_file_to_remote,
        "writes_local_file": writes_local_file,
        "writes_executable_like_file": writes_executable_like_file,
        "has_remote_target": has_remote_target,
        "has_local_target": has_local_target,
        "archive_create": archive_create,
        "archive_extract": archive_extract,
        "compression_or_archiving": compression_or_archiving,
        "creates_scheduled_task": creates_scheduled_task,
        "modifies_registry_autorun": modifies_registry_autorun,
        "enumerates_identity": enumerates_identity,
        "enumerates_network_config": enumerates_network_config,
        "enumerates_users_or_groups": enumerates_users_or_groups,
        "uses_encoded_payload": uses_encoded_payload,
        "uses_obfuscation": uses_obfuscation,
        "executes_inline_code": executes_inline_code,
        "runs_interpreter": runs_interpreter,
        "uses_signed_proxy_binary": uses_signed_proxy_binary,
        "remote_execution_or_session": remote_execution_or_session,
        "service_control": service_control,
        "creates_or_modifies_service": creates_or_modifies_service,
        "reads_credential_store": reads_credential_store,
        "deletes_shadow_copies": deletes_shadow_copies,
    }

    # Helpful compact tags for downstream prompting / residual building.
    feature_tags = [k for k, v in features.items() if isinstance(v, bool) and v]
    features["feature_tags"] = feature_tags

    # Keep a small metadata block for debugging.
    features["debug_context"] = {
        "exe": exe,
        "subcommand": sub_l,
        "n_flags": len(flags_l),
        "n_urls": len(urls),
        "n_file_paths": len(file_paths),
        "n_registry_paths": len(registry_paths),
        "was_deobfuscated": bool(deobfuscated_command),
    }

    return features


if __name__ == "__main__":
    import argparse
    import json

    from parser import parse_command

    argp = argparse.ArgumentParser(description="Build Genos semantic features from a parsed command.")
    argp.add_argument("command", nargs="?", help="Command string to parse and featurize")
    argp.add_argument("--schema", dest="schema_path", default=None, help="Optional path to parser_schema.json")
    args = argp.parse_args()

    if not args.command:
        print("Usage: python3 semantic_features.py '<command>'")
        raise SystemExit(1)

    parsed = parse_command(args.command, schema_path=args.schema_path)
    feats = build_semantic_features(parsed)
    print(json.dumps({"parsed": parsed, "semantic_features": feats}, indent=2))
