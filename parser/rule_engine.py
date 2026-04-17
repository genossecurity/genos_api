"""
rule_engine.py — Genos v1 Rule/Prior Engine

Sits between semantic_features and the specialist model.

Input:  parsed dict  (from parser.py)
        semantic     dict  (from semantic_features.py)

Output:
    {
        "candidate_classes": [...],   # MITRE-style class labels kept in play
        "banned_classes":    [...],   # classes ruled out by hard logic
        "priors":            {...},   # {class_label: float weight 0‒1}
        "evidence":          [...],   # human-readable strings explaining decisions
    }

Design rules:
  - Conservative: under-constrain rather than over-prune.
  - candidate_classes are soft suggestions; the specialist model is NOT limited
    to them—they are used to up-weight, not hard-filter.
  - banned_classes are hard: used only when a class is logically impossible.
  - priors are additive per family; capped at 1.0.
  - evidence is append-only; one string per fired rule.
"""

from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# MITRE-style class label constants
# (plain strings; no external ontology dependency required in v1)
# ─────────────────────────────────────────────────────────────────────────────

# Persistence
CLS_REGISTRY_RUN_KEY          = "persistence:registry_run_key"
CLS_SCHEDULED_TASK            = "persistence:scheduled_task"
CLS_CREATE_MODIFY_SERVICE     = "persistence:create_modify_system_process"

# Execution
CLS_CMD_SCRIPTING             = "execution:command_and_scripting_interpreter"
CLS_WMI_EXEC                  = "execution:windows_management_instrumentation"
CLS_SIGNED_BINARY_PROXY       = "defense_evasion:signed_binary_proxy_execution"

# Defense evasion
CLS_OBFUSCATED_FILES          = "defense_evasion:obfuscated_files_or_information"
CLS_DEOBFUSCATE_DECODE        = "defense_evasion:deobfuscate_decode_files"

# C2 / Exfil / Collection
CLS_INGRESS_TOOL_TRANSFER     = "command_and_control:ingress_tool_transfer"
CLS_ARCHIVE_COLLECTED_DATA    = "collection:archive_collected_data"
CLS_EXFILTRATION_TOOL         = "exfiltration:exfiltration_over_c2_channel"

# Impact
CLS_INHIBIT_SYSTEM_RECOVERY   = "impact:inhibit_system_recovery"

# Discovery
CLS_ACCOUNT_DISCOVERY         = "discovery:account_discovery"
CLS_NETWORK_CONFIG_DISCOVERY  = "discovery:system_network_configuration_discovery"
CLS_REMOTE_SYSTEM_DISCOVERY   = "discovery:remote_system_discovery"

# Lateral movement
CLS_REMOTE_SERVICES           = "lateral_movement:remote_services"

# Defense evasion (expanded)
CLS_MODIFY_REGISTRY           = "defense_evasion:modify_registry"
CLS_IMPAIR_DEFENSES           = "defense_evasion:impair_defenses"
CLS_FILE_PERM_MOD             = "defense_evasion:file_permissions_modification"
CLS_INDICATOR_REMOVAL         = "defense_evasion:indicator_removal"
CLS_HIDE_ARTIFACTS            = "defense_evasion:hide_artifacts"
CLS_MASQUERADING              = "defense_evasion:masquerading"

# Execution (expanded)
CLS_NATIVE_API                = "execution:native_api"

# Discovery (expanded)
CLS_FILE_DIR_DISCOVERY        = "discovery:file_and_directory_discovery"
CLS_QUERY_REGISTRY            = "discovery:query_registry"
CLS_SYSTEM_INFO_DISCOVERY     = "discovery:system_information_discovery"
CLS_PROCESS_DISCOVERY         = "discovery:process_discovery"

# Credential access
CLS_CREDENTIAL_DUMPING        = "credential_access:os_credential_dumping"

# Persistence (expanded)
CLS_EVENT_TRIGGERED_EXEC      = "persistence:event_triggered_execution"
CLS_BOOT_LOGON_AUTOSTART      = "persistence:boot_logon_autostart_execution"
CLS_HIJACK_EXEC_FLOW          = "persistence:hijack_execution_flow"
CLS_ACCOUNT_MGMT              = "persistence:account_manipulation"

# Privilege escalation
CLS_ABUSE_ELEVATION           = "privilege_escalation:abuse_elevation_control"

# Impact (expanded)
CLS_DATA_DESTRUCTION          = "impact:data_destruction"

# Collection (expanded)
CLS_DATA_FROM_LOCAL           = "collection:data_from_local_system"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_candidate(result: dict, cls: str) -> None:
    if cls not in result["candidate_classes"]:
        result["candidate_classes"].append(cls)


def _add_ban(result: dict, cls: str) -> None:
    if cls not in result["banned_classes"]:
        result["banned_classes"].append(cls)


def _add_prior(result: dict, cls: str, weight: float) -> None:
    current = result["priors"].get(cls, 0.0)
    result["priors"][cls] = min(1.0, current + weight)


def _evidence(result: dict, msg: str) -> None:
    result["evidence"].append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Rule families
# ─────────────────────────────────────────────────────────────────────────────

def _rule_registry_persistence(parsed: dict, sem: dict, result: dict) -> None:
    """Registry run-key / autorun write."""
    if not sem.get("modifies_registry_autorun"):
        return

    _add_candidate(result, CLS_REGISTRY_RUN_KEY)
    _add_candidate(result, CLS_SIGNED_BINARY_PROXY)

    _add_prior(result, CLS_REGISTRY_RUN_KEY, 0.85)
    if sem.get("writes_executable_like_file"):
        _add_prior(result, CLS_REGISTRY_RUN_KEY, 0.10)

    reg_paths = parsed.get("registry_paths", [])
    _evidence(result, f"registry_autorun: write to autorun key [{', '.join(reg_paths)}]")

    if sem.get("writes_executable_like_file"):
        _evidence(result, "registry_autorun: payload is executable-like file")


def _rule_scheduled_task(parsed: dict, sem: dict, result: dict) -> None:
    """Scheduled task / job creation."""
    if not sem.get("creates_scheduled_task"):
        return

    _add_candidate(result, CLS_SCHEDULED_TASK)
    _add_candidate(result, CLS_SIGNED_BINARY_PROXY)

    _add_prior(result, CLS_SCHEDULED_TASK, 0.85)
    if sem.get("writes_executable_like_file"):
        _add_prior(result, CLS_SCHEDULED_TASK, 0.10)

    exe = parsed.get("executable", "")
    _evidence(result, f"scheduled_task: {exe} used to create/register a scheduled task")

    if sem.get("writes_executable_like_file"):
        _evidence(result, "scheduled_task: task payload is executable-like")


def _rule_download_ingress(parsed: dict, sem: dict, result: dict) -> None:
    """Download / ingress tool transfer."""
    if not sem.get("downloads_remote_resource"):
        return

    _add_candidate(result, CLS_INGRESS_TOOL_TRANSFER)

    _add_prior(result, CLS_INGRESS_TOOL_TRANSFER, 0.75)

    urls = parsed.get("urls", [])
    ips = parsed.get("ips", [])
    src = urls[:1] or ips[:1] or ["<unknown>"]
    _evidence(result, f"ingress_transfer: remote resource fetched from {src[0]}")

    if sem.get("writes_executable_like_file"):
        _add_candidate(result, CLS_SIGNED_BINARY_PROXY)
        _add_prior(result, CLS_INGRESS_TOOL_TRANSFER, 0.10)
        _evidence(result, "ingress_transfer: downloaded payload is executable-like")

    if sem.get("uses_signed_proxy_binary"):
        _add_candidate(result, CLS_SIGNED_BINARY_PROXY)
        _add_prior(result, CLS_SIGNED_BINARY_PROXY, 0.40)
        _evidence(result, f"signed_proxy: {parsed.get('executable', '')} is a signed proxy binary")

    # curl|bash pattern — inline execution of downloaded content
    if parsed.get("has_pipe") and sem.get("runs_interpreter"):
        _add_candidate(result, CLS_CMD_SCRIPTING)
        _add_prior(result, CLS_CMD_SCRIPTING, 0.50)
        _evidence(result, "ingress_transfer: piped directly to interpreter (curl|bash pattern)")


def _rule_archive_collection(parsed: dict, sem: dict, result: dict) -> None:
    """Archive creation — staging / collection."""
    if not sem.get("archive_create"):
        return

    _add_candidate(result, CLS_ARCHIVE_COLLECTED_DATA)
    _add_prior(result, CLS_ARCHIVE_COLLECTED_DATA, 0.65)

    exe = parsed.get("executable", "")
    _evidence(result, f"archive_collection: {exe} used to create archive (possible staging/collection)")

    if sem.get("compression_or_archiving"):
        _add_prior(result, CLS_ARCHIVE_COLLECTED_DATA, 0.10)

    # archive extract is NOT collection; do not add the class
    if sem.get("archive_extract") and not sem.get("archive_create"):
        _add_ban(result, CLS_ARCHIVE_COLLECTED_DATA)


def _rule_encoded_execution(parsed: dict, sem: dict, result: dict) -> None:
    """Encoded / obfuscated interpreter execution."""
    if not (sem.get("uses_encoded_payload") or sem.get("uses_obfuscation")):
        return

    _add_candidate(result, CLS_OBFUSCATED_FILES)
    _add_candidate(result, CLS_CMD_SCRIPTING)
    _add_candidate(result, CLS_SIGNED_BINARY_PROXY)

    _add_prior(result, CLS_OBFUSCATED_FILES, 0.75)
    _add_prior(result, CLS_CMD_SCRIPTING, 0.60)
    _add_prior(result, CLS_SIGNED_BINARY_PROXY, 0.50)

    encoded_markers = parsed.get("encoded_markers", [])
    _evidence(result, f"encoded_execution: encoded/obfuscated payload detected [{', '.join(encoded_markers)}]")

    if sem.get("executes_inline_code"):
        _add_prior(result, CLS_CMD_SCRIPTING, 0.20)
        _evidence(result, "encoded_execution: inline code execution via interpreter flag")

    deob = parsed.get("deobfuscated_command")
    if deob:
        _add_candidate(result, CLS_DEOBFUSCATE_DECODE)
        _add_prior(result, CLS_DEOBFUSCATE_DECODE, 0.60)
        _evidence(result, f"encoded_execution: payload decoded to → {deob[:120]}")

    if sem.get("downloads_remote_resource"):
        _add_prior(result, CLS_INGRESS_TOOL_TRANSFER, 0.55)
        _evidence(result, "encoded_execution: decoded payload downloads remote resource")


def _rule_wmi_remote_execution(parsed: dict, sem: dict, result: dict) -> None:
    """WMI / remote execution."""
    exe = parsed.get("executable", "")
    subcommand = (parsed.get("subcommand") or "").lower()

    is_wmi_process = exe == "wmic" and subcommand == "process"
    if not (is_wmi_process or (sem.get("remote_execution_or_session") and exe == "wmic")):
        return

    _add_candidate(result, CLS_WMI_EXEC)
    _add_candidate(result, CLS_SIGNED_BINARY_PROXY)

    _add_prior(result, CLS_WMI_EXEC, 0.80)
    _add_prior(result, CLS_SIGNED_BINARY_PROXY, 0.50)

    _evidence(result, f"wmi_exec: wmic process call detected (remote/local execution via WMI)")

    args = parsed.get("positional_args", [])
    if any("create" in a.lower() for a in args):
        _add_prior(result, CLS_WMI_EXEC, 0.15)
        _evidence(result, "wmi_exec: 'create' keyword — spawning new process via WMI")


def _rule_shadow_copy_deletion(parsed: dict, sem: dict, result: dict) -> None:
    """Shadow copy / backup deletion — inhibit recovery."""
    if not sem.get("deletes_shadow_copies"):
        return

    _add_candidate(result, CLS_INHIBIT_SYSTEM_RECOVERY)
    _add_prior(result, CLS_INHIBIT_SYSTEM_RECOVERY, 0.90)

    exe = parsed.get("executable", "")
    _evidence(result, f"shadow_deletion: {exe} used to delete shadow copies (inhibit recovery)")

    # High-confidence: ban classes that are clearly impossible here
    _add_ban(result, CLS_INGRESS_TOOL_TRANSFER)
    _add_ban(result, CLS_REGISTRY_RUN_KEY)
    _add_ban(result, CLS_SCHEDULED_TASK)


def _rule_service_creation(parsed: dict, sem: dict, result: dict) -> None:
    """Service creation or modification — persistence / privilege."""
    if not sem.get("creates_or_modifies_service"):
        return

    _add_candidate(result, CLS_CREATE_MODIFY_SERVICE)
    _add_prior(result, CLS_CREATE_MODIFY_SERVICE, 0.80)

    exe = parsed.get("executable", "")
    sub = parsed.get("subcommand") or ""
    _evidence(result, f"service_creation: {exe} {sub} — creates/modifies a system service")

    if sem.get("writes_executable_like_file"):
        _add_prior(result, CLS_CREATE_MODIFY_SERVICE, 0.10)
        _evidence(result, "service_creation: service binPath points to executable-like file")


def _rule_remote_session(parsed: dict, sem: dict, result: dict) -> None:
    """SSH/SCP/SFTP remote session or file transfer."""
    exe = parsed.get("executable", "")
    if exe not in {"ssh", "scp", "sftp", "rsync"}:
        return
    if not sem.get("remote_execution_or_session") and not sem.get("transfers_file_to_remote"):
        return

    _add_candidate(result, CLS_REMOTE_SERVICES)
    _add_prior(result, CLS_REMOTE_SERVICES, 0.55)

    remote_targets = parsed.get("remote_targets", [])
    target_str = remote_targets[0] if remote_targets else "<unknown>"
    _evidence(result, f"remote_session: {exe} session/transfer to {target_str}")

    if sem.get("transfers_file_to_remote") and sem.get("writes_executable_like_file"):
        _add_prior(result, CLS_EXFILTRATION_TOOL, 0.40)
        _add_candidate(result, CLS_EXFILTRATION_TOOL)
        _evidence(result, "remote_session: executable-like file transferred to remote host")


def _rule_discovery(parsed: dict, sem: dict, result: dict) -> None:
    """Identity / network enumeration."""
    if sem.get("enumerates_identity"):
        _add_candidate(result, CLS_ACCOUNT_DISCOVERY)
        _add_prior(result, CLS_ACCOUNT_DISCOVERY, 0.45)
        _evidence(result, f"discovery: {parsed.get('executable', '')} — identity/account enumeration")

    if sem.get("enumerates_network_config"):
        _add_candidate(result, CLS_NETWORK_CONFIG_DISCOVERY)
        _add_prior(result, CLS_NETWORK_CONFIG_DISCOVERY, 0.45)
        _evidence(result, f"discovery: {parsed.get('executable', '')} — network configuration discovery")


# ─────────────────────────────────────────────────────────────────────────────
# Extended rule families (v1.1) — executable / feature lookup sets
# ─────────────────────────────────────────────────────────────────────────────

_INTERPRETERS = frozenset({
    "powershell", "powershell.exe", "pwsh", "pwsh.exe",
    "cmd", "cmd.exe", "command.com",
    "bash", "sh", "zsh", "ksh", "csh", "tcsh", "dash", "fish",
    "python", "python3", "python2", "python.exe",
    "ruby", "perl", "lua", "node", "php",
    "osascript", "wscript", "cscript", "mshta", "mshta.exe",
})

_SCRIPTING_BUILTINS = frozenset({
    "echo", "printf", "write-host", "write-output", "write-error",
    "write-verbose", "write-debug", "write-information",
    "invoke-expression", "iex", "invoke-command", "icm",
    "invoke-webrequest", "iwr", "invoke-restmethod", "irm",
    "start-process", "new-object", "set-variable",
    "if", "for", "while", "foreach", "switch", "try",
    "set", "export", "eval", "exec", "source", "alias",
})

_PERM_EXES = frozenset({
    "chmod", "chown", "chgrp", "icacls", "cacls", "takeown",
    "setfacl", "chattr",
})

_LOLBIN_PROXY_EXES = frozenset({
    "rundll32", "rundll32.exe", "regsvr32", "regsvr32.exe",
    "msiexec", "msiexec.exe", "certutil",
    "cmstp", "cmstp.exe", "installutil", "installutil.exe",
    "msbuild", "msbuild.exe", "xwizard", "xwizard.exe",
    "odbcconf", "odbcconf.exe", "forfiles", "forfiles.exe",
    "mavinject", "mavinject.exe", "ieexec", "infdefaultinstall",
    "control", "control.exe", "hh", "hh.exe",
    "pcalua", "pcalua.exe", "presentationhost", "presentationhost.exe",
    "syncappvpublishingserver", "syncappvpublishingserver.exe",
    "bitsadmin", "bitsadmin.exe",
})

_FILE_DISCOVERY_EXES = frozenset({
    "dir", "ls", "find", "get-childitem", "gci", "tree",
    "locate", "fd", "where", "where.exe", "test-path",
})

_SYSTEM_INFO_DIRECT = frozenset({
    "systeminfo", "systeminfo.exe", "uname", "ver",
    "hostname", "dmidecode", "lscpu", "lsb_release",
    "sw_vers", "lshw", "lspci", "lsusb", "lsmod",
    "get-computerinfo", "get-wmiobject",
})

_PROCESS_DISCOVERY_EXES = frozenset({
    "ps", "tasklist", "tasklist.exe", "get-process", "top", "htop", "pgrep",
})

_CREDENTIAL_EXES = frozenset({
    "mimikatz", "lazagne", "secretsdump", "pypykatz",
    "procdump", "procdump.exe", "gsecdump", "pwdump",
    "hashdump", "fgdump", "wce",
})

_PRIV_ESC_EXES = frozenset({
    "sudo", "su", "doas", "runas", "pkexec", "pfexec", "gksudo", "kdesudo",
})

_FILE_READ_EXES = frozenset({
    "cat", "type", "more", "less", "head", "tail",
    "get-content", "gc", "strings", "xxd", "hexdump", "od",
})

_ACCOUNT_MGMT_EXES = frozenset({
    "useradd", "groupadd", "adduser", "addgroup",
    "usermod", "groupmod", "dsadd", "dsmod", "dscl",
    "new-aduser", "add-adgroupmember", "new-localuser",
})


def _rule_interpreter_general(parsed: dict, sem: dict, result: dict) -> None:
    """Interpreter execution — fires only for known interpreters, not builtins."""
    exe = parsed.get("executable", "").lower()
    fires = (
        exe in _INTERPRETERS
        or sem.get("runs_interpreter")
        or sem.get("executes_inline_code")
        or bool(parsed.get("inline_code"))
    )
    if not fires:
        return
    _add_candidate(result, CLS_CMD_SCRIPTING)
    _add_prior(result, CLS_CMD_SCRIPTING, 0.25)
    _evidence(result, f"interpreter_general: {exe} — scripting/interpreter execution")


def _rule_registry_broad(parsed: dict, sem: dict, result: dict) -> None:
    """Registry operations with capped, intent-specific priors."""
    exe = parsed.get("executable", "").lower()
    sub = (parsed.get("subcommand") or "").lower()
    reg_paths = parsed.get("registry_paths", [])
    norm = parsed.get("normalized_command", "").lower()

    if sem.get("modifies_registry_autorun"):
        return

    registry_context = bool(reg_paths) or any(
        marker in norm for marker in {"hklm:", "hkcu:", "hkcr:", "hku:", "registry::"}
    )
    if not registry_context and exe not in {
        "reg", "reg.exe",
        "get-itemproperty", "get-item", "get-childitem",
        "set-itemproperty", "new-itemproperty", "remove-itemproperty",
        "set-item", "new-item", "remove-item",
    }:
        return

    query_subcommands = {"query", "export", "save"}
    write_subcommands = {"add", "delete", "import", "copy", "load", "unload"}
    ps_query_cmds = {"get-itemproperty", "get-item", "get-childitem"}
    ps_write_cmds = {
        "set-itemproperty", "new-itemproperty", "remove-itemproperty",
        "set-item", "new-item", "remove-item",
    }

    is_query = (exe in {"reg", "reg.exe"} and sub in query_subcommands) or exe in ps_query_cmds
    is_write = (exe in {"reg", "reg.exe"} and sub in write_subcommands) or exe in ps_write_cmds

    if not is_query and not is_write:
        if any(token in norm for token in {
            " reg query ", "get-itemproperty", "get-childitem", " reg export ", " reg save ",
        }):
            is_query = True
        elif any(token in norm for token in {
            " reg add ", " reg delete ", "set-itemproperty", "new-itemproperty",
            "remove-itemproperty", "new-item ", "set-item ", "remove-item ",
        }):
            is_write = True

    if not is_query and not is_write:
        return

    if is_query:
        _add_candidate(result, CLS_QUERY_REGISTRY)
        _add_prior(result, CLS_QUERY_REGISTRY, 0.28)
        _evidence(result, f"registry_query: {exe} {sub} — registry enumeration/query")

    if is_write:
        _add_candidate(result, CLS_MODIFY_REGISTRY)
        _add_prior(result, CLS_MODIFY_REGISTRY, 0.28)
        _evidence(result, f"registry_write: {exe} {sub} — registry modification")

    _SECURITY_HINTS = {"defender", "firewall", "antivirus", "spynet",
                       "disableantispyware", "disablerealtimemonitoring",
                       "tamperprotection", "securityhealth", "safeboot",
                       "windows defender", "real-time protection"}
    if is_write and any(h in norm for h in _SECURITY_HINTS):
        _add_candidate(result, CLS_IMPAIR_DEFENSES)
        _add_prior(result, CLS_IMPAIR_DEFENSES, 0.35)
        _evidence(result, "registry_security_modification: registry write targets security settings")


def _rule_defense_impairment(parsed: dict, sem: dict, result: dict) -> None:
    """Disabling security tools, firewalls, audit, SELinux, etc."""
    exe = parsed.get("executable", "").lower()
    sub = (parsed.get("subcommand") or "").lower()
    norm = parsed.get("normalized_command", "").lower()
    flags_l = [f.lower() for f in (parsed.get("flags") or [])]

    fires = False
    # PowerShell AV manipulation
    if exe in {"set-mppreference", "disable-windowsoptionalfeature",
               "remove-mppreference", "add-mppreference"}:
        fires = True
    # Stop security services (PowerShell)
    _SEC_SERVICES = {"windefend", "msmpsvc", "sense", "wuauserv",
                     "wscsvc", "securityhealthservice", "mpssvc"}
    if exe in {"stop-service", "disable-service"} and any(
            s in norm for s in _SEC_SERVICES):
        fires = True
    # net stop for security services
    if exe == "net" and sub == "stop" and any(s in norm for s in _SEC_SERVICES):
        fires = True
    # Firewall manipulation
    if exe == "netsh" and "firewall" in norm:
        fires = True
    if exe == "ufw" and sub in {"disable", "delete", "reset"}:
        fires = True
    if exe == "iptables" and ("-f" in flags_l or "--flush" in flags_l
                              or "-x" in flags_l or "-d" in flags_l):
        fires = True
    # SELinux/AppArmor
    if exe == "setenforce" and "0" in (parsed.get("positional_args") or []):
        fires = True
    if exe in {"aa-disable", "aa-teardown"}:
        fires = True
    if exe == "systemctl" and sub in {"disable", "stop"} and any(
            s in norm for s in {"apparmor", "firewalld", "ufw", "selinux"}):
        fires = True
    # Audit manipulation
    if exe == "auditpol" and ("set" in norm or "/set" in norm):
        fires = True
    if exe == "wevtutil" and sub in {"cl", "clear-log"}:
        fires = True
    if not fires:
        return
    _add_candidate(result, CLS_IMPAIR_DEFENSES)
    _add_prior(result, CLS_IMPAIR_DEFENSES, 0.55)
    _evidence(result, f"defense_impairment: {exe} — security tool/policy manipulation")


def _rule_permission_modification(parsed: dict, sem: dict, result: dict) -> None:
    """File/directory permission modification."""
    exe = parsed.get("executable", "").lower()
    if exe not in _PERM_EXES:
        return
    _add_candidate(result, CLS_FILE_PERM_MOD)
    _add_prior(result, CLS_FILE_PERM_MOD, 0.50)
    _evidence(result, f"permission_mod: {exe} — file/directory permission change")


def _rule_indicator_removal(parsed: dict, sem: dict, result: dict) -> None:
    """Log deletion, timestamp tampering, history clearing."""
    exe = parsed.get("executable", "").lower()
    sub = (parsed.get("subcommand") or "").lower()
    norm = parsed.get("normalized_command", "").lower()
    flags_l = [f.lower() for f in (parsed.get("flags") or [])]
    file_paths_l = [p.lower() for p in (parsed.get("file_paths") or [])]

    fires = False
    # touch -t (timestamp manipulation)
    if exe == "touch" and any(f in flags_l for f in {"-t", "-d", "-r"}):
        fires = True
    # Deleting log files
    _LOG_HINTS = {"/var/log/", "\\logs\\", ".log", "auth.log", "syslog",
                  "security.evtx", "system.evtx", "application.evtx",
                  "wtmp", "btmp", "lastlog", "utmp"}
    all_paths = " ".join(file_paths_l) + " " + norm
    if exe in {"rm", "del", "erase", "remove-item", "shred", "truncate"}:
        if any(h in all_paths for h in _LOG_HINTS):
            fires = True
        if any(p.endswith((".log", ".evtx", ".evt")) for p in file_paths_l):
            fires = True
    # Clear-EventLog / wevtutil
    if exe in {"clear-eventlog", "wevtutil"} and (
            sub in {"cl", "clear-log"} or "clear" in norm):
        fires = True
    # History clearing
    if ("histfile" in norm or "bash_history" in norm
            or ".zsh_history" in norm or "history -c" in norm):
        fires = True
    if not fires:
        return
    _add_candidate(result, CLS_INDICATOR_REMOVAL)
    _add_prior(result, CLS_INDICATOR_REMOVAL, 0.45)
    _evidence(result, f"indicator_removal: {exe} — log/evidence manipulation")


def _rule_hide_artifacts(parsed: dict, sem: dict, result: dict) -> None:
    """Hidden files, attrib +h, ADS, etc."""
    exe = parsed.get("executable", "").lower()
    norm = parsed.get("normalized_command", "").lower()
    flags_l = [f.lower() for f in (parsed.get("flags") or [])]

    fires = False
    # attrib +h (Windows hidden attribute)
    if exe in {"attrib", "attrib.exe"} and ("+h" in flags_l or "+h" in norm):
        fires = True
    # Creating hidden files/dirs (Unix dot-prefix)
    if exe in {"mkdir", "touch", "cp", "mv", "copy"} and (
            "/." in norm or "\\." in norm):
        fires = True
    # Alternate Data Streams
    if ":" in norm and any(ext in norm for ext in {".txt:", ".exe:", ".dll:"}):
        fires = True
    # Set-ItemProperty / attrib to hide
    if "hidden" in norm and exe in {"set-itemproperty", "attrib", "chflags"}:
        fires = True
    if not fires:
        return
    _add_candidate(result, CLS_HIDE_ARTIFACTS)
    _add_prior(result, CLS_HIDE_ARTIFACTS, 0.40)
    _evidence(result, f"hide_artifacts: {exe} — hidden file/artifact manipulation")


def _rule_file_dir_discovery(parsed: dict, sem: dict, result: dict) -> None:
    """File and directory enumeration."""
    exe = parsed.get("executable", "").lower()
    norm = parsed.get("normalized_command", "").lower()
    if exe not in _FILE_DISCOVERY_EXES:
        return
    # find -perm is privilege escalation, not pure discovery
    if exe == "find" and "-perm" in norm:
        return
    _add_candidate(result, CLS_FILE_DIR_DISCOVERY)
    _add_prior(result, CLS_FILE_DIR_DISCOVERY, 0.35)
    _evidence(result, f"file_discovery: {exe} — file/directory enumeration")


def _rule_system_info_discovery(parsed: dict, sem: dict, result: dict) -> None:
    """System information gathering."""
    exe = parsed.get("executable", "").lower()
    norm = parsed.get("normalized_command", "").lower()

    fires = exe in _SYSTEM_INFO_DIRECT
    # cat /etc/os-release, /proc/version, etc.
    if exe in {"cat", "type", "get-content"} and any(
            p in norm for p in {"/etc/os-release", "/proc/version",
                                "/proc/cpuinfo", "/etc/hostname",
                                "/etc/machine-id"}):
        fires = True
    if not fires:
        return
    _add_candidate(result, CLS_SYSTEM_INFO_DISCOVERY)
    _add_prior(result, CLS_SYSTEM_INFO_DISCOVERY, 0.40)
    _evidence(result, f"system_info_discovery: {exe} — system information gathering")


def _rule_process_discovery(parsed: dict, sem: dict, result: dict) -> None:
    """Process listing / enumeration."""
    exe = parsed.get("executable", "").lower()
    norm = parsed.get("normalized_command", "").lower()

    fires = exe in _PROCESS_DISCOVERY_EXES
    # wmic process list/get (but not "process call create")
    if exe == "wmic" and (parsed.get("subcommand") or "").lower() == "process":
        if "call" not in norm and "create" not in norm:
            fires = True
    if not fires:
        return
    _add_candidate(result, CLS_PROCESS_DISCOVERY)
    _add_prior(result, CLS_PROCESS_DISCOVERY, 0.40)
    _evidence(result, f"process_discovery: {exe} — process enumeration")


def _rule_credential_access_broad(parsed: dict, sem: dict, result: dict) -> None:
    """Credential access — file reads, credential tools."""
    exe = parsed.get("executable", "").lower()
    fires = sem.get("reads_credential_store") or exe in _CREDENTIAL_EXES
    if not fires:
        return
    _add_candidate(result, CLS_CREDENTIAL_DUMPING)
    _add_prior(result, CLS_CREDENTIAL_DUMPING, 0.55)
    _evidence(result, f"credential_access: {exe} — credential store access/dumping")


def _rule_privilege_escalation(parsed: dict, sem: dict, result: dict) -> None:
    """Sudo, SUID search, elevation control abuse."""
    exe = parsed.get("executable", "").lower()
    norm = parsed.get("normalized_command", "").lower()

    fires = exe in _PRIV_ESC_EXES
    # find -perm +2000/+4000 (SUID/SGID search)
    if exe == "find" and "-perm" in norm and any(
            p in norm for p in {"+4000", "+2000", "/4000", "/2000",
                                "-4000", "-2000", "u+s", "g+s"}):
        fires = True
    # Start-Process -Verb RunAs (PowerShell elevation)
    if exe == "start-process" and "runas" in norm:
        fires = True
    if not fires:
        return
    _add_candidate(result, CLS_ABUSE_ELEVATION)
    _add_prior(result, CLS_ABUSE_ELEVATION, 0.40)
    _evidence(result, f"privilege_escalation: {exe} — elevation control abuse")


def _rule_lolbin_proxy_broad(parsed: dict, sem: dict, result: dict) -> None:
    """LOLBin/signed binary proxy execution (beyond download/encoding)."""
    exe = parsed.get("executable", "").lower()
    lolbin_matches = parsed.get("lolbin_matches", [])

    if exe not in _LOLBIN_PROXY_EXES and not lolbin_matches:
        return
    _add_candidate(result, CLS_SIGNED_BINARY_PROXY)
    _add_prior(result, CLS_SIGNED_BINARY_PROXY, 0.35)
    _evidence(result, f"lolbin_proxy: {exe} — signed binary proxy / LOLBin execution")

    # LOLBins that double as download tools
    if exe in {"certutil", "bitsadmin", "bitsadmin.exe"}:
        _add_candidate(result, CLS_INGRESS_TOOL_TRANSFER)
        _add_prior(result, CLS_INGRESS_TOOL_TRANSFER, 0.25)


def _rule_event_triggered(parsed: dict, sem: dict, result: dict) -> None:
    """Event-triggered execution: trap, udev, accessibility features."""
    exe = parsed.get("executable", "").lower()
    norm = parsed.get("normalized_command", "").lower()

    fires = False
    if exe == "trap":
        fires = True
    if exe in {"atbroker", "atbroker.exe"}:
        fires = True
    if "udev" in norm and ("rules.d" in norm or "rule" in norm):
        fires = True
    # Accessibility feature replacement
    if any(a in norm for a in {"narrator.exe", "osk.exe", "magnify.exe",
                                "sethc.exe", "utilman.exe",
                                "displayswitch.exe"}):
        fires = True
    # WMI event subscription
    if "eventfilter" in norm or "eventconsumer" in norm or "__event" in norm:
        fires = True
    # XDG autostart .desktop
    if "autostart" in norm and ".desktop" in norm:
        fires = True
    if not fires:
        return
    _add_candidate(result, CLS_EVENT_TRIGGERED_EXEC)
    _add_prior(result, CLS_EVENT_TRIGGERED_EXEC, 0.40)
    _evidence(result, f"event_triggered: {exe} — event-triggered execution")


def _rule_boot_logon_broad(parsed: dict, sem: dict, result: dict) -> None:
    """Boot/logon autostart: login scripts, profile.d, init.d, startup folder."""
    norm = parsed.get("normalized_command", "").lower()
    exe = parsed.get("executable", "").lower()

    _AUTOSTART_PATHS = {
        ".bashrc", ".bash_profile", ".profile", ".zshrc", ".cshrc",
        "/etc/profile", "/etc/profile.d/", "profile.d/",
        "/etc/rc.local", "rc.local", "/etc/init.d/", "init.d/",
        "start menu\\programs\\startup", "start menu/programs/startup",
        "\\appdata\\roaming\\microsoft\\windows\\start menu",
        "xdg/autostart",
    }
    fires = any(p in norm for p in _AUTOSTART_PATHS)
    # systemd enable (for persistence)
    if exe == "systemctl" and (parsed.get("subcommand") or "").lower() == "enable":
        fires = True
    if not fires:
        return
    _add_candidate(result, CLS_BOOT_LOGON_AUTOSTART)
    _add_prior(result, CLS_BOOT_LOGON_AUTOSTART, 0.35)
    _evidence(result, "boot_logon: autostart/login script path detected")


def _rule_hijack_execution(parsed: dict, sem: dict, result: dict) -> None:
    """DLL search order hijacking, LD_PRELOAD, PATH manipulation."""
    norm = parsed.get("normalized_command", "").lower()
    exe = parsed.get("executable", "").lower()

    fires = False
    # LD_PRELOAD / LD_LIBRARY_PATH / DYLD_*
    if any(e in norm for e in {"ld_preload", "ld_library_path", "dyld_insert",
                                "dyld_framework", "dyld_library"}):
        fires = True
    # DLL-related patterns
    if any(p in norm for p in {"dll hijack", "dllmain", "sideload",
                                "search order"}):
        fires = True
    # PATH manipulation for hijacking
    if exe in {"export", "set", "setx", "setx.exe"} and "path" in norm:
        fires = True
    if not fires:
        return
    _add_candidate(result, CLS_HIJACK_EXEC_FLOW)
    _add_prior(result, CLS_HIJACK_EXEC_FLOW, 0.35)
    _evidence(result, "hijack_execution: execution flow hijacking pattern")


def _rule_data_from_local(parsed: dict, sem: dict, result: dict) -> None:
    """Reading data from the local file system."""
    exe = parsed.get("executable", "").lower()
    file_paths = parsed.get("file_paths", [])

    if exe not in _FILE_READ_EXES:
        return
    # Only fire if there are actual file paths being read
    if not file_paths and exe not in {"strings", "xxd", "hexdump", "od"}:
        return
    _add_candidate(result, CLS_DATA_FROM_LOCAL)
    _add_prior(result, CLS_DATA_FROM_LOCAL, 0.25)
    _evidence(result, f"data_from_local: {exe} — reading local file data")


def _rule_data_destruction(parsed: dict, sem: dict, result: dict) -> None:
    """Destructive file/disk operations."""
    exe = parsed.get("executable", "").lower()
    norm = parsed.get("normalized_command", "").lower()
    flags_l = [f.lower() for f in (parsed.get("flags") or [])]

    fires = False
    if exe in {"shred", "wipefs", "srm", "sdelete", "sdelete.exe"}:
        fires = True
    if exe == "dd" and any(p in norm for p in {"if=/dev/zero", "if=/dev/null",
                                                "if=/dev/urandom"}):
        fires = True
    if exe == "diskpart" and "clean" in norm:
        fires = True
    # rm -rf / del /f (extreme deletion)
    if exe in {"rm"} and "-rf" in " ".join(flags_l):
        fires = True
    if not fires:
        return
    _add_candidate(result, CLS_DATA_DESTRUCTION)
    _add_prior(result, CLS_DATA_DESTRUCTION, 0.35)
    _evidence(result, f"data_destruction: {exe} — destructive operation")


def _rule_account_management(parsed: dict, sem: dict, result: dict) -> None:
    """Account/group creation and manipulation."""
    exe = parsed.get("executable", "").lower()
    sub = (parsed.get("subcommand") or "").lower()
    norm = parsed.get("normalized_command", "").lower()
    args_l = [a.lower() for a in (parsed.get("positional_args") or [])]

    fires = exe in _ACCOUNT_MGMT_EXES
    # net user /add, net localgroup /add
    if exe == "net" and sub in {"user", "localgroup", "group"}:
        if "/add" in norm or "add" in args_l:
            fires = True
    if not fires:
        return
    _add_candidate(result, CLS_ACCOUNT_MGMT)
    _add_prior(result, CLS_ACCOUNT_MGMT, 0.45)
    _evidence(result, f"account_mgmt: {exe} — account/group manipulation")


def _rule_masquerading_heuristic(parsed: dict, sem: dict, result: dict) -> None:
    """Copy/rename to system paths, suspicious filename patterns."""
    exe = parsed.get("executable", "").lower()
    norm = parsed.get("normalized_command", "").lower()

    _SYSTEM_PATHS = {"\\windows\\system32", "\\windows\\syswow64",
                     "\\system32\\", "\\syswow64\\",
                     "/usr/bin/", "/usr/sbin/", "/usr/local/bin/"}
    fires = False
    if exe in {"copy", "cp", "move", "mv", "copy-item", "move-item",
               "xcopy", "robocopy", "ren", "rename", "rename-item"}:
        if any(sp in norm for sp in _SYSTEM_PATHS):
            fires = True
    if not fires:
        return
    _add_candidate(result, CLS_MASQUERADING)
    _add_prior(result, CLS_MASQUERADING, 0.30)
    _evidence(result, f"masquerading: {exe} — file placed in/moved to system path")


def _rule_native_api_heuristic(parsed: dict, sem: dict, result: dict) -> None:
    """Windows API calls, DllImport, P/Invoke patterns."""
    norm = parsed.get("normalized_command", "").lower()

    _API_PATTERNS = {"virtualalloc", "virtualprotect", "createremotethread",
                     "ntcreatethreadex", "writeprocessmemory", "readprocessmemory",
                     "openprocess", "createprocess", "shellexecute", "winexec",
                     "loadlibrary", "getprocaddress", "dllimport", "p/invoke",
                     "add-type", "interopservices", "reflection.assembly",
                     "reflection.emit", "kernel32", "ntdll", "advapi32"}
    if not any(p in norm for p in _API_PATTERNS):
        return
    _add_candidate(result, CLS_NATIVE_API)
    _add_prior(result, CLS_NATIVE_API, 0.30)
    _evidence(result, "native_api: API call / P/Invoke pattern detected")


def _rule_keyword_heuristic(parsed: dict, sem: dict, result: dict) -> None:
    """Keyword-based heuristic for fragments and unrecognized commands."""
    norm = parsed.get("normalized_command", "").lower()
    if len(norm) < 3:
        return

    # Only fire keyword heuristic if no high-confidence rule already matched
    if result["candidate_classes"] and max(result["priors"].values(), default=0) > 0.5:
        return

    _KEYWORD_MAP = [
        ({"ld_preload", "ld_library", "dyld_"}, CLS_HIJACK_EXEC_FLOW, 0.20),
        ({"atbroker", "sethc", "utilman", "narrator.exe", "osk.exe",
          "magnify.exe"}, CLS_EVENT_TRIGGERED_EXEC, 0.20),
        ({"lsass", "ntds.dit", "mimikatz", "hashdump", "credential"},
         CLS_CREDENTIAL_DUMPING, 0.20),
        ({"defender", "antimalware", "antivirus", "security center"},
         CLS_IMPAIR_DEFENSES, 0.20),
        ({"autorun", "startup folder", "start menu"},
         CLS_BOOT_LOGON_AUTOSTART, 0.20),
        ({"suid", "setuid", "sgid", "setgid"},
         CLS_ABUSE_ELEVATION, 0.20),
        # File/path fragments
        ({".dll", ".exe", ".sys", ".drv"}, CLS_SIGNED_BINARY_PROXY, 0.15),
        ({"\\windows\\", "\\system32\\", "\\syswow64\\"},
         CLS_SIGNED_BINARY_PROXY, 0.15),
        ({"/usr/bin/", "/usr/sbin/", "/bin/", "/sbin/", "/opt/"},
         CLS_CMD_SCRIPTING, 0.15),
        # Registry-related fragments
        ({"hkcu\\", "hklm\\", "hkcr\\", "hku\\", "registry"},
         CLS_MODIFY_REGISTRY, 0.15),
    ]

    for keywords, cls, prior in _KEYWORD_MAP:
        if any(k in norm for k in keywords):
            _add_candidate(result, cls)
            _add_prior(result, cls, prior)
            _evidence(result, f"keyword_heuristic: matched {cls} via keyword")


def _rule_command_content_scan(parsed: dict, sem: dict, result: dict) -> None:
    """Scan normalized_command for known tool patterns in chained/piped commands."""
    norm = parsed.get("normalized_command", "").lower()
    exe = parsed.get("executable", "").lower()

    # Only useful for chained commands (pipe, /c, -c, &&, semicolons)
    if not (parsed.get("has_pipe") or "/c " in norm or "-c " in norm
            or "&&" in norm or "; " in norm):
        return

    _NET_TOOLS = {"ipconfig", "ifconfig", "netstat", "route ", "arp ", "nslookup", "dig "}
    _ID_TOOLS = {"whoami", "net user"}
    _SYSINFO_TOOLS = {"systeminfo", "uname ", "hostname"}
    _PROC_TOOLS = {"tasklist", "get-process", "wmic process"}
    _REG_TOOLS = {" reg ", "regedit"}

    for tool in _NET_TOOLS:
        if tool in norm and tool.strip() != exe:
            _add_candidate(result, CLS_NETWORK_CONFIG_DISCOVERY)
            _add_prior(result, CLS_NETWORK_CONFIG_DISCOVERY, 0.25)
            _evidence(result, f"content_scan: chained command contains {tool.strip()}")
            break

    for tool in _ID_TOOLS:
        if tool in norm and tool.strip() != exe:
            _add_candidate(result, CLS_ACCOUNT_DISCOVERY)
            _add_prior(result, CLS_ACCOUNT_DISCOVERY, 0.25)
            break

    for tool in _SYSINFO_TOOLS:
        if tool in norm and tool.strip() != exe:
            _add_candidate(result, CLS_SYSTEM_INFO_DISCOVERY)
            _add_prior(result, CLS_SYSTEM_INFO_DISCOVERY, 0.25)
            break

    for tool in _PROC_TOOLS:
        if tool in norm and tool.strip() != exe:
            _add_candidate(result, CLS_PROCESS_DISCOVERY)
            _add_prior(result, CLS_PROCESS_DISCOVERY, 0.25)
            break

    for tool in _REG_TOOLS:
        if tool in norm and tool.strip() != exe:
            _add_candidate(result, CLS_MODIFY_REGISTRY)
            _add_prior(result, CLS_MODIFY_REGISTRY, 0.20)
            break


def _rule_powershell_cmdlets(parsed: dict, sem: dict, result: dict) -> None:
    """PowerShell cmdlets that map to specific rule classes."""
    exe = parsed.get("executable", "").lower()

    _PS_CMDLET_MAP = {
        # File operations
        "copy-item": CLS_DATA_FROM_LOCAL,
        "move-item": CLS_DATA_FROM_LOCAL,
        "get-content": CLS_DATA_FROM_LOCAL,
        "new-item": CLS_CMD_SCRIPTING,
        "remove-item": CLS_INDICATOR_REMOVAL,
        "get-childitem": CLS_FILE_DIR_DISCOVERY,
        # Process operations
        "start-process": CLS_CMD_SCRIPTING,
        "stop-process": CLS_IMPAIR_DEFENSES,
        "get-process": CLS_PROCESS_DISCOVERY,
        # Network
        "invoke-webrequest": CLS_INGRESS_TOOL_TRANSFER,
        "invoke-restmethod": CLS_INGRESS_TOOL_TRANSFER,
        "invoke-expression": CLS_CMD_SCRIPTING,
        "invoke-command": CLS_CMD_SCRIPTING,
        # Module/import
        "import-module": CLS_CMD_SCRIPTING,
        # Output (used in scripts)
        "write-host": CLS_CMD_SCRIPTING,
        "write-output": CLS_CMD_SCRIPTING,
        "write-error": CLS_CMD_SCRIPTING,
    }
    cls = _PS_CMDLET_MAP.get(exe)
    if cls is None:
        return
    _add_candidate(result, cls)
    _add_prior(result, cls, 0.25)
    _evidence(result, f"ps_cmdlet: {exe} → {cls}")


def _rule_file_operations(parsed: dict, sem: dict, result: dict) -> None:
    """Copy, move, rm, mkdir and similar file manipulation commands."""
    exe = parsed.get("executable", "").lower()
    norm = parsed.get("normalized_command", "").lower()

    _COPY_CMDS = {"copy", "cp", "xcopy", "robocopy"}
    _MOVE_CMDS = {"move", "mv", "ren", "rename"}
    _DEL_CMDS = {"rm", "del", "erase", "shred", "unlink"}
    _MKDIR_CMDS = {"mkdir", "md"}

    if exe in _COPY_CMDS or exe in _MOVE_CMDS:
        _add_candidate(result, CLS_DATA_FROM_LOCAL)
        _add_prior(result, CLS_DATA_FROM_LOCAL, 0.20)
        _evidence(result, f"file_op: {exe} — file copy/move operation")
    elif exe in _DEL_CMDS:
        _add_candidate(result, CLS_INDICATOR_REMOVAL)
        _add_prior(result, CLS_INDICATOR_REMOVAL, 0.20)
        _evidence(result, f"file_op: {exe} — file deletion")
    elif exe in _MKDIR_CMDS:
        _add_candidate(result, CLS_CMD_SCRIPTING)
        _add_prior(result, CLS_CMD_SCRIPTING, 0.15)
        _evidence(result, f"file_op: {exe} — directory creation")


def _rule_net_command(parsed: dict, sem: dict, result: dict) -> None:
    """The 'net' command covers many sub-operations."""
    exe = parsed.get("executable", "").lower()
    sub = (parsed.get("subcommand") or "").lower()
    if exe != "net":
        return

    _NET_SUB_MAP = {
        "user": CLS_ACCOUNT_DISCOVERY,
        "localgroup": CLS_ACCOUNT_DISCOVERY,
        "group": CLS_ACCOUNT_DISCOVERY,
        "view": CLS_REMOTE_SYSTEM_DISCOVERY,
        "share": CLS_NETWORK_CONFIG_DISCOVERY,
        "use": CLS_REMOTE_SERVICES,
        "start": CLS_CREATE_MODIFY_SERVICE,
        "stop": CLS_IMPAIR_DEFENSES,
        "session": CLS_REMOTE_SYSTEM_DISCOVERY,
        "accounts": CLS_ACCOUNT_DISCOVERY,
        "config": CLS_NETWORK_CONFIG_DISCOVERY,
    }
    cls = _NET_SUB_MAP.get(sub)
    if cls:
        _add_candidate(result, cls)
        _add_prior(result, cls, 0.35)
        _evidence(result, f"net_command: net {sub} → {cls}")
    else:
        # Unknown subcommand — weak generic signal
        _add_candidate(result, CLS_CMD_SCRIPTING)
        _add_prior(result, CLS_CMD_SCRIPTING, 0.15)
        _evidence(result, f"net_command: net {sub} — generic")


def _rule_scripting_builtins(parsed: dict, sem: dict, result: dict) -> None:
    """Shell builtins / scripting constructs (echo, if, for, set, etc).
    Very weak signal — just enough to avoid full fallback."""
    exe = parsed.get("executable", "").lower()
    if exe not in _SCRIPTING_BUILTINS:
        return
    _add_candidate(result, CLS_CMD_SCRIPTING)
    _add_prior(result, CLS_CMD_SCRIPTING, 0.15)
    _evidence(result, f"scripting_builtin: {exe} — shell/script builtin")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Rule strength classification
# ─────────────────────────────────────────────────────────────────────────────
# STRONG: high-confidence deterministic structural signals.
# WEAK:   broad / ambiguous signals — ML expected to resolve.
# Rules not listed default to WEAK.

STRONG_RULES = frozenset({
    _rule_registry_persistence,
    _rule_scheduled_task,
    _rule_download_ingress,
    _rule_encoded_execution,
    _rule_wmi_remote_execution,
    _rule_shadow_copy_deletion,
    _rule_service_creation,
    _rule_defense_impairment,
    _rule_credential_access_broad,
    _rule_account_management,
    _rule_data_destruction,
    _rule_permission_modification,
    _rule_indicator_removal,
})

WEAK_RULES = frozenset({
    _rule_archive_collection,
    _rule_remote_session,
    _rule_discovery,
    _rule_interpreter_general,
    _rule_registry_broad,
    _rule_hide_artifacts,
    _rule_file_dir_discovery,
    _rule_system_info_discovery,
    _rule_process_discovery,
    _rule_privilege_escalation,
    _rule_lolbin_proxy_broad,
    _rule_event_triggered,
    _rule_boot_logon_broad,
    _rule_hijack_execution,
    _rule_data_from_local,
    _rule_masquerading_heuristic,
    _rule_native_api_heuristic,
    _rule_command_content_scan,
    _rule_powershell_cmdlets,
    _rule_file_operations,
    _rule_net_command,
    _rule_scripting_builtins,
    _rule_keyword_heuristic,
})

RULE_FAMILIES = [
    # ── Original 10 rules ──
    _rule_registry_persistence,
    _rule_scheduled_task,
    _rule_download_ingress,
    _rule_archive_collection,
    _rule_encoded_execution,
    _rule_wmi_remote_execution,
    _rule_shadow_copy_deletion,
    _rule_service_creation,
    _rule_remote_session,
    _rule_discovery,
    # ── Extended rules (v1.1) ──
    _rule_interpreter_general,
    _rule_registry_broad,
    _rule_defense_impairment,
    _rule_permission_modification,
    _rule_indicator_removal,
    _rule_hide_artifacts,
    _rule_file_dir_discovery,
    _rule_system_info_discovery,
    _rule_process_discovery,
    _rule_credential_access_broad,
    _rule_privilege_escalation,
    _rule_lolbin_proxy_broad,
    _rule_event_triggered,
    _rule_boot_logon_broad,
    _rule_hijack_execution,
    _rule_data_from_local,
    _rule_data_destruction,
    _rule_account_management,
    _rule_masquerading_heuristic,
    _rule_native_api_heuristic,
    _rule_command_content_scan,
    _rule_powershell_cmdlets,
    _rule_file_operations,
    _rule_net_command,
    _rule_scripting_builtins,
    _rule_keyword_heuristic,  # must be last — weak heuristic
]


def build_rule_result(parsed: dict, semantic_features: dict) -> dict:
    """
    Run all rule families against parsed + semantic output.

    Returns dict with candidate_classes, banned_classes, priors, evidence,
    rule_strength ("strong"|"weak"|"none"), and fired_rules list.
    """
    result: Dict[str, Any] = {
        "candidate_classes": [],
        "banned_classes":    [],
        "priors":            {},
        "evidence":          [],
    }

    fired_rules: List[str] = []
    has_strong = False
    has_weak = False

    for rule_fn in RULE_FAMILIES:
        before = len(result["candidate_classes"]) + len(result["evidence"])
        rule_fn(parsed, semantic_features, result)
        after = len(result["candidate_classes"]) + len(result["evidence"])
        if after > before:
            fired_rules.append(rule_fn.__name__)
            if rule_fn in STRONG_RULES:
                has_strong = True
            else:
                has_weak = True

    # Remove any candidate that was subsequently banned
    result["candidate_classes"] = [
        c for c in result["candidate_classes"]
        if c not in result["banned_classes"]
    ]

    # Determine bucket
    if has_strong:
        strength = "strong"
    elif has_weak:
        strength = "weak"
    else:
        strength = "none"

    result["rule_strength"] = strength
    result["fired_rules"] = fired_rules

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI for manual inspection
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys

    from parser import parse_command
    from semantic_features import build_semantic_features

    argp = argparse.ArgumentParser(description="Run the Genos rule engine on a command.")
    argp.add_argument("command", nargs="?", help="Command string to analyse")
    args = argp.parse_args()

    if not args.command:
        print("Usage: python3 rule_engine.py '<command>'")
        sys.exit(1)

    parsed = parse_command(args.command)
    sem = build_semantic_features(parsed)
    rule_result = build_rule_result(parsed, sem)

    print(json.dumps({
        "parsed":    {k: v for k, v in parsed.items() if v not in (None, [], {}, "")},
        "semantic":  {k: v for k, v in sem.items() if isinstance(v, bool) and v},
        "rule_result": rule_result,
    }, indent=2))
