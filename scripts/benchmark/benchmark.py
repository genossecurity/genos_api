import argparse
import json
import os
import sys
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle, islice

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Ensure Genos directories are in path
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(BENCHMARK_DIR))  # Up to genos_api root
sys.path.insert(0, BASE_DIR)

try:
    from engine import GenosEngine
except ImportError:
    print("[!] ERROR: Could not import 'GenosEngine' from 'engine.py'.")
    print("    Ensure engine.py is in the same directory as benchmark.py.")
    sys.exit(1)

# ==========================================
# PATH RESOLVERS
# ==========================================
def resolve_data_path(filename: str) -> str:
    """Resolve training/test data path with fallback to archive."""
    candidates = [
        os.path.join(BASE_DIR, "data", "training", filename),
        os.path.join(BASE_DIR, "data", "test", filename),
        os.path.join(BASE_DIR, "data", "raw", filename),
        os.path.join(BASE_DIR, "data", "art", filename),
        os.path.join(BASE_DIR, "data", filename),
        filename
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

# ==========================================
# MODULE 1: AUDIT & CONFLICT FINDER
# ==========================================
def run_audit(training_csv="specialist_train.csv", atlas_csv="mitre_atlas_raw.csv"):
    train_path = resolve_data_path(training_csv)
    atlas_path = resolve_data_path(atlas_csv)

    print("=" * 60)
    print("GENOS LABEL REVEAL & ATLAS COVERAGE AUDIT")
    print("=" * 60)

    if not train_path:
        print(f"[!] Training data not found for {training_csv}.")
        return

    df = pd.read_csv(train_path)
    print(f"[*] Analyzing Training File: {train_path}")
    print(f"[*] Total Rows: {len(df)}")
    
    unique_ids = df['mitre_id'].unique()
    print(f"[*] Total Unique IDs: {len(unique_ids)}")
    
    aml_samples = df[df['mitre_id'].str.contains('AML', na=False, case=False)]
    t_samples = df[df['mitre_id'].str.contains('^T[0-9]', na=False, case=False)]
    
    print("-" * 60)
    print(f"[+] ATLAS (AML) Samples Found: {len(aml_samples)}")
    print(f"[+] Enterprise (T-prefix) Samples Found: {len(t_samples)}")
    
    if atlas_path:
        atlas_df = pd.read_csv(atlas_path)
        all_atlas_ids = set(atlas_df['mitre_id'].unique())
        current_labels = set(unique_ids)
        
        missing = all_atlas_ids - current_labels
        coverage = (len(all_atlas_ids) - len(missing)) / len(all_atlas_ids) * 100
        
        print("-" * 60)
        print(f"[*] Total Official ATLAS Techniques: {len(all_atlas_ids)}")
        print(f"[*] Current ATLAS Coverage: {coverage:.2f}%")
        print(f"[*] Missing IDs: {len(missing)}")
        
        if missing:
            print("\n[!] TOP 5 MISSING ATLAS TECHNIQUES:")
            missing_details = atlas_df[atlas_df['mitre_id'].isin(missing)]
            for _, row in missing_details.head(5).iterrows():
                name = row.get('technique_name', 'Unknown')
                print(f"    - {row['mitre_id']}: {name}")
    else:
        print(f"\n[!] ATLAS raw file '{atlas_csv}' not found. Skipping coverage check.")
    print("=" * 60)

# ==========================================
# MODULE 2: FUZZER & CONFUSION MATRIX
# ==========================================
def run_fuzzer(test_csv="specialist_test.csv"):
    test_path = resolve_data_path(test_csv)
    if not test_path:
        print(f"[!] Test data not found for {test_csv}.")
        return

    print("[*] INITIALIZING GENOS ENGINE FOR FUZZING...")
    engine = GenosEngine()
    
    df = pd.read_csv(test_path)
    print(f"[*] Fuzzing {len(df)} samples across classes...")

    y_true = []
    y_pred = []
    
    class_stats = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning"):
        cmd = row['command']
        true_id = str(row['mitre_id'])
        y_true.append(true_id)
        
        if true_id not in class_stats:
            class_stats[true_id] = {"correct": 0, "total": 0}
        class_stats[true_id]["total"] += 1

        res = engine.scan(cmd)
        
        pred_id = "None"
        if res.get('MITRE_codes') and len(res['MITRE_codes']) > 0:
            pred_id = res['MITRE_codes'][0]['code']
        y_pred.append(pred_id)

        if pred_id == true_id:
            class_stats[true_id]["correct"] += 1

    # Fuzzer Gap Analysis
    report = []
    for code, stats in class_stats.items():
        recall = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        report.append({
            "MITRE_ID": code,
            "Total_Samples": stats['total'],
            "Correct": stats['correct'],
            "Recall_Rate": round(recall, 2)
        })

    logs_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    report_df = pd.DataFrame(report).sort_values(by="Recall_Rate", ascending=True)
    gap_csv = os.path.join(logs_dir, 'mitre_gap_analysis.csv')
    report_df.to_csv(gap_csv, index=False)
    print(f"\n[+] Gap Analysis saved to: {gap_csv}")

    # Confusion Matrix Logic
    labels = sorted(list(set(y_true + y_pred)))
    
    print("\n[*] Generating F1 Classification Report...")
    clf_report = classification_report(y_true, y_pred, labels=labels, digits=2, zero_division=0)
    report_txt = os.path.join(logs_dir, "specialist_f1_report.txt")
    with open(report_txt, "w") as f:
        f.write(clf_report)
    print(f"[+] Detailed F1 report saved to '{report_txt}'")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confusions.append((labels[i], labels[j], cm[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    print("\n[!] TOP 10 CROSS-TECHNIQUE CONFUSIONS:")
    for actual, predicted, count in confusions[:10]:
        print(f"    True: {actual:10} mistaken for {predicted:10} | {count} times")

    # Heatmap Generation (Top 20 Problematic Classes)
    if len(confusions) > 0:
        problematic = list(set([c[0] for c in confusions[:15]] + [c[1] for c in confusions[:15]]))
        prob_indices = [labels.index(p) for p in problematic if p in labels]
        
        if prob_indices:
            short_cm = cm[np.ix_(prob_indices, prob_indices)]
            plt.figure(figsize=(14, 12))
            sns.heatmap(short_cm, annot=True, fmt='d', cmap='OrRd', 
                        xticklabels=[labels[i] for i in prob_indices], 
                        yticklabels=[labels[i] for i in prob_indices])
            plt.title('Genos Top Confusions Heatmap')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            hm_path = os.path.join(logs_dir, "specialist_confusion.png")
            plt.savefig(hm_path)
            print(f"[+] Confusion Heatmap saved to '{hm_path}'")

# ==========================================
# MODULE 3: DIRECT ENGINE STRESS TEST
# ==========================================
def run_stress_test(max_requests=200, workers=8):
    print("=" * 60)
    print(f"GENOS DIRECT ENGINE STRESS TEST ({workers} Workers)")
    print("=" * 60)

    print("[*] Loading Engine into Memory...")
    engine = GenosEngine()
    
    # Payload mix from your eval.py
    payloads = [
        "Add-Type -TypeDefinition @' [DllImport(\"kernel32.dll\")] public static extern IntPtr LoadLibrary(string d); '@",
        "[Reflection.Assembly]::Load([System.Convert]::FromBase64String('TVqQAAMAAAAEAAAA//8AALgAAAA...'))",
        "rundll32.exe advpack.dll,LaunchINFSection c:\\temp\\test.inf,DefaultInstall",
        "reg save HKLM\\SAM C:\\windows\\temp\\sam.save",
        "comsvcs.dll, MiniDump (Get-Process lsass).id C:\\windows\\temp\\l.dmp",
        "kubectl get pods -n production --watch",  # Benign control
        "powershell -ExecutionPolicy Bypass -File \\\\server\\share\\script.ps1"
    ]

    req_cycle = islice(cycle(payloads), max_requests)
    
    def process_command(cmd):
        start = time.perf_counter()
        res = engine.scan(cmd)
        latency = (time.perf_counter() - start) * 1000
        return cmd, res, latency

    latencies = []
    print(f"[*] Blasting {max_requests} requests...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_command, cmd) for cmd in req_cycle]
        for future in tqdm(as_completed(futures), total=max_requests, desc="Benchmarking"):
            cmd, res, latency = future.result()
            latencies.append(latency)

    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)

    print("-" * 60)
    print("STRESS TEST RESULTS")
    print("-" * 60)
    print(f"Total Requests Processed : {max_requests}")
    print(f"Average Latency          : {avg_lat:.2f} ms")
    print(f"95th Percentile Latency  : {p95_lat:.2f} ms")
    print(f"99th Percentile Latency  : {p99_lat:.2f} ms")
    print("=" * 60)

# ==========================================
# MAIN ROUTER
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genos Unified Benchmark & Evaluation Tool")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["audit", "fuzz", "stress", "all"], 
        default="all",
        help="Which benchmark suite to run."
    )
    parser.add_argument("--train-csv", default="specialist_train.csv", help="Training data CSV for audit.")
    parser.add_argument("--test-csv", default="specialist_test.csv", help="Test data CSV for fuzzer.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrency for stress test.")
    parser.add_argument("--requests", type=int, default=200, help="Total requests for stress test.")
    
    args = parser.parse_args()

    if args.mode in ["audit", "all"]:
        run_audit(training_csv=args.train_csv)
        print("\n")
        
    if args.mode in ["fuzz", "all"]:
        run_fuzzer(test_csv=args.test_csv)
        print("\n")
        
    if args.mode in ["stress", "all"]:
        run_stress_test(max_requests=args.requests, workers=args.workers)
        print("\n")