import os
import json
import time
import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:6001/scan"  # Change to your actual URL/IP if testing remotely
API_KEY = "MS9R-VMPR-AA8L-Y7PH"         # MUST match the API key expected by api.py
INTERNAL_TOKEN = os.getenv("INTERNAL_TEST_TOKEN", "")
HEADERS = {
    "Content-Type": "application/json", 
    "X-API-Key": API_KEY
}

# Testing Parameters
CONCURRENCY_LIMIT = 20  # How many requests to send simultaneously
TOTAL_REQUESTS = 500    # Total samples to test
MALICIOUS_RATIO = 0.5   # 50% malicious, 50% benign
CONFIDENCE_THRESHOLD = 85.0 # Minimum confidence % to trigger a "Malicious" alert

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Fallback to current directory if not in scripts/benchmark
if not os.path.exists(os.path.join(BASE_DIR, "data")):
    BASE_DIR = os.path.dirname(os.path.dirname(BASE_DIR)) 

BENIGN_CSV = os.path.join(BASE_DIR, "data", "training", "gatekeeper_test.csv")
MALICIOUS_CSV = os.path.join(BASE_DIR, "data", "training", "specialist_test.csv")
LOG_FILE = "live_stress_report.txt"

def load_test_data(total_samples, malicious_ratio):
    """Loads a random, balanced sample of real data from the test CSVs."""
    print("[*] Loading and shuffling test data from CSVs...")
    
    malicious_count = int(total_samples * malicious_ratio)
    benign_count = total_samples - malicious_count
    
    all_cmds = []
    
    # Load Malicious
    if os.path.exists(MALICIOUS_CSV):
        m_df = pd.read_csv(MALICIOUS_CSV).sample(n=malicious_count, replace=True)
        for _, row in m_df.iterrows():
            all_cmds.append({
                "cmd": str(row['command']).strip(), 
                "type": "malicious",
                "true_id": str(row['mitre_id'])
            })
    else:
        print(f"[!] Warning: {MALICIOUS_CSV} not found.")

    # Load Benign
    if os.path.exists(BENIGN_CSV):
        b_df = pd.read_csv(BENIGN_CSV).sample(n=benign_count, replace=True)
        for _, row in b_df.iterrows():
            all_cmds.append({
                "cmd": str(row['command']).strip(), 
                "type": "benign",
                "true_id": "Benign"
            })
    else:
         print(f"[!] Warning: {BENIGN_CSV} not found.")

    # Shuffle the dataset
    df = pd.DataFrame(all_cmds).sample(frac=1).reset_index(drop=True)
    return df.to_dict('records')

async def fetch(session, semaphore, item, results, f):
    """Asynchronous worker to hit the API and log the result."""
    payload = {"command": item["cmd"]}

    # /scan requires api_key in JSON body; /scan/internal may optionally use internal_token.
    if "/scan/internal" in API_URL:
        if INTERNAL_TOKEN:
            payload["internal_token"] = INTERNAL_TOKEN
    else:
        payload["api_key"] = API_KEY
    
    # FIX: Force strict JSON string serialization to prevent aiohttp encoding errors
    payload_str = json.dumps(payload)
    
    async with semaphore:
        start_time = time.perf_counter()
        try:
            # FIX: Send using the 'data' parameter with our strict string, not 'json'
            async with session.post(API_URL, headers=HEADERS, data=payload_str, timeout=15) as resp:
                latency = (time.perf_counter() - start_time) * 1000
                
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Apply Threshold Logic
                    raw_label = data.get("label")
                    conf = data.get("label_confidence", 0.0)
                    
                    # Only consider it Malicious if it breaches the confidence threshold
                    is_mal = (raw_label == "Malicious") and (conf >= CONFIDENCE_THRESHOLD)
                    final_decision = "Malicious" if is_mal else "Benign"
                    
                    # Analytics
                    if item["type"] == "malicious":
                        if is_mal: results["tp"] += 1
                        else: results["fn"] += 1
                    else:
                        if is_mal: results["fp"] += 1
                        else: results["tn"] += 1
                        
                    results["total"] += 1
                    results["latencies"].append(latency)
                    
                    # Log to file
                    log_entry = (
                        f"Request #{results['total']}\n"
                        f"Command Type: {item['type']} (True ID: {item['true_id']})\n"
                        f"Command: {item['cmd']}\n"
                        f"API Label: {raw_label} ({conf}%)\n"
                        f"Final Decision (Threshold {CONFIDENCE_THRESHOLD}%): {final_decision}\n"
                        f"Latency: {latency:.2f}ms\n"
                        f"Full Response: {json.dumps(data, indent=2)}\n"
                        f"{'-'*60}\n\n"
                    )
                    f.write(log_entry)
                    
                else:
                    # FIX: Read the exact error message from FastAPI if it fails again
                    error_detail = await resp.text()
                    print(f"\n[!] Server Error {resp.status}: {error_detail}")
                    print(f"    Failed Command: {item['cmd'][:50]}...")
                    results["errors"] += 1
                    
        except asyncio.TimeoutError:
             print(f"\n[!] Timeout on command: {item['cmd'][:30]}...")
             results["errors"] += 1
        except Exception as e:
            print(f"\n[!] Request failed: {e}")
            results["errors"] += 1

async def run_stress_test():
    test_data = load_test_data(TOTAL_REQUESTS, MALICIOUS_RATIO)
    if not test_data:
        print("[!] No data loaded. Exiting.")
        return

    results = {"total": 0, "fp": 0, "fn": 0, "tp": 0, "tn": 0, "errors": 0, "latencies": []}
    
    # Initialize the log file
    with open(LOG_FILE, "w") as f:
        f.write("="*60 + "\n")
        f.write(f"GENOS V1.1 ASYNC STRESS TEST (Threshold: {CONFIDENCE_THRESHOLD}%)\n")
        f.write("="*60 + "\n\n")

    print(f"[*] Blasting {len(test_data)} requests with concurrency {CONCURRENCY_LIMIT}...")
    
    # Manage concurrent connections
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async with aiohttp.ClientSession() as session:
        with open(LOG_FILE, "a") as f:
            tasks = [fetch(session, semaphore, item, results, f) for item in test_data]
            # Run tasks with a progress bar
            await tqdm.gather(*tasks, desc="API Requests")

    # --- FINAL REPORT ---
    avg_latency = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0
    
    print("\n" + "="*50)
    print(f"{'GENOS V1.1 LIVE STRESS REPORT':^50}")
    print("="*50)
    print(f"Total Completed : {results['total']}")
    print(f"Failed Requests : {results['errors']}")
    print(f"Avg Latency     : {avg_latency:.2f} ms")
    print("-" * 50)
    print(f"True Positives  : {results['tp']} (Attacks Caught)")
    print(f"True Negatives  : {results['tn']} (Normal Traffic Ignored)")
    print(f"False Positives : {results['fp']} (Benign flagged as Malicious)")
    print(f"False Negatives : {results['fn']} (Attacks Missed)")
    print("-" * 50)
    
    precision = results["tp"] / (results["tp"] + results["fp"]) if (results["tp"] + results["fp"]) > 0 else 0
    recall = results["tp"] / (results["tp"] + results["fn"]) if (results["tp"] + results["fn"]) > 0 else 0
    
    print(f"Live Precision  : {precision*100:.2f}% (Threshold: {CONFIDENCE_THRESHOLD}%)")
    print(f"Live Recall     : {recall*100:.2f}% (Threshold: {CONFIDENCE_THRESHOLD}%)")
    print("="*50)
    print(f"[+] Full logs saved to {LOG_FILE}")

if __name__ == "__main__":
    # Required for Windows compatibility with asyncio
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(run_stress_test())