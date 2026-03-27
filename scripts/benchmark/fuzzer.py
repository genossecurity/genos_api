# Save as scripts/benchmark/full_mitre_audit.py
import os
import sys
import torch
import pandas as pd
import json
from tqdm import tqdm

# Resolve paths BEFORE importing engine
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from engine import GenosEngine

print("[*] INITIALIZING GENOS SPECIALIST AUDIT...")
engine = GenosEngine()

# Load the specialist map to decode IDs
specialist_map_path = os.path.join(BASE_DIR, 'config', 'specialist_map.json')
with open(specialist_map_path, 'r') as f:
    mitre_map = json.load(f)

# Load the Gold Standard Test Set
test_set_path = os.path.join(BASE_DIR, 'data', 'test', 'specialist_test_set.csv')
df = pd.read_csv(test_set_path)

# Initialize counters
class_stats = {code: {"correct": 0, "total": 0} for code in mitre_map.keys()}

print(f"[*] Auditing {len(df)} samples across 141 classes...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    cmd = row['command']
    true_id = str(row['mitre_id'])
    
    # Run Inference
    res = engine.scan(cmd)
    
    # Extract the top predicted code
    pred_id = "None"
    if 'MITRE_codes' in res and len(res['MITRE_codes']) > 0:
        pred_id = res['MITRE_codes'][0]['code']

    # Update Statistics
    if true_id in class_stats:
        class_stats[true_id]["total"] += 1
        if pred_id == true_id:
            class_stats[true_id]["correct"] += 1

# 3. Generate the Failure Report
report = []
for code, stats in class_stats.items():
    recall = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
    report.append({
        "MITRE_ID": code,
        "Total_Samples": stats['total'],
        "Correct": stats['correct'],
        "Recall_Rate": round(recall, 2)
    })

report_df = pd.DataFrame(report).sort_values(by="Recall_Rate", ascending=True)

# Create logs directory if needed
logs_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(logs_dir, exist_ok=True)
report_df.to_csv(os.path.join(logs_dir, 'mitre_gap_analysis.csv'), index=False)

# Summary Output
print("\n" + "="*60)
print(f"{'GENOS SPECIALIST PERFORMANCE SUMMARY':^60}")
print("="*60)
print(f"Total Techniques Tested : 141")
print(f"Dead Zones (0% Recall)  : {len(report_df[report_df['Recall_Rate'] == 0])}")
print(f"Elite Classes (100%)    : {len(report_df[report_df['Recall_Rate'] == 100])}")
print("-" * 60)
print("TOP 5 FAILURE GAPS (Fix these first):")
print(report_df.head(5).to_string(index=False))
print("="*60)