import pandas as pd
from engine import GenosEngine
from tqdm import tqdm
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_engine_audit():
    print("[*] Initializing Genos Engine for Full Pipeline Audit...")
    engine = GenosEngine()
    
    # 1. Load Data
    # Malicious Test Set (2,820 samples)
    m_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'test', 'specialist_test_set.csv'))
    
    # Benign Test Set (Let's take 2,820 random samples to keep it balanced)
    b_df = pd.read_csv(
        os.path.join(BASE_DIR, 'data', 'training', 'genos-balanced-good.csv')
    ).sample(n=len(m_df), random_state=42)

    stats = {
        "tier1": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "tier2": {"correct": 0, "total_attempted": 0}
    }

    # 2. Test Malicious Samples (Measuring Recall & Attribution)
    print(f"\n[*] Processing {len(m_df)} Malicious Samples...")
    for _, row in tqdm(m_df.iterrows(), total=len(m_df)):
        cmd = row['command']
        true_mitre = str(row['mitre_id'])
        
        res = engine.scan(cmd)
        
        if res.get('label', '').lower() == 'malicious':
            stats["tier1"]["tp"] += 1
            stats["tier2"]["total_attempted"] += 1
            
            # Check if Tier 2 got the ID right
            mitre_codes = res.get('MITRE_codes') or []
            pred_mitre = mitre_codes[0].get('code', 'None') if mitre_codes else 'None'
            if pred_mitre == true_mitre:
                stats["tier2"]["correct"] += 1
        else:
            stats["tier1"]["fn"] += 1

    # 3. Test Benign Samples (Measuring False Positive Rate)
    print(f"\n[*] Processing {len(b_df)} Benign Samples...")
    for _, row in tqdm(b_df.iterrows(), total=len(b_df)):
        cmd = row['command']
        
        res = engine.scan(cmd)
        
        if res.get('label', '').lower() == 'malicious':
            stats["tier1"]["fp"] += 1
        else:
            stats["tier1"]["tn"] += 1

    # 4. Final Analytics
    t1 = stats["tier1"]
    precision = t1["tp"] / (t1["tp"] + t1["fp"]) if (t1["tp"] + t1["fp"]) > 0 else 0
    recall = t1["tp"] / (t1["tp"] + t1["fn"]) if (t1["tp"] + t1["fn"]) > 0 else 0
    fpr = t1["fp"] / (t1["fp"] + t1["tn"]) if (t1["fp"] + t1["tn"]) > 0 else 0
    
    t2_acc = (stats["tier2"]["correct"] / stats["tier2"]["total_attempted"] * 100) if stats["tier2"]["total_attempted"] > 0 else 0

    print("\n" + "="*60)
    print(f"{'GENOS ENGINE: FINAL PIPELINE SCORECARD':^60}")
    print("="*60)
    print(f"TIER 1 (GATEKEEPER) PERFORMANCE:")
    print(f"  > Detection Rate (Recall): {recall*100:.2f}%")
    print(f"  > False Positive Rate:     {fpr*100:.2f}%")
    print(f"  > Precision:               {precision*100:.2f}%")
    print("-" * 60)
    print(f"TIER 2 (SPECIALIST) PERFORMANCE:")
    print(f"  > Attribution Accuracy:    {t2_acc:.2f}% (On T1 Detections)")
    print("-" * 60)
    print(f"OVERALL ENGINE SCORE:")
    # The true "End-to-End" success rate
    final_score = (stats["tier2"]["correct"] / len(m_df)) * 100
    print(f"  > End-to-End Accuracy:     {final_score:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_engine_audit()