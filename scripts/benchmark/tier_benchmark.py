import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Fix path to import from root genos_api directory
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(BENCHMARK_DIR))  # Up to genos_api root
sys.path.insert(0, BASE_DIR)

# Import your engine components
from engine import GenosEngine

def evaluate_tiers():
    print("[*] Loading Genos Engine...")
    engine = GenosEngine()
    
    # ==========================================
    # 1. EVALUATE TIER 1 (GATEKEEPER) PURELY
    # ==========================================
    gk_test_path = os.path.join(BASE_DIR, "data", "training", "gatekeeper_test.csv")
    sp_test_path = os.path.join(BASE_DIR, "data", "training", "specialist_test.csv")
    
    if os.path.exists(gk_test_path) and os.path.exists(sp_test_path):
        print("\n" + "="*50)
        print("TIER 1 (GATEKEEPER) ISOLATED METRICS")
        print("="*50)
        
        # Load Benign Samples (Label 0)
        gk_df = pd.read_csv(gk_test_path)
        benign_cmds = gk_df['command'].tolist()
        benign_labels = [0] * len(benign_cmds)
        
        # Load Malicious Samples (Label 1)
        sp_df = pd.read_csv(sp_test_path)
        malicious_cmds = sp_df['command'].tolist()
        malicious_labels = [1] * len(malicious_cmds)
        
        # Combine into a unified test set
        all_cmds = benign_cmds + malicious_cmds
        y_true_t1 = benign_labels + malicious_labels
        y_pred_t1 = []
        
        for cmd in tqdm(all_cmds, desc="Testing Gatekeeper"):
            inputs = engine.tokenizer(
                str(cmd).lower().strip(), return_tensors="pt", truncation=True, 
                padding="max_length", max_length=engine.max_length
            ).to(engine.device)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda' if 'cuda' in engine.device.type else 'cpu'):
                    logits = engine.t1(inputs["input_ids"], inputs["attention_mask"])
                    pred = torch.argmax(logits, dim=1).item()
                    y_pred_t1.append(pred)
                    
        acc = accuracy_score(y_true_t1, y_pred_t1)
        # Now average='binary' will work perfectly because we have both 0s and 1s
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_t1, y_pred_t1, average='binary')
        
        print(f"Total Samples Tested : {len(all_cmds)} ({len(benign_cmds)} Benign / {len(malicious_cmds)} Malicious)")
        print(f"Tier 1 Accuracy      : {acc * 100:.2f}%")
        print(f"Tier 1 Precision     : {precision * 100:.2f}% (Confidence when flagging Malicious)")
        print(f"Tier 1 Recall        : {recall * 100:.2f}% (Ability to catch all attacks)")
        print(f"Tier 1 F1-Score      : {f1 * 100:.2f}%")

    # ==========================================
    # 2. EVALUATE TIER 2 (SPECIALIST) PURELY
    # ==========================================
    if os.path.exists(sp_test_path):
        print("\n" + "="*50)
        print("TIER 2 (SPECIALIST) ISOLATED METRICS")
        print("="*50)
        
        # sp_df is already loaded above
        y_true_t2 = []
        y_pred_t2 = []
        
        # Reverse map to get integer IDs for labels
        inv_map = {v: k for k, v in engine.s_map.items()}
        
        for _, row in tqdm(sp_df.iterrows(), desc="Testing Specialist", total=len(sp_df)):
            true_mitre = str(row['mitre_id'])
            if true_mitre not in inv_map:
                continue
                
            y_true_t2.append(true_mitre)
            
            inputs = engine.tokenizer(
                str(row['command']).lower().strip(), return_tensors="pt", truncation=True, 
                padding="max_length", max_length=engine.max_length
            ).to(engine.device)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda' if 'cuda' in engine.device.type else 'cpu'):
                    # BYPASS TIER 1 ENTIRELY - Force Tier 2 Prediction
                    s_logits = engine.t2(inputs["input_ids"], inputs["attention_mask"])
                    pred_idx = torch.argmax(s_logits, dim=1).item()
                    y_pred_t2.append(engine.s_map[pred_idx])
                    
        acc = accuracy_score(y_true_t2, y_pred_t2)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_t2, y_pred_t2, average='weighted', zero_division=0)
        
        print(f"Total Samples Tested : {len(sp_df)} (141 MITRE Classes)")
        print(f"Tier 2 Accuracy      : {acc * 100:.2f}%")
        print(f"Tier 2 Precision     : {precision * 100:.2f}%")
        print(f"Tier 2 Recall        : {recall * 100:.2f}%")
        print(f"Tier 2 F1-Score      : {f1 * 100:.2f}%")

if __name__ == "__main__":
    evaluate_tiers()