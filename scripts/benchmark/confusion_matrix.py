import torch
import torch.nn as nn
import pandas as pd
import json
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

# Ensure Genos directories are in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. MODEL ARCHITECTURE (Synced with Trainer2)
# ==========================================
class SpecialistModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Add use_safetensors=True here
        self.encoder = RobertaModel.from_pretrained(
            "microsoft/codebert-base", 
            use_safetensors=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

# ==========================================
# 2. DIAGNOSTIC ENGINE
# ==========================================
def run_specialist_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Paths
    MODEL_PATH = "models/specialist.pt"
    MAP_PATH = "config/specialist_map.json"
    TEST_DATA_PATH = "data/test/specialist_test_set.csv" # Always test on unseen data
    
    # 1. Load MITRE Map
    with open(MAP_PATH, "r") as f:
        label_map = json.load(f)
    inv_map = {int(v): k for k, v in label_map.items()}
    num_classes = len(label_map)
    
    # 2. Initialize Model
    print(f"[*] Loading Specialist weights from {MODEL_PATH}...")
    model = SpecialistModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. Load Test Data
    df = pd.read_csv(TEST_DATA_PATH)
    y_true, y_pred = [], []
    top3_hits = 0

    print(f"[*] Auditing {len(df)} samples from holdout set...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cmd = str(row['command']).lower().strip()
        true_idx = label_map[row['mitre_id']]
        
        inputs = tokenizer(
            cmd, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=256 # Synced with GenosEngine
        ).to(device)
        
        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.softmax(logits, dim=1)
            
            # Top 1 Prediction for Confusion Matrix
            pred_idx = torch.argmax(logits, dim=1).item()
            
            # Top 3 Check for "Close Calls"
            _, top3_idxs = torch.topk(probs, k=3)
            if true_idx in top3_idxs[0].cpu().numpy():
                top3_hits += 1
            
        y_true.append(true_idx)
        y_pred.append(pred_idx)

    # ==========================================
    # 3. ANALYTICS & VISUALIZATION
    # ==========================================
    
    # Calculate Top-K Metrics
    top1_acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    top3_acc = (top3_hits / len(y_true)) * 100

    print("\n" + "="*50)
    print(f"{'GENOS SPECIALIST PERFORMANCE':^50}")
    print("="*50)
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}% (Safety Net)")
    print("-" * 50)

    # Classification Report (Precision/Recall/F1)
    # We filter target_names to only include classes present in the test set
    present_classes = sorted(list(set(y_true) | set(y_pred)))
    target_names = [inv_map[i] for i in present_classes]
    
    print("\n[!] Generating Per-Class F1 Report...")
    report = classification_report(y_true, y_pred, target_names=target_names, digits=2)
    with open("specialist_f1_report.txt", "w") as f:
        f.write(report)
    print("[+] Detailed report saved to 'specialist_f1_report.txt'")

    # Confusion Analysis
    cm = confusion_matrix(y_true, y_pred)
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((inv_map[i], inv_map[j], cm[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)

    print("\nTOP 10 CROSS-TECHNIQUE CONFUSIONS:")
    for actual, predicted, count in confusions[:10]:
        print(f"  {actual:10} mistaken for {predicted:10} | {count} times")

    # Heatmap of the "Chaos Zone" (Top 25 problematic classes)
    problematic_indices = list(set(
        [label_map[c[0]] for c in confusions[:15]] + 
        [label_map[c[1]] for c in confusions[:15]]
    ))
    
    short_cm = cm[np.ix_(problematic_indices, problematic_indices)]
    short_labels = [inv_map[i] for i in problematic_indices]

    plt.figure(figsize=(14, 12))
    sns.heatmap(short_cm, annot=True, fmt='d', xticklabels=short_labels, yticklabels=short_labels, cmap="YlOrRd")
    plt.title("Genos Specialist: High-Conflict MITRE Boundaries")
    plt.ylabel("Ground Truth (Actual)")
    plt.xlabel("Engine Prediction")
    plt.savefig("specialist_confusion_heatmap.png")
    print("\n[+] Heatmap saved to 'specialist_confusion_heatmap.png'")

if __name__ == "__main__":
    run_specialist_diagnostics()