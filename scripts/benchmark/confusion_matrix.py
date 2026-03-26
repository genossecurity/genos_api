import torch
import pandas as pd
import json
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your Specialist model class here or ensure it's in scope
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.training.trainer2 import SpecialistModel


def resolve_data_path() -> str:
    candidates = [
        os.path.join(BASE_DIR, "data", "training", "trainer1-bad.csv"),
        os.path.join(BASE_DIR, "data", "archive", "training", "trainer1-bad.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find trainer1-bad.csv in expected locations: "
        + ", ".join(candidates)
    )

def generate_diagnostics(model_path, data_path, map_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    # 1. Load Map and Model
    with open(map_path, "r") as f:
        label_map = json.load(f)
    inv_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    
    model = SpecialistModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load Data (Ideally a validation set)
    df = pd.read_csv(data_path)
    y_true, y_pred = [], []

    print(f"[*] Analyzing {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cmd = str(row['command']).lower().strip()
        label = label_map[row['mitre_id']]
        
        inputs = tokenizer(cmd, return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(device)
        
        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            pred = torch.argmax(logits, dim=1).item()
            
        y_true.append(label)
        y_pred.append(pred)

    # 3. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 4. Identify Top Confusions (Ignoring the diagonal/correct matches)
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((inv_map[i], inv_map[j], cm[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "="*50)
    print("TOP 10 MITRE CONFUSIONS (Actual -> Predicted)")
    print("="*50)
    for actual, predicted, count in confusions[:10]:
        print(f"{actual:10} -> {predicted:10} | Occurrences: {count}")
    
    # 5. Visual Summary (Heatmap of top 20 problematic classes)
    # Full 141x141 is too big to plot, so we filter for the most active ones
    problematic_indices = list(set([label_map[c[0]] for c in confusions[:20]] + [label_map[c[1]] for c in confusions[:20]]))
    short_cm = cm[np.ix_(problematic_indices, problematic_indices)]
    short_labels = [inv_map[i] for i in problematic_indices]

    plt.figure(figsize=(12, 10))
    sns.heatmap(short_cm, annot=True, fmt='d', xticklabels=short_labels, yticklabels=short_labels, cmap="YlGnBu")
    plt.title("Genos Specialist: Top MITRE Confusion Heatmap")
    plt.ylabel("Actual MITRE ID")
    plt.xlabel("Predicted MITRE ID")
    plt.savefig("specialist_confusion_matrix.png")
    print("\n[+] Confusion matrix heatmap saved to 'specialist_confusion_matrix.png'")

if __name__ == "__main__":
    generate_diagnostics(
        model_path=os.path.join(BASE_DIR, "models", "specialist.pt"),
        data_path=resolve_data_path(),
        map_path=os.path.join(BASE_DIR, "config", "specialist_map.json"),
    )