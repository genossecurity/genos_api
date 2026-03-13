import torch
import torch.nn as nn
import pandas as pd
import json
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

# --- 1. ARCHITECTURES (Synced with your final trainers) ---

class GatekeeperModel(nn.Module):
    """Tier 1: High-Precision Shield (Matches trainer1.py)"""
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2)
        )

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

class SpecialistModel(nn.Module):
    """Tier 2: High-Resolution Specialist (Matches trainer2_final.py)"""
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),      # Layer 0
            nn.LayerNorm(1024),        # Layer 1
            nn.GELU(),                 # Layer 2
            nn.Dropout(0.3),           # Layer 3
            nn.Linear(1024, 1024),     # Layer 4 (The extra high-res layer)
            nn.GELU(),                 # Layer 5
            nn.Linear(1024, num_classes)# Layer 6 (Output)
        )

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

# --- 2. EVALUATION ENGINE ---

def run_integrated_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Load Mappings
    with open("../config/specialist_map.json", "r") as f:
        s_map = {int(v): k for k, v in json.load(f).items()}

    # Initialize and Load
    print("[*] Loading Gatekeeper (Tier 1)...")
    t1 = GatekeeperModel().to(device)
    t1.load_state_dict(torch.load("../models/gatekeeper.pt", map_location=device))
    
    print("[*] Loading Ultra Specialist (Tier 2)...")
    t2 = SpecialistModel(num_classes=len(s_map)).to(device)
    # Fix: Now matches the 7-layer state_dict in specialist.pt
    t2.load_state_dict(torch.load("../models/specialist.pt", map_location=device))
    
    t1.eval()
    t2.eval()

    def normalize(cmd):
        return str(cmd).lower().strip()

    # Data Source
    b_df = pd.read_csv("../data/benign_final.csv").sample(1000)
    m_df = pd.read_csv("../data/mitre_atlas_raw.csv")

    print("\n" + "="*50 + "\n🚀 GENOS PIPELINE AUDIT\n" + "="*50)

    fps = 0
    for cmd in tqdm(b_df['command'], desc="Checking False Positives"):
        inputs = tokenizer(normalize(cmd), return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(device)
        with torch.no_grad():
            if torch.argmax(t1(inputs['input_ids'], inputs['attention_mask']), dim=1).item() == 1:
                fps += 1

    hits, exact = 0, 0
    for _, row in tqdm(m_df.iterrows(), total=len(m_df), desc="Checking Attack Attribution"):
        inputs = tokenizer(normalize(row['command']), return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(device)
        with torch.no_grad():
            if torch.argmax(t1(inputs['input_ids'], inputs['attention_mask']), dim=1).item() == 1:
                hits += 1
                idx = torch.argmax(t2(inputs['input_ids'], inputs['attention_mask']), dim=1).item()
                if s_map[idx] == row['mitre_id']:
                    exact += 1

    print(f"\n" + "="*50 + f"\n📊 FINAL RESULTS (15-Epoch Potential)\n" + "="*50)
    print(f"False Positive Rate   : {(fps/len(b_df))*100:.2f}%")
    print(f"Overall Detection Rate: {(hits/len(m_df))*100:.2f}%")
    print(f"MITRE Match Accuracy  : {(exact/hits)*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_integrated_audit()