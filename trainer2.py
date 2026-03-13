import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# --- 1. ARCHITECTURE: 7-Layer Ultra Head ---
class SpecialistModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),      # Layer 0
            nn.LayerNorm(1024),        # Layer 1
            nn.GELU(),                 # Layer 2
            nn.Dropout(0.3),           # Layer 3
            nn.Linear(1024, 1024),     # Layer 4 (High-Res Hidden)
            nn.GELU(),                 # Layer 5
            nn.Linear(1024, num_classes)# Layer 6 (Output)
        )

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Load and map labels
    df = pd.read_csv("malicious_augmented.csv")
    unique_ids = sorted(df['mitre_id'].unique().tolist())
    label_map = {mid: i for i, mid in enumerate(unique_ids)}
    with open("specialist_map.json", "w") as f:
        json.dump(label_map, f)
    
    # Balanced Sampling Logic
    labels = [label_map[m] for m in df['mitre_id']]
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1. / class_counts.float()
    samples_weights = weights[labels]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    # Data Loader
    data_list = list(zip(df['command'].tolist(), labels))
    loader = DataLoader(data_list, batch_size=16, sampler=sampler)

    model = SpecialistModel(num_classes=len(label_map)).to(device)
    
    # Differential Learning Rates (95% Match Strategy)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},  # Slow Encoder
        {'params': model.classifier.parameters(), 'lr': 1e-4} # Fast Head
    ], weight_decay=0.01)

    print(f"[*] Training Ultra Specialist on 141 Classes...")
    print(f"[*] Tracking Success % per Epoch...")

    for epoch in range(15):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/15")
        for txt, lbl in loop:
            optimizer.zero_grad()
            
            # Tokenization
            enc = tokenizer(list(txt), padding=True, truncation=True, max_length=96, return_tensors="pt").to(device)
            targets = lbl.to(device)
            
            # Forward
            logits = model(enc['input_ids'], enc['attention_mask'])
            
            # Focal Loss + Label Smoothing
            ce_loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** 2 * ce_loss).mean()
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # --- TRACKING METRICS ---
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item()
            
            # Update Progress Bar with live stats
            current_acc = (correct_predictions / total_samples) * 100
            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

        # End of Epoch Summary
        epoch_acc = (correct_predictions / total_samples) * 100
        avg_loss = total_loss / len(loader)
        print(f"\n[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | Training Success Rate: {epoch_acc:.2f}%")
        print("-" * 60)

        # Save Checkpoint
        torch.save(model.state_dict(), "specialist.pt")

    print("[+] Training Complete. specialist.pt updated with final weights.")

if __name__ == "__main__":
    train()