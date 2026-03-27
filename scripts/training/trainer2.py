import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import os
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def resolve_data_path(filename: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, "data", "training", filename),
        os.path.join(BASE_DIR, "data", "archive", "training", filename),
        filename 
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {filename}")

class SpecialistModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base", use_safetensors=True)
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

class SpecialistDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=256):
        print("[*] Pre-tokenizing data into RAM...")
        texts = [str(item[0]).lower().strip() for item in data_list]
        self.labels = [item[1] for item in data_list]
        self.encodings = tokenizer(
            texts, padding="max_length", truncation=True, 
            max_length=max_len, return_tensors="pt"
        )

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "ids": self.encodings['input_ids'][idx],
            "mask": self.encodings['attention_mask'][idx],
            "lbl": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_booster():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model_save_path = os.path.join(BASE_DIR, "models", "specialist.pt")
    map_path = os.path.join(BASE_DIR, "config", "specialist_map.json")

    # 1. Load the EXISTING Map
    if not os.path.exists(map_path):
        raise FileNotFoundError("Missing specialist_map.json. Cannot fine-tune without original class mapping.")
    
    with open(map_path, "r") as f:
        label_map = json.load(f)
    print(f"[*] Loaded mapping for {len(label_map)} classes.")

    # 2. Load the Data & Auto-Merge Booster
    df_main = pd.read_csv(resolve_data_path("specialist_train_set.csv"))
    
    try:
        booster_path = resolve_data_path("specialist_booster.csv")
        df_booster = pd.read_csv(booster_path)
        df = pd.concat([df_main, df_booster], ignore_index=True)
        print(f"[*] Dynamically merged booster data. Total samples: {len(df)}")
    except FileNotFoundError:
        df = df_main
        print("[!] No specialist_booster.csv found. Training on standard set.")

    labels = [label_map[m] for m in df['mitre_id']]
    data_list = list(zip(df['command'].tolist(), labels))
    
    full_dataset = SpecialistDataset(data_list, tokenizer)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

    # 3. Initialize Model and LOAD PREVIOUS WEIGHTS
    model = SpecialistModel(num_classes=len(label_map)).to(device)
    
    if os.path.exists(model_save_path):
        print(f"[*] Found existing weights. Loading {model_save_path} for precision patch...")
        model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    else:
        print("[!] No weights found. This will be a fresh training run.")

    # 4. Precision Focal Loss Weights
    print("[*] Adjusting boundaries: Dropping T1003 penalty to fix Precision Drop...")
    class_weights = torch.ones(len(label_map)).to(device)
    
    priority_ids = ['T1129', 'T1087', 'T1016', 'T1220']
    for pid in priority_ids:
        if pid in label_map:
            class_weights[label_map[pid]] = 8.0 
            
    # The Precision Fix: Reduce T1003 so the model stops guessing it when confused
    if 'T1003' in label_map:
        class_weights[label_map['T1003']] = 2.0 

    # 5. Polishing Optimizer (Ultra-Low Learning Rates for RTX 6000 speedrun)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 5e-7}, 
        {'params': model.classifier.parameters(), 'lr': 5e-6}
    ], weight_decay=0.01)

    # 6. Training Loop (Single Polishing Epoch)
    epochs = 1 
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Polishing Epoch {epoch+1}/{epochs}")
        for batch in loop:
            optimizer.zero_grad()
            ids, mask, targets = batch['ids'].to(device), batch['mask'].to(device), batch['lbl'].to(device)
            
            logits = model(ids, mask)
            
            ce_loss = F.cross_entropy(logits, targets, weight=class_weights, label_smoothing=0.1, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** 2 * ce_loss).mean()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, targets = batch['ids'].to(device), batch['mask'].to(device), batch['lbl'].to(device)
                logits = model(ids, mask)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        
        val_acc = (val_correct / val_total) * 100
        print(f"\n[Epoch {epoch+1}] Val Acc: {val_acc:.2f}%")
        
        if val_acc >= best_val_acc: # Save if it meets or beats
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"[+] Brain Updated! Precision Patch weights saved.")

if __name__ == "__main__":
    train_booster()