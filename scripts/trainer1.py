import torch
import torch.nn as nn
import pandas as pd
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 1. ARCHITECTURE ---
class GatekeeperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2) # Binary: 0=BENIGN, 1=MALICIOUS
        )

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

class BinaryDataset(Dataset):
    def __init__(self, b_csv, m_csv, tokenizer):
        # Normalization: Force everything to lower
        b_cmds = [str(c).lower().strip() for c in pd.read_csv(b_csv)['command']]
        m_cmds = [str(c).lower().strip() for c in pd.read_csv(m_csv)['command']]
        
        self.texts = b_cmds + m_cmds
        self.labels = [0] * len(b_cmds) + [1] * len(m_cmds)
        self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=96, return_tensors="pt")
        return {
            "ids": enc['input_ids'].squeeze(0),
            "mask": enc['attention_mask'].squeeze(0),
            "lbl": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train():
    device = torch.device("cuda")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    dataset = BinaryDataset("benign_final.csv", "malicious_augmented.csv", tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GatekeeperModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    print("[*] Training Gatekeeper (Tier 1)...")
    for epoch in range(1):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()
            logits = model(batch['ids'].to(device), batch['mask'].to(device))
            loss = criterion(logits, batch['lbl'].to(device))
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(model.state_dict(), "gatekeeper.pt")
    print("[+] Saved gatekeeper.pt")

if __name__ == "__main__":
    train()