import pandas as pd
import torch
import torch.nn as nn
import os
import glob
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 1. THE ARCHITECTURE (Matched to your specific .pt file structure) ---
class GatekeeperModel(nn.Module):
    def __init__(self):
        super(GatekeeperModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Identity(), 
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 2),
            nn.Dropout(0.3), 
            nn.Identity()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

# --- 2. DYNAMIC FILE DETECTION ---
def find_file(pattern):
    matches = glob.glob(os.path.join(BASE_DIR, "data", f"*{pattern}*.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing {pattern} CSV in data/")
    return matches[0]

# --- 3. CONFIGURATION ---
BASE_MALICIOUS = find_file("malicious")
BASE_BENIGN = find_file("benign")
BOOST_DATA = os.path.join(BASE_DIR, "data", "archive", "boost_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "archive", "gatekeeper_old.pt")
OUTPUT_PATH = os.path.join(BASE_DIR, "models", "gatekeeper.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 5e-7 
EPOCHS = 3

# --- 4. DATA PREP ---
class GenosDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

# --- 5. THE NUDGE ---
def train_nudge():
    print(f"[+] Loading boost data from {BOOST_DATA}...")
    boost_df = pd.read_csv(BOOST_DATA)
    mal_df = pd.read_csv(BASE_MALICIOUS)
    ben_df = pd.read_csv(BASE_BENIGN)
    
    # Combine the dataframes
    df = pd.concat([mal_df, boost_df, ben_df]).sample(frac=1).reset_index(drop=True)
    
    # --- DATA CLEANING ---
    print(f"[*] Raw dataset size: {len(df)}")
    df = df.dropna(subset=['text', 'label']) # Drop empty rows
    df['text'] = df['text'].astype(str)      # Force everything to be a string
    df['label'] = df['label'].astype(int)    # Ensure labels are integers
    print(f"[*] Cleaned dataset size: {len(df)}")

    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = GatekeeperModel() 
    
    print(f"[+] Loading weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    model.to(DEVICE)
    
    loader = DataLoader(GenosDataset(df['text'].tolist(), df['label'].tolist(), tokenizer), batch_size=16, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"[*] Starting 3-Epoch Nudge on {DEVICE}...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.6f}")

    print(f"[!] Success. Weights saved to {OUTPUT_PATH}")
    torch.save(model.state_dict(), OUTPUT_PATH)

if __name__ == "__main__":
    train_nudge()