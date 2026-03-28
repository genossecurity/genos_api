import torch
import torch.nn as nn
import pandas as pd
import os
import json
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.amp import autocast, GradScaler # Ensure this import is at the top of your script

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

class GatekeeperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base", use_safetensors=True)
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

class BinaryDataset(Dataset):
    def __init__(self, b_csv, m_csv, tokenizer, max_len=256, phase="Train"):
        print(f"[*] Loading and pre-tokenizing Gatekeeper dataset ({phase})...")
        # Load Benign
        b_df = pd.read_csv(b_csv)
        b_cmds = [str(c).lower().strip() for c in b_df['command']]
        
        # Load Malicious
        m_df = pd.read_csv(m_csv)
        m_cmds = [str(c).lower().strip() for c in m_df['command']]
        
        texts = b_cmds + m_cmds
        self.labels = [0] * len(b_cmds) + [1] * len(m_cmds)
        
        self.encodings = tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=max_len, 
            return_tensors="pt"
        )
        print(f"[+] {phase} Load complete: {len(b_cmds)} Benign | {len(m_cmds)} Malicious")

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "ids": self.encodings['input_ids'][idx],
            "mask": self.encodings['attention_mask'][idx],
            "lbl": torch.tensor(self.labels[idx], dtype=torch.long)
        }
def train():
    # 1. HARDWARE OPTIMIZATION
    torch.backends.cudnn.benchmark = True # Auto-tunes hardware paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Keep memory usage predictable: use micro-batches + grad accumulation.
    effective_batch_size = int(os.getenv("GENOS_T1_EFFECTIVE_BATCH", "256"))
    micro_batch_size = int(os.getenv("GENOS_T1_MICRO_BATCH", "32"))
    if micro_batch_size <= 0:
        raise ValueError("GENOS_T1_MICRO_BATCH must be > 0")
    grad_acc_steps = max(1, effective_batch_size // micro_batch_size)
    effective_batch_size = micro_batch_size * grad_acc_steps
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    train_dataset = BinaryDataset(
        resolve_data_path("gatekeeper_train.csv"),
        resolve_data_path("specialist_train.csv"), 
        tokenizer,
        phase="Train"
    )
    
    val_dataset = BinaryDataset(
        resolve_data_path("gatekeeper_val.csv"),
        resolve_data_path("specialist_val.csv"),
        tokenizer,
        phase="Validation"
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=micro_batch_size,
        shuffle=True, 
        pin_memory=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=micro_batch_size,
        shuffle=False, 
        pin_memory=True,
        num_workers=4
    )

    print(
        f"[*] Batch config: micro={micro_batch_size}, grad_acc={grad_acc_steps}, "
        f"effective={effective_batch_size}"
    )

    # 3. COMPILE MODEL (opt-in; can increase memory use on large transformer graphs)
    raw_model = GatekeeperModel().to(device)
    use_compile = os.getenv("GENOS_T1_USE_COMPILE", "0") == "1"
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(raw_model)
            print("[+] torch.compile() enabled.")
        except Exception:
            model = raw_model
            print("[-] torch.compile() failed. Proceeding without it.")
    else:
        model = raw_model
        print("[*] torch.compile() disabled (set GENOS_T1_USE_COMPILE=1 to enable).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    weights = torch.tensor([1.0, 3.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 4. INITIALIZE AMP SCALER
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    best_val_acc = 0.0
    model_save_path = os.path.join(BASE_DIR, "models", "gatekeeper.pt")
    
    epochs = 5 
    print(f"[*] Hardening Gatekeeper (Tier 1)...")

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        optimizer.zero_grad(set_to_none=True)
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for step, batch in enumerate(loop, start=1):
            ids, mask, labels = batch['ids'].to(device), batch['mask'].to(device), batch['lbl'].to(device)
            
            # 6. AMP AUTOCAST CONTEXT
            with autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                logits = model(ids, mask)
                loss = criterion(logits, labels) / grad_acc_steps
            
            # 7. SCALED BACKWARD PASS
            scaler.scale(loss).backward()

            if step % grad_acc_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            loop.set_postfix(acc=f"{(train_correct / train_total) * 100:.2f}%")

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, labels = batch['ids'].to(device), batch['mask'].to(device), batch['lbl'].to(device)
                
                # Use AMP for inference too
                with autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                    logits = model(ids, mask)
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = (val_correct / val_total) * 100
        print(f"\n[Epoch {epoch+1}] Train Acc: {(train_correct / train_total) * 100:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save raw_model.state_dict() if using torch.compile to avoid saving compiled prefixes
            torch.save(raw_model.state_dict(), model_save_path)
            print(f"[+] Gatekeeper Improved! Saved gatekeeper.pt")
        print("-" * 60)

if __name__ == "__main__":
    train()