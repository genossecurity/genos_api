import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import base64
import re
from transformers import RobertaModel, RobertaTokenizer

# --- ARCHITECTURES ---

class Tier1_Gatekeeper(nn.Module):
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
        return self.classifier(self.encoder(ids, mask).last_hidden_state[:, 0, :])

class Tier2_Specialist(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
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
        return self.classifier(self.encoder(ids, mask).last_hidden_state[:, 0, :])

# --- PIPELINE ENGINE ---

class GenosEngine:
    def __init__(self, t1_path="gatekeeper.pt", t2_path="specialist.pt", map_path="specialist_map.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Load Tier 1
        self.t1 = Tier1_Gatekeeper().to(self.device)
        self.t1.load_state_dict(torch.load(t1_path, map_location=self.device))
        self.t1.eval()
        
        # Load Tier 2
        with open(map_path, "r") as f:
            self.s_map = {int(v): k for k, v in json.load(f).items()}
        self.t2 = Tier2_Specialist(num_classes=len(self.s_map)).to(self.device)
        self.t2.load_state_dict(torch.load(t2_path, map_location=self.device))
        self.t2.eval()

    def universal_decoder(self, text):
        """Pre-processes common encodings."""
        if re.match(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$", text):
            try:
                decoded = base64.b64decode(text).decode('utf-8')
                if len(decoded) > 3: return decoded
            except: pass
        if "\\x" in text:
            try:
                return bytes.fromhex(text.replace("\\x", "")).decode('utf-8')
            except: pass
        return text

    def scan(self, raw_cmd):
        processed_cmd = self.universal_decoder(raw_cmd).lower().strip()
        inputs = self.tokenizer(processed_cmd, return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(self.device)
        
        with torch.no_grad():
            # Tier 1 Gatekeeper
            g_logits = self.t1(inputs['input_ids'], inputs['attention_mask'])
            g_probs = F.softmax(g_logits, dim=1)
            g_conf, g_idx = torch.max(g_probs, dim=1)
            
            if g_idx.item() == 0:
                return {"status": "Benign", "confidence": f"{g_conf.item():.2%}"}
            
            # Tier 2 Specialist
            s_logits = self.t2(inputs['input_ids'], inputs['attention_mask'])
            s_probs = F.softmax(s_logits, dim=1)
            s_conf, s_idx = torch.max(s_probs, dim=1)
            
            return {
                "status": "Malicious",
                "mitre_id": self.s_map[s_idx.item()],