# genos_engine.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import base64
import re
from transformers import RobertaModel, RobertaTokenizer
from scripts.augmentation.boost import apply_boosts  # Import the boosts engine

# =========================
# MODEL ARCHITECTURES
# =========================

class Tier1_Gatekeeper(nn.Module):
    def __init__(self):
        super().__init__()
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
        outputs = self.encoder(ids, mask).last_hidden_state[:, 0, :]
        logits = self.classifier(outputs)
        return logits


# =========================
# GENOS ENGINE
# =========================

class GenosEngine:
    def __init__(self, t1_path="models/gatekeeper.pt", t2_path="models/specialist.pt", map_path="models/specialist_map.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

        # --- Load Tier 1 Gatekeeper ---
        self.t1 = Tier1_Gatekeeper().to(self.device)
        self.t1.load_state_dict(torch.load(t1_path, map_location=self.device), strict=False)
        self.t1.eval()

        # --- Load Tier 2 Specialist ---
        with open(map_path, "r") as f:
            self.s_map = {int(v): k for k, v in json.load(f).items()}
        self.t2 = Tier2_Specialist(num_classes=len(self.s_map)).to(self.device)
        self.t2.load_state_dict(torch.load(t2_path, map_location=self.device), strict=False)
        self.t2.eval()

    # --- DECODER / PREPROCESSING ---
    def universal_decoder(self, text: str) -> str:
        """Decodes Base64 or hex-encoded strings."""
        try:
            # Base64 check
            if re.match(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$", text):
                decoded = base64.b64decode(text).decode('utf-8')
                if len(decoded) > 3:
                    return decoded
        except Exception:
            pass

        # Hex check
        if "\\x" in text:
            try:
                return bytes.fromhex(text.replace("\\x", "")).decode('utf-8')
            except Exception:
                pass

        return text

    def scan(self, raw_cmd):
        processed_cmd = self.universal_decoder(raw_cmd).lower().strip()
        inputs = self.tokenizer(
            processed_cmd, return_tensors="pt", truncation=True,
            padding="max_length", max_length=96
        ).to(self.device)

        with torch.no_grad():
            # Tier 1: Gatekeeper
            g_logits = self.t1(inputs['input_ids'], inputs['attention_mask'])
            g_probs = F.softmax(g_logits, dim=1)
            g_conf, g_idx = torch.max(g_probs, dim=1)

            # Tier 2: Specialist
            s_logits = self.t2(inputs['input_ids'], inputs['attention_mask'])
            s_probs = F.softmax(s_logits, dim=1).squeeze()
            s_probs = apply_boosts(processed_cmd, s_probs, self.s_map)

            # --- Top MITRE codes sorted by confidence ---
            top_vals, top_idxs = torch.topk(s_probs, k=min(5, len(s_probs)))
            top_mitre = [
                {"code": self.s_map[idx.item()], "confidence": val.item()}
                for idx, val in zip(top_idxs, top_vals)
            ]

            return {
                "status": "Malicious" if g_idx.item() == 1 else "Benign",
                "gatekeeper_confidence": g_conf.item(),
                "top_mitre": top_mitre
            }