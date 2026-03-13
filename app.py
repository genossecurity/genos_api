from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
import base64
import re
import json

app = Flask(__name__)

# --- CONFIGURATION ---
VALID_API_KEY = "GENOS_DEV_KEY_2026" # Replace with your actual key management

# --- MODEL ARCHITECTURES ---

class Tier1_Gatekeeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(768, 1024), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(1024, 2)
        )
    def forward(self, ids, mask):
        return self.classifier(self.encoder(ids, mask).last_hidden_state[:, 0, :])

class Tier2_Specialist(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, num_classes)
        )
    def forward(self, ids, mask):
        return self.classifier(self.encoder(ids, mask).last_hidden_state[:, 0, :])

# --- ENGINE ---

class GenosEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Load Tier 1
        self.t1 = Tier1_Gatekeeper().to(self.device)
        self.t1.load_state_dict(torch.load("gatekeeper.pt", map_location=self.device))
        self.t1.eval()
        
        # Load Tier 2
        with open("specialist_map.json", "r") as f:
            self.s_map = {int(v): k for k, v in json.load(f).items()}
        self.t2 = Tier2_Specialist(num_classes=len(self.s_map)).to(self.device)
        self.t2.load_state_dict(torch.load("specialist.pt", map_location=self.device))
        self.t2.eval()

    def analyze(self, raw_cmd):
        cmd = raw_cmd.lower().strip()
        inputs = self.tokenizer(cmd, return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(self.device)
        
        with torch.no_grad():
            # Tier 1 Analysis
            g_logits = self.t1(inputs['input_ids'], inputs['attention_mask'])
            g_probs = F.softmax(g_logits, dim=1)
            g_conf, g_idx = torch.max(g_probs, dim=1)
            
            if g_idx.item() == 0: # Benign
                return "Benign", None, f"{g_conf.item():.2%}"
            
            # Tier 2 Analysis (If Tier 1 is Malicious)
            s_logits = self.t2(inputs['input_ids'], inputs['attention_mask'])
            s_probs = F.softmax(s_logits, dim=1)
            s_conf, s_idx = torch.max(s_probs, dim=1)
            
            return "Malicious", self.s_map[s_idx.item()], f"{s_conf.item():.2%}"

engine = GenosEngine()

# --- ROUTES ---

@app.route('/scan', methods=['POST'])
def scan():
    data = request.get_json()
    
    # 1. Input Validation
    if not data or 'command' not in data or 'api_key' not in data:
        return jsonify({"error": "Invalid input format. Required: {'command': '...', 'api_key': '...'}"}), 400
    
    # 2. API Key Check
    if data['api_key'] != VALID_API_KEY:
        return jsonify({"error": "Unauthorized API Key"}), 401

    # 3. Analyze
    status, mitre_code, confidence = engine.analyze(data['command'])

    # 4. Formatted Output
    if status == "Benign":
        return jsonify({
            "status": "Benign",
            "confidence": confidence
        })
    else:
        return jsonify({
            "status": "Malicious",
            "mitre_id": mitre_code,
            "confidence": confidence
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)