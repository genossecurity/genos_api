import os
import secrets
from functools import wraps

import torch
import torch.nn as nn
import numpy as np
import faiss
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from flask import Flask, jsonify, request

# ── Command normalizer (mirrors stress.py pre-processing) ────────────────────

def normalize_command(cmd):
    if not isinstance(cmd, str):
        return ""
    cmd = cmd.lower()
    cmd = cmd.replace("^", "").replace("`", "")
    cmd = cmd.replace("/", "\\")
    cmd = " ".join(cmd.split())
    cmd = cmd.replace("> nul 2>&1", "").replace(">nul", "")
    cmd = cmd.replace("\\.\\", "\\")
    return cmd.strip()


# ── Model architecture (must match training) ──────────────────────────────────

class SecurityAgentX_Model(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=3):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.classifier_head = nn.Linear(768, num_labels)
        self.decoder_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 768),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier_head(embedding)
        reconstructed = self.decoder_head(embedding)
        return embedding, reconstructed, logits


# ── Engine initialisation ─────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

checkpoint = torch.load(os.path.join(BASE_DIR, "anomaly_engine.pt"), map_location=device)
# num_labels=3 kept to match saved weights; classifier head is unused at inference
model = SecurityAgentX_Model(model_name="microsoft/codebert-base", num_labels=3).to(device)
model.load_state_dict(checkpoint["model_state"], strict=False)
model.eval()
THRESHOLD = checkpoint.get("threshold", 0.001)

# FAISS atlas covers all 1745 MITRE ATT&CK attacks — used for every malicious verdict
atlas_index = faiss.read_index(os.path.join(BASE_DIR, "mitre_atlas.index"))
metadata = pd.read_csv(os.path.join(BASE_DIR, "mitre_atlas_metadata.csv"))

print(f"[+] Genos engine loaded | threshold={THRESHOLD:.8f} | atlas={len(metadata)} techniques | device={device}")

# ── API key auth ──────────────────────────────────────────────────────────────

# Set the API key via the GENOS_API_KEY environment variable.
API_KEY = os.environ.get("GENOS_API_KEY", "changeme")

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.json and request.json.get("api_key")
        if not key or not secrets.compare_digest(key, API_KEY):
            return jsonify(error="Unauthorized"), 401
        return f(*args, **kwargs)
    return decorated


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def hello():
    return jsonify(message="Genos API online")


@app.route("/api/test", methods=["POST"])
def test():
    return jsonify(message="server up!")


@app.route("/api", methods=["POST"])
@require_api_key
def analyze():
    body = request.get_json(silent=True) or {}
    command = body.get("commandline", "")
    if not command:
        return jsonify(error="'commandline' field is required"), 400

    inputs = tokenizer(
        normalize_command(command),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=96,
    ).to(device)

    with torch.no_grad():
        emb, rec, logits = model(inputs["input_ids"], inputs["attention_mask"])
        error = torch.mean((emb - rec) ** 2).item()
        is_anomalous = error > THRESHOLD

    if not is_anomalous:
        return jsonify(
            verdict="BENIGN",
            mitre_code=None,
            label="Benign",
            confidence=None,
            reconstruction_error=f"{error:.8f}",
        )

    # Anomalous — identify the closest of the 1745 MITRE techniques via FAISS atlas
    query_vec = emb.cpu().float().numpy().astype("float32")
    distances, indices = atlas_index.search(query_vec, 1)
    match = metadata.iloc[indices[0][0]]
    # Convert L2 distance to a 0-100 similarity score (lower distance = higher similarity)
    l2_dist = float(distances[0][0])
    similarity = max(0.0, 1.0 / (1.0 + l2_dist))

    return jsonify(
        verdict="MALICIOUS",
        mitre_code=str(match["mitre_id"]),
        label=str(match["technique_name"]),
        confidence=f"{similarity:.2%}",
        reconstruction_error=f"{error:.8f}",
        detection_type="ATLAS_MATCH",
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=6000)
