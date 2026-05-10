# Genos API — `ieee-paper2` Branch

This branch is a **read-only snapshot** of the Genos V1 codebase at the point it was declared IEEE-paper-ready (commit `892f4aa`). It preserves the exact code, data, training scripts, and benchmark scripts that produced the results reported in the second IEEE paper. All active development continues on `main`.

---

## System Overview

Genos is a two-tier, cascaded neural pipeline for real-time malicious command detection and MITRE ATT&CK technique attribution. It is served as a REST API over Flask/Gunicorn.

```
Raw command (string)
        │
        ▼
  Deobfuscation loop (up to 5 layers)
  ├── Regex pattern checks + Shannon entropy > 5.2 bits
  ├── Base64 decode, [char] construction resolution,
  │   string concatenation collapse, PowerShell payload extraction
  └── Stops when text stabilises (delta entropy < 0.01)
        │
        ▼
  Normalise: lowercase + strip
        │
        ▼
  Tokenise: RobertaTokenizer, max_length=256, padding, truncation
        │
        ▼
┌─────────────────────────────┐
│  Tier 1 — Gatekeeper        │  CodeBERT encoder → CLS → Dropout → Linear(768→1024) → GELU → Dropout → Linear(1024→2)
│  Binary: Benign / Malicious │  softmax → argmax → label + confidence (%)
└─────────────┬───────────────┘
              │
    ┌─────────┴──────────┐
    │ Benign             │ Malicious
    ▼                    ▼
Return immediately   ┌─────────────────────────────────┐
{"label":"Benign",   │  Tier 2 — Specialist             │  CodeBERT encoder → CLS → Linear(768→1024) → LayerNorm
 "MITRE_codes":[]}   │  108-class MITRE technique ID    │  → GELU → Dropout(0.3) → Linear(1024→1024) → GELU → Linear(1024→108)
                     │  softmax(logits / T=0.5) → top-5 │  Temperature T=0.5 sharpens the distribution
                     └─────────────────────────────────┘
                              │
                              ▼
              {"label":"Malicious","label_confidence":99.81,
               "MITRE_codes":[{"code":"T1021","confidence":99.83}, ...]}
```

Both models share the same `microsoft/codebert-base` encoder (RoBERTa pre-trained on code, 768-D hidden). They are loaded once at startup and kept resident in GPU memory for the lifetime of the process.

---

## Models

### Tier 1 — Gatekeeper (`models/gatekeeper.pt`)

Binary classifier that decides whether a command is benign or malicious.

- **Architecture**: CodeBERT CLS token → `Dropout(0.2) → Linear(768→1024) → GELU → Dropout(0.2) → Linear(1024→2)`
- **Training**: 5 epochs, AdamW (lr=1e-5, weight_decay=0.01), CrossEntropyLoss with class weights `[1.0, 3.0]` (malicious upweighted 3×), mixed-precision (float16 + GradScaler), gradient accumulation (micro-batch 32, effective batch 256)
- **Data**: `data/training/gatekeeper_train.csv` (12,000 benign) + `data/training/specialist_train.csv` (11,280 malicious), validated on matching val splits
- **Selection**: checkpoint saved on every validation accuracy improvement

### Tier 2 — Specialist (`models/specialist.pt`)

Multi-class classifier that attributes a malicious command to one of 108 MITRE ATT&CK techniques.

- **Architecture**: CodeBERT CLS token → `Linear(768→1024) → LayerNorm(1024) → GELU → Dropout(0.3) → Linear(1024→1024) → GELU → Linear(1024→108)`
- **Training**: Precision-patch fine-tune (1 epoch, loads existing weights). Focal loss with label smoothing 0.1 (`γ=2`). Differential learning rates: encoder lr=5e-7, classifier lr=5e-6. Per-class weights: default=1.0, priority underperformers (T1129, T1087, T1016, T1220) boosted to 8.0, T1003 reduced to 2.0 to fix precision bleed.
- **Classes**: 108 MITRE techniques defined in `config/specialist_map.json` (`{mitre_id: int}`)
- **Data**: `data/training/specialist_train.csv` (11,280 samples), validated on `specialist_val.csv`

---

## Data

| File | Rows | Description |
|---|---|---|
| `data/training/gatekeeper_train.csv` | 12,000 | Benign commands — training |
| `data/training/gatekeeper_val.csv` | 1,500 | Benign commands — validation |
| `data/training/gatekeeper_test.csv` | 1,500 | Benign commands — test |
| `data/training/specialist_train.csv` | 11,280 | Malicious commands — training |
| `data/training/specialist_val.csv` | 1,410 | Malicious commands — validation |
| `data/training/specialist_test.csv` | 1,410 | Malicious commands — test |
| `data/test/specialist_test_set.csv` | 2,820 | Held-out: 20 samples × 141 classes |

All CSVs share the schema `command` (string), `mitre_id` (string — `"Benign"` or a MITRE technique ID). Malicious samples are sourced from Atomic Red Team, MITRE CTI, LOLBAS, and GTFOBins. A subset includes obfuscated variants (Base64, `[char]` constructions, hex escapes, string concatenation) to stress-test the deobfuscation pipeline.

---

## Results at this snapshot

### Tier 2 Specialist — per-class F1 (`specialist_f1_report.txt`)

Evaluated over 2,820 held-out samples (20 per class, 141 techniques):

| Metric | Score |
|---|---|
| Accuracy | 91% |
| Macro avg precision | 0.94 |
| Macro avg recall | 0.91 |
| Macro avg F1 | 0.92 |

Hardest classes (lowest F1): T1016 (0.48), T1129 (0.50), T1218 (0.58), T1059 (0.59), T1220 (0.63) — high recall but low precision (over-prediction under ambiguity). Majority of classes hit F1 ≥ 0.90.

### IEEE pipeline benchmark (`scripts/benchmark/ieee.py`)

Full deployment-aligned evaluation comparing Genos against a TF-IDF + Random Forest baseline:

| Architecture | Top-1 Acc | Top-3 Acc | Macro F1 | Latency |
|---|---|---|---|---|
| TF-IDF + Random Forest (baseline) | reported | reported | reported | reported |
| CodeBERT Specialist + preprocessing | reported | reported | reported | reported |

Additional metrics reported: Tier 1 AUC, Precision, Recall, F1; mean deobfuscation time; end-to-end latency sweep at 0.50 / 0.90 / 0.99 benign traffic ratios; Tier 2 trigger rate; ROC curve saved to `logs/ieee_roc_curve_pipeline.png`.

### Async stress test (`scripts/benchmark/internal_api_test.py` → `live_stress_report.txt`)

500 live requests through the full API stack — 50% malicious / 50% benign, 20 concurrent workers, 85% confidence threshold. Metrics: TP, TN, FP, FN, per-request latency.

---

## API

Served by Gunicorn on `127.0.0.1:6001` (1 sync worker, 300s timeout).

### `POST /scan` — authenticated
```json
{ "api_key": "YOUR_KEY", "command": "net user /domain" }
```
Validates API key against MongoDB. Logs usage. Returns:
```json
{
  "label": "Malicious",
  "label_confidence": 99.81,
  "MITRE_codes": [
    { "code": "T1087", "confidence": 97.43 },
    { "code": "T1069", "confidence": 1.22 }
  ]
}
```
For benign commands `MITRE_codes` is an empty array. Confidence values are percentages (2 d.p.).

### `POST /scan/internal` — token-gated, no DB
```json
{ "internal_token": "...", "command": "..." }
```
Used by benchmark and stress-test scripts. Skips MongoDB entirely.

### `GET /health`
Returns `{"status": "ok"}` once the engine warm-up pass completes.

---

## Running

```bash
source venv/bin/activate
gunicorn -c gunicorn.conf.py app:app
```

The worker loads both CodeBERT models and runs a warm-up inference on startup before accepting traffic. The 300s timeout in `gunicorn.conf.py` covers this load time. `preload_app` is intentionally not set — CUDA cannot survive a fork.

---

## Repository layout

```
app.py                          Flask application + routes
engine.py                       GenosEngine — deobfuscation + two-tier inference
gunicorn.conf.py                Gunicorn config (bind, workers, timeout)
config/
  specialist_map.json           108-class MITRE → int label map
  definitive_mitre_map.json     Full MITRE technique reference
models/
  gatekeeper.pt                 Tier 1 weights
  specialist.pt                 Tier 2 weights
data/training/                  Train / val / test splits
scripts/
  training/trainer1.py          Gatekeeper training script
  training/trainer2.py          Specialist precision-patch script
  benchmark/ieee.py             IEEE pipeline benchmark
  benchmark/internal_api_test.py  Async API stress test
  utils/genos.py                Local REPL shell (no HTTP)
specialist_f1_report.txt        Per-class F1 results
live_stress_report.txt          Stress test output log
notes.md                        Full methodology notes (all six sections)
```
