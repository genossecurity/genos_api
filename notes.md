# Genos V1 — IEEE-Paper2 Branch Notes

Branch: `ieee-paper2` | Commit: `892f4aa` ("ieee ready benchmark")

---

## 1. Training, Test & Validation Data

### Architecture
Two separate models are trained and evaluated independently:
- **Tier 1 — Gatekeeper** (binary classifier: Benign vs Malicious)
- **Tier 2 — Specialist** (multi-class MITRE ATT&CK technique classifier)

### Dataset splits

| Split | File | Rows (excl. header) | Label type |
|---|---|---|---|
| Gatekeeper train | `data/training/gatekeeper_train.csv` | 12,000 | `command`, `mitre_id` (Benign / technique) |
| Gatekeeper val | `data/training/gatekeeper_val.csv` | 1,500 | same |
| Gatekeeper test | `data/training/gatekeeper_test.csv` | 1,500 | same |
| Specialist train | `data/training/specialist_train.csv` | 11,280 | `command`, `mitre_id` |
| Specialist val | `data/training/specialist_val.csv` | 1,410 | same |
| Specialist test | `data/training/specialist_test.csv` | 1,410 | same |
| Held-out specialist | `data/test/specialist_test_set.csv` | 2,820 | same (20 samples per class × 141 classes) |

### Schema
Every CSV has exactly two columns:
- `command` — raw or obfuscated shell/PowerShell/CLI command string
- `mitre_id` — ground-truth label (`"Benign"` for benign; MITRE technique ID e.g. `"T1059"` for malicious)

### How samples are constructed
**Benign samples** — synthetic IT/devops commands (Terraform, grep, PowerShell maintenance scripts, etc.) that do not map to any ATT&CK technique. The gatekeeper CSVs are benign-only; the specialist CSVs are malicious-only.

**Malicious samples** — sourced from Atomic Red Team, MITRE CTI, LOLBAS, and GTFOBins. Commands cover 108 MITRE ATT&CK techniques stored in `config/specialist_map.json`. Techniques are mapped to integer class indices: `{mitre_id: int}`.

**Obfuscation** — a subset of malicious commands include Base64 encoding, PowerShell `[char]` constructions, hex escapes (`\xNN`), string concatenation, and high-entropy blobs. The engine handles deobfuscation at inference time (see Section 6).

---

## 2. Test Methodology

### Specialist per-class evaluation (`specialist_f1_report.txt`)
- **Test set**: held-out split with **20 samples per class**, across **141 MITRE techniques** (2,820 total).
- **Metric suite**: precision, recall, F1 per class; macro/weighted averages at the bottom.
- **Overall results** (from `specialist_f1_report.txt`):

| Metric | Score |
|---|---|
| Accuracy | 0.91 (91%) |
| Macro avg precision | 0.94 |
| Macro avg recall | 0.91 |
| Macro avg F1 | 0.92 |

- Classes with perfect F1 1.00: T1001, T1005, T1006, T1010, T1012, T1020, T1025, T1040, T1041, T1047, T1049, T1056, T1072, T1074, T1091, T1095, T1115, T1119, T1120, T1124, T1132, T1133, T1134, T1137, T1140, T1176, T1195, T1202, T1207, T1216, T1217, T1559, T1563, T1570, T1573, T1578, T1580, T1592, T1595, T1606, T1609, T1610, T1611, T1615, T1619, T1620, T1622, T1648, T1649, T1651, T1654 and more.
- Hardest classes (lowest F1): T1129 (0.50), T1218 (0.58), T1059 (0.59), T1016 (0.48), T1220 (0.63) — these have high recall but low precision, meaning the model over-predicts them.

### IEEE pipeline benchmark (`scripts/benchmark/ieee.py`)
The IEEE benchmark (`ieee.py` — "GENOS IEEE PIPELINE BENCHMARK v3.0") performs a full deployment-aligned evaluation:

1. **Baseline** — TF-IDF (unigrams+bigrams+trigrams, max 10,000 features) + Random Forest (100 estimators, seed 42) trained on `specialist_train.csv`, evaluated on `specialist_test.csv`. Reports Top-1 Acc, Top-3 Acc, Macro F1, latency.

2. **Tier 1 evaluation** — `evaluate_tier1_pipeline()`: runs the full preprocessing pipeline (deobfuscation → lower/strip normalisation → tokenise → forward pass) over all benign (`gatekeeper_test.csv`) + all malicious (`specialist_test.csv`) samples. Reports AUC, Precision, Recall, F1, mean deobfuscation time, obfuscation flag rate, deobfuscation change rate. ROC curve saved to `logs/ieee_roc_curve_pipeline.png`.

3. **Tier 2 evaluation** — `evaluate_tier2_pipeline()`: same preprocessing path, but measures Top-1 Acc, Top-3 Acc, Macro F1, latency per sample against `specialist_test.csv`.

4. **End-to-end pipeline sweep** — `evaluate_full_pipeline_end_to_end()`: constructs a mixed stream of 10,000 commands at three benign traffic ratios (0.50, 0.90, 0.99) and measures: end-to-end latency, Tier 2 trigger rate, obfuscation flag rate, deobfuscation change rate.

All evaluations mirror the exact deployed cascade: preprocess → Tier 1 argmax → conditional Tier 2.

---

## 3. Stress Test Methodology

### Async live API stress test (`scripts/benchmark/internal_api_test.py`)
- **Target**: `http://127.0.0.1:6001/scan` (or `/scan/internal` with internal token)
- **Total requests**: 500 (configurable via `TOTAL_REQUESTS`)
- **Split**: 50% malicious / 50% benign drawn from real test CSVs (`gatekeeper_test.csv`, `specialist_test.csv`) sampled with replacement
- **Concurrency**: 20 simultaneous async workers (aiohttp `asyncio.Semaphore`)
- **Confidence threshold**: 85.0% — only decisions with confidence ≥ 85% are counted as Malicious; below threshold defaults to Benign
- **Metrics collected**: TP, TN, FP, FN, error count, per-request latency (ms)
- **Output log**: `live_stress_report.txt` — every request logged with: command type, true ID, raw command, API label, confidence, final decision, latency, full JSON response
- **Payload format**: `{"api_key": "...", "command": "..."}` serialised as strict JSON string via `json.dumps()` and passed as `data=` (not `json=`) to prevent aiohttp encoding issues

Sample results observed in `live_stress_report.txt`:
- T1021 command → Malicious 99.81%, 63ms
- Benign grep command → Benign 99.17%, 109ms
- T1046 docker scan command → Malicious 99.80%, 165ms

### Direct engine stress test (`stress_test_responses.txt`)
- Mode: `direct_engine` — bypasses HTTP entirely, calls `GenosEngine.scan()` directly
- Used to validate raw model performance without network overhead
- Typical latencies: 14–40ms per inference
- Confidence consistently ≥ 99.9% on unambiguous commands (e.g. `Add-Type -TypeDefinition` → T1129 at 99.12%)

---

## 4. Training Methodology

### Tier 1 — Gatekeeper (`scripts/training/trainer1.py`)
Base model: **CodeBERT** (`microsoft/codebert-base`, RobertaModel backbone, 768-dim CLS token)

**Architecture**:
```
RobertaModel (codebert-base)
└── CLS token → Dropout(0.2) → Linear(768→1024) → GELU → Dropout(0.2) → Linear(1024→2)
```

**Training procedure**:
1. Load benign CSVs (`gatekeeper_train/val.csv`) and malicious CSVs (`specialist_train/val.csv`) → merge into binary labels (0=benign, 1=malicious)
2. Pre-tokenise entire dataset into RAM using `RobertaTokenizer` (`max_length=256`, `padding="max_length"`, `truncation=True`)
3. Gradient accumulation: effective batch 256, micro-batch 32, `grad_acc_steps = 8`
4. Mixed-precision training: `torch.amp.autocast(float16)` + `GradScaler` on CUDA
5. Optional `torch.compile()` via env var `GENOS_T1_USE_COMPILE=1`
6. Train for **5 epochs**; save checkpoint on every validation accuracy improvement
7. Best model saved to `models/gatekeeper.pt`

### Tier 2 — Specialist (`scripts/training/trainer2.py`)
Base model: **CodeBERT** (`microsoft/codebert-base`)

**Architecture**:
```
RobertaModel (codebert-base)
└── CLS token → Linear(768→1024) → LayerNorm(1024) → GELU → Dropout(0.3) → Linear(1024→1024) → GELU → Linear(1024→N_classes)
```
where `N_classes = len(specialist_map.json)` (108 classes in this branch).

**Training procedure** (precision patch / fine-tune mode):
1. Loads existing `models/specialist.pt` weights — this is a polishing run, not a cold start
2. Loads pre-split data from `specialist_train/val.csv`; maps `mitre_id` → integer via `specialist_map.json`
3. Trains for **1 epoch** (single polishing epoch)
4. **Focal Loss** with label smoothing: `ce_loss = F.cross_entropy(..., label_smoothing=0.1, reduction='none')`, then `loss = ((1 - exp(-ce_loss))^2 * ce_loss).mean()`
5. Per-class loss weighting: most classes = 1.0; priority underperforming classes (T1129, T1087, T1016, T1220) boosted to weight **8.0**; T1003 deliberately reduced to **2.0** to fix precision bleed
6. Saves on any validation accuracy improvement to `models/specialist.pt`

---

## 5. Hyperparameters

### Gatekeeper (trainer1.py)

| Parameter | Value |
|---|---|
| Base model | `microsoft/codebert-base` |
| Tokenizer max length | 256 |
| Micro batch size | 32 (env: `GENOS_T1_MICRO_BATCH`) |
| Effective batch size | 256 (env: `GENOS_T1_EFFECTIVE_BATCH`) |
| Gradient accumulation steps | 8 (= effective / micro) |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Weight decay | 0.01 |
| Loss function | CrossEntropyLoss |
| Class weights | `[1.0, 3.0]` (benign, malicious) |
| Epochs | 5 |
| Mixed precision | float16 via `torch.amp.autocast` + `GradScaler` |
| DataLoader workers | 4 |
| DataLoader pin_memory | True |
| Model selection | Best validation accuracy |
| `torch.compile` | Off by default (set `GENOS_T1_USE_COMPILE=1` to enable) |

### Specialist (trainer2.py — precision patch)

| Parameter | Value |
|---|---|
| Base model | `microsoft/codebert-base` |
| Tokenizer max length | 256 |
| Batch size | 16 |
| Optimizer | AdamW (param groups) |
| Encoder LR | 5e-7 |
| Classifier LR | 5e-6 |
| Weight decay | 0.01 |
| Loss function | Focal loss (γ=2) with label smoothing 0.1 |
| Priority class weight | 8.0 (T1129, T1087, T1016, T1220) |
| T1003 class weight | 2.0 (precision fix) |
| Default class weight | 1.0 |
| Epochs | 1 (polishing run) |
| DataLoader workers | 7 |
| DataLoader pin_memory | True |
| Model selection | Best (≥) validation accuracy |

### Engine inference

| Parameter | Value |
|---|---|
| Max token length | 256 (env: `GENOS_MAX_TOKENS`) |
| Mixed precision | float16 (CUDA) / bfloat16 (CPU) via `autocast` |
| Temperature scaling (Tier 2) | 0.5 (logits divided before softmax to sharpen distribution) |
| Top-K MITRE predictions returned | 5 |
| Max deobfuscation layers | 5 |
| Entropy threshold for obfuscation | > 5.2 bits |

---

## 6. How the App Works (Step-by-Step)

### Startup
1. `app.py` imports `GenosEngine` from `engine.py`
2. `GenosEngine.__init__()` runs:
   - Detects CUDA / CPU device
   - Loads `RobertaTokenizer` from `microsoft/codebert-base`
   - Resolves asset paths with fallback chains for `gatekeeper.pt`, `specialist.pt`, `config/specialist_map.json`
   - Instantiates `Tier1_Gatekeeper` and loads `gatekeeper.pt` weights → `.eval()`
   - Instantiates `Tier2_Specialist` (N=108 classes) and loads `specialist.pt` weights → `.eval()`
3. App runs a warm-up inference: `engine.scan("warmup")` — forces GPU kernel compilation and model loading to complete before the first real request
4. Sets `_engine_ready = True`
5. Gunicorn serves on `127.0.0.1:6001` (1 sync worker, 300s timeout to cover the load time)

### Request flow — `/scan` (authenticated)

```
POST /scan  {"api_key": "...", "command": "..."}
      │
      ▼
1. Input validation — check "api_key" and "command" keys present
2. MongoDB lookup — keys_collection.find_one({"key": api_key})
      │  invalid → 401
      ▼
3. _run_inference(command):
      │
      ├─ Auto Base64 decode attempt (base64.b64decode, validate=True)
      │   success → use decoded string; failure → use original
      │
      ├─ engine.scan(command):
      │     ├─ Deobfuscation loop (up to 5 layers):
      │     │     is_obfuscated() checks regex patterns + entropy > 5.2
      │     │     deobfuscate_layer(): universal_decoder → decode_embedded_base64
      │     │       → extract_powershell_payload → deobfuscate_char_constructions
      │     │       → clean_concatenation → pyminusone (if installed)
      │     │     Stops if: text unchanged OR entropy delta < 0.01
      │     │
      │     ├─ Normalise: lower().strip()
      │     │
      │     ├─ Tokenise: RobertaTokenizer(max_length=256, padding, truncation) → GPU
      │     │
      │     ├─ Tier 1 (Gatekeeper) forward pass:
      │     │     logits = t1(input_ids, attention_mask)
      │     │     probs = softmax(logits)
      │     │     label = "Malicious" if argmax==1 else "Benign"
      │     │     label_confidence = max(probs) × 100
      │     │
      │     └─ If Malicious → Tier 2 (Specialist) forward pass:
      │           logits = t2(input_ids, attention_mask)
      │           probs = softmax(logits / 0.5)  ← temperature scaling T=0.5
      │           top5 = topk(probs, k=5)
      │           MITRE_codes = [{code, confidence×100}, ...]
      │
      ├─ Normalise confidences via _to_percentage() (raw prob × 100, 2dp)
      │
4. Log usage to MongoDB: usage_collection.update_one(inc req_count)
      │
5. Return JSON response:
      {
        "label": "Malicious" | "Benign",
        "label_confidence": float (%),
        "MITRE_codes": [{"code": "T1059", "confidence": float}, ...]
      }
```

For **benign** commands, `MITRE_codes` is an empty array (Tier 2 is never called).

### Request flow — `/scan/internal`
Same as `/scan` but skips MongoDB entirely. Authenticates via `INTERNAL_TEST_TOKEN` env var (checked against `data.internal_token`). Used by the async stress tester and benchmark scripts.

### Other endpoints
- `GET /health` — returns `{"status": "ok"}` if `_engine_ready` is True, else `{"status": "loading"}`
- Direct engine CLI — `scripts/utils/genos.py` provides a local REPL shell that loads `GenosEngine` directly and prints colour-coded results without any HTTP layer

### Gunicorn configuration (`gunicorn.conf.py`)
- Bind: `127.0.0.1:6001` (env: `GENOS_API_BIND`)
- Workers: 1 (single worker — prevents duplicate GPU model loads; CUDA cannot be forked)
- Worker class: `sync`
- Timeout: 300s (covers model load + warm-up on startup)
- `preload_app = False` (not set) — worker initialises models itself, avoids CUDA fork issue
