# Genos API

Genos is a command-analysis API with a three-class gatekeeper, an 11-label multi-label family specialist, and a behavior encoder, served over Gunicorn and Flask.

The system was developed as part of an IEEE research programme. See the [scientific validation report](artifacts/scientific_validation/REPORT.md) for measured results and unresolved validity gaps.

---

## How the engine works

The core inference logic lives in the `genos/` package (`genos/engine.py` as the router, with `genos/gatekeeper.py`, `genos/specialist.py`, `genos/evidence.py`, `genos/deobfuscator.py`, and `genos/baseline.py` as dedicated components). The Flask API in `app.py` wraps it.

### Startup

When the process starts:

1. `python-dotenv` loads `.env` from the working directory.
2. `GenosEngine` is constructed once — no hot-reloading of models.
3. The engine resolves its asset paths in this order for each file:
   - absolute path (if given)
   - relative to `os.getcwd()`
   - relative to the directory containing `genos/engine.py`
   - each fallback candidate in turn
4. The specialist label map is loaded from the first file that exists:
   - `map_path` argument passed by the caller
   - `config/specialist_map.json` ← current live path
   - `models/specialist_map.json`
   - If none found: built dynamically by reading `mitre_id` values from the raw MITRE CSV and sorting them
5. The shared CodeBERT tokenizer/config are loaded from `GENOS_CODEBERT_PATH` (with an optional local-only mode).
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

Uses the local CodeBERT directory configured by `GENOS_CODEBERT_PATH`:

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

The default `GENOS_VIEW_POLICY=mean` averages raw and decoded gatekeeper scores when decoding changes the command. The family specialist receives the deobfuscated command. `raw` and `decoded` remain available for gatekeeper experiments. All commands receive behavior and family analysis.

Scores are uncalibrated model estimates unless a validation-fitted, model-bound `GENOS_CALIBRATION_PATH` is supplied. Legacy `confidence` fields use percentages (0–100); `score_type` declares their interpretation. Behavior action tags use the reported `GENOS_BEHAVIOR_ACTION_THRESHOLD` (default 0.5), which has not yet been selected on independent labels.

Checkpoints load strictly. Behavior load failures stop initialization unless `GENOS_ALLOW_BEHAVIOR_FALLBACK=1` explicitly permits the reported heuristic baseline. Responses include model/implementation hashes, representation, preprocessing policy, fallback status, and truncation flags.

See [scientific validation workflow](docs/scientific_validation.md) for the audit, grouped datasets, experiments, annotation protocol, and remaining limitations. Existing benchmark scores are development evidence and do not establish independent operational accuracy.

### Tier 2 — Family Specialist

The default specialist is a multi-label linear model over char and word TF-IDF n-grams. It emits probabilities for Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Discovery, Lateral Movement, Command-and-Control / Payload Retrieval, Exfiltration, Impact, and Benign Admin. Its model rows and runtime response contain family labels, not technique IDs.

Runtime artifact: `models/family_specialist_tfidf.joblib`. It uses the deobfuscated command, lowercased and stripped. `GENOS_SPECIALIST_MODE=mitre` explicitly selects the retained legacy technique ranker for secondary top-5 comparisons; it is not the default.

### Engine output schema

`GenosEngine.scan()` returns:

```json
{
  "label": "Malicious",
  "label_confidence": 99.81,
  "score_type": "uncalibrated_model_estimate",
  "deobfuscated_cmd": "invoke-expression ...",
  "specialist_mode": "family",
  "attack_families": {
    "decision_threshold": 0.5,
    "predicted_families": [
      { "family": "Execution", "probability": 82.4, "selected": true }
    ],
    "all_family_scores": ["one score for each of the 11 families"]
  }
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
- Default responses contain multi-label `attack_families`; `MITRE_codes` is only emitted when the legacy `mitre` specialist mode is explicitly selected
- `deobfuscated_cmd` is `null` when the input was not flagged as obfuscated

---

## Models

The gatekeeper uses CodeBERT by default. The primary specialist is an 11-family calibrated linear TF-IDF model. Behavior analysis remains a separate CodeBERT encoder.

| File | Purpose |
|---|---|
| `models/gatekeeper.pt` | Tier 1 — 3-class CodeBERT gatekeeper (Benign / Context_Dependent / Malicious) |
| `models/family_specialist_tfidf.joblib` | Default specialist — calibrated multi-label 11-family classifier |
| `models/specialist_tfidf_char_rf.pkl` | Legacy secondary MITRE technique ranker, only used in explicit `mitre` mode |
| `models/archive/specialist_tfidf_rf.pkl` | Archived Tier 2 word-level TF-IDF + RF alternative |
| `config/specialist_map.json` | Legacy MITRE-mode label map |
| `config/gatekeeper_meta.json` | Gatekeeper class-map and training metadata read at startup |

Model files are local runtime assets and are excluded from Git by `.gitignore`; provide the active checkpoint files separately when deploying. Historical and experimental checkpoints are grouped under `models/archive/`, and retraining outputs go under `models/experiments/`.

An experimental calibrated linear TF-IDF gatekeeper can be selected without loading the CodeBERT gate checkpoint:

```bash
GENOS_GATEKEEPER_BACKEND=tfidf
GENOS_GATEKEEPER_TFIDF_PATH=models/experiments/tfidf_gatekeeper_20261004_seed42/gatekeeper_tfidf.joblib
```

CodeBERT remains the default gatekeeper. The default specialist mode is `family`; the legacy technique ranker remains available only when `GENOS_SPECIALIST_MODE=mitre`. Behavior still uses its own CodeBERT encoder.

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

### Local Hugging Face weights

The gatekeeper and behavior model checkpoints in `models/` still need the
CodeBERT tokenizer/config files. Download that shared Hugging Face artifact once
into the repository, then startup reads it from disk:

```bash
venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/codebert-base', local_dir='models/codebert-base')"
```

Set these values in `.env`:

```dotenv
GENOS_CODEBERT_PATH=models/codebert-base
GENOS_HF_LOCAL_ONLY=1
```

With local-only mode enabled, missing files fail during startup instead of
triggering a network download. A direct Hugging Face cache snapshot can also be
used as `GENOS_CODEBERT_PATH`.

### API request builder — `/api`

Open `/api` (also linked from the scanner) to enter a command, select optional
Tier 2 analysis and IOC types, copy a shell-safe curl request, or run it in the
page. Response examples are illustrative; the Run buttons show actual results.

### `GET /api/scan` — customizable analysis

The default request runs Tier 1 and automatically detects and decodes obfuscation:

```bash
curl --get --silent --show-error 'http://127.0.0.1:6001/api/scan' \
  --data-urlencode 'command=cat /etc/shadow'
```

Example response (confidence values are illustrative):

```json
{
  "label": "Context_Dependent",
  "label_confidence": 99.0,
  "deobfuscated_cmd": null
}
```

| Query parameter | Default | Meaning |
|---|---|---|
| `command` | Required | Command text; use `--data-urlencode` for spaces and special characters. |
| `tier2` | `false` | `true` adds stage analysis for non-benign commands. `1` and `0` are also accepted. |
| `iocs` | Omitted | `all`, or a comma-separated selection of `urls,domains,ips,ports,files,registry`. |

Tier 2 and IOC extraction are independent and only run when requested. Tier 1 is
always included. This endpoint analyzes each request without baseline shortcuts
or baseline sighting writes. Existing POST scanner routes retain their full response.

```bash
curl --get --silent --show-error 'http://127.0.0.1:6001/api/scan' \
  --data-urlencode 'command=cat /etc/shadow' \
  --data-urlencode 'tier2=true' \
  --data-urlencode 'iocs=urls,ips,files'
```

This adds `tier2: {"status": "completed", "stage": "Credential Access",
"stage_confidence": 98.4}` and an `iocs` object containing only the requested keys.
All confidence values use a 0–100 scale. Benign commands skip Tier 2 even when
requested, returning `status: "skipped_benign"` and null stage fields.
`deobfuscated_cmd` is null when no obfuscation was detected; selected IOC types
with no matches return empty arrays. Omit `iocs` to exclude IOC data entirely.
Invalid, duplicated, or unknown query parameters return HTTP 400. Responses use
`Cache-Control: no-store`.

### `GET /health`

```json
{ "status": "ok" }
```

The app loads model weights and completes a warm-up inference before the web
server begins serving pages. If initialization fails, the process exits instead
of serving an unready scanner.

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
  "specialist_mode": "family",
  "attack_families": {
    "predicted_families": [
      { "family": "Discovery", "probability": 97.43, "selected": true }
    ]
  }
}
```

Response (benign):

```json
{
  "label": "Benign",
  "label_confidence": 99.99,
  "specialist_mode": "family",
  "attack_families": {
    "predicted_families": [
      { "family": "Benign Admin", "probability": 99.9, "selected": true }
    ]
  }
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
| `GENOS_VIEW_POLICY` | `genos/engine.py` | `mean` | Raw/decoded distribution fusion policy |
| `GENOS_CALIBRATION_PATH` | `genos/engine.py` | unset | Validation-fitted artifact bound to the runtime |
| `GENOS_BEHAVIOR_POLICY_PATH` | `genos/engine.py` | unset | Validation-selected per-action thresholds |
| `GENOS_ALLOW_BEHAVIOR_FALLBACK` | `genos/engine.py` | `0` | Explicitly allow reported heuristic fallback |
| `GENOS_MAX_TOKENS` | `genos/engine.py` | `256` | Tokeniser max sequence length |
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

from genos.engine import GenosEngine

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
| `worker_class` | `gthread` | Serve pages alongside inference; scans share an inference lock |
| `threads` | `4` | Request concurrency; configurable with `GENOS_API_THREADS` (minimum 2) |
| `timeout` | `300` | Worker heartbeat timeout |
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
| `scripts/evaluation/compare_tfidf_codebert_gatekeepers.py` | Paired grouped comparison, including obfuscated and runtime-view slices |
| `scripts/data/build_tactic_family_dataset.py` | Build template-grouped, multi-label family splits from source tactic metadata |
| `scripts/training/train_family_specialist.py` | Train the 11-label calibrated family specialist |
| `scripts/training/trainer1.py` | Train the three-class gatekeeper on audited grouped splits |
| `scripts/training/train_tfidf_gatekeeper.py` | Train an experimental calibrated CPU linear gatekeeper |
| `scripts/training/train_behavior_encoder.py` | Train raw or structured behavior representations |
| `scripts/training/trainer_tfidf.py` | Train raw or structured MITRE RF baselines |

These trainers default to the prepared task directories under `data/derived/scientific_v2/` and write isolated experiments under `models/experiments/`. Original legacy data remains under `data/training/`. Labels are still weak supervision until independent reviews are completed.

---

## Repository layout

```
app.py                              Flask application and route handling
gunicorn.conf.py                    Gunicorn runtime configuration
requirements.txt                    Python dependencies
.env.example                        Environment variable template

genos/                              Core engine package (import as `genos.*`)
  __init__.py                       Public package exports
  engine.py                         GenosEngine router — coordinates the full scan pipeline
  gatekeeper.py                     Tier 1 triage: Benign vs Suspicious classification
  specialist.py                     Tier 2 specialist: 11 MITRE ATT&CK tactic families + behavior
  evidence.py                       IOC extraction and analyst evidence summaries
  deobfuscator.py                   Standalone multi-layer deobfuscation logic
  baseline.py                       Stateful baseline store and novelty scoring
  scientific_validation.py          Shared dataset audits and evaluation metrics

config/
  specialist_map.json               Legacy MITRE-mode technique map
  definitive_mitre_map.json         Legacy integer technique map
  label_map.json                    Human-readable label definitions
  gatekeeper_meta.json              Active gatekeeper metadata
  meta/                             Historical training metadata and config snapshots

models/
  gatekeeper.pt                     Active Tier 1 checkpoint
  behavior_encoder.pt               Active behavior checkpoint
  family_specialist_tfidf.joblib    Active 11-family multi-label specialist
  specialist_tfidf_char_rf.pkl      Legacy secondary MITRE ranker
  archive/                          Historical and experimental model files
  experiments/                      Isolated retraining runs and metadata

data/
  training/                         Source and legacy datasets, grouped by task/version
    genos_dataset/                  Gatekeeper datasets and supervised patches
    genos_behavior/                 Behavior labels
    genos_residual_expanded/        Legacy MITRE technique dataset
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
