# Genos API

Flask API that triages a single command line: is it benign, malicious, or does it need context, and if not benign, which attack families and stage it matches.

Results are research-grade and unvalidated on independent data. See the [validation report](artifacts/scientific_validation/REPORT.md).

## Pipeline

```
command
  │
  ├─ 1. Deobfuscate        entropy > 5.2 or known patterns → up to 5 decode passes
  │
  ├─ 2. Tier 1 gatekeeper  CodeBERT → Benign / Malicious / Context_Dependent
  │                        raw and decoded scores averaged (GENOS_VIEW_POLICY=mean)
  │
  ├─ 3. Routing            rule features can adjust the verdict and routing
  │
  ├─ 4. Tier 2             non-benign only (Benign skips it)
  │     ├─ family specialist   TF-IDF + linear SVM → 11 attack families
  │     └─ behavior encoder    CodeBERT → attack stage + action tags
  │
  └─ 5. Evidence           URLs, domains, IPs, ports, files, registry paths
```

The POST routes also check a baseline store first: commands known to be stable return `Benign` without running the models. `GET /api/scan` never uses the baseline.

## Models

Weights are not in Git. See [MODELS.md](MODELS.md) for what each file does.

| File | Role |
|---|---|
| `models/gatekeeper.pt` | Tier 1 verdict |
| `models/family_specialist_tfidf.joblib` | Tier 2 attack families |
| `models/behavior_encoder.pt` | Tier 2 attack stage and action tags |
| `models/codebert-base/` | Shared tokenizer and config (`microsoft/codebert-base`) |

```bash
hf download genos-security/genos --local-dir models
hf download microsoft/codebert-base --local-dir models/codebert-base
```

The `.pt` and `.joblib` files are pickles; only load files you trust.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, install it first from the [PyTorch index](https://download.pytorch.org/whl/), then run the command above. CPU works too. `pyminusone` is optional and adds PowerShell AST deobfuscation.

Create `.env`:

```dotenv
GENOS_CODEBERT_PATH=models/codebert-base
GENOS_HF_LOCAL_ONLY=1
GENOS_SPECIALIST_MODE=family
GENOS_FAMILY_SPECIALIST_PATH=models/family_specialist_tfidf.joblib
```

## Run

```bash
gunicorn -c gunicorn.conf.py app:app     # 127.0.0.1:6001
curl -s http://127.0.0.1:6001/health
```

Startup loads the models and runs a warm-up scan before serving. If a checkpoint fails to load the process exits.

Gunicorn runs one worker (one model copy in memory) with 4 threads; inference is serialized by a lock. Put Nginx or Caddy in front for TLS and rate limiting, and don't bind to `0.0.0.0`.

## API

| Route | Purpose |
|---|---|
| `GET /` | Web scanner |
| `GET /api` | Request builder page |
| `GET /api/scan` | Customizable scan (below) |
| `POST /scan`, `POST /api/scan` | Full scan result |
| `GET /api/families` | The 11 family labels |
| `GET /health` | Readiness, device, specialist mode |

### `GET /api/scan`

```bash
curl --get 'http://127.0.0.1:6001/api/scan' \
  --data-urlencode 'command=cat /etc/shadow' \
  --data-urlencode 'tier2=true' \
  --data-urlencode 'iocs=urls,ips,files'
```

| Parameter | Default | Meaning |
|---|---|---|
| `command` | required | Command to scan |
| `tier2` | `false` | `true` adds `tier2.stage` and `tier2.stage_confidence`. Benign commands return `status: "skipped_benign"`. |
| `iocs` | omitted | `all`, or any of `urls,domains,ips,ports,files,registry` |

Returns `label`, `label_confidence` (0-100), and `deobfuscated_cmd` (null if not obfuscated). Unknown or repeated parameters return 400. Responses are `Cache-Control: no-store`.

### `POST /scan`

```bash
curl -X POST http://127.0.0.1:6001/scan \
  -H 'Content-Type: application/json' \
  -d '{"command": "net user /domain"}'
```

Optional `"include"` object turns response sections on or off: `evidence`, `mitre`, `families`, `analysis`, `ioc`, `meta`. Form-encoded `command` also works. There is no authentication; keep it behind your proxy.

Main response fields:

| Field | Meaning |
|---|---|
| `label` | `Benign`, `Malicious`, or `Context_Dependent` |
| `label_confidence` | 0-100 model score; see `score_type` |
| `attack_families` | `predicted_families` plus `all_family_scores` for the 11 families |
| `attack_stage`, `behavior` | Stage and action tags (non-benign only) |
| `deobfuscated_cmd`, `decoded_payload` | Decoded text when obfuscation was found |
| `evidence`, `ioc_summary` | Extracted indicators |
| `action` | `pass` or `requires_context` |

Scores are uncalibrated unless `GENOS_CALIBRATION_PATH` is set. `Context_Dependent` means the command can't be judged without more context.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GENOS_CODEBERT_PATH` | `models/codebert-base` | Tokenizer/backbone directory |
| `GENOS_HF_LOCAL_ONLY` | `0` | `1` blocks network fallback at startup |
| `GENOS_SPECIALIST_MODE` | `family` | `mitre` selects the legacy technique ranker |
| `GENOS_FAMILY_SPECIALIST_PATH` | `models/family_specialist_tfidf.joblib` | Specialist file |
| `GENOS_VIEW_POLICY` | `mean` | `raw`, `decoded`, or `mean` gatekeeper scoring |
| `GENOS_MAX_TOKENS` | `256` | Token limit (max 512) |
| `GENOS_GATEKEEPER_BACKEND` | `codebert` | `tfidf` selects the experimental CPU gatekeeper (needs `GENOS_GATEKEEPER_TFIDF_PATH`) |
| `GENOS_BEHAVIOR_ACTION_THRESHOLD` | `0.5` | Action tag cutoff |
| `GENOS_CALIBRATION_PATH` | unset | Validation-fitted calibration artifact |
| `GENOS_ALLOW_BEHAVIOR_FALLBACK` | `0` | Allow heuristic behavior fallback if the checkpoint fails |
| `GENOS_API_BIND` | `127.0.0.1:6001` | Gunicorn bind address |
| `GENOS_API_THREADS` | `4` | Gunicorn threads (min 2) |

## Layout

```
app.py              Flask routes
genos/              engine, gatekeeper, specialist, deobfuscator, evidence, baseline
config/             label maps and gatekeeper metadata
parser/             command parser and rule engine
scripts/            data prep, training, evaluation, ops
tests/              unit tests
docs/               validation workflow
models/             weights (git-ignored)
```

Training and evaluation workflows are in [docs/scientific_validation.md](docs/scientific_validation.md).

## Tests

```bash
python -m unittest discover tests
```
