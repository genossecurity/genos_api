# Genos Directory Structure

```
genos_api/
├── app.py                  # Main Flask API server
├── engine.py               # Inference engine with two-tier classification
├── gunicorn.conf.py        # Gunicorn WSGI configuration
├── gunicorn.ctl            # Gunicorn PID/control file
├── README.md               # Project documentation
├── STRUCTURE.md            # This file
│
├── config/                 # Configuration files (JSON)
│   ├── definitive_mitre_map.json    # MITRE ATT&CK to category mapping
│   ├── label_map.json               # Label definitions
│   └── specialist_map.json          # Specialist model class mapping
│
├── data/                   # Reference datasets (CSV)
│   └── art/
│       └── mitre_atlas_raw.csv      # Raw MITRE ATLAS source data
│
├── models/                 # Model checkpoints (Git LFS tracked)
│   ├── gatekeeper.pt       # Tier 1 classifier (benign vs malicious)
│   └── specialist.pt       # Tier 2 specialist model (MITRE ATT&CK classification)
│
├── scripts/                # Grouped scripts by function
│   ├── benchmark/
│   │   ├── ieee.py                  # IEEE benchmark evaluation
│   │   └── internal_api_test.py     # Internal API endpoint tests
│   └── ops/
│       └── reload_api.sh            # Unified ops script (start/reload/nginx/status)
│
└── __pycache__/            # Python bytecode (ignored)
```

## Organization Principles

- **Root**: Entry Points & Core Components
  - `app.py` - Flask REST API server
  - `engine.py` - Two-tier inference engine for command classification
  - `gunicorn.conf.py` - Production WSGI configuration
  - `gunicorn.ctl` - Gunicorn runtime PID/control file

- **config/**: Configuration Files
  - JSON mappings for labels and MITRE ATT&CK codes
  - Specialist model class-to-label mappings

- **data/**: Reference Datasets
  - `data/art/` - Raw MITRE ATLAS source data used for reference and mapping

- **models/**: Neural Network Weights (Git LFS)
  - `gatekeeper.pt` - Tier 1 benign/malicious classifier
  - `specialist.pt` - Tier 2 MITRE ATT&CK technique classifier

- **scripts/**: Executable Scripts (Grouped by Role)
  - `scripts/benchmark/`: benchmark and API test runners (`ieee.py`, `internal_api_test.py`)
  - `scripts/ops/`: operational service script (`reload_api.sh`)

## Scripts Reference

### scripts/benchmark/ieee.py

This script runs benchmark evaluation against the IEEE dataset, measuring detection accuracy and MITRE attribution quality. It belongs to the **evaluation cycle** as an external benchmark for model quality assessment.

### scripts/benchmark/internal_api_test.py

This script tests the live API endpoints for correctness and latency, sending known-malicious and known-benign commands to `/scan` and validating verdicts. It belongs to the **evaluation/deployment operations cycle** as a post-deploy smoke test.

### scripts/ops/reload_api.sh

This is the single operational control script for the API. It consolidates prior start, restart, and nginx-reload workflows into one entrypoint with modes: `reload` (default full restart + nginx reload + endpoint verification), `start` (foreground Gunicorn launch), `nginx` (nginx-only reload + health test on active API port), and `status` (active health check on ports 6001/6000). It belongs to the **evaluation/deployment operations cycle**, providing a single reproducible command surface for local or server runtime control.
## Core Components

### engine.py - Inference Engine

The `engine.py` module implements a two-tier neural network classification system for command-line analysis:

#### Architecture

**Tier 1: Gatekeeper Model** (`Tier1_Gatekeeper`)
- Purpose: Binary classification of commands as benign vs malicious
- Backbone: Microsoft CodeBERT (`microsoft/codebert-base`)
- Output: 2-class probabilities (benign/malicious) with confidence score
- Architecture:
  - CodeBERT encoder (768-dim embeddings)
  - FC layer: 768 → 1024
  - LayerNorm + GELU activation
  - Output layer: 1024 → 2 (benign/malicious)
  - Dropout: 0.3

**Tier 2: Specialist Model** (`Tier2_Specialist`)
- Purpose: Fine-grained MITRE ATT&CK technique classification
- Backbone: Microsoft CodeBERT
- Output: Probabilities for each MITRE technique in specialist_map.json
- Architecture:
  - CodeBERT encoder (768-dim embeddings)
  - Two-layer classifier with 1024-dim hidden layer
  - GELU activations and dropout (0.3)
  - Output: variable classes (depends on specialist_map.json)

#### GenosEngine Class

The `GenosEngine` class orchestrates inference:

**Initialization:**
- Loads both Tier 1 and Tier 2 models from checkpoints
- Loads specialist class mapping (specialist_map.json, specialist.pt)
- Instantiates RobertaTokenizer for text preprocessing
- Detects GPU availability and moves models to appropriate device

**Key Methods:**

`universal_decoder(text: str) → str`
- Detects and decodes Base64-encoded strings
- Detects and decodes hex-escaped strings (\x format)
- Returns decoded text if decodable, original text otherwise
- Handles malformed encodings gracefully

`scan(raw_cmd) → dict`
- Main inference method accepting raw command strings
- Preprocessing pipeline:
  1. Decode (Base64/hex)
  2. Convert to lowercase
  3. Tokenize with CodeBERT tokenizer (96 token max length)
- Tier 1 classification: Determines if malicious/benign
- Tier 2 classification: Generates probabilities for all MITRE techniques
- Returns top-5 MITRE techniques with confidence scores

**Output Format:**
```json
{
  "status": "Malicious|Benign",
  "gatekeeper_confidence": 0.95,
  "top_mitre": [
    {"code": "T1053.005", "confidence": 0.87},
    {"code": "T1219", "confidence": 0.72},
    ...
  ]
}
```

#### Dependencies

- PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)
- Transformers library (RobertaModel, RobertaTokenizer)
- JSON for loading specialist mappings

#### Workflow Integration

- Used by [app.py](app.py#L1) for REST API inference requests
- Loaded once on server startup for efficient batch inference
- Supports GPU acceleration for production deployments

Model files are tracked using Git LFS:
```
*.pt filter=lfs diff=lfs merge=lfs -text
```

To clone this repo with LFS files:
```bash
git lfs install
git clone <repo-url>
git lfs pull  # Download LFS files
```

## File Paths (From Scripts)

When running scripts from **scripts/** subfolders, use project-root-relative resolution:
- Benchmarks (`scripts/benchmark/*.py`) resolves config/models via `../../config` and `../../models`

When running from **root** directory, directly use:
- `config/specialist_map.json`
- `models/gatekeeper.pt`
- `models/specialist.pt`
