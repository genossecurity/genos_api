# Genos Directory Structure

```
genos_api/
├── app.py                  # Main Flask API server
├── engine.py              # Inference engine with two-tier classification
├── gunicorn.conf.py       # Gunicorn WSGI configuration
├── requirements.txt       # Python dependencies
├── req_res.txt            # Request/response examples
├── README.md              # Project documentation
├── STRUCTURE.md           # This file
│
├── config/                # Configuration files (JSON)
│   ├── definitive_mitre_map.json    # MITRE ATT&CK to category mapping
│   ├── label_map.json               # Label definitions
│   └── specialist_map.json          # Specialist model class mapping
│
├── data/                  # Training & evaluation datasets (CSV)
│   ├── training/          # Primary training datasets (loaded by trainer scripts)
│   │   ├── trainer1-good.csv        # Benign samples for Tier 1 (Gatekeeper) training
│   │   └── trainer1-bad.csv         # Malicious samples for Tier 1 (Gatekeeper) training
│   ├── archive/           # Historical, backup, and experimental datasets
│   │   ├── training/
│   │   │   ├── trainer1-bad.csv
│   │   │   └── trainer1-good.csv
│   │   ├── raw/
│   │   ├── art/
│   │   ├── benign_final.csv
│   │   ├── boost_data.csv
│   │   ├── genos_v1_master_benign.csv
│   │   ├── genos_v1_master_malicious.csv
│   │   ├── malicious_augmented.csv
│   │   ├── mitre_atlas_metadata.csv
│   │   ├── mitre_atlas_raw.csv
│   │   └── synthetic_benign_baseline.csv
│
├── models/                # Model checkpoints (Git LFS tracked)
│   ├── gatekeeper.pt      # Tier 1 classifier (benign vs malicious)
│   ├── specialist.pt      # Tier 2 specialist model (MITRE ATT&CK classification)
│   ├── mitre_atlas.index  # FAISS vector index
│   ├── specialist_map.json
│   └── archive/
│       └── gatekeeper_old.pt
│
├── scripts/               # Grouped scripts by function
│   ├── data_augmentation/
│   │   ├── augment.py     # Benign dataset prep with admin-noise injection
│   │   └── boost.py       # Inference probability boost heuristics
│   ├── benchmarks/
│   │   └── stress.py      # Unified benchmark (offline audit + fuzzing + API benchmark)
│   ├── data/
│   │   └── source.py      # MITRE source map generation utility
│   ├── training/
│   │   ├── trainer1.py    # Tier 1 (Gatekeeper) training script
│   │   └── trainer2.py    # Tier 2 (Specialist) training script
│   ├── ops/
│   │   ├── reload_api.sh  # API reload/restart script
│   │   └── run_gunicorn.sh # Gunicorn server launch script
│   ├── utils/
│   │   ├── genos.py       # Local CLI shell for engine testing
│   │   └── last_push.py   # Final gatekeeper fine-tuning utility
│   └── __pycache__/
│
└── __pycache__/          # Python bytecode (ignored)
```

## Organization Principles

- **Root**: Entry Points & Core Components
  - `app.py` - Flask REST API server
  - `engine.py` - Two-tier inference engine for command classification
  - `gunicorn.conf.py` - Production WSGI configuration
  
- **config/**: Configuration Files
  - JSON mappings for labels and MITRE ATT&CK codes
  - Specialist model class-to-label mappings

- **data/**: Datasets (organized by purpose)
  - `data/training/` - Active training datasets loaded by trainer scripts (Tier 1/Tier 2 inputs)
  - `data/archive/` - Historical, experimental, and backup datasets (preserved for reference or debugging)

- **models/**: Neural Network Weights (Git LFS)
  - `gatekeeper.pt` - Tier 1 benign/malicious classifier
  - `specialist.pt` - Tier 2 MITRE ATT&CK technique classifier
  - `mitre_atlas.index` - FAISS vector index for similarity search

- **scripts/**: Executable Scripts (Grouped by Role)
  - `scripts/training/`: model training (`trainer1.py`, `trainer2.py`)
  - `scripts/data_augmentation/`: data and inference augmentation (`augment.py`, `boost.py`)
  - `scripts/benchmarks/`: unified benchmark runner (`stress.py`)
  - `scripts/data/`: source-to-map data processing (`source.py`)
  - `scripts/ops/`: operational service scripts (`run_gunicorn.sh`, `reload_api.sh`)
  - `scripts/utils/`: local utility scripts (`genos.py`, `last_push.py`)

## Scripts Reference

### scripts/data/source.py

This script reads raw MITRE-labeled command data and generates a deterministic label-index mapping (`definitive_mitre_map.json`) used by downstream training and analysis. It is part of the **data generation/preparation cycle**, specifically the label-engineering stage that standardizes class IDs before model training.

### scripts/data_augmentation/augment.py

This script builds a benign training dataset by loading the synthetic benign baseline, normalizing commands, injecting random admin-noise commands, shuffling, and writing `benign_final.csv`. It belongs to the **data generation cycle**, focused on synthetic expansion of benign samples to improve Tier 1 robustness.

### scripts/data_augmentation/boost.py

This module applies rule-based post-processing to Tier 2 probabilities (MITRE techniques), boosting or penalizing classes based on command patterns such as encoded PowerShell, LOLBins, or credential-theft indicators. It participates in the **evaluation/inference calibration cycle**, where model outputs are adjusted to better reflect domain heuristics at prediction time.

### scripts/training/trainer1.py

This script trains the Tier 1 Gatekeeper binary classifier (benign vs malicious) using `benign_final.csv` and `malicious_augmented.csv`, tracks epoch accuracy, and saves `models/gatekeeper.pt`. It is part of the **training cycle**, specifically first-stage detection model fitting.

### scripts/training/trainer2.py

This script trains the Tier 2 Specialist multi-class MITRE classifier using balanced sampling, focal-style loss behavior, and differential learning rates between encoder and head, while regenerating `config/specialist_map.json` and saving `models/specialist.pt`. It belongs to the **training cycle**, specifically second-stage attribution model fitting.

### scripts/benchmarks/stress.py

This is the single, final benchmark script for the project. It combines three evaluation modes in one place: offline model audit (false positives, detection rate, MITRE attribution), fuzzing resilience tests (obfuscation-aware robustness checks), and live API benchmarking (latency and verdict accuracy against `/scan`). It belongs to the **evaluation cycle** as the unified quality gate for checkpoint validation and deployment verification.

### scripts/ops/run_gunicorn.sh

This operational script activates the project environment, frees the configured port, and starts the Flask app under Gunicorn using the repository config. It is part of the **evaluation/deployment operations cycle**, enabling reproducible local or server runtime for inference and benchmark execution.

### scripts/ops/reload_api.sh

This script force-restarts Gunicorn, clears existing listeners, relaunches the API, and waits for `/health` to return success so startup readiness is verified automatically. It belongs to the **evaluation/deployment operations cycle**, supporting controlled rollouts and restart validation.

### scripts/utils/genos.py

This utility provides an interactive local shell that loads `GenosEngine`, accepts commands, and prints live benign/malicious decisions with top MITRE prediction and confidence values. It is in the **evaluation cycle** as a manual smoke-test and qualitative inspection tool for inference behavior.

### scripts/utils/last_push.py

This utility performs a short gatekeeper fine-tuning “nudge” pass by combining malicious, benign, and boost datasets, then loading archived weights and saving an updated `gatekeeper.pt`. It is part of the **training cycle**, typically used as a final refinement step before evaluation or deployment.

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
- Applies confidence boosts using `boost.py` engine
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

Large model and index files are tracked using Git LFS:
```
*.pt filter=lfs diff=lfs merge=lfs -text
*.index filter=lfs diff=lfs merge=lfs -text
```

To clone this repo with LFS files:
```bash
git lfs install
git clone <repo-url>
git lfs pull  # Download LFS files
```

## File Paths (From Scripts)

When running scripts from **scripts/** subfolders, use project-root-relative resolution:
- Training (`scripts/training/*.py`) resolves data/models via `../../data` and `../../models`
- Benchmarks (`scripts/benchmarks/*.py`) resolves config/data/models via `../../config`, `../../data`, and `../../models`
- Augmentation/Data scripts resolve files via `../../data` (and `../../config` where needed)

When running from **root** directory, directly use:
- `config/specialist_map.json`
- `data/benign_final.csv`
- `models/gatekeeper.pt`
