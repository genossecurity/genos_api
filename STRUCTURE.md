# Genos Directory Structure

```
genos_dev/
├── app.py                  # Main Flask API server
├── genos.py               # CLI/utility functions
├── source.py              # Source data processing
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore (with LFS config)
├── README.md              # Project documentation
├── STRUCTURE.md           # This file
│
├── config/                # Configuration files (JSON)
│   ├── definitive_mitre_map.json
│   ├── label_map.json
│   └── specialist_map.json
│
├── data/                  # Training & evaluation datasets (CSV)
│   ├── benign_final.csv
│   ├── malicious_augmented.csv
│   ├── mitre_atlas_metadata.csv
│   ├── mitre_atlas_raw.csv
│   └── synthetic_benign_baseline.csv
│
├── models/                # Model checkpoints (Git LFS tracked)
│   ├── gatekeeper.pt      # Tier 1 classifier
│   ├── specialist.pt      # Tier 2 specialist model
│   └── mitre_atlas.index  # FAISS vector index
│
├── scripts/               # Training & evaluation scripts
│   ├── train_autoencoder.py
│   ├── eval.py
│   ├── stress.py
│   ├── inference_engine.py
│   ├── augment.py
│   ├── benign.py
│   ├── post_eval.py
│   └── trainer*.py
│
└── __pycache__/          # Python bytecode (ignored)
```

## Organization Principles

- **Root**: Only Entry Points
  - `app.py` - Flask API server
  - `genos.py` - CLI utilities
  - `source.py` - Data processing

- **config/**: Configuration Files
  - Model-specific mappings (JSON)
  - Label/specialist maps

- **data/**: Datasets
  - Training/evaluation CSVs
  - All command data

- **models/**: Neural Network Weights (Git LFS)
  - `.pt` checkpoints (Gatekeeper, Specialist)
  - `.index` FAISS vector index

- **scripts/**: Executable Scripts
  - Training: `train_autoencoder.py`, `trainer*.py`
  - Evaluation: `eval.py`, `stress.py`, `post_eval.py`
  - Inference: `inference_engine.py`
  - Utilities: `augment.py`, `benign.py`

## Git LFS Configuration

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

When running scripts from **scripts/** directory, reference paths as:
- Config: `../config/specialist_map.json`
- Data: `../data/benign_final.csv`
- Models: `../models/gatekeeper.pt`

When running from **root** directory, directly use:
- `config/specialist_map.json`
- `data/benign_final.csv`
- `models/gatekeeper.pt`
