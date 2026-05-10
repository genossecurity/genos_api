# Genos API — `ieee-paper2` Branch

This branch is a **read-only snapshot** of the Genos V1 codebase at the point it was declared
IEEE-paper-ready (commit `892f4aa` — "ieee ready benchmark").

Its sole purpose is to preserve the exact code, data splits, training scripts, benchmark scripts,
and model architecture that produced the results reported in the second IEEE paper. Nothing on
this branch should be modified; all active development continues on `main`.

## What this branch contains

| Path | Description |
|---|---|
| `app.py` | Flask API server — `/scan`, `/scan/internal`, `/health` endpoints |
| `engine.py` | `GenosEngine` — two-tier CodeBERT inference with deobfuscation pipeline |
| `scripts/training/trainer1.py` | Gatekeeper (binary) training — 5-epoch AdamW + AMP |
| `scripts/training/trainer2.py` | Specialist (multi-class MITRE) precision-patch training — 1-epoch focal loss |
| `scripts/benchmark/ieee.py` | IEEE pipeline benchmark — TF-IDF baseline, Tier 1 AUC/ROC, Tier 2 Top-K, end-to-end latency sweep |
| `scripts/benchmark/internal_api_test.py` | Async live stress test — 500 requests, 20 concurrent workers, 85% confidence threshold |
| `data/training/` | Pre-split train/val/test CSVs for both Gatekeeper and Specialist |
| `config/specialist_map.json` | 108-class MITRE technique → integer label mapping |
| `models/gatekeeper.pt` | Trained Tier 1 binary classifier weights |
| `models/specialist.pt` | Trained Tier 2 MITRE technique classifier weights |
| `specialist_f1_report.txt` | Per-class F1 report — 91% accuracy, 0.92 macro F1 over 2,820 held-out samples |
| `live_stress_report.txt` | Output log from the async API stress test |
| `notes.md` | Comprehensive notes covering data, methodology, hyperparameters, and app flow |

## Key results at this snapshot

- **Tier 1 Gatekeeper**: binary benign/malicious classification on CodeBERT (`microsoft/codebert-base`)
- **Tier 2 Specialist**: 108-class MITRE ATT&CK technique classifier — 91% accuracy, macro F1 0.92
- **Stress test**: 500-request async benchmark at 20 concurrent workers, 85% confidence threshold
- **IEEE benchmark**: full deployment-aligned pipeline evaluation including TF-IDF RF baseline comparison, Tier 1 ROC/AUC, end-to-end latency sweep at 0.50 / 0.90 / 0.99 benign traffic ratios

See `notes.md` for a full breakdown of all six areas: data splits, test methodology, stress test methodology, training methodology, hyperparameters, and step-by-step app flow.
