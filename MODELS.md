# Models

Everything the Genos pipeline loads at runtime. Weights are not stored in Git
(`models/` is git-ignored). Download them from Hugging Face:

```bash
hf download genos-security/genos --local-dir models
hf download microsoft/codebert-base --local-dir models/codebert-base
```

| File | Used by | What it does |
|---|---|---|
| `models/gatekeeper.pt` | Tier 1, [genos/gatekeeper.py](genos/gatekeeper.py) | CodeBERT fine-tune. Classifies a command as Benign, Context_Dependent, or Malicious. Benign commands skip Tier 2. Class map and training metadata come from `config/gatekeeper_meta.json`. |
| `models/family_specialist_tfidf.joblib` | Tier 2, [genos/specialist.py](genos/specialist.py) | Calibrated multi-label TF-IDF (char and word) linear classifier. Assigns the command to one or more of 11 behavior families. Default specialist (`GENOS_SPECIALIST_MODE=family`). |
| `models/family_specialist_tfidf.json` | Tier 2 | Training configuration and feature/calibration metadata for the specialist. |
| `models/behavior_encoder.pt` | Tier 2, [genos/specialist.py](genos/specialist.py) | CodeBERT fine-tune. Predicts the attack stage (13 classes) and action tags (9 multi-label), `max_length` 256. Loaded at startup by `app.py`. |
| `models/behavior_encoder.json` | Tier 2 | Stage and action label maps, backbone name, and test metrics for the encoder. |
| `models/codebert-base/` | Tier 1 and the behavior encoder | Shared `microsoft/codebert-base` tokenizer and config (downloaded from Hugging Face, not ours). Set by `GENOS_CODEBERT_PATH`. |

Small label and config files the models depend on live in `config/` and are
tracked in Git.

## Not required

- `models/specialist_tfidf_char_rf.pkl`: legacy MITRE technique ranker, used only
  when `GENOS_SPECIALIST_MODE=mitre`.
- `models/archive/` and `models/experiments/`: historical checkpoints and
  training runs. They are not loaded by the API and are not on Hugging Face.

## Security

`.pt`, `.joblib`, and `.pkl` files are Python pickles. Loading one executes code
embedded in it, so only load files from sources you trust.
