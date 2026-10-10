# 11-family command-text curation, 2026-10-04

The original family dataset was already mapped to 11 output families. This pass changes 12 training rows whose visible command behavior contradicts the weak source-derived family label. The corrections and command-specific reasons are in `data/derived/family_specialist_v1/relabel_train_command_text_curated.csv`. This is a Codex command-text curation pass, not independent human review or proof of malicious intent.

`data/derived/reviewed_v1/accepted.jsonl` contains gatekeeper verdicts rather than 11-family annotations, so it was not used as a family-label source.

The source train, validation, and test files are preserved. The corrected copy is `data/derived/family_specialist_v2_curated/`; its `correction_audit.jsonl` records before/after labels and provenance. Validation and test labels are byte-for-byte unchanged. The corrected split still has 37,846 training rows and 11 families. Template-group disjointness is checked before training.

The review queues are `data/derived/family_specialist_v1/relabel_train_review.csv` (220 distinct training templates, balanced across rare families) and `data/derived/family_specialist_v1/relabel_validation_review.csv` (200 distinct validation templates prioritized by model disagreement). Their `corrected_family_labels` fields remain blank. The latter includes model scores and is a prioritization aid rather than a blind annotation form. Do not use test predictions to select corrections.

To apply later corrections, fill `corrected_family_labels` with a JSON array of one or more of the exact family names and add a reason in `review_notes`. Leave unreviewed rows blank. Benign Admin must be the only label on its row. Then run:

```bash
venv/bin/python scripts/data/apply_family_label_reviews.py \
  --reviews data/derived/family_specialist_v1/relabel_train_review.csv \
            data/derived/family_specialist_v1/relabel_validation_review.csv \
  --output-dir data/derived/family_specialist_next
venv/bin/python scripts/training/train_family_specialist.py \
  --data-dir data/derived/family_specialist_next \
  --output-dir models/experiments/family_specialist_next
venv/bin/python scripts/training/train_family_codebert.py \
  --data-dir data/derived/family_specialist_next \
  --output-dir models/experiments/family_codebert_next --epochs 4
```

The apply script rejects unknown labels, duplicate corrections, template mismatches, and test edits. At least one changed label is required. Training artifacts remain experiments until a separately justified runtime promotion.

## Current experiment

The corrected runs are `models/experiments/family_specialist_20261004_curated_seed42/` and `models/experiments/family_codebert_20261004_curated_seed42/`. Both used seed 42 and a fixed 0.5 decision threshold. CodeBERT used four epochs and selected the fourth by validation macro-F1. Results are against the same weak-label validation and test rows as the original runs:

| Model | Validation macro-F1 | Validation top-1 | Test macro-F1 | Test top-1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF original | 0.6043 | 0.9365 | 0.5699 | 0.9371 |
| TF-IDF corrected | 0.6053 | 0.9363 | 0.5710 | 0.9378 |
| CodeBERT original | 0.5278 | 0.9230 | 0.5035 | 0.9114 |
| CodeBERT corrected | 0.5168 | 0.9241 | 0.5111 | 0.9169 |

TF-IDF's validation change is negligible. CodeBERT's validation macro-F1 is lower after correction despite a higher test score. The test labels are weak and these rows have already been examined in prior comparisons, so test differences are descriptive. Most weak labels remain unreviewed, and rare-family support remains small.

## Runtime promotion

The corrected TF-IDF checkpoint and matching metadata were copied to the local API's default paths, `models/family_specialist_tfidf.joblib` and `models/family_specialist_tfidf.json`, at the user's request. The active local checkpoint SHA-256 is `8e66c5a2a86ea06b151dd88ab2e5da8b9ae38832d2843813ce0aa49aa5387e44`. The original checkpoint remains in `models/experiments/family_specialist_20261004_seed42/`. The `models/` directory is gitignored, so another environment needs the same artifact copied there separately. This promotion changes which family model the local API loads on startup; it does not establish operational accuracy beyond the weak-label evaluation above.
