# TF-IDF Gatekeeper Comparison

## Setup

- Same frozen grouped `scientific_v2` train/validation/test splits as CodeBERT.
- Both models used the same 250-row `gatekeeper_benign_core_patch_v2a.jsonl` training patch; no patch examples were added to validation or test.
- Test set: 7,236 rows, 2,441 recorded holdout groups. Labels are legacy weak supervision.
- TF-IDF: lowercase/strip, union of `char_wb` 2-5 n-grams and word 1-2 n-grams (token pattern retains flags such as `-enc` and `/c`), `LinearSVC(class_weight=balanced)` with sigmoid `CalibratedClassifierCV`; three `StratifiedGroupKFold` folds over training groups.
- CodeBERT checkpoint: seed 42, five epochs, completed checkpoint at `models/experiments/retrain-20261004-104842/gatekeeper/seed_42/gatekeeper.pt`.
- TF-IDF fit time: 8.72 seconds. The model, metadata, and per-example predictions are saved under `models/experiments/tfidf_gatekeeper_20261004_seed42/`.

## Results

| Test view | N | CodeBERT accuracy | TF-IDF accuracy | CodeBERT macro-F1 | TF-IDF macro-F1 | TF-IDF minus CodeBERT macro-F1, grouped 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 7,236 | 0.9168 | 0.9028 | 0.7362 | 0.6953 | -0.0409 [-0.0676, -0.0160] |
| Runtime mean view | 7,236 | 0.9167 | 0.9027 | 0.7360 | 0.6951 | -0.0409 [-0.0658, -0.0172] |
| Obfuscated, raw view | 136 | 0.8750 | 0.8676 | 0.7483 | 0.7209 | -0.0274 [-0.1526, 0.0921] |
| Obfuscated, runtime mean view | 136 | 0.8676 | 0.8603 | 0.7367 | 0.7108 | -0.0259 [-0.1508, 0.0969] |

Raw-view paired accuracy difference was -0.0140 [-0.0219, -0.0058] for TF-IDF minus CodeBERT. Runtime-mean paired accuracy difference was -0.0140 [-0.0222, -0.0062]. Intervals use 2,000 paired bootstrap resamples of recorded holdout groups.

TF-IDF test per-class F1: Benign 0.9524, Malicious 0.7561, Context_Dependent 0.3775. CodeBERT seed-42 trainer metrics: Benign 0.9626, Malicious 0.7785, Context_Dependent 0.4676.

The existing label-rule baseline scored accuracy 0.8223 and macro-F1 0.3857 on this test file. It shares rule families with the weak labels and measures teacher agreement, not true intent. The prepared rows do not retain per-example rule firings, so this comparison cannot establish absence of lexical/rule leakage.

## Obfuscation And Cost

Genos' obfuscation detector flagged 136 test commands; only one was changed by the current deobfuscation passes. The small obfuscated slice does not support a reliable ranking: both paired macro-F1 intervals cross zero. The detector-defined slice is not an independent obfuscation annotation.

Single-command gate latency, including tokenizer/vectorizer work: CodeBERT on CUDA, median 5.79 ms and p95 5.86 ms; TF-IDF on CPU, median 1.02 ms and p95 1.09 ms. These are gate-only timings, not full API latency. The optional TF-IDF gate does not remove the separate CodeBERT behavior encoder from the full Genos pipeline.

## Interpretation

TF-IDF is a fast, inspectable CPU gate, but it is not equivalent to CodeBERT on this split. CodeBERT's macro-F1 advantage is modest overall and statistically consistent under the recorded group bootstrap; the context-dependent class accounts for much of the absolute model weakness. On the obfuscated subset, results are inconclusive. All metrics are against weak labels from a single prepared environment, not independently reviewed ground truth or production behavior.

The TF-IDF backend is optional and CodeBERT remains the default. To test the alternative gate:

```bash
GENOS_GATEKEEPER_BACKEND=tfidf \
GENOS_GATEKEEPER_TFIDF_PATH=models/experiments/tfidf_gatekeeper_20261004_seed42/gatekeeper_tfidf.joblib
```

Comparison report and per-row predictions: `models/experiments/tfidf_gatekeeper_20261004_seed42/codebert_comparison_with_latency.json` and its `_predictions.jsonl` companion. Rule baseline: `models/experiments/tfidf_gatekeeper_20261004_seed42/rule_baseline.json`.
