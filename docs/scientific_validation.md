# Scientific validation

The runtime default now uses a multi-label 11-family TF-IDF specialist. This improves the output taxonomy, not the quality of the weak labels; independent operational accuracy is not established.

## Runtime contract

- Tier 1 labels are `Benign` and `Context_Dependent`; direct abuse is folded into `Context_Dependent` so every non-benign command receives specialist review. Specialist families and routing features retain the high-signal detail.
- Suspicious commands receive behavior and family analysis; benign commands skip the specialist. Family mode does not load the MITRE map/model and does not return `MITRE_codes`. Set `GENOS_SPECIALIST_MODE=mitre` only for legacy technique-ranking comparisons. Heuristic behavior fallback is disabled unless `GENOS_ALLOW_BEHAVIOR_FALLBACK=1`.
- `GENOS_VIEW_POLICY=mean` averages raw/decoded gatekeeper distributions; `raw` and `decoded` support experiments. Mean pooling is a declared engineering default, not a validated optimum. The family specialist receives the deobfuscated command. The behavior encoder retains its own configured representation.
- The API passes the original text to the engine. Bare Base64 decoding and wrapped-payload decoding are centralized. The gatekeeper retains training's lowercase/strip normalization. MITRE and behavior representations retain their own training formats.
- Legacy `confidence` and `class_probabilities` fields remain percentage-valued model estimates. `score_type` distinguishes uncalibrated estimates from validation temperature scaling. Temperature scaling preserves argmax; it is not a fix for wrong labels. Behavior action thresholds remain explicitly reported defaults pending validation.
- `GENOS_BEHAVIOR_POLICY_PATH` accepts per-action thresholds selected on validation exports by `scripts/evaluation/select_action_thresholds.py`. The artifact must match the runtime signature. Actions lacking positive or negative validation examples retain the explicit 0.5 default. Select this policy before fitting temperature calibration, so calibration binds to the finalized pipeline. Action probability estimates remain uncalibrated.
- Model heads load strictly. Metadata maps, token limits, and new MITRE artifact hashes/representations are checked. Legacy MITRE artifacts without metadata are explicitly marked unverified.
- `provenance` records checkpoint, implementation, metadata hashes, input format, view policy, optional deobfuscator availability, and execution details. Calibration files must match the runtime signature exactly.
- Family scores are independent calibrated probabilities, not a softmax distribution; multiple families may be selected. Technique-level top-5 remains a separate legacy secondary metric and must not be interpreted as family-classifier output.
- Parser/rule features are extracted evidence, not a causal explanation of the learned verdict. The UI labels scores accordingly.
- `deobfuscation_trace` records bounded transform views and stop reason. `evidence.indicator_records` records the view, span, and decoder for observed indicators. Stable-baseline bypasses have a null model confidence and `baseline_policy_no_model_score` score type; they skip only when cheap routing and IOC checks are clear.

## Audit and grouped datasets

`data/derived/scientific_v2/{gatekeeper,mitre,behavior}/manifest.json` records input/output SHA-256 hashes, label counts, quarantines, group overlap, and unseen evaluation labels. Original data is preserved. Generated data and local annotations are git-ignored; compact reports live in `artifacts/scientific_validation/`.

Gatekeeper source splits contained 40 train/validation, 80 train/test, and 10 validation/test overlapping normalized commands. There were 44 differing label sets among the 80 train/test overlaps. The training patch overlaps 194/500 expanded-benign and 84/100 routine-admin development examples.

The prepared gatekeeper corpus quarantines 131 rows with conflicting labels and 181 development-exposed rows. Remaining train/validation/test counts are 57,891 / 7,237 / 7,236. All prepared tasks have zero normalized command and recorded-group overlap. These are future-training splits: the active models have seen some reassigned rows and must not be evaluated on them as unseen data.

`artifacts/scientific_validation/cross_component_exposure.json` also audits between tasks: 276 specialist test commands occur in gatekeeper training, and 798 gatekeeper test commands occur in specialist training. These experiments evaluate individual components. A final end-to-end study needs a common source/template-grouped holdout excluded from every component’s training and development data.

Where explicit family/source groups are absent, the builder uses coarse first-executable groups and marks their origin as inferred. Connected family/source groups stay together. This reduces overlap but does not prove template, campaign, or source independence. Group sizes change class prevalence; report per-class metrics and macro-F1, not accuracy alone. Review these groups before a definitive study. Previously used test sets are development data for future tuning.

Reproduce into a new empty directory:

```bash
python3 scripts/data/prepare_scientific_splits.py --kind gatekeeper \
  --output-dir data/derived/new_study/gatekeeper \
  --benchmark-jsonl data/derived/scientific_v1/development_benchmarks.jsonl
# Repeat --kind mitre and --kind behavior into their own directories.
```

The exclusion snapshot was read from git's historical development-case generator without restoring the intentionally deleted benchmark scripts. Pass all additional known development cases through `--benchmark-jsonl`. No matching algorithm can discover unrecorded historical exposure.

## Independent annotation

The user confirmed that the prepared queues should be used and independent reviews are still needed. `artifacts/scientific_validation/annotation_status.json` records their hashes and pending status. MITRE and behavior queues are identical within each split; the same command review can support both tasks. Across the gatekeeper and specialist queues, nine commands cross validation/test roles. Keep tuning task-specific and do not describe their union as an untouched end-to-end test.

`data/derived/reviewed_v1/accepted.jsonl` contains 300 provisional gatekeeper labels generated by `scripts/ops/authorized_single_reviewer.py`. That script writes both reviewer passes from one authorized workflow. Its agreement score must not be treated as independent reviewer agreement or as operational accuracy ground truth.

Each dataset directory contains blind `annotation_queue.jsonl` (test) and `annotation_validation_queue.jsonl` files: no weak labels or model outputs. Obtain separate reviews from two analysts who did not author the labeling rules. Record observable behavior separately from authorization and verdict:

- `observable_behavior`: actions visible in the command, without guessing intent.
- `authorization`: `authorized`, `unauthorized`, or `unknown`; retain supporting source context.
- `verdict`: `Benign` or `Context_Dependent`, with a rationale. Document the boundary between routine operations and commands requiring context before reviewing.
- `mitre_codes`: all supported techniques; use `[]` when none applies. Do not infer maliciousness from a technique's presence.
- `annotator_id`, command ID, family, and source provenance: retain these for agreement and grouped evaluation.

Use a third analyst to resolve disagreements. Software does not adjudicate substantive conflicts:

```bash
venv/bin/python scripts/evaluation/audit_annotations.py \
  --review-a review_a.jsonl --review-b review_b.jsonl \
  --adjudications adjudications.jsonl --output-dir data/derived/reviewed_v1
```

This reports full agreement and verdict Cohen's kappa, and separates accepted examples from unresolved ones. Reviewer independence remains an attestation. Freeze reviewed test labels before model/threshold selection; use separate reviewed validation data for tuning.

## Controlled training

The current family model, dataset audit, grouped confidence intervals, per-family metrics, and known label conflicts are recorded in [family specialist results](../artifacts/scientific_validation/family_specialist_11_label_results.md). Rebuild the family dataset from the prepared source pools with `scripts/data/build_tactic_family_dataset.py`, then train with `scripts/training/train_family_specialist.py`. The emitted rows contain family labels and command/template data only; technique IDs are used only to convert the source labels and are omitted from the family model artifacts.

All active trainers check normalized-command conflicts and all supplied group axes before training. Supplemental patches and soft-label files are included in the gatekeeper check. Training writes experiment artifacts, not active runtime checkpoints.

```bash
venv/bin/python scripts/training/trainer_tfidf.py \
  --data-dir data/derived/scientific_v2/mitre --input-format raw \
  --seed 42 --output-dir models/experiments/my_study/mitre_raw_42
# Repeat with structured and seeds 43, 44 using distinct output directories.

venv/bin/python scripts/training/train_behavior_encoder.py \
  --data-dir data/derived/scientific_v2/behavior --input-format raw \
  --seed 42 --epochs 4 --output models/experiments/my_study/behavior_raw_42/behavior_encoder.pt
# Repeat with structured, identical training settings, and additional seeds.

venv/bin/python scripts/training/trainer1.py \
  --data-dir data/derived/scientific_v2/gatekeeper \
  --stage-loss-weight 0 --seed 42 \
  --output-dir models/experiments/my_study/gatekeeper_42
```

`GENOS_CODEBERT_BACKBONE` can point at a local pretrained CodeBERT directory containing config and safetensors. Use the same pretrained initialization across ablations; never initialize from a checkpoint already exposed to the evaluation examples. Local pilot runs used cached pretrained safetensors. The one-epoch behavior pilots are feasibility experiments, not convergence studies. The follow-up protocol in `artifacts/scientific_validation/full_training_protocol.json` fixes five gatekeeper epochs and four behavior epochs per representation; both full behavior runs use `--amp`. These are fixed budgets, not proof of convergence.

The MITRE trainer preserves the real `classes_` mapping and reports unseen-label examples rather than silently dropping them. It saves per-example scores for paired comparisons:

```bash
venv/bin/python scripts/evaluation/compare_ablations.py \
  --left models/experiments/my_study/mitre_raw_42/char_rf_test_predictions.jsonl \
  --right models/experiments/my_study/mitre_structured_42/char_rf_test_predictions.jsonl \
  --output artifacts/scientific_validation/my_comparison.json

venv/bin/python scripts/evaluation/rule_baseline.py \
  --dataset data/derived/scientific_v2/gatekeeper/gatekeeper_3class_test.csv \
  --output artifacts/scientific_validation/my_rule_baseline.json
```

The optional linear gatekeeper experiment uses the same grouped Gatekeeper splits and benign training patch, with group-aware cross-validated sigmoid calibration:

```bash
venv/bin/python scripts/training/train_tfidf_gatekeeper.py \
  --data-dir data/derived/scientific_v2/gatekeeper \
  --train-patch data/training/genos_dataset/gatekeeper_benign_core_patch_v2a.jsonl \
  --output-dir models/experiments/tfidf_gatekeeper_seed42

venv/bin/python scripts/evaluation/compare_tfidf_codebert_gatekeepers.py \
  --codebert-checkpoint-dir models/experiments/retrain-20261004-104842/gatekeeper/seed_42 \
  --tfidf-model models/experiments/tfidf_gatekeeper_seed42/gatekeeper_tfidf.joblib \
  --output artifacts/scientific_validation/tfidf_gatekeeper_comparison.json
```

CodeBERT remains the default. To try the fitted CPU gate in the runtime, set `GENOS_GATEKEEPER_BACKEND=tfidf` and `GENOS_GATEKEEPER_TFIDF_PATH` to the joblib artifact. This switches only gatekeeper verdict inference; behavior still uses its separate CodeBERT encoder.

The 2026-10-10 command-family migration collapses `Malicious` into `Context_Dependent` across the gatekeeper splits and patch. Exact routine commands are labeled `Benign`; process/network inspection, interpreters, remote transfer, and direct abuse are `Context_Dependent`. This is training coverage, not a runtime allowlist. The migrated two-class TF-IDF artifact is `models/experiments/gatekeeper_two_class_tfidf_20261010_v2/`; it reports validation/test macro-F1 of 0.865/0.875 with 0.88 ms median single-command gate latency. CodeBERT remains the Tier 1 backend; retraining it requires a PyTorch version accepted by Transformers' secure checkpoint loader. No active CodeBERT checkpoint was changed.

Do not mark every binary as benign. A binary-only prior is appropriate only for a reviewed inventory of low-risk commands and flags. Dual-use tools such as `ps`, `ss`, `ip`, `crontab`, `python`, `powershell`, `curl`, `wget`, `ssh`, `scp`, and `rsync` need context-sensitive examples and high-signal negative examples.

For a repeatable multi-seed run across the three components, use `scripts/ops/model_training_orchestrator.py`. It audits hashes, class counts, and split disjointness before training, then runs the trainers and creates `benchmark.json` from their validation/test metrics. Gatekeeper training includes the existing 250-example `gatekeeper_benign_core_patch_v2a.jsonl` by default; the preflight verifies it does not overlap the frozen evaluation splits, and the trainer records its hash. Use `--no-gatekeeper-train-patch` to omit it. The default is a dry run:

```bash
venv/bin/python scripts/ops/model_training_orchestrator.py \
  --phase all --run-dir models/experiments/retrain_2026_10_04
```

Add `--run` to execute. Defaults train seeds 42, 43, and 44; MITRE and behavior each compare raw and structured inputs; CUDA AMP is enabled for behavior training and ignored on CPU (`--no-amp` disables it). Progress streams to the terminal and `training.log`. Training has a six-hour wall-clock budget by default (`--max-runtime-hours`); it finishes the current fixed-epoch model before stopping at a job boundary. An interrupted partial checkpoint is preserved, and rerunning the same command starts that model in a numbered retry directory while skipping completed seeds. Use `--phase prepare --run` or `--phase all --prepare-data --run` to create new grouped splits under the run directory before training. The benchmark report identifies candidates using validation metrics only, keeps test metrics descriptive, and includes paired grouped MITRE representation comparisons when both prediction exports exist. Benchmark values retain their recorded weak-label basis; the script does not relabel data or promote checkpoints.

The regex baseline shares rule families with the weak labels. Its score measures agreement with those labels, not independently verified intent. Group-bootstrap intervals condition on the supplied groups/labels and one trained seed; report variation across seeds separately.

## Pipeline evaluation and calibration

Export the deployed preprocessing policy, not just a bare model's logits. Declare every training source, including patches. Non-development exports reject detected exposure. Hash records establish which files were checked, not that omitted training sources never existed.

```bash
venv/bin/python scripts/evaluation/evaluate_pipeline.py \
  --dataset independent_validation.jsonl --split validation \
  --training-data actual_training.csv --training-data actual_patch.jsonl \
  --view-policy mean --output data/derived/validation_predictions.json

venv/bin/python scripts/evaluation/calibrate_scores.py \
  --validation data/derived/validation_predictions.json \
  --test data/derived/test_predictions.json \
  --output data/derived/gatekeeper_calibration.json
```

Repeat evaluation separately for raw, decoded, and mean policies. `scripts/evaluation/compare_view_policies.py --raw raw.json --decoded decoded.json --mean mean.json --output comparison.json` compares these exports and reconstructs the retired 0.55/0.03 risk-selection baseline without enabling it in production. Compare them on validation data, then evaluate the selected policy once on a frozen test. Use `--component mitre` for single-label technique calibration and `--component behavior` for stage calibration. Merge separately fitted components with the fitter’s `--existing` option; signatures must match. Multilabel action scores remain uncalibrated. The fitter never fits to test data, checks validation/test separation and provenance, and reports NLL, multiclass Brier score, ECE, and reliability bins before/after scaling. An export already calibrated cannot be refitted. No calibration artifact is enabled for the existing runtime because independent validation is missing.

A deployable artifact is bound to the full runtime signature. Changing model weights, code, representation, view policy, or execution details invalidates it and requires a new validation export. Report the label basis used for fitting; weak-label calibration does not establish correctness against human ground truth.

For independently reviewed technique sets (including `[]`), use `scripts/evaluation/evaluate_mitre_relevance.py --annotations accepted.jsonl --predictions pipeline_export.json --output relevance.json`. It reports exact-code micro precision/recall, positive-command hit rate, and the rate of candidate techniques on no-technique examples. Missing labels are rejected rather than interpreted as negative examples.

## Release evidence still required

- Independent reviewed validation/test sets with source and template provenance, including routine benign traffic, dual-use cases, direct abuse, and no-technique examples.
- Fresh gatekeeper training on clean splits and a converged multi-seed behavior comparison. The current `pwd` smoke-test failure is retained in the runtime report.
- Behavior coverage and accuracy by verdict, parser failure and truncation slices, long/multiline/case-sensitive commands, wrapper transformations, and distribution shift.
- Precision/recall at deployment prevalence; benign-to-malicious and benign-to-context rates with group uncertainty; multilabel MITRE relevance and no-technique false positives.
- Validation-selected action thresholds or abstention policy, calibration on representative independent data, and an untouched final test.

Runtime and infrastructure tests: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python -m unittest discover -s scripts/evaluation -p 'test_*.py' -v`.

Repeat the real-checkpoint structural smoke check with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python scripts/evaluation/smoke_runtime.py --api --output /tmp/runtime_smoke.json`. This records predictions, including errors on development examples, without using those examples to estimate accuracy. Optional checkpoint and view-policy arguments allow isolated experimental checks.
