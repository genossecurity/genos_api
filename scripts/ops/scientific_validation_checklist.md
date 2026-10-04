# Scientific validation checklist

This checklist captures the required next steps before any deployment claim or runtime promotion.

## 1) Independent review of prepared queues

- [ ] Gatekeeper validation queue reviewed by two independent analysts
- [ ] Gatekeeper test queue reviewed by two independent analysts
- [ ] MITRE validation queue reviewed by two independent analysts
- [ ] MITRE test queue reviewed by two independent analysts
- [ ] Behavior validation queue reviewed by two independent analysts
- [ ] Behavior test queue reviewed by two independent analysts
- [ ] Disagreements adjudicated by a third independent reviewer
- [ ] Reviewed outputs saved under `data/derived/reviewed_v1/`
- [ ] Reviewer identities and provenance recorded

Required rubric:
- `observable_behavior`: actions visible in the command, without guessing intent
- `authorization`: `authorized`, `unauthorized`, or `unknown`
- `verdict`: `Benign`, `Malicious`, or `Context_Dependent`
- `mitre_codes`: list of supported techniques, or `[]`
- `annotator_id`, `command_id`, `family`, `source provenance`

Use the audit command:

```bash
venv/bin/python scripts/evaluation/audit_annotations.py \
  --review-a review_a.jsonl \
  --review-b review_b.jsonl \
  --adjudications adjudications.jsonl \
  --output-dir data/derived/reviewed_v1
```

## 2) Freeze reviewed labels

- [ ] Manage reviewed validation labels as frozen
- [ ] Manage reviewed test labels as frozen
- [ ] Do not use reviewed test data for threshold tuning
- [ ] Use reviewed validation data only for threshold or abstention selection
- [ ] Record the label basis used for final threshold fitting

## 3) Provenance and grouped holdout review

- [ ] Review family/source/template provenance for grouped validation and test data
- [ ] Confirm a clean group-wise holdout excluded from all training and development data
- [ ] Confirm zero normalized-command overlap remains in the final scientific splits
- [ ] Confirm no cross-task leakage remains in final benchmark design
- [ ] Record unresolved group uncertainty before making final claims

## 4) Clean retraining

- [ ] Retrain gatekeeper on clean grouped splits
- [ ] Retrain behavior models on clean grouped splits
- [ ] Keep experimental results in `models/experiments/...`
- [ ] Keep active runtime checkpoints unchanged until the final study is frozen
- [ ] Verify fixed experimental protocol is followed per `artifacts/scientific_validation/full_training_protocol.json`

Representative commands:

```bash
venv/bin/python scripts/training/trainer1.py \
  --data-dir data/derived/scientific_v2/gatekeeper \
  --stage-loss-weight 0 \
  --seed 42 \
  --output-dir models/experiments/my_study/gatekeeper_42
```

```bash
venv/bin/python scripts/training/train_behavior_encoder.py \
  --data-dir data/derived/scientific_v2/behavior \
  --input-format raw \
  --seed 42 \
  --epochs 4 \
  --output models/experiments/my_study/behavior_raw_42/behavior_encoder.pt
```

## 5) Validation-only threshold tuning

- [ ] Select action thresholds on reviewed validation data only
- [ ] Select abstention policy on reviewed validation data only
- [ ] Compute benign-to-malicious and benign-to-context trade-offs
- [ ] Record prevalence-aware metrics for deployment scenarios
- [ ] Do not modify threshold policy after final test evaluation

## 6) Calibration

- [ ] Fit calibration on reviewed validation data only
- [ ] Evaluate the calibrated pipeline once on the untouched final test
- [ ] Record NLL, multiclass Brier score, ECE, and reliability bins
- [ ] Do not refit after final test evaluation

Command:

```bash
venv/bin/python scripts/evaluation/calibrate_scores.py \
  --validation data/derived/validation_predictions.json \
  --test data/derived/test_predictions.json \
  --output data/derived/gatekeeper_calibration.json
```

## 7) Final verification

- [ ] Run evaluation regression tests
- [ ] Run runtime smoke test
- [ ] Confirm checkpoint signatures and preprocessing policy match the runtime
- [ ] Confirm no development examples are used to estimate deployment accuracy
- [ ] Confirm the `pwd` failure is documented, not hidden

Commands:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
venv/bin/python -m unittest discover -s scripts/evaluation -p 'test_*.py' -v
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
venv/bin/python scripts/evaluation/smoke_runtime.py --api --output /tmp/runtime_smoke.json
```

## 8) Deployment gate

- [ ] Independent reviewed labels are complete
- [ ] Provenance and group review is complete
- [ ] Thresholds are selected on validation only
- [ ] Calibration is fitted on validation only and evaluated once on final test
- [ ] Final test remains untouched
- [ ] Runtime smoke passes
- [ ] Infrastructure checks pass

Only then should the project consider promotion or deployment claims.
