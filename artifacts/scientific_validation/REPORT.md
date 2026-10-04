# Scientific validation implementation and results

The runtime changes and initial experiments are complete; fresh five-epoch gatekeeper training and the fixed four-epoch behavior comparison are now in progress under `full_training_protocol.json`. Independent ground truth and production validity are not established. Active checkpoints have not been replaced by experimental models.

## Implemented

- Preserve `Context_Dependent` in the API and UI; behavior runs for every command.
- Use a declared raw/decoded/mean view policy and normalized score fusion across all learned outputs.
- Centralize Base64 preprocessing in the engine.
- Strictly load requested checkpoints and label maps; require explicit heuristic fallback.
- Remove dead thresholds, obsolete override regexes, and the embedded development harness.
- Return hashes, preprocessing/representation settings, actual model/fallback state, calibration status, and truncation flags.
- Reject overlapping/conflicting training inputs, including supplemental patches; retire benchmark-fed patch builders.
- Provide frozen grouped datasets, separate blind validation/test annotation queues, reviewer agreement/adjudication tooling, regex baselines, view-policy comparisons, paired uncertainty, MITRE no-technique/multilabel relevance, and validation-only calibration.

## Dataset checks

The gatekeeper splits now contain 57,891 training, 7,237 validation, and 7,236 test rows. 131 conflicting-label rows and 181 development-exposed rows were quarantined. All three tasks have zero normalized command or recorded-group overlap. Inferred first-executable groups are provisional and need source/template review. New splits cannot validate old checkpoints that have seen reassigned examples. Cross-component exposure is also present: 276 specialist test commands occur in gatekeeper training and 798 gatekeeper test commands occur in specialist training. These are component experiments, not an untouched end-to-end evaluation; see `cross_component_exposure.json`.

## MITRE experiment

New character TF-IDF + 400-tree RF models, three seeds, identical grouped splits, weak legacy labels. Macro-F1 is over the complete configured label map. Two test examples have a technique absent from training.

| Seed | RAW top-1 | Structured top-1 | Structured − RAW | Conditional 95% interval |
|---|---:|---:|---:|---|
| 42 | 37.88% | 34.83% | -3.05 pp | [-5.70, -0.40] pp |
| 43 | 39.21% | 34.57% | -4.64 pp | [-7.28, -1.85] pp |
| 44 | 39.34% | 34.44% | -4.90 pp | [-7.42, -2.12] pp |

Structured parser features hurt top-1 generalization in this specific setup. This does not establish that RAW is best in deployment. The 755 test rows have 755 inferred groups, so these intervals effectively resample individual examples; they do not account for undetected template/source relationships. These results are now development evidence if used to choose future models.

## Behavior pilot

Fresh CodeBERT initialization, one epoch, seed 42, batch size 8, identical grouped splits. This is a feasibility pilot rather than a converged or multi-seed comparison.

| Input | Stage accuracy | Stage macro-F1 | Action micro-F1 |
|---|---:|---:|---:|
| raw | 46.36% | 0.234 | 0.322 |
| structured | 45.17% | 0.221 | 0.978 |

Action labels derive from the parser/rule features supplied in structured input. High action F1 therefore supports the ability to reproduce those labels; it does not validate independent behavioral understanding.

## Verification

28 automated regression tests passed. Real local checkpoints load strictly. Flask API normalization, `/`, `/demo`, `/health`, raw/Base64 inference, and normalized gate/MITRE/behavior distributions passed integration checks. The frontend JavaScript compiled successfully; no browser rendering test was run. Python compilation, frozen data hashes, training rejection of overlapping legacy data, and git whitespace checks passed.

## Unresolved findings

The active model still predicts `pwd` as Malicious (52.36% uncalibrated score), with behavior Persistence. This failure is preserved in `runtime_smoke.json`; no rule was added to hide it.

Prepared annotation queues remain pending independent reviews, as confirmed by the user. Their hashes and cross-task overlap are recorded in `annotation_status.json`; no human labels have been invented.

Before making deployment-accuracy claims: obtain independent reviewed labels and provenance; review inferred groups; retrain the gatekeeper; run converged behavior experiments across seeds; fit the implemented action-threshold policy and any abstention policy on independent validation; fit and evaluate calibration; and evaluate once on an untouched final test. The current MITRE model always ranks known candidates and has no learned no-technique decision. No calibration artifact has been enabled.

Commands and annotation rubric: [scientific validation workflow](../../docs/scientific_validation.md).
