# 11-Family Specialist Results

## Dataset

The specialist now uses the 11 requested outputs: Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Discovery, Lateral Movement, Command-and-Control / Payload Retrieval, Exfiltration, Impact, and Benign Admin.

Family targets were derived from the local ATT&CK catalog's tactic associations and weak Gatekeeper Benign labels. Technique IDs are omitted from the emitted model rows and are not used as model classes or inference output. The catalog mapping omitted 500 attack rows whose techniques only map to excluded tactics. Nineteen exact commands with both Benign and attack-source labels were quarantined. The remaining data was split jointly by parser residual-template groups, not by the old independently assigned task splits:

| Split | Rows | Template groups |
|---|---:|---:|
| Train | 37,846 | 36,707 |
| Validation | 4,740 | 4,598 |
| Test | 4,740 | 4,600 |

There are 1,767 multi-label commands across the splits. Labels are still weak supervision; neither source label nor the tactic mapping is independently human-verified.

## Test Metrics

Fixed per-label decision threshold: 0.5. No thresholds were selected on test. CIs are 1,000 bootstrap resamples of residual-template groups.

| Family | Support | Precision | Recall | F1 | Group-bootstrap 95% CI for F1 |
|---|---:|---:|---:|---:|---:|
| Execution | 92 | 0.620 | 0.478 | 0.540 | [0.438, 0.628] |
| Persistence | 144 | 0.767 | 0.618 | 0.685 | [0.609, 0.751] |
| Privilege Escalation | 147 | 0.730 | 0.497 | 0.591 | [0.500, 0.667] |
| Defense Evasion | 280 | 0.770 | 0.561 | 0.649 | [0.596, 0.697] |
| Credential Access | 59 | 0.816 | 0.525 | 0.639 | [0.513, 0.750] |
| Discovery | 115 | 0.736 | 0.583 | 0.650 | [0.571, 0.726] |
| Lateral Movement | 7 | 1.000 | 0.143 | 0.250 | [0.000, 0.667] |
| Command-and-Control / Payload Retrieval | 39 | 0.808 | 0.538 | 0.646 | [0.500, 0.776] |
| Exfiltration | 9 | 1.000 | 0.111 | 0.200 | [0.000, 0.545] |
| Impact | 21 | 0.500 | 0.381 | 0.432 | [0.207, 0.625] |
| Benign Admin | 4,027 | 0.979 | 0.993 | 0.986 | [0.983, 0.988] |

Macro-F1: **0.570 [0.509, 0.628]**. Top-1 accuracy (any true family): **0.937 [0.928, 0.946]**. Top-2 accuracy (any true family): **0.960 [0.953, 0.967]**.

Benign Admin vs rest: precision 0.979, recall 0.993, F1 0.986; TP 3,998, FP 85, FN 29, TN 628.

Most frequent top-1 confusions include Defense Evasion → Benign Admin (46), Privilege Escalation → Benign Admin (27), Persistence → Benign Admin (18), and Execution ↔ Defense Evasion (15 and 13). Rare-family estimates are weak: test support is seven for Lateral Movement and nine for Exfiltration.

## Secondary Technique Metric

Technique-level top-5 remains a separate legacy metric, not a family-classifier output. Existing legacy raw-model results on their original weak-label grouped test split were 0.612 (seed 42), 0.613 (seed 43), and 0.612 (seed 44). This is a different label task and test artifact; it is not directly comparable to family top-1/top-2 or family F1.

## Runtime And Limitations

The family model is a multi-label char/word TF-IDF model with per-family sigmoid calibration using group-isolated training folds. Single-command family inference measured 2.92 ms median and 3.10 ms p95 on CPU, excluding API, behavior, and HTTP work. The rest of Genos still runs the gatekeeper and behavior encoder; this does not make the full API a sub-3-ms CPU system.

The family mode is now the default and omits `MITRE_codes`; `GENOS_SPECIALIST_MODE=mitre` explicitly selects the legacy technique ranker. The weak-label smoke examples reveal unresolved contradictions: `pwd` is Gatekeeper-Malicious but family-Benign Admin, and a benign-labeled `curl` download is also predicted Benign Admin. Do not treat these metrics as validated malicious-intent accuracy. Independent review and more examples for rare families are needed before operational claims.
