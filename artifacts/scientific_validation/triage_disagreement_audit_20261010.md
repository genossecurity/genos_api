# Command triage disagreement audit — 2026-10-10

## Scope

The active CodeBERT gatekeeper and active TF-IDF family specialist were run on 22 command probes and 300 rows from `data/derived/reviewed_v1/accepted.jsonl`. The gate was run without baseline shortcuts. The family model was run for every row to expose disagreements, including commands that the normal runtime would classify Benign and skip.

The 300 accepted rows are **provisional single-reviewer labels**. `scripts/ops/authorized_single_reviewer.py` generated both reviewer passes, and all 300 rows lack source context. They contain 275 `Context_Dependent`, 25 `Benign`, and no `Malicious` labels. Their 203 binary label disagreements with the gate are triage candidates, not an accuracy estimate. The 100% agreement in `data/derived/reviewed_v1/agreement.json` does not represent independent reviewer agreement.

## Findings

- All five bare short probes (`pwd`, `ls`, `hostname`, `env`, `id`) were routed Suspicious by the learned gate, with no high-risk feature override. The family model selected `Benign Admin` for all five.
- `pwd` had 22.25% gate Benign probability; `bash -c pwd` had 85.5%. `ls` had 45.56%; `ls -la` had 76.1%. These flips are evidence of input-form sensitivity, not proof of the underlying training cause.
- The active training data labels `pwd`, `ls`, `hostname`, `env`, and `id` Benign. `ps aux` and `crontab -l` have context-sensitive or conflicting source labels. Adding a literal `pwd` allowlist would hide the model failure without resolving similar cases.
- The existing experimental TF-IDF gate predicts `pwd`, `ls`, `hostname`, and `env` Benign, but its published weak-label macro-F1 is lower than CodeBERT's overall. This small probe set is insufficient to promote it.
- `python -c "print(1)"`, `powershell Get-Process`, and ordinary `scp` triggered broad deterministic feature overrides. Those rules represent dual-use behaviors and need a separately reviewed alert policy. They are a different issue from `pwd`, which had no override.
- Across all 322 rows, 34 gate/family disagreements were found. A model-blind queue of 236 unique commands was produced for targeted error review. Because the queue was selected using model output, it is **not** an untouched test set.

## Reproduce and resolve

Run `scripts/evaluation/audit_triage_disagreements.py` with `scripts/evaluation/triage_probe_cases.jsonl` and any additional command corpus. The script records gate probabilities, routing policy, family scores, provisional label status, and writes an optional blind review queue. Runtime responses now expose `triage_consistency.status=gate_family_disagreement` when the suspicious gate and benign-only family model conflict; this does not change the verdict.

Use the blind queue for targeted error analysis with two genuinely independent reviewers and recorded execution context where available. Maintain a separate random, source/template-grouped holdout that has not been selected using model predictions. After the verdict rubric is fixed, compare retrained gate candidates on short-command and wrapper slices, then tune broad routing overrides and thresholds on validation data. Report false-alert and miss rates on the untouched holdout at deployment prevalence before changing the active checkpoint or policy.
