# Real Benign Provenance Inputs

Put real benign operational source files under `data/provenance/raw/`.

These files are intentionally ignored by git because they may contain:

- local shell history
- internal hostnames
- environment-specific paths
- operational command sequences from private systems

Supported source formats for `config/benign_provenance_sources.json`:

- `history`: shell-history style plain text
- `text`: one command per line
- `jsonl`: one JSON object per line with at least `command`

Recommended source types to start with:

- `local_shell_history`
- `local_kubernetes_history`
- `local_docker_history`
- `runbook_commands`
- `cicd_job_commands`
- `official_kubernetes_docs`
- `official_docker_docs`
- `official_cloud_docs`
- `package_maintenance_docs`

Recommended workflow:

1. Export one small trusted benign source into `data/provenance/raw/`, for example `my_dev_shell_history.txt`.
2. Sanitize obvious secrets before ingestion with `scripts/data/sanitize_shell_history.py`.
3. Reference the sanitized file from `config/benign_provenance_sources.json`.
4. Build the corpus with `scripts/data/build_real_benign_provenance_corpus.py`.
5. Evaluate the active checkpoint on the holdout with `scripts/benchmark/real_benign_holdout_benchmark.py` before retraining.

Example sanitization step:

```bash
cd /home/sam/genos/genos_api
source venv/bin/activate

python3 scripts/data/sanitize_shell_history.py \
	data/provenance/raw/my_dev_shell_history_raw.txt \
	data/provenance/raw/my_dev_shell_history.txt
```

The sanitizer masks obvious secrets while preserving command structure. By default it redacts:

- common token formats and bearer values
- `--token` / `--password` style flag values
- `KEY=value` assignments for common secret names
- private IPv4 addresses
- internal-looking domains such as `.internal`, `.corp`, `.svc`, `.cluster`
- usernames in `/home/<user>` paths and `user@host` targets

Minimal JSONL row example:

```json
{"command":"kubectl get pods -n prod","source_type":"official_kubernetes_docs","label_basis":"routine_operational_provenance","provenance_source":"kubernetes_docs","source_uri":"https://kubernetes.io/docs/","holdout_group":"official_kubernetes_docs","source_name":"official_kubernetes_docs_examples"}
```