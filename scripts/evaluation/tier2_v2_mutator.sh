#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-artifacts/tier2_v2_mutator_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to reuse existing output directory: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "== Parser validation =="
PYTHONPATH=parser venv/bin/python parser/validate_parser.py

echo "== Rule and specialist-prior validation =="
PYTHONPATH=parser venv/bin/python parser/validate_hybrid_pipeline.py

echo "== Tier 2 v2 mutation matrix =="
PYTHONPATH=parser venv/bin/python scripts/evaluation/tier2_v2_mutator.py \
  --output "$RUN_DIR/mutation_matrix.jsonl"

echo "== Mutation report =="
cat "$RUN_DIR/mutation_matrix.jsonl"

echo
echo "Review this matrix before retraining Tier 2. Output: $RUN_DIR"
