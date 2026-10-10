#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x venv/bin/python ]]; then
  echo "Missing venv/bin/python; run this from the Genos repository." >&2
  exit 1
fi

RUN_DIR="${RUN_DIR:-models/experiments/gatekeeper_two_class_codebert_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to reuse existing output directory: $RUN_DIR" >&2
  exit 1
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export GENOS_CODEBERT_BACKBONE="${GENOS_CODEBERT_BACKBONE:-models/codebert-base}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Repository: $ROOT_DIR"
echo "Output:     $RUN_DIR"
echo "Backbone:   $GENOS_CODEBERT_BACKBONE"
echo
echo "GPU processes before training:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

echo
echo "== Training two-class CodeBERT gatekeeper =="
venv/bin/python scripts/training/trainer1.py \
  --data-dir data/derived/scientific_v2/gatekeeper \
  --output-dir "$RUN_DIR" \
  --epochs "${GENOS_T1_EPOCHS:-5}" \
  --train-patch-jsonl data/training/genos_dataset/gatekeeper_benign_core_patch_v2a.jsonl \
  --num-workers 0 \
  --micro-batch-size "${GENOS_T1_MICRO_BATCH:-4}" \
  --effective-batch-size "${GENOS_T1_EFFECTIVE_BATCH:-128}" \
  --max-len "${GENOS_T1_MAX_LEN:-128}" \
  --log-interval 200

echo
echo "== Checkpoint smoke test =="
HF_HUB_OFFLINE="$HF_HUB_OFFLINE" TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
GENOS_CODEBERT_BACKBONE="$GENOS_CODEBERT_BACKBONE" \
GENOS_SPECIALIST_MODE=family \
venv/bin/python scripts/evaluation/smoke_runtime.py \
  --gatekeeper "$RUN_DIR/gatekeeper.pt" \
  --gatekeeper-meta "$RUN_DIR/gatekeeper_meta.json" \
  --output "$RUN_DIR/smoke_runtime.json"

echo
echo "== Unit tests and diff checks =="
venv/bin/python -m unittest discover -s tests -q
git diff --check

echo
echo "== Two-class metadata =="
venv/bin/python - "$RUN_DIR/gatekeeper_meta.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
meta = json.loads(path.read_text(encoding="utf-8"))
labels = meta.get("label_names")
if labels != ["Benign", "Context_Dependent"]:
    raise SystemExit(f"Unexpected label schema: {labels!r}")
print(json.dumps({
    "checkpoint": str(path.with_name("gatekeeper.pt")),
    "label_names": labels,
    "num_classes": meta.get("num_classes"),
    "test_metrics": meta.get("test_metrics"),
}, indent=2))
PY

echo
echo "Completed successfully: $RUN_DIR"
