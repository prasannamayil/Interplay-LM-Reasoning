#!/bin/bash
# HTCondor executable: prepare FineWeb-Edu 10BT for lingua (download + parquet->jsonl
# + terashuf shuffle + 32 chunks). Heavy CPU/network/disk; run on a big-CPU node.
# Output: ${DATA_DIR}/fineweb_edu_10bt_shuffled/  (chunks + val).
set -uo pipefail

PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
DATA_DIR="${DATA_DIR:-/fast/pmayilvahanan/lm_datasets}"

source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/lingua:${PYTHONPATH:-}"

echo "=== FineWeb-Edu 10BT prep -> ${DATA_DIR} ==="
python --version
bash "${PROJECT_ROOT}/scripts/prepare_fineweb_data.sh" "${DATA_DIR}"
echo "FINEWEB_PREP_DONE"
