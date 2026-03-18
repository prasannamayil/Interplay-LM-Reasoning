#!/bin/bash
# =============================================================================
# Pass@1 evaluation for ALL checkpoints in a BD3LM run (process+outcome;
# extra steps not penalized). Calls run_eval.sh once per checkpoint.
# =============================================================================
#
# Usage:
#   bash dllm/examples/gsm_infinity/run_eval_all_checkpoints_pass1.sh
#
# Or override paths:
#   CHECKPOINTS_ROOT=/path/to/run OUTPUT_BASE=/path/to/results bash .../run_eval_all_checkpoints_pass1.sh
#
# Defaults:
#   CHECKPOINTS_ROOT = /fast/bthambiraja/projects/Interplay-LM-Reasoning/dllm/saves/gsm_infinity/a2d_bd3lm_400M_bs32_20260313_173005
#   OUTPUT_BASE     = $PROJECT_ROOT/results/dllm_eval/a2d_bd3lm_400M_bs32_20260313_173005_pass1
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/fast/pmayilvahanan/Interplay-LM-Reasoning}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-/fast/bthambiraja/projects/Interplay-LM-Reasoning/dllm/saves/gsm_infinity/a2d_bd3lm_400M_bs32_20260313_173005}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/results/dllm_eval/a2d_bd3lm_400M_bs32_20260313_173005_pass1}"

export PROJECT_ROOT

if [ ! -d "${CHECKPOINTS_ROOT}" ]; then
    echo "Error: CHECKPOINTS_ROOT not found: ${CHECKPOINTS_ROOT}"
    exit 1
fi

# Discover all checkpoint-* and checkpoint-final directories
CHECKPOINTS=()
for d in "${CHECKPOINTS_ROOT}"/checkpoint-* "${CHECKPOINTS_ROOT}"/checkpoint-final; do
    [ -d "$d" ] || continue
    CHECKPOINTS+=( "$(basename "$d")" )
done
# Sort so checkpoint-1500, 3000, ... checkpoint-final order is sensible (checkpoint-final last)
CHECKPOINTS=( $(printf '%s\n' "${CHECKPOINTS[@]}" | sort -V) )

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "Error: No checkpoint directories found under ${CHECKPOINTS_ROOT}"
    exit 1
fi

echo "=============================================="
echo "Pass@1 eval for all checkpoints (process+outcome)"
echo "=============================================="
echo "CHECKPOINTS_ROOT: ${CHECKPOINTS_ROOT}"
echo "OUTPUT_BASE:      ${OUTPUT_BASE}"
echo "Checkpoints (${#CHECKPOINTS[@]}): ${CHECKPOINTS[*]}"
echo "=============================================="

for ckpt in "${CHECKPOINTS[@]}"; do
    MODEL_PATH="${CHECKPOINTS_ROOT}/${ckpt}"
    OUTPUT_DIR="${OUTPUT_BASE}/${ckpt}"
    echo ""
    echo ">>> Evaluating ${ckpt} -> ${OUTPUT_DIR}"
    bash "${SCRIPT_DIR}/run_eval.sh" "${MODEL_PATH}" bd3lm "${OUTPUT_DIR}" 1
done

echo ""
echo "=============================================="
echo "All checkpoints done. Results under: ${OUTPUT_BASE}/*/metrics.jsonl"
echo "=============================================="
