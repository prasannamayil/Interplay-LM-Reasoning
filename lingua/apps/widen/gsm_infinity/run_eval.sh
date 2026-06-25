#!/bin/bash
# =============================================================================
# Evaluate a single apps.widen checkpoint on GSM-Infinity (pass@128)
# =============================================================================
# Usage:
#   bash lingua/apps/widen/gsm_infinity/run_eval.sh <ckpt_dir> <output_dir> [n_samples]
#
# Example:
#   bash lingua/apps/widen/gsm_infinity/run_eval.sh \
#       lingua/saves/gsm_infinity/mtp_400M_gsm_.../checkpoints/0000005000 \
#       results/mtp_eval/mtp_400M_gsm_.../checkpoint-5000
# =============================================================================

set -e

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"

CKPT_DIR="${1:?Error: checkpoint directory required}"
OUTPUT_DIR="${2:?Error: output directory required}"
N_SAMPLES="${3:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_GEN_LEN="${MAX_GEN_LEN:-1024}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEST_DIR="${TEST_DIR:-$PROJECT_ROOT/data/composition_hf/test_small}"

source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH}"

cd "${LINGUA_ROOT}"

python -m apps.widen.gsm_infinity.eval_pass128 \
    --ckpt_dir "${CKPT_DIR}" \
    --test_dir "${TEST_DIR}" \
    --n_samples "${N_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_gen_len "${MAX_GEN_LEN}" \
    --temperature "${TEMPERATURE}" \
    --max_tokens "${MAX_TOKENS}" \
    --batch_size "${BATCH_SIZE}"
