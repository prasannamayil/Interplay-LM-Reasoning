#!/bin/bash
# =============================================================================
# Build HARD-skewed tokenized diffusion data, MATCHED to the widen AR line.
#   - same source shards: data/composition_hf/train/{2..10}
#   - same per-op WEIGHTS as scripts/widen_line/build_hard_skewed_data.py
#       op2-4 = 0.2  /  op5-7 = 0.3  /  op8-10 = 0.5  (per-op within band)
#   - same text format (precache _compose_text_batch already emits
#       "<question> {problem} {question} </question> <solution> {body} </solution> <answer> {gold} </answer>")
#   - tokenizer a2d_qwen2_400M (= widen qwen2_400M base + <mask>)
#   - seq_len 1024  (widen GSM seq_len; NOT the diffusion default 2048)
#   - no_pack (one example per row, EOS appended) — matches the existing dllm recipe
# CPU-only job. Idempotent: precache_data.py skips if the output already exists.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

TOKEN_BUDGET="${TOKEN_BUDGET:-5B}"
SEQ_LEN="${SEQ_LEN:-1024}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/composition_hard_dllm_seq${SEQ_LEN}}"
A2D_MODEL_CONFIG="${DLLM_ROOT}/model_configs/a2d_qwen2_400M"
# per-op fractions: 0.2 easy (op2-4) / 0.3 medium (op5-7) / 0.5 hard (op8-10)
OP_WEIGHTS="${OP_WEIGHTS:-2:0.0667,3:0.0667,4:0.0667,5:0.10,6:0.10,7:0.10,8:0.1667,9:0.1667,10:0.1667}"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
cd "${DLLM_ROOT}"

echo "[build_hard_data] python: $(which python)"
echo "[build_hard_data] budget=${TOKEN_BUDGET} seq_len=${SEQ_LEN} -> ${OUTPUT_DIR}"
echo "[build_hard_data] op_weights=${OP_WEIGHTS}"

python examples/gsm_infinity/precache_data.py \
    --raw_data_dir "${PROJECT_ROOT}/data/composition_hf/train" \
    --tokenizer_path "${A2D_MODEL_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --token_budget "${TOKEN_BUDGET}" \
    --op_min 2 --op_max 10 \
    --op_weights "${OP_WEIGHTS}" \
    --seq_length "${SEQ_LEN}" \
    --num_proc 64 \
    --no_pack

echo "BUILD_HARD_DATA_DONE ${OUTPUT_DIR}"
