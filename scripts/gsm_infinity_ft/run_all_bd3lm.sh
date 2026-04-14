#!/bin/bash
# =============================================================================
# BD3LM (A2D Pythia-2.8b) -- Train + Multi-Checkpoint Eval on GSM-Infinity
# =============================================================================
# Finetune A2D-Pythia-2.8b with BD3LM (block diffusion) on GSM-Infinity
# 10B nopack data. Runs ONE block_size config (set via BLOCK_SIZE env var,
# default 16), then evaluates ALL checkpoints with pass@1 and the best
# with pass@128.
#
# To run both block sizes, invoke twice:
#   BLOCK_SIZE=16 bash scripts/gsm_infinity_ft/run_all_bd3lm.sh
#   BLOCK_SIZE=1  bash scripts/gsm_infinity_ft/run_all_bd3lm.sh
#
# Changes from v1:
#   - Added --mask_prompt_loss True (only train on solution/answer tokens)
#   - Collator now uses NoAttentionMaskWrapper + label_pad_token_id=pad_id
#   - Save every 500 steps, keep 25 checkpoints
#   - Multi-checkpoint eval: pass@1 sweep + pass@128 on best
#   - Split into per-block-size invocations (no sequential dependency)
#
# Prerequisites: run scripts/gsm_infinity_ft/run_data_prep.sh first.
#
# Usage:
#   BLOCK_SIZE=16 bash scripts/gsm_infinity_ft/run_all_bd3lm.sh
#   BLOCK_SIZE=1  bash scripts/gsm_infinity_ft/run_all_bd3lm.sh
# =============================================================================

set -euo pipefail

BLOCK_SIZE="${BLOCK_SIZE:-16}"

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
MODEL_PATH="${DLLM_ROOT}/.models/a2d/pythia-2.8b"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft/pythia-2.8b-bd3lm-bs${BLOCK_SIZE}"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft}"

if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    echo "Run: bash scripts/gsm_infinity_ft/run_data_prep.sh"
    exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: A2D model not found at ${MODEL_PATH}"
    echo "Run: bash scripts/gsm_infinity_ft/run_data_prep.sh"
    exit 1
fi

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

# Move old results out of the way (if any)
if [[ -d "${OUTPUT_DIR}" ]]; then
    OLD="${OUTPUT_DIR}_old_$(date +%Y%m%d_%H%M%S)"
    echo "Moving old results: ${OUTPUT_DIR} -> ${OLD}"
    mv "${OUTPUT_DIR}" "${OLD}"
fi
EVAL_OUTPUT="${PROJECT_ROOT}/results/gsm_infinity_ft/eval/pythia-2.8b-bd3lm-bs${BLOCK_SIZE}"
if [[ -d "${EVAL_OUTPUT}" ]]; then
    OLD="${EVAL_OUTPUT}_old_$(date +%Y%m%d_%H%M%S)"
    echo "Moving old eval: ${EVAL_OUTPUT} -> ${OLD}"
    mv "${EVAL_OUTPUT}" "${OLD}"
fi

# =========================================================================
# Train BD3LM Pythia-2.8b (block_size=${BLOCK_SIZE})
# =========================================================================
echo "============================================================"
echo "Training BD3LM A2D-Pythia-2.8b (block_size=${BLOCK_SIZE})"
echo "  Model:   ${MODEL_PATH}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Output:  ${OUTPUT_DIR}"
echo "============================================================"

cd "${DLLM_ROOT}"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/gsm_infinity/pt_bd3lm.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_args "${DATASET_PATH}" \
    --load_preprocessed_data True \
    --max_length 2048 \
    --insert_eos True \
    --block_size "${BLOCK_SIZE}" \
    --max_steps 10000 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --bf16 True \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 25 \
    --eval_strategy "no" \
    --report_to wandb \
    --run_name "pythia-2.8b-bd3lm-bs${BLOCK_SIZE}-gsm-infinity" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Evaluate ALL checkpoints (pass@1 sweep + pass@128 on best)
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating all BD3LM bs${BLOCK_SIZE} checkpoints"
echo "============================================================"

cd "${PROJECT_ROOT}"
bash scripts/gsm_infinity_ft/run_eval_all_checkpoints.sh \
    "${OUTPUT_DIR}" \
    bd3lm \
    "${EVAL_OUTPUT}" \
    "${BLOCK_SIZE}"

echo ""
echo "============================================================"
echo "All done! Results at:"
echo "  Model:   ${OUTPUT_DIR}"
echo "  Eval:    ${EVAL_OUTPUT}/"
echo "============================================================"
