#!/bin/bash
# =============================================================================
# Script 3: BD3LM (A2D Pythia-2.8b) -- Train + Eval on GSM-Infinity
# =============================================================================
# Finetune A2D-Pythia-2.8b with BD3LM (block diffusion) on GSM-Infinity
# 10B nopack data. Two runs:
#   1. block_size=16 (standard block diffusion)
#   2. block_size=1  (per-token, degenerates toward MDLM-like)
# Then evaluate both on test_small with pass@1.
#
# Prerequisites: run scripts/gsm_infinity_ft/run_data_prep.sh first.
# Estimated time: ~24-30h on 8x H100
#
# Usage:
#   bash scripts/gsm_infinity_ft/run_all_bd3lm.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia"
MODEL_PATH="${DLLM_ROOT}/.models/a2d/pythia-2.8b"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft}"

# Verify prerequisites
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

# Common training args
COMMON_ARGS=(
    --dataset_args "${DATASET_PATH}"
    --load_preprocessed_data True
    --max_length 2048
    --insert_eos True
    --max_steps 10000
    --learning_rate 2e-5
    --weight_decay 0.1
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --max_grad_norm 1.0
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 4
    --bf16 True
    --gradient_checkpointing True
    --logging_steps 10
    --save_steps 1000
    --save_total_limit 10
    --eval_strategy "no"
    --report_to wandb
    --attn_implementation sdpa
)

cd "${DLLM_ROOT}"

# =========================================================================
# Train BD3LM block_size=16
# =========================================================================
BS16_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft/pythia-2.8b-bd3lm-bs16"

echo "============================================================"
echo "Training BD3LM A2D-Pythia-2.8b (block_size=16)"
echo "  Model:   ${MODEL_PATH}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Output:  ${BS16_DIR}"
echo "============================================================"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/gsm_infinity/pt_bd3lm.py \
    --model_name_or_path "${MODEL_PATH}" \
    --block_size 16 \
    --run_name "pythia-2.8b-bd3lm-bs16-gsm-infinity" \
    --output_dir "${BS16_DIR}" \
    "${COMMON_ARGS[@]}"

echo "Training complete (block_size=16): ${BS16_DIR}"

# =========================================================================
# Train BD3LM block_size=1
# =========================================================================
BS1_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft/pythia-2.8b-bd3lm-bs1"

echo ""
echo "============================================================"
echo "Training BD3LM A2D-Pythia-2.8b (block_size=1)"
echo "  Model:   ${MODEL_PATH}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Output:  ${BS1_DIR}"
echo "============================================================"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/gsm_infinity/pt_bd3lm.py \
    --model_name_or_path "${MODEL_PATH}" \
    --block_size 1 \
    --run_name "pythia-2.8b-bd3lm-bs1-gsm-infinity" \
    --output_dir "${BS1_DIR}" \
    "${COMMON_ARGS[@]}"

echo "Training complete (block_size=1): ${BS1_DIR}"

# =========================================================================
# Eval BD3LM block_size=16 on test_small (pass@1)
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating BD3LM block_size=16 checkpoint-final (pass@1)"
echo "============================================================"

EVAL_BS16="${PROJECT_ROOT}/results/gsm_infinity_ft/eval/pythia-2.8b-bd3lm-bs16"

BLOCK_SIZE_BD3LM=16 bash "${DLLM_ROOT}/examples/gsm_infinity/run_eval.sh" \
    "${BS16_DIR}/checkpoint-final" \
    bd3lm \
    "${EVAL_BS16}/checkpoint-final" \
    1

# =========================================================================
# Eval BD3LM block_size=1 on test_small (pass@1)
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating BD3LM block_size=1 checkpoint-final (pass@1)"
echo "============================================================"

EVAL_BS1="${PROJECT_ROOT}/results/gsm_infinity_ft/eval/pythia-2.8b-bd3lm-bs1"

BLOCK_SIZE_BD3LM=1 bash "${DLLM_ROOT}/examples/gsm_infinity/run_eval.sh" \
    "${BS1_DIR}/checkpoint-final" \
    bd3lm \
    "${EVAL_BS1}/checkpoint-final" \
    1

echo ""
echo "============================================================"
echo "All done! Results at:"
echo "  BS16 model: ${BS16_DIR}"
echo "  BS16 eval:  ${EVAL_BS16}/checkpoint-final/metrics.jsonl"
echo "  BS1 model:  ${BS1_DIR}"
echo "  BS1 eval:   ${EVAL_BS1}/checkpoint-final/metrics.jsonl"
echo "============================================================"
