#!/bin/bash
# =============================================================================
# AR (Pythia-2.8b) -- Train + Multi-Checkpoint Eval on GSM-Infinity
# =============================================================================
# Finetune Pythia-2.8b as a causal LM on GSM-Infinity 10B nopack data,
# then evaluate ALL checkpoints with pass@1 and the best with pass@128.
#
# Changes from v1:
#   - Added weight_decay=0.1, max_grad_norm=1.0 (match small model config)
#   - Added --mask_prompt_loss (only train on solution/answer tokens)
#   - Save every 500 steps, keep 25 checkpoints
#   - Multi-checkpoint eval: pass@1 sweep + pass@128 on best
#
# Prerequisites: run scripts/gsm_infinity_ft/run_data_prep.sh first.
# Estimated time: ~12-16h training + ~4-8h eval on 8x H100
#
# Usage:
#   bash scripts/gsm_infinity_ft/run_all_ar.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
MODEL="${PROJECT_ROOT}/.models/pretrained/pythia-2.8b"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft/pythia-2.8b-ar"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft}"

if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
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
EVAL_OUTPUT="${PROJECT_ROOT}/results/gsm_infinity_ft/eval/pythia-2.8b-ar"
if [[ -d "${EVAL_OUTPUT}" ]]; then
    OLD="${EVAL_OUTPUT}_old_$(date +%Y%m%d_%H%M%S)"
    echo "Moving old eval: ${EVAL_OUTPUT} -> ${OLD}"
    mv "${EVAL_OUTPUT}" "${OLD}"
fi

# =========================================================================
# Train AR Pythia-2.8b
# =========================================================================
echo "============================================================"
echo "Training AR Pythia-2.8b on GSM-Infinity"
echo "  Model:   ${MODEL}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Output:  ${OUTPUT_DIR}"
echo "============================================================"

cd "${PROJECT_ROOT}"

torchrun --nproc_per_node=8 \
    scripts/gsm_infinity_ft/finetune_pythia_ar.py \
    --model_name_or_path "${MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 10000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.03 \
    --save_steps 500 \
    --save_total_limit 25 \
    --logging_steps 10 \
    --bf16

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Evaluate ALL checkpoints (pass@1 sweep + pass@128 on best)
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating all AR checkpoints"
echo "============================================================"

bash scripts/gsm_infinity_ft/run_eval_all_checkpoints.sh \
    "${OUTPUT_DIR}" \
    ar \
    "${EVAL_OUTPUT}"

echo ""
echo "============================================================"
echo "All done! Results at:"
echo "  Model:   ${OUTPUT_DIR}"
echo "  Eval:    ${EVAL_OUTPUT}/"
echo "============================================================"
