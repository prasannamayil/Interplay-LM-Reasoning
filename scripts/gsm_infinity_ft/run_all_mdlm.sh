#!/bin/bash
# =============================================================================
# Script 2: MDLM (A2D Pythia-2.8b) -- Train + Eval on GSM-Infinity
# =============================================================================
# Finetune A2D-Pythia-2.8b with MDLM (masked diffusion) on GSM-Infinity
# 10B nopack data, then evaluate on test_small with pass@1.
#
# Prerequisites: run scripts/gsm_infinity_ft/run_data_prep.sh first.
# Estimated time: ~12-16h on 8x H100
#
# Usage:
#   bash scripts/gsm_infinity_ft/run_all_mdlm.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia"
MODEL_PATH="${DLLM_ROOT}/.models/a2d/pythia-2.8b"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft/pythia-2.8b-mdlm"

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

# =========================================================================
# Train MDLM Pythia-2.8b
# =========================================================================
echo "============================================================"
echo "Training MDLM A2D-Pythia-2.8b on GSM-Infinity"
echo "  Model:   ${MODEL_PATH}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Output:  ${OUTPUT_DIR}"
echo "============================================================"

cd "${DLLM_ROOT}"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/gsm_infinity/pt_mdlm.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_args "${DATASET_PATH}" \
    --load_preprocessed_data True \
    --max_length 2048 \
    --insert_eos True \
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
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --eval_strategy "no" \
    --report_to wandb \
    --run_name "pythia-2.8b-mdlm-gsm-infinity" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Eval MDLM on test_small (pass@1)
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating MDLM checkpoint-final (pass@1)"
echo "============================================================"

EVAL_OUTPUT="${PROJECT_ROOT}/results/gsm_infinity_ft/eval/pythia-2.8b-mdlm"

bash "${DLLM_ROOT}/examples/gsm_infinity/run_eval.sh" \
    "${OUTPUT_DIR}/checkpoint-final" \
    mdlm \
    "${EVAL_OUTPUT}/checkpoint-final" \
    1

echo ""
echo "============================================================"
echo "All done! Results at:"
echo "  Model:   ${OUTPUT_DIR}"
echo "  Eval:    ${EVAL_OUTPUT}/checkpoint-final/metrics.jsonl"
echo "============================================================"
