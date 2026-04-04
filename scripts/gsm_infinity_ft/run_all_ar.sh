#!/bin/bash
# =============================================================================
# Script 1: AR (Pythia-2.8b) -- Train + Eval on GSM-Infinity
# =============================================================================
# Finetune Pythia-2.8b as a causal LM on GSM-Infinity 10B nopack data,
# then evaluate on test_small with pass@1.
#
# Prerequisites: run scripts/gsm_infinity_ft/run_data_prep.sh first.
# Estimated time: ~12-16h on 8x H100
#
# Usage:
#   bash scripts/gsm_infinity_ft/run_all_ar.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia"
MODEL="EleutherAI/pythia-2.8b"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft/pythia-2.8b-ar"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft}"

# Verify data exists
if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    echo "Run: bash scripts/gsm_infinity_ft/run_data_prep.sh"
    exit 1
fi

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

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
    --warmup_ratio 0.03 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --bf16

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Eval AR on test_small (pass@1)
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating AR checkpoint-final (pass@1)"
echo "============================================================"

EVAL_OUTPUT="${PROJECT_ROOT}/results/gsm_infinity_ft/eval/pythia-2.8b-ar"

bash "${DLLM_ROOT}/examples/gsm_infinity/run_eval.sh" \
    "${OUTPUT_DIR}/checkpoint-final" \
    ar \
    "${EVAL_OUTPUT}/checkpoint-final" \
    1

echo ""
echo "============================================================"
echo "All done! Results at:"
echo "  Model:   ${OUTPUT_DIR}"
echo "  Eval:    ${EVAL_OUTPUT}/checkpoint-final/metrics.jsonl"
echo "============================================================"
