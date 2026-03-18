#!/bin/bash
# =============================================================================
# Finetune Pythia-BD3LM (diffusion) on Alpaca -- 8x H100 GPUs
# =============================================================================
# Uses existing dllm SFT infrastructure with the A2D-converted Pythia model.
#
# Usage:
#   bash scripts/finetune/run_finetune_bd3lm.sh [MODEL_SIZE] [BLOCK_SIZE]
#   MODEL_SIZE: 1.4b, 2.8b (default), 6.9b
#   BLOCK_SIZE: 32 (default)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
SIZE="${1:-2.8b}"
BLOCK_SIZE="${2:-32}"
MODEL_PATH="${DLLM_ROOT}/.models/a2d/pythia-${SIZE}"
DATASET_SPEC="${DATASET_SPEC:-tatsu-lab/alpaca}"
DATASET_TAG="${DATASET_TAG:-alpaca}"
LOAD_PREPROCESSED_DATA="${LOAD_PREPROCESSED_DATA:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
MAX_LENGTH="${MAX_LENGTH:-512}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs${BLOCK_SIZE}-${DATASET_TAG}}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: A2D model not found at ${MODEL_PATH}"
    echo "Run: bash scripts/finetune/run_convert_pythia.sh"
    exit 1
fi

echo "============================================================"
echo "Finetuning Pythia-${SIZE} BD3LM (bs=${BLOCK_SIZE}) on SFT dataset"
echo "  Model:  ${MODEL_PATH}"
echo "  Dataset: ${DATASET_SPEC}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs:   8"
echo "============================================================"

cd "$DLLM_ROOT"
export PYTHONPATH="${DLLM_ROOT}:${PYTHONPATH:-}"

CMD=(
    accelerate launch
    --config_file scripts/accelerate_configs/zero2.yaml
    examples/a2d/bd3lm/sft.py
    --model_name_or_path "${MODEL_PATH}"
    --dataset_args "${DATASET_SPEC}"
    --output_dir "${OUTPUT_DIR}"
    --max_length "${MAX_LENGTH}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --learning_rate 2e-5
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 4
    --block_size "${BLOCK_SIZE}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --logging_steps 10
    --report_to wandb
    --run_name "pythia-${SIZE}-bd3lm-bs${BLOCK_SIZE}-${DATASET_TAG}"
)

if [[ "${LOAD_PREPROCESSED_DATA}" == "1" ]]; then
    CMD+=(--load_preprocessed_data)
fi

"${CMD[@]}"

echo "Done: ${OUTPUT_DIR}"
