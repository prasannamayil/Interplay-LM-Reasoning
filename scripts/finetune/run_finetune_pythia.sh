#!/bin/bash
# =============================================================================
# Finetune Pythia (AR) on Alpaca -- 8x H100 GPUs
# =============================================================================
# Usage:
#   bash scripts/finetune/run_finetune_pythia.sh [MODEL_SIZE]
#   MODEL_SIZE: 1.4b, 2.8b (default), 6.9b
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SIZE="${1:-2.8b}"
MODEL="EleutherAI/pythia-${SIZE}"
DATASET_SPEC="${DATASET_SPEC:-tatsu-lab/alpaca}"
DATASET_TAG="${DATASET_TAG:-alpaca}"
LOAD_PREPROCESSED_DATA="${LOAD_PREPROCESSED_DATA:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LEARNING_RATE="${PYTHIA_LR:-2e-5}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-${DATASET_TAG}}"
NGPUS=8

echo "============================================================"
echo "Finetuning Pythia-${SIZE} on SFT dataset"
echo "  Model:  ${MODEL}"
echo "  Dataset: ${DATASET_SPEC}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs:   ${NGPUS}"
echo "============================================================"

cd "$PROJECT_ROOT"

CMD=(
    torchrun --nproc_per_node="${NGPUS}"
    scripts/finetune/finetune_pythia.py
    --model_name_or_path "${MODEL}"
    --dataset "${DATASET_SPEC}"
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 4
    --learning_rate "${LEARNING_RATE}"
    --warmup_ratio 0.03
    --save_steps "${SAVE_STEPS}"
    --eval_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --max_length "${MAX_LENGTH}"
)

if [[ "${LOAD_PREPROCESSED_DATA}" == "1" ]]; then
    CMD+=(--load_preprocessed_data)
fi

if [[ "${NO_FSDP:-0}" == "1" ]]; then
    CMD+=(--no_fsdp)
fi

"${CMD[@]}"

echo "Done: ${OUTPUT_DIR}"
