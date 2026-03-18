#!/bin/bash
# =============================================================================
# Finetune Mamba (AR/SSM) on Alpaca -- 8x H100 GPUs
# =============================================================================
# Usage:
#   bash scripts/finetune/run_finetune_mamba.sh [MODEL_SIZE]
#
# MODEL_SIZE: 1.4b (default), 2.8b
#
# If Mamba loss stays very high vs Pythia, try a higher LR:
#   MAMBA_LR=1e-4 bash scripts/finetune/run_finetune_mamba.sh 2.8b
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SIZE="${1:-2.8b}"
MODEL="state-spaces/mamba-${SIZE}-hf"
DATASET_SPEC="${DATASET_SPEC:-tatsu-lab/alpaca}"
DATASET_TAG="${DATASET_TAG:-alpaca}"
LOAD_PREPROCESSED_DATA="${LOAD_PREPROCESSED_DATA:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
MAX_LENGTH="${MAX_LENGTH:-512}"
# Allow overriding output dir for LR sweeps or custom dataset tags.
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-${DATASET_TAG}}"
NGPUS=8
# Mamba can need a higher LR than Pythia; override with MAMBA_LR=5e-5 or 1e-4 if loss is stuck high
LEARNING_RATE="${MAMBA_LR:-2e-5}"

echo "============================================================"
echo "Finetuning Mamba-${SIZE} on SFT dataset"
echo "  Model:  ${MODEL}"
echo "  Dataset: ${DATASET_SPEC}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs:   ${NGPUS}"
echo "  LR:     ${LEARNING_RATE}"
echo "============================================================"

cd "$PROJECT_ROOT"

CMD=(
    torchrun --nproc_per_node="${NGPUS}"
    scripts/finetune/finetune_mamba.py
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

"${CMD[@]}"

echo "Done: ${OUTPUT_DIR}"
