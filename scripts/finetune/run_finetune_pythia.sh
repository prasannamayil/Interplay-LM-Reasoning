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
OUTPUT_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-alpaca"
NGPUS=8

echo "============================================================"
echo "Finetuning Pythia-${SIZE} on Alpaca"
echo "  Model:  ${MODEL}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs:   ${NGPUS}"
echo "============================================================"

cd "$PROJECT_ROOT"

torchrun --nproc_per_node=${NGPUS} \
    scripts/finetune/finetune_pythia.py \
    --model_name_or_path "${MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --save_steps 200 \
    --save_total_limit 20 \
    --max_length 512 \
    --bf16

echo "Done: ${OUTPUT_DIR}"
