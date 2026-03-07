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
OUTPUT_DIR="${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-alpaca"
NGPUS=8
# Mamba can need a higher LR than Pythia; override with MAMBA_LR=5e-5 or 1e-4 if loss is stuck high
LEARNING_RATE="${MAMBA_LR:-2e-5}"

echo "============================================================"
echo "Finetuning Mamba-${SIZE} on Alpaca"
echo "  Model:  ${MODEL}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs:   ${NGPUS}"
echo "  LR:     ${LEARNING_RATE}"
echo "============================================================"

cd "$PROJECT_ROOT"

torchrun --nproc_per_node=${NGPUS} \
    scripts/finetune/finetune_mamba.py \
    --model_name_or_path "${MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio 0.03 \
    --save_steps 200 \
    --save_total_limit 20 \
    --max_length 512

echo "Done: ${OUTPUT_DIR}"
