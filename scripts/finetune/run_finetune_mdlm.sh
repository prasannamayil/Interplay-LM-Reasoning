#!/bin/bash
# =============================================================================
# Finetune Pythia-MDLM (diffusion) on Alpaca -- 8x H100 GPUs
# =============================================================================
# Uses existing dllm SFT infrastructure with the A2D-converted Pythia model.
#
# Usage:
#   bash scripts/finetune/run_finetune_mdlm.sh [MODEL_SIZE]
#   MODEL_SIZE: 1.4b, 2.8b (default), 6.9b
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
SIZE="${1:-2.8b}"
MODEL_PATH="${DLLM_ROOT}/.models/a2d/pythia-${SIZE}"
OUTPUT_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-mdlm-alpaca"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: A2D model not found at ${MODEL_PATH}"
    echo "Run: bash scripts/finetune/run_convert_pythia.sh"
    exit 1
fi

echo "============================================================"
echo "Finetuning Pythia-${SIZE} MDLM on Alpaca"
echo "  Model:  ${MODEL_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs:   8"
echo "============================================================"

cd "$DLLM_ROOT"
export PYTHONPATH="${DLLM_ROOT}:${PYTHONPATH:-}"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/a2d/mdlm/sft.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_args "tatsu-lab/alpaca" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length 512 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_steps 200 \
    --save_total_limit 20 \
    --bf16 True \
    --logging_steps 10 \
    --report_to wandb \
    --run_name "pythia-${SIZE}-mdlm-alpaca"

echo "Done: ${OUTPUT_DIR}"
