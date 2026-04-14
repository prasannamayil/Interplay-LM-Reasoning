#!/bin/bash
# =============================================================================
# BD3LM (A2D Pythia-160M) -- Train + Eval on GSM-Infinity
# =============================================================================
# Finetune A2D-Pythia-160M with BD3LM on GSM-Infinity 10B data.
# Uses same hyperparameters as the working Qwen2-100M from-scratch setup:
#   - lr=1e-4, warmup=0.05, batch=512 seqs/step (~1M tokens/step)
#   - 10K steps = ~10B tokens
#   - Block size set via BLOCK_SIZE env var (default 16)
#
# Run separately for each block size:
#   BLOCK_SIZE=16 bash scripts/gsm_infinity_ft_160m/run_all_bd3lm.sh
#   BLOCK_SIZE=32 bash scripts/gsm_infinity_ft_160m/run_all_bd3lm.sh
#
# Usage:
#   BLOCK_SIZE=16 bash scripts/gsm_infinity_ft_160m/run_all_bd3lm.sh
# =============================================================================

set -euo pipefail

BLOCK_SIZE="${BLOCK_SIZE:-16}"

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
MODEL_PATH="${DLLM_ROOT}/.models/a2d/pythia-160m"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_160m/pythia-160m-bd3lm-bs${BLOCK_SIZE}"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-160m}"

if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Dataset not found. Run: bash scripts/gsm_infinity_ft/run_data_prep.sh"
    exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: A2D model not found. Run: bash scripts/gsm_infinity_ft_160m/run_data_prep.sh"
    exit 1
fi

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ -d "${OUTPUT_DIR}" ]]; then
    OLD="${OUTPUT_DIR}_old_$(date +%Y%m%d_%H%M%S)"
    echo "Moving old results: ${OUTPUT_DIR} -> ${OLD}"
    mv "${OUTPUT_DIR}" "${OLD}"
fi

# BD3LM doubles effective seq length (x_t + x_0), so halve batch size, double grad accum
# to keep same effective batch of 512 seqs/step
echo "============================================================"
echo "Training BD3LM A2D-Pythia-160M (block_size=${BLOCK_SIZE})"
echo "  Model:   ${MODEL_PATH}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Batch:   32 x 2 x 8 GPUs = 512 seqs/step (~1M tokens/step)"
echo "  Steps:   10,000 (~10B tokens)"
echo "============================================================"

cd "${DLLM_ROOT}"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/gsm_infinity/pt_bd3lm.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_args "${DATASET_PATH}" \
    --load_preprocessed_data True \
    --max_length 2048 \
    --insert_eos True \
    --block_size "${BLOCK_SIZE}" \
    --max_steps 10000 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --bf16 True \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 25 \
    --eval_strategy "no" \
    --report_to wandb \
    --run_name "pythia-160m-bd3lm-bs${BLOCK_SIZE}-gsm-infinity" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Evaluate: pass@1 sweep + pass@16 on 8 checkpoints
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating all BD3LM bs${BLOCK_SIZE} checkpoints"
echo "============================================================"

EVAL_OUTPUT="${PROJECT_ROOT}/results/gsm_infinity_ft_160m/eval/pythia-160m-bd3lm-bs${BLOCK_SIZE}"

cd "${PROJECT_ROOT}"
PASS_K=16 bash scripts/gsm_infinity_ft/run_eval_all_checkpoints.sh \
    "${OUTPUT_DIR}" \
    bd3lm \
    "${EVAL_OUTPUT}" \
    "${BLOCK_SIZE}"

echo ""
echo "============================================================"
echo "All done! Results at:"
echo "  Model:   ${OUTPUT_DIR}"
echo "  Eval:    ${EVAL_OUTPUT}/"
echo "============================================================"
