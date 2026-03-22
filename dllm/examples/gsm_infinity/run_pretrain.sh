#!/bin/bash
# =============================================================================
# GSM-Infinity Pre-training Run Script for DLLM Variants
# =============================================================================
# Pre-trains A2D-MDLM or A2D-BD3LM (~100M params) on GSM-Infinity data.
#
# Usage:
#   # Step 1: Preprocess data (only once)
#   bash dllm/examples/gsm_infinity/run_pretrain.sh preprocess
#
#   # Step 2: Train A2D-MDLM
#   bash dllm/examples/gsm_infinity/run_pretrain.sh mdlm
#
#   # Step 3: Train A2D-BD3LM
#   bash dllm/examples/gsm_infinity/run_pretrain.sh bd3lm
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

# GPU Configuration
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

# Accelerate config (zero2 recommended for ~100M model)
ACCEL_CONFIG="${ACCEL_CONFIG:-zero2}"

# Data budget and output path for pre-tokenized dataset
TOKEN_BUDGET="${TOKEN_BUDGET:-10B}"
TOKENIZED_DATA="${PROJECT_ROOT}/data/composition_hf_dllm_${TOKEN_BUDGET}"

# Wandb
export WANDB_PROJECT="${WANDB_PROJECT:-dllm-gsm-infinity}"

# =============================================================================
# Environment Setup
# =============================================================================
setup_env() {
    echo "[Setup] Activating environment..."
    source "${VENV}"
    export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
    # HuggingFace cache on lustre (avoids home directory quota)
    export HF_HOME="${PROJECT_ROOT}/.hf_cache"
    export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
    cd "${DLLM_ROOT}"
    echo "[Setup] Python: $(which python)"
    echo "[Setup] GPUs: ${GPU_LIST} (${NPROC} processes)"
}

# =============================================================================
# Step 0: Convert model config (if not already done)
# =============================================================================
convert_config() {
    echo "=============================================="
    echo "Converting Qwen2 100M config to A2D-Qwen2"
    echo "=============================================="

    if [ -f "${DLLM_ROOT}/model_configs/a2d_qwen2_100M/config.json" ]; then
        echo "Config already exists, skipping."
        return
    fi

    python examples/gsm_infinity/convert_config.py \
        --src_dir "${PROJECT_ROOT}/model_configs/qwen2_100M" \
        --output_dir "${DLLM_ROOT}/model_configs/a2d_qwen2_100M"
}

# =============================================================================
# Step 1: Pre-cache tokenized data (single process, no DDP timeout issues)
# =============================================================================
precache() {
    echo "=============================================="
    echo "Preprocessing ${TOKEN_BUDGET} tokenized data -> ${TOKENIZED_DATA}"
    echo "=============================================="

    python examples/gsm_infinity/precache_data.py \
        --raw_data_dir "${PROJECT_ROOT}/data/composition_hf/train" \
        --tokenizer_path "${DLLM_ROOT}/model_configs/a2d_qwen2_100M" \
        --output_dir "${TOKENIZED_DATA}" \
        --token_budget "${TOKEN_BUDGET}" \
        --op_min 2 --op_max 10 \
        --seq_length 2048 \
        --num_proc 64 \
        --no_pack
}

# =============================================================================
# Step 2a: Train A2D-MDLM
# =============================================================================
train_mdlm() {
    echo "=============================================="
    echo "Training A2D-MDLM 100M on GSM-Infinity"
    echo "=============================================="
    echo "GPUs: ${NPROC}, Accel config: ${ACCEL_CONFIG}"

    RUN_NAME="a2d_mdlm_100M_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${DLLM_ROOT}/saves/gsm_infinity/${RUN_NAME}"

    accelerate launch \
        --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
        --num_processes "${NPROC}" \
        examples/gsm_infinity/pt_mdlm.py \
        --model_name_or_path "${DLLM_ROOT}/model_configs/a2d_qwen2_100M" \
        --dataset_args "${TOKENIZED_DATA}" \
        --load_preprocessed_data True \
        --max_length 2048 \
        --insert_eos True \
        --max_steps 10000 \
        --learning_rate 1e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --bf16 True \
        --gradient_checkpointing False \
        --logging_steps 10 \
        --save_steps 500 \
        --save_total_limit 25 \
        --eval_strategy "no" \
        --report_to wandb \
        --run_name "${RUN_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        "$@"

    echo "Training complete! Output: ${OUTPUT_DIR}"
}

# =============================================================================
# Step 2b: Train A2D-BD3LM
# =============================================================================
train_bd3lm() {
    echo "=============================================="
    echo "Training A2D-BD3LM 100M on GSM-Infinity"
    echo "=============================================="
    echo "GPUs: ${NPROC}, Accel config: ${ACCEL_CONFIG}"

    RUN_NAME="a2d_bd3lm_100M_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${DLLM_ROOT}/saves/gsm_infinity/${RUN_NAME}"

    accelerate launch \
        --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
        --num_processes "${NPROC}" \
        examples/gsm_infinity/pt_bd3lm.py \
        --model_name_or_path "${DLLM_ROOT}/model_configs/a2d_qwen2_100M" \
        --dataset_args "${TOKENIZED_DATA}" \
        --load_preprocessed_data True \
        --max_length 2048 \
        --insert_eos True \
        --max_steps 10000 \
        --learning_rate 1e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 2 \
        --block_size 32 \
        --attn_implementation flex_attention \
        --bf16 True \
        --gradient_checkpointing True \
        --logging_steps 10 \
        --save_steps 500 \
        --save_total_limit 25 \
        --eval_strategy "no" \
        --report_to wandb \
        --run_name "${RUN_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        "$@"

    echo "Training complete! Output: ${OUTPUT_DIR}"
}

# =============================================================================
# Main dispatch
# =============================================================================
COMMAND="${1:-help}"
shift 2>/dev/null || true

case "${COMMAND}" in
    precache)
        setup_env
        convert_config
        precache
        ;;
    mdlm)
        setup_env
        convert_config
        precache
        train_mdlm "$@"
        ;;
    bd3lm)
        setup_env
        convert_config
        precache
        train_bd3lm "$@"
        ;;
    all)
        setup_env
        convert_config
        precache
        train_mdlm "$@"
        train_bd3lm "$@"
        ;;
    help|*)
        echo "Usage: $0 {precache|mdlm|bd3lm|all} [extra_args...]"
        echo ""
        echo "Commands:"
        echo "  precache    Pre-cache 10B tokenized data (single process, run once)"
        echo "  mdlm        Train A2D-MDLM variant (auto-runs precache first)"
        echo "  bd3lm       Train A2D-BD3LM variant (auto-runs precache first)"
        echo "  all         Run precache + both training variants"
        echo ""
        echo "Environment variables:"
        echo "  GPU_LIST       GPU IDs (default: 0,1,2,3,4,5,6,7)"
        echo "  ACCEL_CONFIG   Accelerate config name (default: zero2)"
        echo "  TOKEN_BUDGET   Token budget (default: 10B)"
        echo "  WANDB_PROJECT  Wandb project name (default: dllm-gsm-infinity)"
        ;;
esac


