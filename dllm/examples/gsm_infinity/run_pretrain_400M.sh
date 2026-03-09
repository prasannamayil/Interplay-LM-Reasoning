#!/bin/bash
# =============================================================================
# GSM-Infinity Pre-training — 400M Diffusion Models (A2D-MDLM / A2D-BD3LM)
# =============================================================================
# Architecture: 1024 hidden × 26 layers × 4096 FFN ≈ 398M params
# Data: 10B tokens of GSM-Infinity (op range 2-10, all templates)
# Hardware: 8x H100 GPUs
#
# Usage:
#   # Step 1: Preprocess data (shared with other sizes; skip if already done)
#   bash dllm/examples/gsm_infinity/run_pretrain_400M.sh precache
#
#   # Step 2a: Train A2D-MDLM 400M
#   bash dllm/examples/gsm_infinity/run_pretrain_400M.sh mdlm
#
#   # Step 2b: Train A2D-BD3LM 400M
#   bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm
#
#   # Both models sequentially
#   bash dllm/examples/gsm_infinity/run_pretrain_400M.sh all
#
# Note on memory: the 400M model is ~4x the memory of the 100M model.
#   If OOM, reduce per_device_train_batch_size and increase
#   gradient_accumulation_steps proportionally, or switch ACCEL_CONFIG=zero3.
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-/fast/pmayilvahanan/Interplay-LM-Reasoning}"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

# GPU Configuration
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

# Accelerate config — zero2 usually sufficient; use zero3 if OOM
ACCEL_CONFIG="${ACCEL_CONFIG:-zero2}"

# Data budget and output path for pre-tokenized dataset
TOKEN_BUDGET="${TOKEN_BUDGET:-10B}"
TOKENIZED_DATA="${PROJECT_ROOT}/data/composition_hf_dllm_${TOKEN_BUDGET}"

# Model config dirs
AR_MODEL_CONFIG="${PROJECT_ROOT}/model_configs/qwen2_400M"
A2D_MODEL_CONFIG="${DLLM_ROOT}/model_configs/a2d_qwen2_400M"

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
    export HF_HOME="${PROJECT_ROOT}/.hf_cache"
    export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
    cd "${DLLM_ROOT}"
    echo "[Setup] Python:   $(which python)"
    echo "[Setup] GPUs:     ${GPU_LIST} (${NPROC} processes)"
    echo "[Setup] Model:    ${A2D_MODEL_CONFIG}"
}

# =============================================================================
# Step 0: Convert model config (if not already done)
# =============================================================================
convert_config() {
    echo "=============================================="
    echo "Converting Qwen2 400M config to A2D-Qwen2"
    echo "=============================================="

    if [[ -f "${A2D_MODEL_CONFIG}/config.json" ]]; then
        echo "Config already exists at ${A2D_MODEL_CONFIG}, skipping."
        return
    fi

    python examples/gsm_infinity/convert_config.py \
        --src_dir "${AR_MODEL_CONFIG}" \
        --output_dir "${A2D_MODEL_CONFIG}"
}

# =============================================================================
# Step 1: Pre-cache tokenized data
# Tokenized data is shared across model sizes — skip if already cached.
# =============================================================================
precache() {
    echo "=============================================="
    echo "Preprocessing ${TOKEN_BUDGET} tokenized data -> ${TOKENIZED_DATA}"
    echo "=============================================="

    if [[ -d "${TOKENIZED_DATA}" ]]; then
        echo "Tokenized data already exists at ${TOKENIZED_DATA}, skipping."
        return
    fi

    python examples/gsm_infinity/precache_data.py \
        --raw_data_dir "${PROJECT_ROOT}/data/composition_hf/train" \
        --tokenizer_path "${A2D_MODEL_CONFIG}" \
        --output_dir "${TOKENIZED_DATA}" \
        --token_budget "${TOKEN_BUDGET}" \
        --op_min 2 --op_max 10 \
        --seq_length 2048 \
        --num_proc 64
}

# =============================================================================
# Step 2a: Train A2D-MDLM 400M
# Batch: 16/GPU × accum=4 × 8 GPUs × 2048 tokens ≈ 1M tokens/step
# gradient_checkpointing=True (wider + deeper than 100M)
# =============================================================================
train_mdlm() {
    echo "=============================================="
    echo "Training A2D-MDLM 400M on GSM-Infinity"
    echo "  hidden=1024, layers=26, FFN=4096  →  ~398M params"
    echo "  batch=16/GPU, accum=4, grad_ckpt=True"
    echo "=============================================="
    echo "GPUs: ${NPROC}, Accel config: ${ACCEL_CONFIG}"

    RUN_NAME="a2d_mdlm_400M_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${DLLM_ROOT}/saves/gsm_infinity/${RUN_NAME}"

    accelerate launch \
        --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
        --num_processes "${NPROC}" \
        examples/gsm_infinity/pt_mdlm.py \
        --model_name_or_path "${A2D_MODEL_CONFIG}" \
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
# Step 2b: Train A2D-BD3LM 400M
# Batch: 8/GPU × accum=8 × 8 GPUs × 2048 tokens ≈ 1M tokens/step
# flex_attention exploits block-sparse mask for faster attention
# =============================================================================
train_bd3lm() {
    local BS="${BLOCK_SIZE:-16}"
    echo "=============================================="
    echo "Training A2D-BD3LM 400M on GSM-Infinity"
    echo "  hidden=1024, layers=26, FFN=4096  →  ~398M params"
    echo "  block_size=${BS}, grad_ckpt=True"
    echo "=============================================="
    echo "GPUs: ${NPROC}, Accel config: ${ACCEL_CONFIG}"

    RUN_NAME="a2d_bd3lm_400M_bs${BS}_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${DLLM_ROOT}/saves/gsm_infinity/${RUN_NAME}"

    accelerate launch \
        --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
        --num_processes "${NPROC}" \
        examples/gsm_infinity/pt_bd3lm.py \
        --model_name_or_path "${A2D_MODEL_CONFIG}" \
        --dataset_args "${TOKENIZED_DATA}" \
        --load_preprocessed_data True \
        --max_length 2048 \
        --insert_eos True \
        --max_steps 5000 \
        --learning_rate 1e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size 128 \
        --gradient_accumulation_steps 1 \
        --block_size "${BS}" \
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
        echo "  mdlm        Train A2D-MDLM 400M  (~398M params)"
        echo "  bd3lm       Train A2D-BD3LM 400M (~398M params)"
        echo "  all         Run precache + both training variants"
        echo ""
        echo "Environment variables:"
        echo "  GPU_LIST       GPU IDs (default: 0,1,2,3,4,5,6,7)"
        echo "  ACCEL_CONFIG   Accelerate config name (default: zero2; try zero3 if OOM)"
        echo "  BLOCK_SIZE     BD3LM block size (default: 16; try 8, 16, 32)"
        echo "  TOKEN_BUDGET   Token budget (default: 10B)"
        echo "  WANDB_PROJECT  Wandb project name (default: dllm-gsm-infinity)"
        echo ""
        echo "Examples:"
        echo "  BLOCK_SIZE=8  bash $0 bd3lm"
        echo "  BLOCK_SIZE=16 bash $0 bd3lm"
        echo "  BLOCK_SIZE=32 bash $0 bd3lm"
        echo "  BLOCK_SIZE=32 bash $0 bd3lm --max_steps 20000  # train longer"
        ;;
esac
