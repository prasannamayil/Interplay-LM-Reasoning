#!/bin/bash
# =============================================================================
# GSM-Infinity Pre-training — 400M Transformer (NTP) — 30B tokens
# =============================================================================
# Data: 30B tokens (uniform ~3.3B/op), ~30K steps
# =============================================================================

set -e

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG="${PROJECT_ROOT}/LLaMA-Factory/examples/gsm_infinity/pt_400M_30B.yaml"
RUN_NAME="pt_400M_ar_30B_$(date +%Y%m%d_%H%M%S)"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29500}"

export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-pretrain}"
export WANDB_RUN_NAME="${RUN_NAME}"
export DISABLE_VERSION_CHECK=1

echo "=============================================="
echo "GSM-Infinity Pre-training — 400M AR — 30B tokens (~30K steps)"
echo "=============================================="
echo "Run name: ${RUN_NAME}"
echo "Config:   ${CONFIG}"
echo "GPUs:     ${GPU_LIST}"
echo ""

module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.1-cu12.x 2>/dev/null || true

source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT}"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

cd "${PROJECT_ROOT}/LLaMA-Factory"

llamafactory-cli train "${CONFIG}" \
    run_name="${RUN_NAME}" \
    output_dir="saves/gsm_infinity/${RUN_NAME}"

echo "Training complete! Output: LLaMA-Factory/saves/gsm_infinity/${RUN_NAME}"
