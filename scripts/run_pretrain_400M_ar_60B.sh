#!/bin/bash
# =============================================================================
# GSM-Infinity Pre-training — 400M Transformer (NTP) — 60B tokens (all data)
# =============================================================================
# Data: 60B tokens (natural distribution — NOT uniform), ~60K steps
# op3 is underrepresented (3B) while op2/9/10 are overrepresented (8B each)
# =============================================================================

set -e

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG="${PROJECT_ROOT}/LLaMA-Factory/examples/gsm_infinity/pt_400M_60B.yaml"
RUN_NAME="pt_400M_ar_60B_$(date +%Y%m%d_%H%M%S)"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29500}"

export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-pretrain}"
export WANDB_RUN_NAME="${RUN_NAME}"
export DISABLE_VERSION_CHECK=1

echo "=============================================="
echo "GSM-Infinity Pre-training — 400M AR — 60B tokens (~60K steps)"
echo "=============================================="
echo "Run name: ${RUN_NAME}"
echo "Config:   ${CONFIG}"
echo "GPUs:     ${GPU_LIST}"
echo ""
echo "WARNING: 60B uses ALL available data with NATURAL (non-uniform) distribution."
echo "  op3 only has 3B tokens (5%) while op2/9/10 have 8B each (13.3%)."
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
