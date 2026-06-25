#!/bin/bash
# =============================================================================
# FineWeb-Edu pre-training — apps.widen architecture zoo (realistic-prose line)
# =============================================================================
# Usage:
#   bash lingua/apps/widen/configs_fineweb/run_pretrain.sh <arch> [size]
#     <arch> = dense | gqa | moe | looped | sliding | linear | tokenformer
#     [size] = 400M (default)   (config: widen_<size>_fineweb.yaml)
#
# Env: GPU_LIST, DATA_DIR (/fast/pmayilvahanan/lm_datasets/), WANDB_PROJECT,
#      MASTER_PORT, EXTRA_OVERRIDES
# Requires FineWeb data prepared (scripts/widen_line/fineweb_prep.sh).
# =============================================================================
set -e

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="${1:?arch required: dense|gqa|moe|looped|sliding|linear|tokenformer}"
SIZE="${2:-400M}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"
DATA_DIR="${DATA_DIR:-/fast/pmayilvahanan/lm_datasets/}"

CONFIG="${SCRIPT_DIR}/widen_${SIZE}_fineweb.yaml"
RUN_NAME="widen_${ARCH}_${SIZE}_fineweb_$(date +%Y%m%d_%H%M%S)"
DUMP_DIR="${LINGUA_ROOT}/saves/fineweb/${RUN_NAME}"
export WANDB_PROJECT="${WANDB_PROJECT:-lingua-fineweb}"

[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}"; exit 1; }

case "${ARCH}" in
    dense)       OVR="model.arch_type=dense" ;;
    gqa)         OVR="model.arch_type=gqa model.n_kv_heads=2" ;;
    moe)         OVR="model.arch_type=moe" ;;
    looped)      OVR="model.arch_type=looped" ;;
    sliding)     OVR="model.arch_type=sliding model.sliding_window=512" ;;
    linear)      OVR="model.arch_type=linear" ;;
    tokenformer) OVR="model.arch_type=tokenformer" ;;
    *) echo "Unknown arch '${ARCH}'"; exit 1 ;;
esac

module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29621}"

mkdir -p "${DUMP_DIR}"
echo "FineWeb widen arch=${ARCH} size=${SIZE}  ovr=${OVR}  dump=${DUMP_DIR}"
cd "${LINGUA_ROOT}"
torchrun \
    --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m apps.widen.train \
    config="${CONFIG}" \
    name="${RUN_NAME}" \
    dump_dir="${DUMP_DIR}" \
    data.root_dir="${DATA_DIR}" \
    ${OVR} \
    ${EXTRA_OVERRIDES:-}
echo "Training complete -> ${DUMP_DIR}"
