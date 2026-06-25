#!/bin/bash
# =============================================================================
# GSM-Infinity pre-training — apps.widen architecture zoo (Exp 3: widen the line)
# =============================================================================
# Trains one of several AR architectures on the IDENTICAL GSM-Infinity data
# (op 2-10 ID, test 2-20) so they can be compared on the same ID->OOD line.
#
# Usage:
#   bash lingua/apps/widen/gsm_infinity/run_pretrain.sh <arch> [config]
#     <arch>   = dense | gqa | moe | looped | sliding | linear | tokenformer
#     [config] = path to a config yaml (default: configs/widen_400M_gsm.yaml)
#
#   bash lingua/apps/widen/gsm_infinity/run_pretrain.sh preprocess   # data only
#
# Env vars:
#   GPU_LIST (0,1,2,3,4,5,6,7)  TOKEN_BUDGET (10B)  WANDB_PROJECT  MASTER_PORT
#   EXTRA_OVERRIDES  (extra OmegaConf dotlist args appended verbatim)
# =============================================================================
set -e

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

TOKEN_BUDGET="${TOKEN_BUDGET:-10B}"
RAW_DATA="${PROJECT_ROOT}/data/composition_hf/train"
TOKENIZER_PATH="${PROJECT_ROOT}/model_configs/qwen2_400M"
PROCESSED_DATA="${PROJECT_ROOT}/data/composition_lingua/gsm_infinity"
export WANDB_PROJECT="${WANDB_PROJECT:-lingua-gsm-infinity}"

setup_env() {
    module load cuda/12.1 2>/dev/null || true
    module load cudnn/8.9.1-cu12.x 2>/dev/null || true
    source "${VENV}"
    export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29611}"
    echo "[Setup] Python: $(which python)"; python --version
}

preprocess() {
    if [[ -d "${PROCESSED_DATA}" ]] && ls "${PROCESSED_DATA}"/gsm_infinity.chunk.*.jsonl 1>/dev/null 2>&1; then
        echo "[Preprocess] Data already exists at ${PROCESSED_DATA}, skipping."
        return
    fi
    echo "[Preprocess] GSM-Infinity -> ${PROCESSED_DATA} (budget ${TOKEN_BUDGET})"
    python -m apps.gsm_infinity.preprocess_data \
        --raw_data_dir "${RAW_DATA}" \
        --output_dir "${PROCESSED_DATA}" \
        --op_min 2 --op_max 10 \
        --token_budget "${TOKEN_BUDGET}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --lines_per_chunk 10000
}

# Map arch -> OmegaConf model overrides
arch_overrides() {
    case "$1" in
        dense)       echo "model.arch_type=dense" ;;
        gqa)         echo "model.arch_type=gqa model.n_kv_heads=2" ;;
        moe)         echo "model.arch_type=moe" ;;
        looped)      echo "model.arch_type=looped" ;;
        sliding)     echo "model.arch_type=sliding model.sliding_window=512" ;;
        linear)      echo "model.arch_type=linear" ;;
        tokenformer) echo "model.arch_type=tokenformer" ;;
        *) echo "UNKNOWN_ARCH" ;;
    esac
}

train() {
    local ARCH="$1"
    local CONFIG="${2:-${SCRIPT_DIR}/configs/widen_400M_gsm.yaml}"
    local OVR; OVR="$(arch_overrides "${ARCH}")"
    if [[ "${OVR}" == "UNKNOWN_ARCH" ]]; then
        echo "Unknown arch '${ARCH}'. Use dense|gqa|moe|looped|sliding|linear|tokenformer"; exit 1
    fi
    local RUN_NAME="widen_${ARCH}_400M_gsm_$(date +%Y%m%d_%H%M%S)"
    local DUMP_DIR="${LINGUA_ROOT}/saves/gsm_infinity/${RUN_NAME}"
    mkdir -p "${DUMP_DIR}"
    echo "=============================================="
    echo "Training apps.widen arch=${ARCH}  config=${CONFIG}"
    echo "  overrides: ${OVR}"
    echo "  GPUs: ${GPU_LIST} (${NPROC})   dump: ${DUMP_DIR}"
    echo "=============================================="
    cd "${LINGUA_ROOT}"
    torchrun \
        --nproc_per_node="${NPROC}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m apps.widen.train \
        config="${CONFIG}" \
        name="${RUN_NAME}" \
        dump_dir="${DUMP_DIR}" \
        data.root_dir="${PROJECT_ROOT}/data/composition_lingua" \
        data.tokenizer.path="${TOKENIZER_PATH}" \
        ${OVR} \
        ${EXTRA_OVERRIDES:-}
    echo "Training complete -> ${DUMP_DIR}"
}

COMMAND="${1:-}"
case "${COMMAND}" in
    preprocess)
        setup_env; cd "${LINGUA_ROOT}"; preprocess ;;
    dense|gqa|moe|looped|sliding|linear|tokenformer)
        setup_env; cd "${LINGUA_ROOT}"; preprocess; train "${COMMAND}" "${2:-}" ;;
    *)
        echo "Usage: $0 {preprocess|dense|gqa|moe|looped|sliding|linear|tokenformer} [config]" ;;
esac
