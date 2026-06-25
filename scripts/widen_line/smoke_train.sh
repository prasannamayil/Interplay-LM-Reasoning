#!/bin/bash
# Smoke-test ONE apps.widen architecture on 1 GPU: ~30 steps of training on
# GSM-Infinity (tiny debug config) + a tiny generation eval. Asserts the run
# completes and a checkpoint is produced. Designed to be called from smoke_all.sh.
#
# Usage: bash smoke_train.sh <arch> <port> <out_root>
set -uo pipefail

ARCH="${1:?arch required}"
PORT="${2:-29711}"
OUT_ROOT="${3:-/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning/results/widen_line/smoke}"

PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
CONFIG="${LINGUA_ROOT}/apps/widen/gsm_infinity/configs/widen_debug_gsm.yaml"
TOKENIZER_PATH="${PROJECT_ROOT}/model_configs/qwen2_400M"
DATA_ROOT="${PROJECT_ROOT}/data/composition_lingua"
RUN_NAME="widen_smoke_${ARCH}"
DUMP_DIR="${OUT_ROOT}/${RUN_NAME}"

# arch -> model overrides (mirror run_pretrain.sh)
case "${ARCH}" in
    dense)       OVR="model.arch_type=dense" ;;
    gqa)         OVR="model.arch_type=gqa model.n_kv_heads=2" ;;
    moe)         OVR="model.arch_type=moe" ;;
    looped)      OVR="model.arch_type=looped" ;;
    sliding)     OVR="model.arch_type=sliding model.sliding_window=128" ;;
    linear)      OVR="model.arch_type=linear" ;;
    tokenformer) OVR="model.arch_type=tokenformer" ;;
    *) echo "[${ARCH}] UNKNOWN ARCH"; exit 2 ;;
esac

echo "############################################################"
echo "# SMOKE arch=${ARCH}  port=${PORT}  dump=${DUMP_DIR}"
echo "############################################################"
rm -rf "${DUMP_DIR}"
mkdir -p "${DUMP_DIR}"
cd "${LINGUA_ROOT}"

echo "[${ARCH}] ---- TRAIN ----"
torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="${PORT}" \
    -m apps.widen.train \
    config="${CONFIG}" \
    name="${RUN_NAME}" \
    dump_dir="${DUMP_DIR}" \
    data.root_dir="${DATA_ROOT}" \
    data.tokenizer.path="${TOKENIZER_PATH}" \
    ${OVR}
TRAIN_RC=$?
if [[ ${TRAIN_RC} -ne 0 ]]; then
    echo "[${ARCH}] TRAIN_FAILED rc=${TRAIN_RC}"; echo "RESULT ${ARCH} FAIL train"; exit 0
fi

# find latest checkpoint
CKPT="$(ls -d "${DUMP_DIR}"/checkpoints/[0-9]* 2>/dev/null | sort | tail -1)"
if [[ -z "${CKPT}" ]]; then
    echo "[${ARCH}] NO_CHECKPOINT"; echo "RESULT ${ARCH} FAIL no_ckpt"; exit 0
fi
echo "[${ARCH}] checkpoint: ${CKPT}"

echo "[${ARCH}] ---- EVAL (tiny generation, op2, n=2) ----"
python -m apps.widen.gsm_infinity.eval_pass128 \
    --ckpt_dir "${CKPT}" \
    --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
    --n_samples 2 \
    --output_dir "${DUMP_DIR}/eval_op2" \
    --max_gen_len 64 \
    --temperature 0.7 \
    --max_tokens 4096 \
    --batch_size 2 \
    --op_levels 2
EVAL_RC=$?
if [[ ${EVAL_RC} -ne 0 ]]; then
    echo "[${ARCH}] EVAL_FAILED rc=${EVAL_RC}"; echo "RESULT ${ARCH} FAIL eval"; exit 0
fi
if [[ -f "${DUMP_DIR}/eval_op2/metrics.jsonl" ]]; then
    echo "[${ARCH}] eval metrics written"; echo "RESULT ${ARCH} PASS"
else
    echo "[${ARCH}] eval metrics MISSING"; echo "RESULT ${ARCH} FAIL eval_metrics"
fi
exit 0
