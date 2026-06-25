#!/bin/bash
# HTCondor executable for the preliminary "widen the line" experiment.
# Trains ONE architecture on GSM-Infinity (matched config, resumable) then evals
# pass@1 across ID+OOD ops. FIXED dump dir per arch => preemption + resubmit RESUMES
# from the latest checkpoint (no wasted compute). Reads $ARCH from the environment.
#
#   ARCH = dense | gqa | moe | looped | tokenformer   (eval-faithful archs)
set -uo pipefail

ARCH="${ARCH:?set ARCH in environment}"
PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
CONFIG="${LINGUA_ROOT}/apps/widen/gsm_infinity/configs/widen_exp_gsm.yaml"
TOKENIZER_PATH="${PROJECT_ROOT}/model_configs/qwen2_400M"
DATA_ROOT="${PROJECT_ROOT}/data/composition_lingua"
EXP_ROOT="${PROJECT_ROOT}/results/widen_line/exp_gsm"
RUN_NAME="widen_exp_${ARCH}"
DUMP_DIR="${EXP_ROOT}/${RUN_NAME}"          # FIXED (resumable)
EVAL_OUT="${EXP_ROOT}/eval/${ARCH}"
OPS="2,4,6,8,10,12,14,16,18,20"

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR="127.0.0.1"
# unique port per arch to avoid clashes if two land on one node
case "${ARCH}" in
    dense) PORT=29731;; gqa) PORT=29732;; moe) PORT=29733;;
    looped) PORT=29734;; tokenformer) PORT=29735;; *) PORT=29739;;
esac
case "${ARCH}" in
    gqa) OVR="model.arch_type=gqa model.n_kv_heads=2";;
    dense|moe|looped|tokenformer) OVR="model.arch_type=${ARCH}";;
    *) echo "bad ARCH ${ARCH}"; exit 2;;
esac

echo "############ widen exp arch=${ARCH} dump=${DUMP_DIR} ############"
nvidia-smi -L || true
mkdir -p "${DUMP_DIR}"
cd "${LINGUA_ROOT}"

echo "[${ARCH}] ---- TRAIN (resumable) ----"
torchrun --nproc_per_node=1 --master_addr="${MASTER_ADDR}" --master_port="${PORT}" \
    -m apps.widen.train \
    config="${CONFIG}" \
    name="${RUN_NAME}" \
    dump_dir="${DUMP_DIR}" \
    data.root_dir="${DATA_ROOT}" \
    data.tokenizer.path="${TOKENIZER_PATH}" \
    ${OVR}
TRC=$?
if [[ ${TRC} -ne 0 ]]; then echo "[${ARCH}] TRAIN rc=${TRC} (will resume on resubmit)"; exit ${TRC}; fi

CKPT="$(ls -d "${DUMP_DIR}"/checkpoints/[0-9]* 2>/dev/null | sort | tail -1)"
if [[ -z "${CKPT}" ]]; then echo "[${ARCH}] NO_CHECKPOINT"; exit 3; fi
echo "[${ARCH}] final checkpoint: ${CKPT}"

echo "[${ARCH}] ---- EVAL pass@1 ops ${OPS} ----"
python -m apps.widen.gsm_infinity.eval_pass128 \
    --ckpt_dir "${CKPT}" \
    --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
    --n_samples 1 \
    --output_dir "${EVAL_OUT}" \
    --max_gen_len 768 \
    --temperature 0.0 \
    --max_tokens 16384 \
    --batch_size 64 \
    --op_levels "${OPS}"
ERC=$?
if [[ ${ERC} -ne 0 ]]; then echo "[${ARCH}] EVAL rc=${ERC}"; exit ${ERC}; fi
echo "[${ARCH}] DONE eval metrics -> ${EVAL_OUT}/metrics.jsonl"
echo "WIDEN_EXP_DONE ${ARCH}"
