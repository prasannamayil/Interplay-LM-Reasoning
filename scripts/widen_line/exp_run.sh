#!/bin/bash
# HTCondor executable for the "widen the line" experiment (multi-checkpoint version).
# Trains ONE architecture on GSM-Infinity (matched config, 12k steps, resumable) then
# evals pass@1 at SEVERAL checkpoints -> a SPREAD of (ID, OOD) points per arch so the
# OOD-vs-ID line can actually be seen (not just one cluster point).
# FIXED dump dir per arch => preemption + resubmit RESUMES from the latest checkpoint.
#   ARCH = dense | gqa | moe | looped | tokenformer   (eval-faithful archs)
set -uo pipefail

ARCH="${ARCH:?set ARCH in environment}"
PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
CONFIG="${LINGUA_ROOT}/apps/widen/gsm_infinity/configs/widen_exp_gsm.yaml"
TOKENIZER_PATH="${PROJECT_ROOT}/model_configs/qwen2_400M"
DATA_ROOT="${PROJECT_ROOT}/data/composition_lingua"
EXP_ROOT="${PROJECT_ROOT}/results/widen_line/exp_gsm"
RUN_NAME="widen_exp2_${ARCH}"               # v2 = 12k steps + multi-ckpt eval (fresh dump)
DUMP_DIR="${EXP_ROOT}/${RUN_NAME}"          # FIXED (resumable; same steps => consistent LR sched)
OPS="2,4,6,8,10,12,14,16,18,20"
EVAL_CKPTS="${EVAL_CKPTS:-2000 4000 8000 12000}"   # checkpoints to eval = the ID-spread points
MAX_EX="${MAX_EX:-100}"                      # examples/op (speed)

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR="127.0.0.1"
case "${ARCH}" in
    dense) PORT=29731;; gqa) PORT=29732;; moe) PORT=29733;;
    looped) PORT=29734;; tokenformer) PORT=29735;; *) PORT=29739;;
esac
case "${ARCH}" in
    gqa) OVR="model.arch_type=gqa model.n_kv_heads=2";;
    dense|moe|looped|tokenformer) OVR="model.arch_type=${ARCH}";;
    *) echo "bad ARCH ${ARCH}"; exit 2;;
esac

echo "############ widen exp2 arch=${ARCH} dump=${DUMP_DIR} ############"
nvidia-smi -L || true
mkdir -p "${DUMP_DIR}"
cd "${LINGUA_ROOT}"

echo "[${ARCH}] ---- TRAIN (12k steps, resumable) ----"
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

echo "[${ARCH}] ---- EVAL pass@1 at checkpoints: ${EVAL_CKPTS} ----"
for STEP in ${EVAL_CKPTS}; do
    CK="${DUMP_DIR}/checkpoints/$(printf '%010d' "${STEP}")"
    OUT="${EXP_ROOT}/eval/${ARCH}/ckpt-${STEP}"
    if [[ ! -d "${CK}" ]]; then echo "[${ARCH}] skip ckpt ${STEP} (missing)"; continue; fi
    if [[ -s "${OUT}/metrics.jsonl" ]]; then echo "[${ARCH}] ckpt ${STEP} already evaluated"; continue; fi
    echo "[${ARCH}] eval ckpt ${STEP} -> ${OUT}"
    python -m apps.widen.gsm_infinity.eval_pass128 \
        --ckpt_dir "${CK}" \
        --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
        --n_samples 1 \
        --output_dir "${OUT}" \
        --max_gen_len 768 \
        --temperature 0.0 \
        --max_tokens 16384 \
        --batch_size 64 \
        --op_levels "${OPS}" \
        --max_examples "${MAX_EX}" || { echo "[${ARCH}] eval ckpt ${STEP} FAILED"; exit 4; }
done
echo "[${ARCH}] DONE all-ckpt eval -> ${EXP_ROOT}/eval/${ARCH}/"
echo "WIDEN_EXP_DONE ${ARCH}"
